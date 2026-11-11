"""Build once on the CPU Dataset notebook; save the Linux/T4 runtime in that Dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

from .common import HERE, elf_arch, extract, fingerprint, sha256, transfer, write_json


def command(arguments, *, cwd, env=None, log=None):
    print('Build:', ' '.join(map(str, arguments[:5])), flush=True)
    log = Path(log or Path(cwd) / 'build.log')
    with log.open('ab') as output:
        result = subprocess.run(list(map(str, arguments)), cwd=cwd, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text(errors='replace')[-5000:]
        raise RuntimeError(f'Build command failed ({result.returncode}); {log}\n{tail}')


def checkout(spec, destination, log):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / '.git').exists():
        command(['git', 'init', '-q', destination], cwd=destination, log=log)
    remotes = subprocess.check_output(['git', 'remote'], cwd=destination, text=True).splitlines()
    if 'origin' not in remotes:
        command(['git', 'remote', 'add', 'origin', spec['repository']], cwd=destination, log=log)
    elif subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=destination, text=True).strip() != spec['repository']:
        raise RuntimeError('Build checkout has an unexpected origin; existing files were retained: ' + str(destination))
    try:
        current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        current = ''
    if current != spec['commit']:
        command(['git', 'fetch', '--depth', '1', 'origin', spec['commit']], cwd=destination, log=log)
        command(['git', 'checkout', '--detach', spec['commit']], cwd=destination, log=log)
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
    if current != spec['commit']:
        raise RuntimeError('Pinned source checkout failed: ' + str(destination))
    return destination


def copy_runtime(build, binary, destination):
    destination.mkdir(parents=True, exist_ok=True)
    candidates = list(build.rglob(binary))
    candidates = [p for p in candidates if p.is_file() and not p.is_symlink()]
    if len(candidates) != 1:
        raise RuntimeError('Expected one built ' + binary + ', found ' + str(candidates))
    elf_arch(candidates[0])
    shutil.copy2(candidates[0], destination / binary)
    for library in build.rglob('*.so*'):
        if library.is_file() and library.name.startswith(('libggml', 'libllama', 'libwhisper', 'libmtmd')):
            target = destination / library.name
            if not target.exists():
                # Dereference into one real versioned file; preserve its alias as a local link.
                if library.is_symlink():
                    real = library.resolve()
                    if not (destination / real.name).exists():
                        shutil.copy2(real, destination / real.name)
                    target.symlink_to(real.name)
                else:
                    shutil.copy2(library, target)


def toolkit(work, sources, env):
    """NVIDIA's pinned redistributables provide nvcc without a GPU or root access."""
    root = work / 'cuda'
    ready = root / '.complete'
    selected = [r for r in sources['runtime_assets'] if r['id'].startswith('cuda_') or r['id'] in ('libcublas', 'libnvvm')]
    identity = fingerprint(selected)
    if not ready.exists() or ready.read_text() != identity:
        root.mkdir(exist_ok=True)
        for record in selected:
            marker = root / ('.asset-' + record['id'])
            if marker.exists() and marker.read_text() == record['sha256']:
                continue
            archive = transfer(record, work / record['name'])
            if record['id'] == 'cuda_driver_stub':
                # NVIDIA's tiny driver-dev package supplies a link-time stub;
                # no driver is installed or included in the saved runtime.
                entries = subprocess.check_output(['ar', 't', archive], text=True).splitlines()
                name = next(n for n in entries if n.startswith('data.tar'))
                data_archive = work / ('driver-' + name)
                with data_archive.open('wb') as data:
                    subprocess.run(['ar', 'p', archive, name], stdout=data, check=True)
                with tarfile.open(data_archive) as source:
                    member = next(m for m in source.getmembers() if m.isfile() and m.name.endswith('/lib/stubs/libcuda.so'))
                    destination = root / 'lib/stubs/libcuda.so'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream, destination.open('wb') as output:
                        shutil.copyfileobj(stream, output)
                data_archive.unlink()
                archive.unlink()
                marker.write_text(record['sha256'])
                continue
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    if not member.isfile():
                        continue
                    pieces = Path(member.name).parts[1:]
                    if not pieces or '..' in pieces or Path(member.name).is_absolute():
                        raise RuntimeError('Invalid CUDA redistributable path.')
                    relative = Path(*pieces)
                    # Static cuBLAS is enormous and is not used by the dynamic backend.
                    if relative.suffix == '.a' and record['id'] == 'libcublas':
                        continue
                    if not (pieces[0] in ('bin', 'include', 'lib', 'lib64', 'nvvm') or pieces[-1].startswith(('LICENSE', 'EULA'))):
                        continue
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream, target.open('wb') as output:
                        shutil.copyfileobj(stream, output)
                    target.chmod(member.mode & 0o777)
                # Recreate only same-directory shared-library aliases.
                for member in source.getmembers():
                    if not member.issym():
                        continue
                    relative = Path(*Path(member.name).parts[1:])
                    if Path(member.linkname).name != member.linkname:
                        continue
                    target = root / relative
                    if target.parent.exists() and not target.exists():
                        target.symlink_to(member.linkname)
            archive.unlink()  # This is the builder's own verified download cache.
            marker.write_text(record['sha256'])
        ready.write_text(identity)
    # NVIDIA archives can encode SONAME aliases as hard links. Reconstruct
    # them from the verified ELF files too, including the link-only driver stub.
    for library in (root / 'lib').rglob('*.so*'):
        if not library.is_file() or library.is_symlink():
            continue
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        soname = re.search(r'\(SONAME\).*\[([^\]]+)\]', dynamic)
        if soname and Path(soname[1]).name == soname[1]:
            alias = library.parent / soname[1]
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(library.name)
    (root / 'lib64').mkdir(exist_ok=True)
    for p in (root / 'lib').glob('*'):
        alias = root / 'lib64' / p.name
        if not alias.exists() and not alias.is_symlink():
            alias.symlink_to(Path('../lib') / p.name)
    env.update(CUDACXX=str(root / 'bin/nvcc'), CUDAToolkit_ROOT=str(root), CUDA_PATH=str(root))
    env['PATH'] = str(root / 'bin') + os.pathsep + env['PATH']
    return root


def build(work, *, jobs=2, cuda=True):
    """No model loading or GPU job is performed by this function."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources = json.loads((HERE / 'sources.json').read_text())
    for name, digest in sources['patches'].items():
        if sha256(HERE / name) != digest:
            raise RuntimeError('Runtime build source changed; regenerate both notebooks before building: ' + name)
    identity = fingerprint(sources)
    # Opaque extension prevents Kaggle's Dataset processing from auto-unpacking
    # the archive and discarding the exact file whose checksum we saved.
    archive = work / 'music-studio-runtime.bundle'
    receipt = work / 'runtime-build.json'
    if archive.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('source_fingerprint') == identity and previous.get('build_mode') == ('cuda' if cuda else 'cpu'):
            from .common import verify
            verify(archive, previous)
            return previous
    if platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('Build Music Studio on Kaggle x86-64; ARM binaries cannot serve this notebook.')
    if shutil.disk_usage(work).free < 7 * 1024**3:
        raise RuntimeError('The first CPU runtime build needs at least 7 GiB measured free disk. Reuse an existing Music Studio Dataset to skip rebuilding.')
    jobs = max(1, min(int(jobs), os.cpu_count() or 2))
    env = dict(os.environ)
    log = work / 'runtime-build.log'
    env.update(CARGO_HOME=str(work / 'cargo'), RUSTUP_HOME=str(work / 'rustup'), CARGO_BUILD_JOBS=str(jobs))
    installer = transfer(sources['rustup'], work / 'rustup-init')
    installer.chmod(0o700)
    command([installer, '-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', sources['rust_toolchain']], cwd=work, env=env, log=log)
    env['PATH'] = str(work / 'cargo/bin') + os.pathsep + env['PATH']
    node = transfer(sources['node'], work / sources['node']['name'])
    node_dir = work / 'node'
    # Re-extraction repairs an interrupted, owned toolchain extraction.
    extract(node, node_dir)
    node_root = next(p for p in node_dir.iterdir() if (p / 'bin/node').exists())
    env['PATH'] = str(node_root / 'bin') + os.pathsep + env['PATH']
    # A private build environment leaves Kaggle's Torch/scientific installation intact.
    tools = work / 'build-tools'
    if not (tools / 'bin/cmake').exists():
        command([sys.executable, '-m', 'venv', tools], cwd=work, log=log)
        command([tools / 'bin/python', '-m', 'pip', 'install', 'cmake==3.31.6', 'ninja==1.11.1.3'], cwd=work, log=log)
    env['PATH'] = str(tools / 'bin') + os.pathsep + env['PATH']
    studio = checkout(sources['studio'], work / 'studio-source', log)
    from .patches import apply
    apply(studio)
    command(['cargo', 'build', '--locked', '--release', '-p', 'music-server', '--config', 'profile.release.strip=true'], cwd=studio, env=env, log=log)
    command(['npm', '--prefix', 'app', 'ci', '--no-audit', '--no-fund'], cwd=studio, env=env, log=log)
    command(['npm', '--prefix', 'app', 'run', 'build'], cwd=studio, env=env, log=log)
    bundle = work / 'bundle'
    bundle.mkdir(exist_ok=True)
    shutil.copy2(studio / 'target/release/music-server', bundle / 'music-server')
    elf_arch(bundle / 'music-server')
    shutil.copytree(studio / 'app/dist', bundle / 'ui', dirs_exist_ok=True)
    notice = bundle / 'licenses'
    notice.mkdir(exist_ok=True)
    shutil.copy2(studio / 'LICENSE', notice / 'YuE2-Studio-MIT.txt')
    if (studio / 'licenses').exists():
        shutil.copytree(studio / 'licenses', notice / 'studio', dirs_exist_ok=True)
    # Free only this build's compiler intermediates before the C++/CUDA build.
    shutil.rmtree(studio / 'target')
    shutil.rmtree(studio / 'app/node_modules')
    cuda_root = toolkit(work, sources, env) if cuda else None
    common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
              '-DCMAKE_BUILD_RPATH=$ORIGIN', '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
              '-DGGML_CCACHE=OFF', '-DGGML_CUDA=' + ('ON' if cuda else 'OFF')]
    if cuda:
        common += ['-DCMAKE_CUDA_ARCHITECTURES=75-real', '-DCMAKE_CUDA_COMPILER=' + str(cuda_root / 'bin/nvcc'), '-DCUDAToolkit_ROOT=' + str(cuda_root)]
        common += ['-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,' + str(cuda_root / 'lib') + ':' + str(cuda_root / 'lib/stubs')]
    engine = checkout(sources['engine'], work / 'engine-source', log)
    command(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'], cwd=engine, log=log)
    command(['cmake', '-S', engine, '-B', engine / 'build-kaggle', *common], cwd=work, env=env, log=log)
    command(['cmake', '--build', engine / 'build-kaggle', '--target', 'yue-server', '--parallel', jobs], cwd=work, env=env, log=log)
    copy_runtime(engine / 'build-kaggle', 'yue-server', bundle / 'engine')
    tools_source = checkout(sources['tools'], work / 'tools-source', log)
    command(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1', 'engine/ggml', 'engine/vendor/vst3sdk'], cwd=tools_source, log=log)
    target = tools_source / 'engine/build-kaggle'
    # Auxiliary MIDI/caption tools use the CPU build verified here. Keep the
    # CUDA music engine independent; do not advertise untested trainer/GPU paths.
    cpu_common = [flag for flag in common if not flag.startswith(('-DGGML_CUDA=', '-DCMAKE_CUDA', '-DCUDAToolkit_ROOT=', '-DCMAKE_EXE_LINKER_FLAGS='))]
    cpu_common += ['-DGGML_CUDA=OFF']
    command(['cmake', '-S', tools_source / 'engine', '-B', target, *cpu_common], cwd=work, env=env, log=log)
    command(['cmake', '--build', target, '--target', 'ace-midi', 'ace-caption', '--parallel', jobs], cwd=work, env=env, log=log)
    copy_runtime(target, 'ace-midi', bundle / 'midi')
    copy_runtime(target, 'ace-caption', bundle / 'caption')
    whisper = checkout(sources['whisper'], work / 'whisper-source', log)
    command(['cmake', '-S', whisper, '-B', whisper / 'build-kaggle', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_CUDA=OFF', '-DGGML_NATIVE=OFF', '-DWHISPER_BUILD_TESTS=OFF', '-DCMAKE_BUILD_RPATH=$ORIGIN',
             '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'], cwd=work, env=env, log=log)
    command(['cmake', '--build', whisper / 'build-kaggle', '--target', 'whisper-cli', '--parallel', jobs], cwd=work, env=env, log=log)
    copy_runtime(whisper / 'build-kaggle', 'whisper-cli', bundle / 'whisper')
    for label, source, names in [
        ('yue2.cpp', engine, ('LICENSE', 'ggml/LICENSE', 'vendor/cpp-httplib/LICENSE')),
        ('HOT-Step', tools_source, ('LICENSE', 'engine/LICENSE', 'engine/ggml/LICENSE', 'engine/vendor/cpp-httplib/LICENSE')),
        ('whisper.cpp', whisper, ('LICENSE',)),
    ]:
        for name in names:
            original = source / name
            if original.is_file():
                target = notice / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
    shutil.copy2(HERE / 'NOTICES.md', notice / 'MUSIC-STUDIO-NOTICES.md')
    if cuda:
        libs = bundle / 'cuda'
        libs.mkdir(exist_ok=True)
        for p in (cuda_root / 'lib').glob('*.so*'):
            if p.is_file() and p.name.startswith(('libcublas', 'libcudart')):
                if p.is_symlink():
                    if not (libs / p.resolve().name).exists():
                        shutil.copy2(p.resolve(), libs / p.resolve().name)
                    alias = libs / p.name
                    if not alias.exists() and not alias.is_symlink():
                        alias.symlink_to(p.resolve().name)
                elif not (libs / p.name).exists():
                    shutil.copy2(p, libs / p.name)
        for p in cuda_root.rglob('*'):
            if p.is_file() and p.name.startswith(('LICENSE', 'EULA')):
                shutil.copy2(p, notice / ('NVIDIA-' + p.name))
    for record in sources['runtime_assets']:
        if record['id'] not in ('onnxruntime', 'llama.cpp'):
            continue
        downloaded = transfer(record, work / record['name'])
        extracted = extract(downloaded, work / ('unpack-' + record['id']))
        if record['id'] == 'onnxruntime':
            folder = bundle / 'onnx'
            folder.mkdir(exist_ok=True)
            for p in extracted.rglob('libonnxruntime.so*'):
                if p.is_file():
                    shutil.copy2(p, folder / p.name)
            for p in extracted.rglob('LICENSE*'):
                if p.is_file():
                    shutil.copy2(p, notice / ('ONNX-' + p.name))
        else:
            copy_runtime(extracted, 'llama-server', bundle / 'writer')
            for p in extracted.rglob('*'):
                if p.is_file() and p.name.startswith(('LICENSE', 'NOTICE', 'COPYING')):
                    destination = notice / 'llama.cpp' / p.relative_to(extracted)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, destination)
        downloaded.unlink()
    cloudflared = transfer(sources['cloudflared'], bundle / 'cloudflared')
    cloudflared.chmod(0o700)
    elf_arch(cloudflared)
    write_json(bundle / 'build-manifest.json', dict(profile='music-studio-v1', source_fingerprint=identity,
               sources=sources, architecture='x86_64', cuda_architectures=[75], build_mode='cuda' if cuda else 'cpu',
               system=platform.platform(), glibc=platform.libc_ver(), tools={'midi': 'cpu', 'caption': 'cpu', 'writer': 'cpu',
               'recognition': 'cpu', 'stems': 'cpu', 'training': 'not provisioned; no trainer weights bundled'}))
    with tarfile.open(archive.with_suffix('.partial'), 'w:gz', dereference=False) as output:
        output.add(bundle, arcname='runtime')
    archive.with_suffix('.partial').replace(archive)
    result = dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive),
                  architecture='x86_64', cuda_architectures=[75], source_fingerprint=identity, build_mode='cuda' if cuda else 'cpu',
                  compatibility={'libc': list(platform.libc_ver()), 'cuda': sources['cuda_version']})
    write_json(receipt, result)
    return result
