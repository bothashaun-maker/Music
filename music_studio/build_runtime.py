"""Build the pinned Linux x86-64/T4 runtime once, on CPU, in resumable stages.

Nothing here loads a model or starts a GPU job. The build is split into named
stages, each with a completed-marker keyed by the exact inputs it consumed, so a
rerun after an interruption resumes at the failing stage instead of repeating
cargo, npm and the CUDA compile. Models are never compiled: they are downloaded
by the Dataset builder and verified against pinned sizes and checksums.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

from .common import HERE, PROFILE, bind_parent, elf_arch, extract, fingerprint, sha256, transfer, write_json


def failure_summary(log, limit=40):
    """Report the error lines, not the warnings that happen to be last.

    A C++ build ends with thousands of -Wmaybe-uninitialized notes, so printing
    only the log tail hid the actual failing unit or target.
    """
    lines = Path(log).read_text(errors='replace').splitlines()
    markers = ('error:', 'Error ', 'FAILED:', 'fatal:', 'CMake Error', 'undefined reference',
               'cannot find', 'No such file', 'ninja: error', 'ninja: build stopped')
    hits = [line for line in lines if any(m in line for m in markers) and 'warning:' not in line]
    if not hits:
        hits = lines[-limit:]
    return '\n'.join(hits[-limit:])


def cpu_cmake_options(options):
    """Configure HOT-Step helpers as genuinely CPU-only targets.

    HOT-Step's CMake adds GGML_USE_CUDA whenever CUDAToolkit is found, even
    when GGML_CUDA=OFF. This builder puts its pinned CUDA toolkit on PATH for
    the separate trainer build; without disabling package discovery, that
    toolkit leaks into the CPU helper configuration and gpu.h then requires
    cuda_runtime.h while compiling with the host C++ compiler.
    """
    result = [flag for flag in options if not flag.startswith((
        '-DGGML_CUDA=', '-DCMAKE_CUDA', '-DCUDAToolkit_ROOT=',
        '-DCMAKE_EXE_LINKER_FLAGS='))]
    result.extend(('-DGGML_CUDA=OFF', '-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE'))
    return result


def command(arguments, *, cwd, env=None, log=None):
    print('Build:', ' '.join(map(str, arguments[:5])), flush=True)
    log = Path(log or Path(cwd) / 'build.log')
    with log.open('ab') as output:
        parent = os.getpid()
        child = subprocess.Popen(list(map(str, arguments)), cwd=cwd, env=env,
                                 stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=lambda: bind_parent(parent))
        try:
            result = child.wait()
        finally:
            if child.poll() is None:
                stop_group(child)
    if result:
        summary = failure_summary(log)
        print('Build failed; errors from', log, flush=True)
        print(summary, flush=True)
        raise RuntimeError(f'Build command failed ({result}); {log}\n{summary}')


def stop_group(child):
    """Stop this build's own process group; never signal the notebook kernel."""
    import signal
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def write_tar_file(source, member, destination):
    """Extract one prevalidated regular file while preserving executable bits."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.extractfile(member) as stream, destination.open('wb') as output:
        shutil.copyfileobj(stream, output)
    os.chmod(destination, member.mode & 0o777)
    return destination


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
        if library.is_file() and library.name.startswith(('libggml', 'libllama', 'libmtmd')):
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


# Files the toolkit must contain after extraction. CMake's CUDA compiler ABI
# check links a test binary, and device linking needs both of the static
# runtime libraries, so they cannot be discarded as "static".
TOOLKIT_REQUIRED = {
    'cuda_nvcc': ('bin/nvcc', 'bin/ptxas', 'bin/cudafe++'),
    'cuda_cudart': ('lib/libcudart_static.a', 'lib/libcudadevrt.a'),
}
# Multi-hundred-megabyte static BLAS/FFT libraries are genuinely unused by the
# dynamic engine backend. The small static CUDA runtime libraries are not.
HEAVY_STATIC = ('libcublas', 'libcufft', 'libcusolver', 'libcusparse', 'libcurand')
HEAVY_STATIC_BYTES = 256 * 1024 * 1024


def toolkit_member_needed(relative, size):
    """Keep every CUDA file except the giant static accelerator libraries."""
    if Path(relative).suffix != '.a':
        return True
    return not (Path(relative).name.lower().startswith(HEAVY_STATIC) or size > HEAVY_STATIC_BYTES)


def toolkit_missing(root, record_id):
    """Files this archive should have produced but that are absent on disk."""
    missing = [name for name in TOOLKIT_REQUIRED.get(record_id, ()) if not (root / name).is_file()]
    if record_id == 'cuda_cudart' and not any(root.glob('lib/libcudart.so*')):
        missing.append('lib/libcudart.so')
    return tuple(missing)


def toolkit(work, sources, env):
    """NVIDIA's pinned redistributables provide nvcc without a GPU or root access."""
    root = work / 'cuda'
    ready = root / '.complete'
    selected = [r for r in sources['runtime_assets'] if r['id'].startswith('cuda_') or r['id'] in ('libcublas', 'libnvvm')]
    identity = fingerprint(selected)
    compilers = tuple(root / name for name in ('bin/nvcc', 'bin/ptxas', 'bin/cudafe++'))
    # CMake's CUDA ABI check links against the static runtime, so a toolkit that
    # lost those libraries is not complete even when nvcc exists.
    static_runtime = (root / 'lib/libcudart_static.a', root / 'lib/libcudadevrt.a')
    complete = (ready.exists() and ready.read_text() == identity
                and all(p.is_file() for p in compilers + static_runtime))
    # Older notebook builds wrote CUDA archive bytes without restoring their
    # executable modes. Re-extract the pinned nvcc archive on such caches.
    executable_ok = complete and all(os.access(p, os.X_OK) for p in compilers)
    if not complete or not executable_ok:
        root.mkdir(exist_ok=True)
        # If this root was marked complete by an older extraction, re-open the
        # asset marker repair path even though the top-level .complete stamp
        # matches. The archive marker alone doesn't prove executable modes or
        # that every required library survived the extraction filter.
        for record in selected:
            marker = root / ('.asset-' + record['id'])
            missing = toolkit_missing(root, record['id'])
            if marker.exists() and marker.read_text() == record['sha256'] and not missing:
                if record['id'] == 'cuda_nvcc' and not all(os.access(p, os.X_OK) for p in compilers):
                    # Earlier notebook builds extracted CUDA files without mode
                    # bits. The archive marker is valid but nvcc is unusable; fetch
                    # this small pinned archive again and repair the owned files.
                    print('Repairing cached CUDA toolkit: nvcc executable mode is missing.', flush=True)
                else:
                    continue
            elif missing and marker.exists():
                print('Repairing cached CUDA toolkit: ' + record['id'] + ' is missing ' +
                      ', '.join(missing) + '.', flush=True)
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
            if record['id'] == 'cuda_nvcc':
                # Remove only this builder-owned toolkit executable directory
                # before repairing bad cached mode bits from older notebook runs.
                shutil.rmtree(root / 'bin', ignore_errors=True)
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    if not member.isfile():
                        continue
                    pieces = Path(member.name).parts[1:]
                    if not pieces or '..' in pieces or Path(member.name).is_absolute():
                        raise RuntimeError('Invalid CUDA redistributable path.')
                    relative = Path(*pieces)
                    # Static cuBLAS is enormous and is not used by the dynamic
                    # backend. libcudart_static.a and libcudadevrt.a are small
                    # and required, so only the heavy static libraries are skipped.
                    if not toolkit_member_needed(relative, member.size):
                        continue
                    # CUDA archives mark nvcc/ptxas/cudafe/fatbinary executable.
                    write_tar_file(source, member, root / relative)
            archive.unlink()
            marker.write_text(record['sha256'])
        for p in (root / 'lib').glob('*.so*'):
            if p.is_file() and not p.is_symlink() and '.so.' in p.name:
                alias = p.with_name(p.name.split('.so.')[0] + '.so')
                if not alias.exists() and not alias.is_symlink():
                    alias.symlink_to(Path('../lib') / p.name)
        if not complete or not executable_ok:
            ready.write_text(identity)
    env.update(CUDACXX=str(root / 'bin/nvcc'), CUDAToolkit_ROOT=str(root), CUDA_PATH=str(root))
    env['PATH'] = str(root / 'bin') + os.pathsep + env['PATH']
    validate_cuda_compiler(root, env, work)
    return root


def validate_cuda_compiler(root, env, work):
    """Fail with the real nvcc error before CMake's opaque ABI try_compile."""
    compiler = Path(root) / 'bin/nvcc'
    if not compiler.is_file():
        raise RuntimeError('Pinned CUDA toolkit is missing nvcc: ' + str(compiler))
    if not os.access(compiler, os.X_OK):
        raise RuntimeError('Pinned CUDA toolkit nvcc is not executable: ' + str(compiler))
    version = subprocess.run([str(compiler), '--version'], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if version.returncode:
        raise RuntimeError('Pinned nvcc --version failed (' + str(version.returncode) + '):\n' + version.stdout[-8000:])
    probe = Path(work) / 'cuda-compiler-probe'
    probe.mkdir(parents=True, exist_ok=True)
    source = probe / 'probe.cu'
    output = probe / 'probe.o'
    host = shutil.which('c++', path=env.get('PATH')) or shutil.which('g++', path=env.get('PATH'))
    if not host:
        raise RuntimeError('A host C++ compiler (c++/g++) is required to compile CUDA code.')
    source.write_text('extern "C" __global__ void music_studio_cuda_probe() {}\n', newline='\n')
    result = subprocess.run([str(compiler), '-ccbin', host, '-arch=sm_75', '-c', str(source), '-o', str(output)],
                            cwd=probe, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Pinned nvcc CUDA compile probe failed (' + str(result.returncode) + '):\n' +
                           version.stdout[-3000:] + '\n' + result.stdout[-12000:])
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('Pinned nvcc probe exited successfully but did not produce an object file.')
    # Compiling alone does not link the CUDA runtime. CMake's compiler ABI check
    # does, so link a real device-code binary here to catch missing
    # libcudart_static.a/libcudadevrt.a before the opaque engine configure fails.
    link_source = probe / 'link.cu'
    link_output = probe / 'probe'
    link_source.write_text('extern "C" __global__ void music_studio_cuda_probe() {}\n'
                           'int main() { music_studio_cuda_probe<<<1, 1>>>(); return 0; }\n', newline='\n')
    linked = subprocess.run([str(compiler), '-ccbin', host, '-arch=sm_75', str(link_source), '-o', str(link_output)],
                            cwd=probe, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if linked.returncode:
        raise RuntimeError('Pinned nvcc CUDA link probe failed (' + str(linked.returncode) + '):\n' +
                           version.stdout[-3000:] + '\n' + linked.stdout[-12000:])
    if not link_output.is_file():
        raise RuntimeError('Pinned nvcc link probe exited successfully but produced no binary.')
    print('CUDA preflight: nvcc compiled and linked an SM75 probe with the pinned static runtime.', flush=True)


def build_env(work, sources):
    """One deterministic environment, so any stage can run after an interruption."""
    env = dict(os.environ)
    env.update(CARGO_HOME=str(work / 'cargo'), RUSTUP_HOME=str(work / 'rustup'),
               RUSTUP_TOOLCHAIN=sources['rust_toolchain'])
    node_dir = work / 'node'
    if node_dir.is_dir():
        roots = [p for p in node_dir.iterdir() if (p / 'bin/node').exists()]
        if roots:
            env['PATH'] = str(roots[0] / 'bin') + os.pathsep + env['PATH']
    tools = work / 'build-tools'
    if (tools / 'bin').is_dir():
        env['PATH'] = str(tools / 'bin') + os.pathsep + env['PATH']
        env['PYTHONPATH'] = str(tools) + os.pathsep + env.get('PYTHONPATH', '')
    if (work / 'cargo/bin').is_dir():
        env['PATH'] = str(work / 'cargo/bin') + os.pathsep + env['PATH']
    return env


def engine_stage(work, sources, env, log, jobs, cuda):
    """Configure and compile the generation engine."""
    print('Engine configure env:', 'CUDACXX=' + env.get('CUDACXX', 'unset'),
          'CUDAToolkit_ROOT=' + env.get('CUDAToolkit_ROOT', 'unset'), flush=True)
    stage_engine(work, sources, env, log, jobs, cuda)


def build(work, *, jobs=2, cuda=True):
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
        raise RuntimeError('The CPU runtime build needs at least 7 GiB measured free disk. Reuse an existing Music Studio Dataset to skip rebuilding.')
    jobs = max(1, min(int(jobs), os.cpu_count() or 2))
    log = work / 'runtime-build.log'

    stage(work, 'toolchain', dict(sources=sources, profile=PROFILE), lambda: stage_toolchain(work, sources, log))
    env = build_env(work, sources)
    env['CARGO_BUILD_JOBS'] = str(jobs)
    stage(work, 'studio', sources['studio'], lambda: stage_studio(work, sources, env, log))
    if cuda:
        # toolkit() is idempotent (marker-guarded) but it must run on every
        # attempt: it exports CUDACXX/CUDAToolkit_ROOT/CUDA_PATH and puts the
        # pinned nvcc on PATH for the engine and trainer configures. Wrapping it
        # in a stage made a completed run skip the export and broke CMake ABI
        # detection at engine configure.
        toolkit(work, sources, env)
    stage(work, 'engine', dict(sources=sources, jobs=jobs, cuda=cuda),
          lambda: engine_stage(work, sources, env, log, jobs, cuda))
    stage(work, 'helpers-cpu', sources['tools'], lambda: stage_helpers(work, sources, env, log, jobs))
    if cuda:
        stage(work, 'helpers-train', dict(sources=sources, jobs=jobs),
              lambda: stage_trainer(work, sources, env, log, jobs))
    stage(work, 'downloads', [r for r in sources['runtime_assets']
                             if r['id'] in ('onnxruntime', 'llama.cpp', 'audiocpp', 'roformer')] + [sources['cloudflared']],
          lambda: stage_downloads(work, sources, log))
    return assemble(work, sources, identity, cuda)


def stage(work, name, inputs, produce):
    """Run one stage unless a marker says it already finished with these inputs."""
    stamp = work / f'stage-{name}.json'
    identity = fingerprint(inputs)
    if stamp.is_file():
        try:
            if json.loads(stamp.read_text()).get('identity') == identity:
                print(f'Stage {name}: already complete for these sources.', flush=True)
                return
        except ValueError:
            pass
    print(f'Stage {name}: starting.', flush=True)
    produce()
    write_json(stamp, dict(identity=identity, name=name))
    print(f'Stage {name}: complete.', flush=True)


def stage_toolchain(work, sources, log):
    env = build_env(work, sources)
    installer = transfer(sources['rustup'], work / 'rustup-init')
    installer.chmod(0o700)
    command([installer, '-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', sources['rust_toolchain']],
            cwd=work, env=env, log=log)
    env = build_env(work, sources)
    node = transfer(sources['node'], work / sources['node']['name'])
    extract(node, work / 'node')
    tools = work / 'build-tools'
    if not (tools / 'bin/cmake').exists() or not (tools / 'bin/ninja').exists():
        shutil.rmtree(tools, ignore_errors=True)
        # Kaggle's /usr/bin/python3 omits ensurepip, so --target uses the working
        # Kaggle pip instead of replacing Torch/scientific packages.
        command([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
                 '--target', tools, 'cmake==3.31.6', 'ninja==1.11.1.3'], cwd=work, log=log)
    if not (tools / 'bin/cmake').is_file() or not (tools / 'bin/ninja').is_file():
        raise RuntimeError('Pinned CMake/Ninja installation did not create the expected build tools.')


def stage_studio(work, sources, env, log):
    studio = checkout(sources['studio'], work / 'studio-source', log)
    from .patches import apply
    apply(studio)
    bundle = work / 'bundle'
    bundle.mkdir(exist_ok=True)
    command(['cargo', 'build', '--locked', '--release', '-p', 'music-server',
             '--config', 'profile.release.strip=true'], cwd=studio, env=env, log=log)
    command(['npm', '--prefix', 'app', 'ci', '--no-audit', '--no-fund'], cwd=studio, env=env, log=log)
    command(['npm', '--prefix', 'app', 'run', 'build'], cwd=studio, env=env, log=log)
    shutil.copy2(studio / 'target/release/music-server', bundle / 'music-server')
    elf_arch(bundle / 'music-server')
    shutil.copytree(studio / 'app/dist', bundle / 'ui', dirs_exist_ok=True)
    notice = bundle / 'licenses'
    notice.mkdir(exist_ok=True)
    shutil.copy2(studio / 'LICENSE', notice / 'YuE2-Studio-MIT.txt')
    if (studio / 'licenses').exists():
        shutil.copytree(studio / 'licenses', notice / 'studio', dirs_exist_ok=True)
    # Free only this build's compiler intermediates.
    shutil.rmtree(studio / 'target', ignore_errors=True)
    shutil.rmtree(studio / 'app/node_modules', ignore_errors=True)


def cmake_flags(work, sources, cuda):
    common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
              '-DCMAKE_BUILD_RPATH=$ORIGIN', '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
              '-DGGML_CCACHE=OFF', '-DGGML_CUDA=' + ('ON' if cuda else 'OFF')]
    if cuda:
        root = work / 'cuda'
        common += ['-DCMAKE_CUDA_ARCHITECTURES=75-real', '-DCMAKE_CUDA_COMPILER=' + str(root / 'bin/nvcc'),
                   '-DCUDAToolkit_ROOT=' + str(root)]
        common += ['-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,' + str(root / 'lib') + ':' + str(root / 'lib/stubs')]
    return common


def stage_engine(work, sources, env, log, jobs, cuda):
    bundle = work / 'bundle'
    engine = checkout(sources['engine'], work / 'engine-source', log)
    command(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'], cwd=engine, log=log)
    built = engine / 'build-kaggle'
    command(['cmake', '-S', engine, '-B', built, *cmake_flags(work, sources, cuda)], cwd=work, env=env, log=log)
    command(['cmake', '--build', built, '--target', 'yue-server', '--parallel', jobs], cwd=work, env=env, log=log)
    copy_runtime(built, 'yue-server', bundle / 'engine')


def stage_helpers(work, sources, env, log, jobs):
    """CPU helpers the studio launches: caption, MIDI and the VST host."""
    bundle = work / 'bundle'
    tools_source = checkout(sources['tools'], work / 'tools-source', log)
    command(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1', 'engine/ggml'], cwd=tools_source, log=log)
    command(['git', 'submodule', 'update', '--init', '--depth', '1', 'engine/vendor/vst3sdk'], cwd=tools_source, log=log)
    command(['git', 'submodule', 'update', '--init', '--depth', '1', 'base', 'pluginterfaces', 'public.sdk'],
            cwd=tools_source / 'engine/vendor/vst3sdk', log=log)
    target = tools_source / 'engine/build-cpu'
    options = cpu_cmake_options(cmake_flags(work, sources, True))
    command(['cmake', '-S', tools_source / 'engine', '-B', target, *options], cwd=work, env=env, log=log)
    # ace-midi is launched by the studio (YUE_MIDI_BIN) and was missing from the
    # earlier bundle while the manifest still advertised a CPU MIDI helper.
    command(['cmake', '--build', target, '--target', 'ace-caption', 'ace-midi', 'vst-host', '--parallel', jobs],
            cwd=work, env=env, log=log)
    copy_runtime(target, 'ace-caption', bundle / 'caption')
    copy_runtime(target, 'ace-midi', bundle / 'midi')
    copy_runtime(target, 'vst-host', bundle / 'vst')


def stage_trainer(work, sources, env, log, jobs):
    bundle = work / 'bundle'
    tools_source = work / 'tools-source'
    trainer = tools_source / 'engine/build-training'
    command(['cmake', '-S', tools_source / 'engine', '-B', trainer, *cmake_flags(work, sources, True)],
            cwd=work, env=env, log=log)
    command(['cmake', '--build', trainer, '--target', 'ace-train', '--parallel', jobs], cwd=work, env=env, log=log)
    copy_runtime(trainer, 'ace-train', bundle / 'training')
    (bundle / 'training/ace-train').rename(bundle / 'training/music-train')


def stage_downloads(work, sources, log):
    bundle = work / 'bundle'
    notice = bundle / 'licenses'
    for record in sources['runtime_assets']:
        if record['id'] not in ('onnxruntime', 'llama.cpp', 'audiocpp', 'roformer'):
            continue
        downloaded = transfer(record, work / record['name'])
        if record['id'] == 'onnxruntime':
            folder = bundle / 'onnx'
            folder.mkdir(exist_ok=True)
            with zipfile.ZipFile(downloaded) as wheel:
                for name in wheel.namelist():
                    base = Path(name).name
                    if base.startswith('libonnxruntime') and '.so' in base:
                        with wheel.open(name) as source, (folder / base).open('wb') as output:
                            shutil.copyfileobj(source, output)
                    elif base == 'LICENSE':
                        destination = notice / 'ONNX-Runtime-LICENSE'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with wheel.open(name) as source, destination.open('wb') as output:
                            shutil.copyfileobj(source, output)
            versioned = folder / 'libonnxruntime.so.1.30.0'
            if not versioned.is_file():
                raise RuntimeError('Pinned ONNX Runtime wheel has no expected 1.30.0 shared library.')
            alias = folder / 'libonnxruntime.so'
            if not alias.exists():
                alias.symlink_to(versioned.name)
        elif record['id'] in ('audiocpp', 'roformer'):
            extracted = extract(downloaded, work / ('unpack-' + record['id']))
            binary = 'audiocpp_server' if record['id'] == 'audiocpp' else 'bs_roformer-cli'
            folder = bundle / ('audio' if record['id'] == 'audiocpp' else 'roformer')
            matches = [p for p in extracted.rglob(binary) if p.is_file()]
            if len(matches) != 1:
                raise RuntimeError('Pinned helper archive has an unexpected layout: ' + binary)
            elf_arch(matches[0])
            # Keep each helper's own GGML libraries beside it; no shared loader path.
            shutil.copytree(matches[0].parent, folder, symlinks=True, dirs_exist_ok=True)
        else:
            extracted = extract(downloaded, work / ('unpack-' + record['id']))
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


def assemble(work, sources, identity, cuda):
    bundle = work / 'bundle'
    notice = bundle / 'licenses'
    notice.mkdir(parents=True, exist_ok=True)
    for label, source, names in [
        ('yue2.cpp', work / 'engine-source', ('LICENSE', 'ggml/LICENSE', 'vendor/cpp-httplib/LICENSE')),
        ('HOT-Step', work / 'tools-source', ('LICENSE', 'engine/LICENSE', 'engine/ggml/LICENSE',
                                             'engine/vendor/cpp-httplib/LICENSE', 'engine/vendor/vst3sdk/LICENSE.txt')),
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
        cuda_root = work / 'cuda'
        for p in (cuda_root / 'lib').glob('*.so*'):
            if p.is_file() and p.name.startswith(('libcublas', 'libcudart')):
                if p.is_symlink():
                    real = p.resolve()
                    if not (libs / real.name).exists():
                        shutil.copy2(real, libs / real.name)
                    alias = libs / p.name
                    if not alias.exists() and not alias.is_symlink():
                        alias.symlink_to(real.name)
                elif not (libs / p.name).exists():
                    shutil.copy2(p, libs / p.name)
        for p in cuda_root.rglob('*'):
            if p.is_file() and p.name.startswith(('LICENSE', 'EULA')):
                shutil.copy2(p, notice / ('NVIDIA-' + p.name))
    for required, label in (('music-server', 'service'), ('ui', 'UI'), ('engine/yue-server', 'engine'),
                            ('caption/ace-caption', 'caption'), ('midi/ace-midi', 'MIDI'),
                            ('writer/llama-server', 'writer'), ('onnx/libonnxruntime.so', 'ONNX Runtime'),
                            ('cloudflared', 'tunnel')):
        if not (bundle / required).exists():
            raise RuntimeError('Completed runtime is missing ' + label + ': ' + required)
    write_json(bundle / 'build-manifest.json', dict(
        profile=PROFILE, source_fingerprint=identity, sources=sources,
        architecture='x86_64', cuda_architectures=[75], build_mode='cuda' if cuda else 'cpu',
        system=platform.platform(), glibc=platform.libc_ver(),
        tools={'midi': 'cpu', 'caption': 'cpu', 'writer': 'cpu',
               'recognition': 'excluded', 'stems': 'cpu',
               'training': 'cuda-sm75; INT8 ConvRot base; no lyric aligner',
               'vst': 'native Linux VST3 offline host'}))
    archive = work / 'music-studio-runtime.bundle'
    with tarfile.open(archive.with_suffix('.partial'), 'w:gz', dereference=False) as output:
        output.add(bundle, arcname='runtime')
    archive.with_suffix('.partial').replace(archive)
    result = dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive),
                  architecture='x86_64', cuda_architectures=[75], source_fingerprint=identity,
                  build_mode='cuda' if cuda else 'cpu',
                  compatibility={'libc': list(platform.libc_ver()), 'cuda': sources['cuda_version']})
    write_json(work / 'runtime-build.json', result)
    return result
