"""Start the saved native studio; model preparation belongs to the CPU builder."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .common import bind_parent, elf_arch, extract, find_pack, fingerprint, notebook_process, scrub, sha256, write_json


def request(url, *, method='GET', payload=None, headers=None, timeout=10):
    body = json.dumps(payload).encode() if payload is not None else None
    all_headers = {'Content-Type': 'application/json'} if body is not None else {}
    all_headers.update(headers or {})
    call = urllib.request.Request(url, data=body, method=method, headers=all_headers)
    # Keep loopback requests local, independent of Kaggle proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if url.startswith('http://127.0.0.1:') else urllib.request.build_opener()
    with opener.open(call, timeout=timeout) as response:
        data = response.read()
        return json.loads(data) if 'json' in response.headers.get('Content-Type', '') else data


def port_free(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            raise RuntimeError(f'Port {port} is already in use. Stop the other notebook startup cell first; its process was left untouched.') from None


def check_gpu(index):
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap', '--format=csv,noheader,nounits'], text=True)
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError('Select GPU T4 ×2 in Kaggle settings and restart this cell. CUDA song generation is required.') from None
    cards = [dict(zip(('index', 'name', 'memory_total_mib', 'memory_free_mib', 'driver', 'compute'), map(str.strip, line.split(',')))) for line in rows.strip().splitlines()]
    chosen = next((c for c in cards if int(c['index']) == index), None)
    if not chosen or chosen['compute'] != '7.5':
        raise RuntimeError('This saved runtime targets T4/Turing SM75. Select GPU T4 ×2, or build a separate runtime for another architecture.')
    if float(chosen['memory_free_mib']) < 9000:
        raise RuntimeError('GPU has less than 9,000 MiB free. Stop the previous music/movie/image engine first; no unrelated process was killed.')
    if int(chosen['driver'].split('.')[0]) < 580:
        raise RuntimeError('This CUDA 13 build requires an NVIDIA driver from the 580 series or newer; use the current Kaggle GPU image.')
    return cards


def owned_child(arguments, *, env, cwd, log):
    output = open(log, 'ab', buffering=0)

    parent = os.getpid()
    try:
        return subprocess.Popen(list(map(str, arguments)), env=env, cwd=cwd, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=lambda: bind_parent(parent))
    finally:
        output.close()


def stop(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def tunnel_address(log):
    """A reserved Quick Tunnel name alone does not prove an edge connection."""
    if not re.search(r'Registered tunnel connection\b', log):
        return None
    urls = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', log)
    return urls[-1] if urls else None


def link_plugins(input_root, runtime):
    """Expose attached Linux VST3 bundles in the host's application plugin folder."""
    folder = runtime / 'vst/vst3'
    folder.mkdir(exist_ok=True)
    chosen = {}
    for bundle in Path(input_root).rglob('*.vst3'):
        binaries = list((bundle / 'Contents/x86_64-linux').glob('*.so'))
        if not bundle.is_dir() or not binaries:
            continue
        for binary in binaries:
            elf_arch(binary)
        if bundle.name in chosen and chosen[bundle.name] != bundle.resolve():
            raise RuntimeError('Two attached Linux plugins have the same bundle name: ' + bundle.name)
        chosen[bundle.name] = bundle.resolve()
    for previous in folder.iterdir():
        if not previous.is_symlink():
            raise RuntimeError('Unexpected files in the runtime plugin-link folder were retained: ' + str(previous))
        if previous.name not in chosen or previous.resolve() != chosen[previous.name]:
            previous.unlink()
    for name, target in chosen.items():
        if not (folder / name).is_symlink():
            (folder / name).symlink_to(target, target_is_directory=True)


def materialize(pack, manifest, runtime, state):
    """Place only links in temporary state; saved library files contain no model links."""
    for record in manifest['files']:
        destination = state / record['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = (pack / record['name']).resolve()
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.is_symlink():
            destination.unlink()  # A model link in this notebook's temporary state.
        elif destination.exists():
            raise RuntimeError('Existing temporary asset differs; retained for review: ' + str(destination))
        destination.symlink_to(source)
    for name, target in [('karaoke/runtime/onnx', 'onnx'), ('assistant/runtime/cpu', 'writer'), ('audiocpp/runtime/audiocpp-cpu', 'audio'), ('separation/runtime/bsroformer-cpu', 'roformer')]:
        destination = state / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(runtime / target, target_is_directory=True)


def restore_library(output, input_root, selection='auto'):
    """Stage a complete backup, then publish it with a resumable, verified commit."""
    output = Path(output)
    stage = output / '.restore-stage'
    owner = stage / '.owner.json'
    ready = stage / '.ready.json'
    if (output / 'library.sqlite').exists() or selection == 'none':
        return
    if stage.exists():
        if not owner.is_file() or json.loads(owner.read_text()).get('target') != str(output.resolve()):
            raise RuntimeError('Unknown restore staging files were retained: ' + str(stage))
        if not ready.is_file():
            # Only the marked, unpublished copy is disposable. Working files stay.
            shutil.rmtree(stage)
    if not ready.is_file():
        if selection != 'auto':
            candidates = [Path(selection)]
        else:
            candidates = sorted({p.parent for p in Path(input_root).rglob('library.sqlite')
                                 if (p.parent / 'media').is_dir() and p.parent.name == 'music_studio'})
        if not candidates:
            return
        if len(candidates) != 1:
            raise RuntimeError('Several saved Music Studio libraries are attached. Select one RESTORE_LIBRARY_PATH, or none.')
        source = candidates[0].resolve()
        if not (source / 'library.sqlite').is_file() or not (source / 'media').is_dir():
            raise RuntimeError('Restore folder must contain library.sqlite and media.')
        names = [name for name in ('media', 'adapters', 'uploads', 'training', 'skins', 'videos', 'processing', 'midi') if (source / name).exists()]
        if (source / 'studio-settings.json').is_file() and not (output / 'studio-settings.json').exists():
            names.append('studio-settings.json')
        for name in names:
            destination = output / name
            if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
                raise RuntimeError('Restore would merge into existing work: ' + name)
        originals = [source / 'library.sqlite']
        for name in names:
            path = source / name
            originals += [path, *path.rglob('*')] if path.is_dir() else [path]
        if any(p.is_symlink() for p in originals):
            raise RuntimeError('Restore library contains symlinks; no user files were copied.')
        required = sum(p.stat().st_size for p in originals if p.is_file())
        if shutil.disk_usage(output).free < required + 512 * 1024**2:
            raise RuntimeError('Insufficient disk to restore the saved library; source remains untouched.')
        stage.mkdir()
        write_json(owner, {'target': str(output.resolve()), 'source': str(source)})
        for name in names:
            if (source / name).is_dir():
                shutil.copytree(source / name, stage / name)
            else:
                shutil.copy2(source / name, stage / name)
        # SQLite backup includes committed WAL pages without modifying attached input.
        import sqlite3
        with sqlite3.connect((source / 'library.sqlite').as_uri() + '?mode=ro', uri=True) as old:
            with sqlite3.connect(stage / 'library.sqlite') as new:
                old.backup(new)
        names.append('library.sqlite')  # Published last: the service cannot open a partial restore.
        files = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for name in names for p in ([stage / name] if (stage / name).is_file() else (stage / name).rglob('*')) if p.is_file()}
        write_json(ready, {'names': names, 'files': files})
    receipt = json.loads(ready.read_text())
    for name in receipt['names']:
        pending, target = stage / name, output / name
        if pending.exists():
            if target.exists():
                if not target.is_dir() or any(target.iterdir()):
                    raise RuntimeError('Restore target contains work; retained for review: ' + str(target))
                target.rmdir()
            pending.replace(target)
        # A move interrupted before its acknowledgement is safe to resume, only
        # when the moved content still matches this transaction's exact bytes.
        expected = {n: r for n, r in receipt['files'].items() if n == name or n.startswith(name + '/')}
        present = {str(p.relative_to(output)) for p in ([target] if target.is_file() else target.rglob('*')) if p.is_file()}
        if present != set(expected) or target.is_symlink():
            raise RuntimeError('Restored files changed during recovery; no current work was overwritten.')
        for relative, record in expected.items():
            path = output / relative
            if path.is_symlink() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise RuntimeError('Restored content differs; retained for review: ' + relative)
    shutil.rmtree(stage)
    print('Restored saved Music Studio library.', flush=True)


def settings(output, *, key, gpu):
    path = output / 'studio-settings.json'
    if path.exists():
        previous = json.loads(path.read_text())
    else:
        previous = dict(configuration={'selections': [
            dict(capability='music_generation', mode='local', local_engine='yue2-cpp', cloud_model=None),
            dict(capability='prompt_enhancement', mode='local', local_engine=None, cloud_model=None),
            dict(capability='cover_art', mode='local', local_engine=None, cloud_model=None)]},
            lyrics_sync=dict(enabled=False, provider='none', runtime='cpu'),
            assistant=dict(provider='managed', managed_model='gemma-4-e4b-q4_0'),
            cover_auto=False, selected_profile_id='quality-q8', selected_component_ids=None,
            separation=dict(runtime='cpu', engine='htdemucs', stems=['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'], overlap=0.25))
    previous['lyrics_sync'] = dict(enabled=False, provider='none', runtime='cpu')
    previous['network'] = {'enabled': True, 'key': key}
    # Explicit CUDA prevents an engine failure becoming an unnoticed CPU render.
    previous['engine_options'] = dict(backend='cuda', keep_loaded=False, max_batch=1, max_seq=16384,
                                      vae_core=512, vae_halo=16, disable_flash_attention=False, clamp_fp16=True, gpu=gpu)
    write_json(path, previous)
    return path


def install_runtime(pack, manifest, cache):
    record = manifest['runtime']
    location = cache / record['sha256'][:16]
    runtime = location / 'runtime'
    stamp = location / '.verified.json'
    if stamp.exists() and json.loads(stamp.read_text()).get('archive_sha256') != record['sha256']:
        raise RuntimeError('Runtime cache identity differs; retained for review: ' + str(location))
    if not stamp.exists():
        if location.exists():
            raise RuntimeError('An interrupted runtime extraction exists. Retained for diagnosis: ' + str(location))
        stage = cache / (record['sha256'][:16] + '.partial')
        marker = stage / '.extracting.json'
        if stage.exists():
            proof = marker if marker.is_file() else stage / '.verified.json'
            if not proof.is_file() or json.loads(proof.read_text()).get('archive_sha256') != record['sha256']:
                raise RuntimeError('Unknown files in extraction staging directory were preserved: ' + str(stage))
            shutil.rmtree(stage)
        stage.mkdir()
        write_json(marker, {'archive_sha256': record['sha256']})
        extract(pack / record['name'], stage)
        staged_runtime = stage / 'runtime'
        expected = json.loads((staged_runtime / 'build-manifest.json').read_text())
        if expected['source_fingerprint'] != record['source_fingerprint']:
            raise RuntimeError('Runtime archive has the wrong source identity.')
        for binary in ['music-server', 'engine/yue-server', 'audio/audiocpp_server', 'roformer/bs_roformer-cli', 'caption/ace-caption', 'training/music-train', 'vst/vst-host', 'writer/llama-server', 'cloudflared']:
            elf_arch(staged_runtime / binary)
            (staged_runtime / binary).chmod(0o700)
        write_json(stage / '.verified.json', {'archive_sha256': record['sha256']})
        stage.replace(location)
        (location / '.extracting.json').unlink(missing_ok=True)
    return runtime


def launch(dataset='auto', *, output_root='/kaggle/working/music_studio', gpu=0, port=8791,
           input_root='/kaggle/input', runtime_root='/kaggle/temp/music_studio_runtime', public=True, require_gpu=True, restore='auto'):
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    lock = (output / '.studio.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Music Studio is already running. Stop its existing startup cell first.') from None
    service = tunnel = None
    status_path = output / 'startup-status.json'
    try:
        write_json(status_path, {'stage': 'verifying-dataset'})
        cards = check_gpu(gpu) if require_gpu else []
        pack, manifest = find_pack(dataset, input_root)
        restore_library(output, input_root, restore)
        cache = Path(runtime_root).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        runtime = install_runtime(pack, manifest, cache)
        link_plugins(input_root, runtime)
        state = cache / ('state-' + manifest['runtime']['source_fingerprint'][:16])
        state.mkdir(exist_ok=True)
        materialize(pack, manifest, runtime, state)
        # Persist every writable project area, keeping models outside saved output.
        for name in ('adapters', 'skins', 'videos', 'training/datasets', 'training/runs', 'processing/vst', 'midi/files', 'logs'):
            target, link = output / name, state / name
            target.mkdir(parents=True, exist_ok=True)
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            elif not link.is_symlink() or link.resolve() != target.resolve():
                raise RuntimeError('Unexpected project-state path was retained: ' + str(link))
        port_free(port)
        port_free(18087)
        key = secrets.token_hex(32)
        saved = settings(output, key=key, gpu=gpu)
        env = dict(os.environ)
        env.update(YUE_STUDIO_DATA_ROOT=str(state), YUE_LIBRARY_ROOT=str(output), YUE_STUDIO_SETTINGS_PATH=str(saved),
                   YUE_MODELS_ROOT=str(state / 'models/yue2-cpp'), YUE_ENGINE_ROOT=str(runtime / 'engine'),
                   YUE_ENGINE_BIN=str(runtime / 'engine/yue-server'), YUE_UI_DIR=str(runtime / 'ui'),
                   YUE_STUDIO_PORT=str(port), YUE_ENGINE_PORT='18087', YUE_ENGINE_HOST='127.0.0.1',
                   YUE_ENGINE_BASE_URL='http://127.0.0.1:18087', YUE_BIND_ADDR='127.0.0.1',
                   YUE_AUDIOCPP_SERVER=str(runtime / 'audio/audiocpp_server'), YUE_CAPTION_BIN=str(runtime / 'caption/ace-caption'),
                   YUE_TRAIN_BIN=str(runtime / 'training/music-train'), MUSIC_VST_HOST_BIN=str(runtime / 'vst/vst-host'),
                   CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                   ORT_DYLIB_PATH=str(runtime / 'onnx/libonnxruntime.so'))
        # Do not put a different engine's libggml on this process's loader path.
        # Each executable was built with $ORIGIN; only NVIDIA's shared runtime is common.
        env['LD_LIBRARY_PATH'] = str(runtime / 'cuda') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        executables = ['music-server', 'engine/yue-server', 'audio/audiocpp_server', 'roformer/bs_roformer-cli',
                       'caption/ace-caption', 'training/music-train', 'vst/vst-host', 'writer/llama-server']
        linked = subprocess.run(['ldd', *[str(runtime / name) for name in executables]],
                                env=env, capture_output=True, text=True)
        if linked.returncode or 'not found' in linked.stdout:
            raise RuntimeError('Saved studio runtime cannot load on this Kaggle image: ' + linked.stdout[-2000:] + linked.stderr[-1000:])
        write_json(output / 'runtime-identity.json', dict(runtime=manifest['runtime'], models=manifest['catalog_fingerprint'], gpus=cards))
        print('Verified saved models. Starting the native CUDA studio; no song starts automatically.', flush=True)
        service = owned_child([runtime / 'music-server'], env=env, cwd=runtime, log=logs / 'studio.log')
        base = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 300
        last_report = 0
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise RuntimeError('Studio exited. ' + scrub((logs / 'studio.log').read_text(errors='replace')[-4000:]))
            try:
                health = request(base + '/setup/status')
                if health.get('ready') and (health.get('engine_ready') or not require_gpu):
                    write_json(output / 'ready-check.json', health)
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            if time.monotonic() - last_report > 20:
                print('Waiting for studio/model startup; details in music_studio/logs/studio.log.', flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        else:
            raise RuntimeError('CUDA engine did not become ready in five minutes. See studio.log; no CPU rendering was substituted.')
        # Browser/API traffic forwarded by Cloudflare is always key-gated.
        try:
            request(base + '/health', headers={'X-Forwarded-For': '203.0.113.1'})
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                raise
        else:
            raise RuntimeError('Network authentication guard did not reject an unauthenticated request.')
        request(base + '/health', headers={'X-Forwarded-For': '203.0.113.1', 'Authorization': 'Bearer ' + key})
        address = base
        if public:
            tunnel_log = logs / 'cloudflare.log'
            tunnel_log.write_text('')
            tunnel = owned_child([runtime / 'cloudflared', 'tunnel', '--no-autoupdate', '--protocol', 'http2', '--url', base],
                                 env=env, cwd=runtime, log=tunnel_log)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if tunnel.poll() is not None:
                    raise RuntimeError('Cloudflare exited. ' + scrub(tunnel_log.read_text(errors='replace')[-2000:]))
                registered = tunnel_address(tunnel_log.read_text(errors='replace'))
                if registered:
                    address = registered
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Cloudflare did not register a tunnel connection. See logs/cloudflare.log and rerun startup.')
        write_json(status_path, dict(stage='ready', public=public, studio_port=port))
        print('\nOPEN MUSIC STUDIO: ' + address + '/?key=' + key, flush=True)
        print('Keep this cell running. Create starts only when you press it in Studio.', flush=True)
        print('Songs, library and logs: ' + str(output), flush=True)
        print('Song generation and LoRA training use GPU ' + str(gpu) + ' one job at a time. Stems, writer, MIDI and style description use CPU.', flush=True)
        while service.poll() is None:
            if tunnel is not None and tunnel.poll() is not None:
                raise RuntimeError('Cloudflare stopped. Saved songs are retained; restart this startup cell for a fresh link.')
            time.sleep(2)
        raise RuntimeError('Studio stopped unexpectedly. See logs/studio.log.')
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        write_json(status_path, dict(stage='stopped' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed', reason=scrub(error)))
        raise
    finally:
        stop(tunnel)
        stop(service)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    notebook_process()
    launch(dataset=os.environ.get('MUSIC_STUDIO_DATASET_PATH', 'auto'), gpu=int(os.environ.get('MUSIC_STUDIO_GPU', '0')),
           restore=os.environ.get('MUSIC_STUDIO_RESTORE_LIBRARY', 'auto'))
