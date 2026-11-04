"""Isolated setup for the original native engine, retaining Kaggle's CUDA stack."""
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request

from kaggle_yue2.runtime import KaggleYuE2, command, pip_install
from kaggle_yue2.recovery import sha256_file

MODEL_REVISION = '8f24312f187763b9854adeec87e437e2b03bbff1'
WHEEL_SHA256 = '8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641'
REQUIREMENTS = ['transformers==4.57.6', 'huggingface-hub==0.36.2', 'gradio==5.49.1',
                'pillow==11.3.0', 'soundfile==0.13.1', 'accelerate==1.13.0',
                'tiktoken==0.12.0', 'safetensors>=0.7.0',
                'mir_eval==0.8.2', 'pretty_midi==0.2.10', 'mido==1.3.3']

# --- ComfyUI canvas host ------------------------------------------------------
# The canvas route runs the SAME overlay environment and the SAME native engine
# as the Gradio route; only the wrapper differs. The checkout and its extra
# dependencies are session-only; model weights always come from the dataset.
COMFYUI_REPO = 'https://github.com/comfyanonymous/ComfyUI.git'
COMFYUI_COMMIT = '423cbd20511becb1f16122ab518e92e58bfa03d9'  # v0.43.2
COMFY_PORT = 8188


def support_root():
    return Path(__file__).resolve().parents[1]


def request_comfy_release(port=COMFY_PORT, timeout=10):
    """Ask a running ComfyUI server to release YuE2 weights before engine handover."""
    import urllib.error
    import urllib.request
    try:
        request = urllib.request.Request(f'http://127.0.0.1:{int(port)}/yue2/release', data=b'', method='POST')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            released = json.loads(response.read().decode()).get('released', False)
        if released:
            print('The ComfyUI server released the YuE2 engine; the lock is free.', flush=True)
        return bool(released)
    except (OSError, ValueError, urllib.error.URLError):
        return False  # No ComfyUI server or no pack: nothing to release here.


def _pinned_checkout(url, commit, target, env=None):
    if not target.exists():
        target.mkdir(parents=True)
        command(['git', 'init', target], env=env)
        command(['git', '-C', str(target), 'remote', 'add', 'origin', url], env=env)
    head = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, env=env)
    if head.returncode == 0:
        if head.stdout.strip() != commit:
            command(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', commit], env=env)
            command(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], env=env)
    else:
        command(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', commit], env=env)
        command(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], env=env)
    head = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, env=env)
    if head.stdout.strip() != commit:
        raise RuntimeError(f'ComfyUI checkout {head.stdout.strip()} differs from the pinned {commit}')


def write_comfy_pack(comfy_dir):
    """Install the KaggleYuE2 node pack: engine wrapper nodes + two canvases."""
    from .comfy_canvas import build_all
    from .comfy_web import OPEN_WORKFLOW_JS
    from .cover_upload_web import COVER_UPLOAD_JS
    comfy_dir = Path(comfy_dir)
    pack = comfy_dir / 'custom_nodes' / 'KaggleYuE2'
    (pack / 'web').mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('comfy_pack.py').read_text()
    compile(source, 'comfy_pack.py', 'exec')  # Fail setup, never the browser session.
    (pack / '__init__.py').write_text(source)
    (pack / 'web' / 'open_workflow.js').write_text(OPEN_WORKFLOW_JS)
    (pack / 'web' / 'cover_upload.js').write_text(COVER_UPLOAD_JS)
    graphs = build_all()
    for name, graph in graphs.items():
        (pack / 'web' / f'yue2-{name}.json').write_text(json.dumps(graph, indent=1))
    # Also install as user workflows so they appear under ComfyUI's Workflows menu.
    workflows = comfy_dir / 'user' / 'default' / 'workflows'
    workflows.mkdir(parents=True, exist_ok=True)
    titles = {'song': 'YuE2 Full Song.json', 'cover': 'YuE2 Cover Restyle.json'}
    for name, graph in graphs.items():
        (workflows / titles[name]).write_text(json.dumps(graph, indent=1))
    return pack


def install_comfyui(root, outputs, env):
    """Session-only pinned ComfyUI + the KaggleYuE2 pack, under Kaggle constraints."""
    runtime_root = Path(root)
    comfy_dir = runtime_root / 'ComfyUI'
    env = dict(env)
    env.update({'YUE2_SUPPORT_DIR': str(support_root()),
                'YUE2_ENGINE_SCRIPT': str(support_root() / 'native_yue2' / 'engine.py')})
    runtime = KaggleYuE2(runtime_root)
    _pinned_checkout(COMFYUI_REPO, COMFYUI_COMMIT, comfy_dir, env=env)
    write_comfy_pack(comfy_dir)
    constraint = runtime_root / 'kaggle-constraints.txt'
    if not constraint.is_file():
        import importlib.metadata
        pinned = []
        for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'scipy'):
            try:
                pinned.append(f'{name}=={importlib.metadata.version(name)}')
            except importlib.metadata.PackageNotFoundError:
                pass
        pinned.append('transformers==4.57.6')
        constraint.write_text('\n'.join(pinned) + '\n')
    # Protect native package pins as well as the image's CUDA/scientific stack.
    protected = runtime_root / 'comfy-constraints.txt'
    protected.write_text(constraint.read_text() + '\n' + '\n'.join(REQUIREMENTS) + '\n')
    asset_root = env.get('YUE2_ASSET_DATASET')
    wheelhouse = Path(asset_root) / 'wheelhouse' if asset_root else None
    # ComfyUI pins wheels the YuE2 wheelhouse never contained (frontend pack and
    # friends); the saved wheelhouse stays first and PyPI fills only the gaps.
    pip_install(runtime.python, ['-r', comfy_dir / 'requirements.txt', '-c', protected],
                env=env, wheelhouse=wheelhouse, purpose='the pinned ComfyUI packages')
    (runtime_root / 'comfy-checkout.txt').write_text(f'{COMFYUI_REPO} @ {COMFYUI_COMMIT}\n')
    print(f'ComfyUI {COMFYUI_COMMIT[:12]} installed with the KaggleYuE2 node pack at {comfy_dir}', flush=True)
    return comfy_dir, env


def install(root='/kaggle/temp/yue2_native', outputs='/kaggle/working/yue2_native/runs',
            *, assets='auto', require_gpu=True):
    runtime = KaggleYuE2(root)
    outputs = Path(outputs).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    temp = runtime.root / 'tmp'
    temp.mkdir(exist_ok=True)
    env = runtime.environment()
    env.update({'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
                'HF_HOME': str(runtime.root / 'huggingface'),
                'HF_HUB_CACHE': str(runtime.root / 'huggingface/hub'),
                'HF_XET_CACHE': str(runtime.root / 'huggingface/xet'),
                'XDG_CACHE_HOME': str(runtime.root / 'cache'),
                'TORCH_HOME': str(runtime.root / 'cache/torch'),
                'TORCHINDUCTOR_CACHE_DIR': str(runtime.root / 'cache/inductor'),
                'TRITON_CACHE_DIR': str(runtime.root / 'cache/triton'),
                'GRADIO_TEMP_DIR': str(temp / 'gradio'),
                'YUE2_COVER_ROOT': str(outputs.parent / 'cover_sources'),
                'YUE2_OUTPUT_ROOT': str(outputs), 'YUE2_RUNTIME_ROOT': str(runtime.root), 'PIP_NO_CACHE_DIR': '1',
                'GRADIO_ANALYTICS_ENABLED': 'False'})
    from .assets import find_assets, activate_assets
    dataset = find_assets(assets)
    if dataset is not None:
        env = activate_assets(dataset, runtime.root, env)
        print(f'Using saved Kaggle assets: {dataset}; model downloads disabled; saved wheels reused.', flush=True)
    for label, path in [('runtime', runtime.root), ('outputs', outputs)]:
        free = shutil.disk_usage(path).free / 2**30
        print(f'{label}: {path}; {free:.1f} GiB free', flush=True)
        needed = (3 if dataset is not None else 16) if label == 'runtime' else 2
        if free < needed:
            raise RuntimeError(f'{path} needs at least {needed} GiB free; found {free:.1f}')
    print('Saved input models stay read-only; the session overlay is temporary. Song outputs must be saved.', flush=True)
    from .process import stop_owned, check_gpu_room
    if require_gpu:
        stop_owned(runtime.root, Path(__file__).with_name('engine.py'), outputs)
        check_gpu_room()
    runtime.prepare_environment()
    constraints = []
    for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 'numba', 'llvmlite'):
        try:
            constraints.append(f'{name}=={importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            pass
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('FFmpeg is required for cover audio decoding; select a Kaggle image that includes it')
    if not any(p.startswith('torchaudio==') for p in constraints):
        raise RuntimeError('Cover transcription needs the image-matched TorchAudio; use a current Kaggle GPU image')
    if not any(p.startswith('torch==') for p in constraints):
        raise RuntimeError('Select the Kaggle GPU image; its preinstalled PyTorch is required')
    constraint = runtime.root / 'kaggle-constraints.txt'
    constraint.write_text('\n'.join(constraints) + '\n')
    wheelhouse = dataset / 'wheelhouse' if dataset is not None else None
    overlay = ['-r', dataset / 'overlay-requirements.txt'] if dataset is not None else []
    pip_install(runtime.python, [*overlay, '-c', constraint, *REQUIREMENTS], env=env,
                wheelhouse=wheelhouse, purpose='the saved YuE2 wheels')
    wheel = runtime.root / 'yue2_infer-0.1.5-py3-none-any.whl'
    if dataset is not None:
        shutil.copyfile(dataset / 'wheelhouse' / wheel.name, wheel)
    if not wheel.exists():
        partial = wheel.with_suffix('.part')
        try:
            url = f'https://huggingface.co/m-a-p/YuE2-3B/resolve/{MODEL_REVISION}/{wheel.name}'
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out)
            if sha256_file(partial) != WHEEL_SHA256:
                raise RuntimeError('Native inference wheel checksum mismatch')
            os.replace(partial, wheel)
        finally:
            partial.unlink(missing_ok=True)
    if sha256_file(wheel) != WHEEL_SHA256:
        raise RuntimeError('Native inference wheel checksum mismatch')
    # Release metadata pins its development CUDA stack; preserve Kaggle's matched stack.
    command([runtime.python, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', wheel], env=env)
    probe = '''
import json, torch, gradio, transformers, huggingface_hub
import yue2.pipeline, yue2.cuda_graph, yue2.nar, yue2.modeling_vae
import torchaudio, scipy, mir_eval, pretty_midi, mido
from transformers.models.bart.modeling_bart import BartDecoder
import inspect
assert 'embed_tokens' in inspect.signature(BartDecoder).parameters, 'SheetSage2 requires the pinned BartDecoder API'
if REQUIRE_GPU:
    assert torch.cuda.is_available(), 'Select GPU T4 x2 in Kaggle settings'
    assert torch.cuda.device_count() >= 2, 'This restored configuration requires GPU T4 x2'
assert transformers.__version__ == '4.57.6'
assert gradio.__version__ == '5.49.1'
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
 'gradio': gradio.__version__, 'hub': huggingface_hub.__version__,
 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
'''
    probe = probe.replace('REQUIRE_GPU', repr(require_gpu))
    result = command([runtime.python, '-c', probe], env=env, capture=True)
    print(result.stdout, flush=True)
    (runtime.root / 'hardware.json').write_text(result.stdout)
    packages = command([runtime.python, '-m', 'pip', 'freeze'], env=env, capture=True).stdout
    (runtime.root / 'installed-packages.txt').write_text(packages)
    # Retain enough runtime information with the eventual saved outputs.
    meta = outputs.parent / 'runtime'
    meta.mkdir(exist_ok=True)
    shutil.copyfile(runtime.root / 'installed-packages.txt', meta / 'installed-packages.txt')
    shutil.copyfile(runtime.root / 'hardware.json', meta / 'hardware.json')
    print('Ready. Setup starts no song or diagnostic. Open the UI, select your duration, and generate your song.', flush=True)
    return str(runtime.python), env
