"""Isolated setup for the original native engine, retaining Kaggle's CUDA stack."""
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request

from kaggle_yue2.runtime import KaggleYuE2, command
from kaggle_yue2.recovery import sha256_file

MODEL_REVISION = '8f24312f187763b9854adeec87e437e2b03bbff1'
WHEEL_SHA256 = '8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641'
REQUIREMENTS = ['transformers==4.57.6', 'huggingface-hub==0.36.2', 'gradio==5.49.1',
                'pillow==11.3.0', 'soundfile==0.13.1', 'accelerate==1.13.0',
                'tiktoken==0.12.0', 'safetensors>=0.7.0',
                'mir_eval==0.8.2', 'pretty_midi==0.2.10', 'mido==1.3.3']


def install(root='/kaggle/temp/yue2_native', outputs='/kaggle/working/yue2_native/runs'):
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
    for label, path in [('runtime', runtime.root), ('outputs', outputs)]:
        free = shutil.disk_usage(path).free / 2**30
        print(f'{label}: {path}; {free:.1f} GiB free', flush=True)
        needed = 16 if label == 'runtime' else 2
        if free < needed:
            raise RuntimeError(f'{path} needs at least {needed} GiB free; found {free:.1f}')
    print('Runtime/model files are temporary. Only song artifacts belong in saved Kaggle outputs.', flush=True)
    from .process import stop_owned, check_gpu_room
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
    command([runtime.python, '-m', 'pip', 'install', '--no-cache-dir', '-c', constraint, *REQUIREMENTS], env=env)
    wheel = runtime.root / 'yue2_infer-0.1.5-py3-none-any.whl'
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
assert torch.cuda.is_available(), 'Select GPU T4 x2 in Kaggle settings'
assert torch.cuda.device_count() >= 2, 'This restored configuration requires GPU T4 x2'
assert transformers.__version__ == '4.57.6'
assert gradio.__version__ == '5.49.1'
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
 'gradio': gradio.__version__, 'hub': huggingface_hub.__version__,
 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
'''
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
