"""Build and consume a read-only Kaggle input bundle; never silently fetch on reuse."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from kaggle_yue2.recovery import sha256_file

SCHEMA = 1
BINARY_PACKAGES = ('torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 'numba', 'llvmlite')
MODELS = {
    'music': ('m-a-p/YuE2-3B', '8f24312f187763b9854adeec87e437e2b03bbff1'),
    'vae': ('m-a-p/YuE2-Vae', '152733a19ad43aa67e367f9b5503ef8075bb5126'),
    'melody': ('m-a-p/SheetSage2', 'eab522a8168e8b8b8c4856bf8609cd86198f01fe'),
    'mert': ('m-a-p/MERT-v2-FullSong', 'd8ba1c745e733b3908ce6ad16ebeb17ac7600a42'),
    'lyrics': ('openai/whisper-large-v3-turbo', '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'),
}
# No demos, alternate PyTorch/ONNX weights, images or audio examples.
PATTERNS = ['*.json', '*.py', '*.tiktoken', '*.safetensors', '*.txt', '*.model',
            'LICENSE*', 'README.md', 'THIRD_PARTY_NOTICES.md', 'licenses/*']


def image_identity():
    versions = {}
    for name in BINARY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name).split('+')[0]
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': list(sys.version_info[:2]), 'machine': platform.machine(),
            'system': platform.system(), 'binary_packages': versions}


def find_assets(selection='auto', input_root='/kaggle/input'):
    if selection is None or str(selection).strip().lower() == 'online':
        return None
    if str(selection) != 'auto':
        selected = Path(selection).expanduser().resolve()
        if not (selected / 'assets.json').is_file():
            raise RuntimeError(f'No complete YuE2 assets.json at {selected}. Attach the saved builder output.')
        return selected
    root = Path(input_root)
    # New and legacy Kaggle input mount layouts, including notebook outputs.
    candidates = sorted({p.parent.resolve() for p in root.rglob('assets.json')}) if root.exists() else []
    valid = []
    for path in candidates:
        try:
            if json.loads((path / 'assets.json').read_text()).get('kind') == 'yue2-kaggle-assets':
                valid.append(path)
        except (ValueError, OSError):
            continue
    if len(valid) != 1:
        raise RuntimeError('Attach ONE saved YuE2 assets dataset/output using Add Input, then rerun. '
                           'AUTO will not download replacements. If multiple versions are attached, '
                           'set ASSET_DATASET to the exact folder containing assets.json. '
                           'Use the separate asset-builder notebook for first-time setup.')
    return valid[0]


def validate_assets(root, *, full=False, check_image=True):
    root = Path(root).resolve()
    info = json.loads((root / 'assets.json').read_text())
    if info.get('kind') != 'yue2-kaggle-assets' or info.get('schema') != SCHEMA or info.get('models') != {k: list(v) for k,v in MODELS.items()}:
        raise RuntimeError('Unsupported or outdated YuE2 dataset; rebuild with the current asset builder')
    if check_image and info.get('image') != image_identity():
        raise RuntimeError('Kaggle Python/platform or binary package versions differ from the saved dataset. '
                           'Use the same Kaggle image version as the builder, or rebuild assets on this image. '
                           'No network fallback was attempted.')
    files = info.get('files', {})
    required = {'wheelhouse', 'hub', 'gradio'}
    if (not files or 'overlay-requirements.txt' not in files
            or not required <= {Path(name).parts[0] for name in files}):
        raise RuntimeError('Incomplete assets manifest')
    for name, record in files.items():
        file = root / name
        if Path(name).is_absolute() or not file.resolve().is_relative_to(root):
            raise RuntimeError('Invalid dataset manifest path')
        if not file.is_file() or file.stat().st_size != record['bytes']:
            raise RuntimeError(f'Missing/incomplete dataset file: {name}. Reattach a complete saved version.')
        # Native loaders hash all music/VAE weights and MERT checks its parent.
        # Hash small files on reuse; full validation is available without any GPU.
        if (full or not name.endswith('.safetensors')) and sha256_file(file) != record['sha256']:
            raise RuntimeError(f'Dataset checksum mismatch: {name}')
    for repo, revision in MODELS.values():
        snapshot = root / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        if not (snapshot / 'config.json').is_file() or not list(snapshot.glob('*.safetensors')):
            raise RuntimeError(f'Missing pinned model snapshot: {repo}')
    return info


def activate_assets(root, runtime_root, env):
    """Return child environment; models remain on read-only input, no multi-GB copy."""
    root, runtime_root = Path(root).resolve(), Path(runtime_root)
    validate_assets(root)
    from kaggle_yue2.runtime import PIP_WHEEL, PIP_SHA256
    source = root / 'wheelhouse' / PIP_WHEEL
    if not source.is_file() or sha256_file(source) != PIP_SHA256:
        raise RuntimeError('Missing or invalid saved bootstrap pip wheel')
    target = runtime_root / 'bootstrap' / PIP_WHEEL
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    gradio = runtime_root / 'huggingface/gradio/frpc'
    gradio.mkdir(parents=True, exist_ok=True)
    for source in (root / 'gradio').iterdir():
        if source.is_file():
            destination = gradio / source.name
            shutil.copyfile(source, destination)
            destination.chmod(0o700)  # /kaggle/input may be mounted noexec/read-only.
    env = dict(env)
    env.update({'YUE2_ASSET_DATASET': str(root), 'HF_HUB_CACHE': str(root / 'hub'),
                'HUGGINGFACE_HUB_CACHE': str(root / 'hub'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_MODULES_CACHE': str(runtime_root / 'modules'),
                'PIP_NO_INDEX': '1', 'PIP_FIND_LINKS': str(root / 'wheelhouse'),
                'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    return env


def build_assets(destination, runtime_root):
    """Run in the installed overlay on Kaggle, without loading any GPU model."""
    from huggingface_hub import snapshot_download
    from kaggle_yue2.runtime import PIP_WHEEL
    from .setup import WHEEL_SHA256, REQUIREMENTS
    destination, runtime_root = Path(destination).resolve(), Path(runtime_root)
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / 'assets.json'
    if complete.exists():
        validate_assets(destination, full=True)
        print(f'Existing complete dataset verified: {destination}', flush=True)
        return
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        raise RuntimeError('Asset builder requires Internet; do not run it in dataset-reuse mode')
    wheels = destination / 'wheelhouse'
    wheels.mkdir(exist_ok=True)
    for source in (runtime_root / 'bootstrap' / PIP_WHEEL,
                   runtime_root / 'yue2_infer-0.1.5-py3-none-any.whl'):
        shutil.copyfile(source, wheels / source.name)
    if sha256_file(wheels / 'yue2_infer-0.1.5-py3-none-any.whl') != WHEEL_SHA256:
        raise RuntimeError('Native inference wheel checksum mismatch')
    # Freeze the overlay rather than downloading a second Torch/CUDA stack.
    installed = json.loads(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'list', '--local', '--format=json'], text=True))
    locked = sorted(f"{p['name']}=={p['version']}" for p in installed
                    if p['name'].lower().replace('_', '-') not in {'pip', 'yue2-infer'})
    if locked:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(wheels),
                        *locked], check=True)
    (destination / 'overlay-requirements.txt').write_text('\n'.join(locked) + '\n')
    (destination / 'kaggle-image.json').write_text(json.dumps(image_identity(), indent=2) + '\n')
    # Resolve in a FRESH overlay, not the builder that already has the packages.
    # Do this before spending time/bandwidth on model downloads.
    with tempfile.TemporaryDirectory(prefix='offline-wheel-check-', dir=runtime_root) as check:
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', check], check=True)
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(Path(check) / 'bin/python'),
                        'install', '--dry-run', '--no-index', '--find-links', str(wheels),
                        '-c', str(runtime_root / 'kaggle-constraints.txt'),
                        '-r', str(destination / 'overlay-requirements.txt'), *REQUIREMENTS], check=True)
    hub = destination / 'hub'
    for label, (repo, revision) in MODELS.items():
        print(f'Saving {label}: {repo} @ {revision}. No inference or GPU required.', flush=True)
        snapshot_download(repo, revision=revision, cache_dir=str(hub), allow_patterns=PATTERNS,
                          max_workers=2)
    # Keep HF cache refs/snapshots for nested SheetSage/MERT Hub lookups. Turn
    # snapshot links into ordinary files; Kaggle output packaging must not lose
    # links or save a duplicate blobs tree. Hard-link each blob before removing the blobs tree; no duplicate disk allocation.
    for link in sorted(hub.rglob('*')):
        if link.is_symlink():
            source = link.resolve(strict=True)
            if not source.is_relative_to(hub.resolve()):
                raise RuntimeError('Unexpected external Hub cache link')
            temporary = link.with_name(link.name + '.materializing')
            temporary.unlink(missing_ok=True)
            os.link(source, temporary)
            os.replace(temporary, link)
    for blobs in hub.glob('models--*/blobs'):
        shutil.rmtree(blobs)
    shutil.rmtree(hub / '.locks', ignore_errors=True)
    # Pre-cache the executable too; Gradio must not download it on every launch.
    from gradio.tunneling import Tunnel, BINARY_PATH, BINARY_URL, CHECKSUMS
    Tunnel.download_binary()
    if sha256_file(BINARY_PATH) != CHECKSUMS[BINARY_URL]:
        raise RuntimeError('Gradio tunnel binary checksum mismatch')
    gradio = destination / 'gradio'
    gradio.mkdir(exist_ok=True)
    shutil.copyfile(BINARY_PATH, gradio / Path(BINARY_PATH).name)
    (destination / 'README.txt').write_text(
        'Private YuE2 Kaggle assets. Attach this output/dataset to the generator.\n'
        'Includes upstream model cards/licenses; their original terms apply.\n'
        'No song generation was run to create these assets.\n')
    records = {str(p.relative_to(destination)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
               for p in sorted(destination.rglob('*')) if p.is_file() and p != complete}
    manifest = {'kind': 'yue2-kaggle-assets', 'schema': SCHEMA,
                'models': {k: list(v) for k,v in MODELS.items()}, 'image': image_identity(), 'files': records}
    partial = complete.with_suffix('.partial')
    partial.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(partial, complete)
    validate_assets(destination, full=False)
    total = sum(record['bytes'] for record in records.values()) / 2**30
    print(f'COMPLETE: {destination} ({total:.2f} GiB). Save Version with outputs. '
          'Create a PRIVATE dataset from that saved output, or attach the saved notebook output directly. '
          'Do not download/re-upload it through your computer.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    parser.add_argument('--runtime-root', required=True)
    args = parser.parse_args()
    build_assets(args.build, args.runtime_root)
