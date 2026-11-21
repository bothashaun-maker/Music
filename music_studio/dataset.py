"""Create ONE private Music Studio Dataset through Kaggle's normal uploader.

The notebook is only a driver: it verifies credentials, picks measured scratch
space and calls :func:`create_dataset` once. Nothing here downloads a model or
compiles code until authentication and free space have both been confirmed.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .common import (GIB, HERE, PROFILE, catalog, choose_scratch, fingerprint,
                     scrub, transfer, validate_manifest, verify, write_json)

# Checked in order; Kaggle's scratch sizes vary by image and session, so the
# builder measures instead of assuming /kaggle/temp is always the largest.
SCRATCH_CANDIDATES = ('/kaggle/temp', '/tmp', '/kaggle/working')
BUILD_REQUIRED = 8 * GIB          # compile gate (7 GiB) plus toolkit extraction
DOWNLOAD_RESERVE = GIB            # largest cached model plus working headroom


def username(api):
    """The authenticated Kaggle username, from the SDK's own configuration."""
    values = getattr(api, 'config_values', None)
    if isinstance(values, dict) and values.get('username'):
        return str(values['username'])
    for source in (os.environ.get('KAGGLE_USERNAME'), ):
        if source:
            return str(source)
    stored = Path.home() / '.kaggle' / 'username'
    if stored.is_file():
        return stored.read_text().strip()
    return ''


def authenticated_api():
    """Fail before any download or compilation when credentials are missing."""
    if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
        except Exception:
            pass
    if not os.environ.get('KAGGLE_API_TOKEN') and not os.environ.get('KAGGLE_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            os.environ['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
        except Exception:
            pass
    try:
        installed = importlib.metadata.version('kaggle')
    except importlib.metadata.PackageNotFoundError:
        installed = ''
    if installed != '2.0.2':
        # Dedicated helper directory: never replace Kaggle's Torch/scientific packages.
        package_dir = Path('/kaggle/working/music_studio_dataset/.api')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target', str(package_dir), 'kaggle==2.0.2'])
        sys.path.insert(0, str(package_dir))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as error:
        raise RuntimeError('Kaggle authentication failed. Enable the private Kaggle Secret KAGGLE_API_TOKEN and rerun. ' + scrub(error)) from None
    return api


def identity(owner, manifest):
    if not isinstance(owner, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise ValueError('Set DATASET_OWNER to your Kaggle username if it cannot be detected from your private credentials.')
    suffix = fingerprint({'catalog': manifest['catalog_fingerprint'], 'runtime': manifest['runtime']['source_fingerprint'],
                          'compatibility': manifest['runtime'].get('compatibility', {})})[:12]
    return owner + '/music-studio-dataset-v1-' + suffix, suffix


def validate_catalog(data):
    """Reject a truncated or edited catalog before a 25 GB build starts."""
    if data.get('schema') != 1 or data.get('profile') != PROFILE:
        raise RuntimeError('The embedded Music Studio catalog is not the expected profile.')
    assets = data.get('assets')
    if not isinstance(assets, list) or not assets:
        raise RuntimeError('The embedded Music Studio catalog lists no assets.')
    names = set()
    for record in assets:
        name = record.get('name', '')
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise RuntimeError('Invalid catalog asset name: ' + repr(name))
        if name in names:
            raise RuntimeError('Duplicate catalog asset name: ' + name)
        names.add(name)
        if not isinstance(record.get('bytes'), int) or record['bytes'] <= 0:
            raise RuntimeError('Invalid catalog size for ' + name)
        digest = str(record.get('sha256', ''))
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RuntimeError('Invalid catalog checksum for ' + name)
        if not str(record.get('source', '')).startswith(('https://', 'http://')):
            raise RuntimeError('Invalid catalog source for ' + name)
        if not record.get('path'):
            raise RuntimeError('Missing saved path for ' + name)
    return data


def reuse_runtime(input_root):
    """Reuse the runtime already saved in an attached Music Studio Dataset."""
    root = Path(input_root)
    if not root.exists():
        return None
    for path in sorted(root.rglob('music-studio-manifest.json')):
        try:
            data = validate_manifest(json.loads(path.read_text()))
            verify(path.parent / data['runtime']['name'], data['runtime'])
            return path.parent / data['runtime']['name'], data['runtime']
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def clean_build_intermediates(work):
    """Only directories owned by this notebook's isolated compiler build."""
    work = Path(work).resolve()
    if not (work / 'runtime-build.json').is_file():
        raise RuntimeError('No completed build receipt; compiler files were retained for diagnosis.')
    for name in ('bundle', 'studio-source', 'engine-source', 'tools-source', 'whisper-source', 'cargo', 'rustup', 'cuda', 'node', 'build-tools', 'unpack-onnxruntime', 'unpack-llama.cpp'):
        path = work / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def build_runtime_record(scratch, jobs=2):
    """Compile the x86-64/SM75 runtime once on CPU and return its saved record."""
    from .build_runtime import build
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print('Building the pinned Linux x86-64/T4 runtime on CPU. This is the long step; the log is',
          scratch / 'runtime-build.log', flush=True)
    runtime = build(scratch, jobs=jobs)
    runtime_path = scratch / runtime['name']
    verify(runtime_path, runtime)
    clean_build_intermediates(scratch)
    print(f'Saved runtime verified: {runtime["name"]} {runtime["bytes"] / GIB:.2f} GiB', flush=True)
    return runtime_path, runtime


def dataset_manifest(data, runtime):
    return dict(schema=1, profile=PROFILE, complete=True, catalog_fingerprint=fingerprint(data),
                files=data['assets'], runtime=runtime,
                model_bytes=sum(r['bytes'] for r in data['assets']))


def publish(manifest, runtime_path, *, output_root, input_root='/kaggle/input', owner='', api=None, cache_root=None):
    """One native create request, streaming one verified model file at a time."""
    validate_manifest(manifest)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if api is None:
        api = authenticated_api()
    handle, suffix = identity(owner or username(api), manifest)
    url = 'https://www.kaggle.com/datasets/' + handle
    receipt = root / 'dataset-upload.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get('submitted') and old.get('handle') == handle:
            print('Previously submitted private Music Studio Dataset:', url, flush=True)
            return url
    signature = inspect.signature(api.upload_files)
    if not {'folder', 'request'}.issubset(signature.parameters):
        raise RuntimeError('Unsupported Kaggle uploader API; no model downloads started.')
    sources = {}
    for record in manifest['files']:
        for search in (Path(input_root), root):
            if search.exists():
                for found in search.rglob(record['name']):
                    try:
                        verify(found, record)
                        sources[record['name']] = found.resolve()
                        break
                    except RuntimeError:
                        continue
                if record['name'] in sources:
                    break
    cache = Path(cache_root or root / '.download-cache').resolve()
    if cache.is_symlink():
        raise RuntimeError('Model cache must not be a symlink.')
    cache.mkdir(parents=True, exist_ok=True)
    missing = [r for r in manifest['files'] if r['name'] not in sources]
    peak = max((r['bytes'] for r in missing), default=0)
    free = shutil.disk_usage(cache).free
    print(f'Music Studio Dataset: {len(manifest["files"])} saved assets; {manifest["model_bytes"] / 1e9:.2f} GB plus the compiled runtime.', flush=True)
    print(f'Largest missing file: {peak / 1e9:.2f} GB. Measured free cache space: {free / 1e9:.2f} GB.', flush=True)
    if free < peak + 512 * 1024**2:
        raise RuntimeError('Need space for the largest missing model plus 512 MiB; the builder does not hold a second full copy of the Dataset.')
    verify(runtime_path, manifest['runtime'])
    metadata = dict(title='Music Studio Dataset ' + suffix, id=handle, licenses=[{'name': 'other'}],
                    description='Private Music-Studio runtime and models. Individual component terms apply. No model is relicensed. See music-studio-manifest.json and MUSIC-STUDIO-NOTICES.md.')
    original = api.upload_files
    uploaded = []
    finished = False

    def stream_upload(*args, **kwargs):
        nonlocal finished
        bound = signature.bind(*args, **kwargs)
        request = bound.arguments['request']

        def send(folder, expected):
            call = signature.bind(*args, **kwargs)
            call.arguments['folder'] = str(folder)
            before = len(request.files)
            original(*call.args, **call.kwargs)
            if len(request.files) != before + expected:
                raise RuntimeError('Kaggle did not return all expected file tokens.')

        records = [manifest['runtime']] + sorted(manifest['files'], key=lambda r: r['name'] not in sources)
        for record in records:
            if record is manifest['runtime']:
                source, owned = Path(runtime_path), False
            else:
                source = sources.get(record['name'], cache / record['name'])
                owned = record['name'] not in sources
                if owned:
                    transfer(record, source)
            verify(source, record)
            with tempfile.TemporaryDirectory(prefix='.upload-one-', dir=root) as temporary:
                # The normal SDK reads the actual bytes behind this absolute symlink.
                (Path(temporary) / record['name']).symlink_to(source.resolve())
                print('Uploading verified asset:', record['name'], flush=True)
                send(temporary, 1)
            uploaded.append(record['name'])
            write_json(root / 'upload-progress.json', dict(handle=handle, uploaded_files=uploaded, dataset_created=False))
            if owned:
                source.unlink()  # Never delete an input, existing pack or user source.
        with tempfile.TemporaryDirectory(prefix='.upload-notices-', dir=root) as temporary:
            folder = Path(temporary)
            write_json(folder / 'music-studio-manifest.json', manifest)
            shutil.copy2(HERE / 'NOTICES.md', folder / 'MUSIC-STUDIO-NOTICES.md')
            send(folder, 2)
        finished = True

    try:
        api.upload_files = stream_upload
        # Preserve the user's working direct create_new pattern; add no private
        # datasets.get/status prerequisite and never version/overwrite a Dataset.
        with tempfile.TemporaryDirectory(prefix='.dataset-request-', dir=root) as temporary:
            write_json(Path(temporary) / 'dataset-metadata.json', metadata)
            result = api.dataset_create_new(temporary, public=False, quiet=False, convert_to_csv=False, dir_mode='skip')
        field = lambda name: result.get(name) if isinstance(result, dict) else getattr(result, name, None)
        if result is None or field('error') or field('error_message') or str(field('status')).lower() in ('error', 'failed', 'failure'):
            raise RuntimeError('Kaggle did not confirm Dataset creation: ' + scrub(field('error') or field('error_message') or field('status') or 'empty response'))
        if not finished:
            raise RuntimeError('The Dataset uploader did not finish; no success receipt was written.')
        write_json(root / 'music-studio-manifest.json', manifest)
        write_json(receipt, dict(handle=handle, url=url, submitted=True, requested_private=True, catalog_fingerprint=manifest['catalog_fingerprint']))
        print('PRIVATE MUSIC STUDIO DATASET SUBMITTED:', url, flush=True)
        print('Wait for processing. In Music-Studio.ipynb use Add Input → Your Datasets → this ONE Music Studio Dataset.', flush=True)
        print('The Dataset saves tools and models. Save generated songs separately with Kaggle output files.', flush=True)
        return url
    except (Exception, SystemExit) as error:
        message = scrub(error)
        write_json(root / 'dataset-error.json', dict(stage='private-dataset-creation', reason=message, creation_confirmed=False,
                    existing_sources_untouched=True, uploaded_temporary_files_may_be_released=True))
        raise RuntimeError(message + ' Creation is not confirmed; check the Dataset page before retrying after an interrupted final request.') from None
    finally:
        api.upload_files = original


def _builder_lock(root):
    """One builder per output directory. Kaggle is Linux; other hosts skip locking."""
    import contextlib
    try:
        import fcntl
    except ImportError:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def locked():
        with (Path(root) / '.builder.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Music Studio Dataset builder is already running.') from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return locked()


def create_dataset(catalog_data, receipt_dir='/kaggle/working/music_studio_dataset', input_root='/kaggle/input',
                   cache_root=None, owner='', jobs=2, api=None, build_scratch=None):
    """Build and submit ONE private Music Studio Dataset.

    Mirrors the other automatic Dataset builders: validate the saved catalog,
    authenticate first, reuse an attached verified runtime when possible,
    measure scratch space, then stream one verified file at a time.
    """
    validate_catalog(catalog_data)
    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=True)
    with _builder_lock(root):
        api = api or authenticated_api()
        owner = owner or username(api)
        if not owner:
            raise RuntimeError('Set DATASET_OWNER to your Kaggle username; no build or download started.')
        print('Authenticated as', owner, '- one private Dataset will be created.', flush=True)
        previous = reuse_runtime(input_root)
        if previous:
            runtime_path, runtime = previous
            print('Reusing the verified runtime from an attached Dataset:', runtime_path.name, flush=True)
        else:
            scratch = Path(build_scratch or choose_scratch(BUILD_REQUIRED, SCRATCH_CANDIDATES, 'music-studio-build'))
            runtime_path, runtime = build_runtime_record(scratch, jobs)
        manifest = dataset_manifest(catalog_data, runtime)
        if cache_root is None:
            cache_root = choose_scratch(max(r['bytes'] for r in catalog_data['assets']) + DOWNLOAD_RESERVE,
                                        SCRATCH_CANDIDATES, 'music-studio-downloads')
        url = publish(manifest, runtime_path, output_root=root, input_root=input_root, owner=owner,
                      api=api, cache_root=cache_root)
        return dict(handle='/'.join(url.rsplit('/', 2)[-2:]), url=url,
                    receipt=str(root / 'dataset-upload.json'))


def create(output_root='/kaggle/working/music_studio_dataset', input_root='/kaggle/input', owner='', jobs=2, api=None):
    """Compatibility wrapper for `python -m music_studio.dataset`."""
    return create_dataset(catalog(), output_root, input_root=input_root, owner=owner, jobs=jobs, api=api)


if __name__ == '__main__':
    create(owner=os.environ.get('MUSIC_STUDIO_DATASET_OWNER', ''), jobs=int(os.environ.get('MUSIC_STUDIO_BUILD_JOBS', '2')))
