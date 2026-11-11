"""Verified transfers, saved-pack validation and archive handling. Standard library only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.error
import urllib.request

PROFILE = 'music-studio-v2'
HERE = Path(__file__).resolve().parent


def bind_parent(expected=None):
    """Bind the launcher or its child to its actual Linux parent, checking the race."""
    import ctypes
    import signal
    expected = os.getppid() if expected is None else int(expected)
    if expected <= 1:
        raise RuntimeError('The notebook parent exited before startup.')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Could not bind Music Studio to its notebook parent')
    if os.getppid() != expected:
        os.kill(os.getpid(), signal.SIGTERM)
        raise RuntimeError('The notebook parent exited during startup.')


def notebook_process():
    """Install cleanup signals before binding a module entry point to Jupyter."""
    import signal
    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    bind_parent(os.environ.get('MUSIC_STUDIO_PARENT_PID', os.getppid()))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def catalog():
    return json.loads((HERE / 'catalog.json').read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(path, record):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
        raise RuntimeError('Verification failed: ' + str(path))
    return path


def transfer(record, destination, reserve=512 * 1024**2):
    """Resume owned partial downloads, then verify both length and digest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verify(destination, record)
    partial = destination.with_name(destination.name + '.partial')
    if partial.is_symlink() or destination.is_symlink():
        raise RuntimeError('Download destination must not be a symlink.')
    last = None
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == record['bytes']:
            try:
                verify(partial, record)
                partial.replace(destination)
                return destination
            except RuntimeError:
                partial.unlink()
                offset = 0
        elif offset > record['bytes']:
            partial.unlink()
            offset = 0
        if shutil.disk_usage(destination.parent).free < record['bytes'] - offset + reserve:
            raise RuntimeError('Insufficient measured free disk for ' + destination.name)
        headers = {'User-Agent': 'Music-Studio/1'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(record['source'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected range response for ' + destination.name)
                written = offset if resumed else 0
                reported = written
                with partial.open('ab' if resumed else 'wb') as output:
                    while chunk := response.read(8 * 1024**2):
                        written += len(chunk)
                        if written > record['bytes']:
                            raise RuntimeError('Server sent more bytes than the pinned asset.')
                        output.write(chunk)
                        if written - reported >= 256 * 1024**2:
                            print(f'{destination.name}: {written / record["bytes"]:.0%}', flush=True)
                            reported = written
            verify(partial, record)
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            last = error
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError('Download did not verify: ' + destination.name + ': ' + str(last))


def extract(archive, destination):
    """Reject path traversal, escaping links, devices and archives that exhaust disk."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        required = sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(destination).free < required + 128 * 1024**2:
            raise RuntimeError('Insufficient disk to unpack ' + str(archive))
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.isdev() or member.isfifo():
                raise RuntimeError('Unsafe archive member: ' + member.name)
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                if link.is_absolute():
                    raise RuntimeError('Absolute link in archive: ' + member.name)
                base = name.parent if member.issym() else PurePosixPath('.')
                target = (destination / base / link).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise RuntimeError('Escaping link in archive: ' + member.name)
        source.extractall(destination, filter='data')
    return destination


def elf_arch(path):
    with open(path, 'rb') as stream:
        header = stream.read(20)
    if header[:4] != b'\x7fELF' or header[4] != 2 or header[5] != 1:
        raise RuntimeError('Expected a 64-bit little-endian Linux binary: ' + str(path))
    machine = int.from_bytes(header[18:20], 'little')
    if machine != 62:
        raise RuntimeError(f'Expected x86-64 ELF machine 62; found {machine}: {path}')
    return 'x86_64'


def validate_manifest(manifest):
    expected = catalog()
    if manifest.get('profile') != PROFILE or manifest.get('schema') != 1 or manifest.get('complete') is not True:
        raise RuntimeError('Attach the Music Studio Dataset made by Music-Studio-Dataset.ipynb.')
    if manifest.get('catalog_fingerprint') != fingerprint(expected):
        raise RuntimeError('The Dataset belongs to a different Music-Studio model catalog.')
    actual = manifest.get('files', [])
    wanted = {(a['name'], a['bytes'], a['sha256'], a['path']) for a in expected['assets']}
    got = {(a.get('name'), a.get('bytes'), a.get('sha256'), a.get('path')) for a in actual}
    if got != wanted or len(actual) != len(wanted):
        raise RuntimeError('The Dataset model allowlist is incomplete or changed.')
    runtime = manifest.get('runtime', {})
    if runtime.get('architecture') != 'x86_64' or runtime.get('cuda_architectures') != [75]:
        raise RuntimeError('The saved runtime is not the x86-64/T4 Music-Studio build.')
    if runtime.get('build_mode') != 'cuda':
        raise RuntimeError('The saved runtime does not contain the required CUDA build.')
    source = json.loads((HERE / 'sources.json').read_text())
    if runtime.get('source_fingerprint') != fingerprint(source):
        raise RuntimeError('Runtime source identity differs from this notebook.')
    for record in [runtime] + actual:
        name = record.get('name', '')
        if not name or PurePosixPath(name).name != name or not isinstance(record.get('bytes'), int):
            raise RuntimeError('Invalid saved file identity.')
        digest = record.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RuntimeError('Invalid saved file checksum.')
    return manifest


def find_pack(selection='auto', input_root='/kaggle/input'):
    if selection != 'auto':
        choices = [Path(selection)]
    else:
        root = Path(input_root)
        choices = sorted({p.parent for p in root.rglob('music-studio-manifest.json')}) if root.exists() else []
    valid = []
    for path in choices:
        try:
            data = json.loads((path / 'music-studio-manifest.json').read_text())
            validate_manifest(data)
            valid.append((path.resolve(), data))
        except (ValueError, OSError, RuntimeError):
            if selection != 'auto':
                raise
    if len(valid) != 1:
        raise RuntimeError(f'Found {len(valid)} matching Music Studio Datasets. Attach one, or set DATASET_PATH explicitly.')
    path, data = valid[0]
    for record in [data['runtime']] + data['files']:
        print('Verifying saved asset:', record['name'], flush=True)
        verify(path / record['name'], record)
    return path, data


def scrub(error):
    import re
    message = str(error)
    for key in ('KAGGLE_API_TOKEN', 'KAGGLE_KEY', 'HF_TOKEN', 'OPENROUTER_API_KEY'):
        secret = os.environ.get(key)
        if secret:
            message = message.replace(secret, '[REDACTED]')
    message = re.sub(r'(?i)Bearer\s+[^\s,;]+', 'Bearer [REDACTED]', message)
    return re.sub(r'(?i)([?&]key=)[^\s&]+', r'\1[REDACTED]', message)[:2000]
