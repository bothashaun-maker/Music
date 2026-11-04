"""Lossless resumable cover uploads; independent of the model and generation nodes."""
import asyncio
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading

CHUNK_BYTES = 1024 * 1024
MAX_BYTES = 2 * 1024 ** 3
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.opus',
                    '.aiff', '.aif', '.webm', '.wma'}


def register_routes(server, input_directory, temporary_directory):
    """Register small requests on the existing Comfy server; never load a model."""
    from aiohttp import web
    locks = {}
    publish_lock = threading.Lock()
    inputs = Path(input_directory)
    parts_root = Path(temporary_directory) / 'yue2-cover-uploads'
    inputs.mkdir(parents=True, exist_ok=True)
    parts_root.mkdir(parents=True, exist_ok=True)

    def specification(values):
        if not hasattr(values, 'get'):
            raise ValueError('Invalid upload metadata')
        ident = str(values.get('upload', ''))
        if not re.fullmatch('[a-f0-9]{32}', ident):
            raise ValueError('Invalid upload ID')
        size = int(values.get('size', 0))
        count = int(values.get('n', 0))
        if not 0 < size <= MAX_BYTES or count != math.ceil(size / CHUNK_BYTES):
            raise ValueError('Choose a nonempty audio file no larger than 2 GiB')
        raw = str(values.get('name', '')).replace('\\', '/').rsplit('/', 1)[-1]
        extension = Path(raw).suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            raise ValueError('Choose a WAV, FLAC, MP3 or other supported audio file')
        stem = re.sub(r'[^A-Za-z0-9 _.()\[\]-]', '_', Path(raw).stem)[:100].strip(' .') or 'song'
        return parts_root / ident, {'name': stem + extension, 'size': size, 'n': count}

    def prepare(directory, metadata):
        directory.mkdir(exist_ok=True)
        path = directory / 'request.json'
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError('This upload ID belongs to a different file; select the file again')
        else:
            temporary = directory / 'request.writing'
            temporary.write_text(json.dumps(metadata))
            temporary.replace(path)

    def complete(directory):
        receipt = directory / 'complete.json'
        if not receipt.is_file():
            return None
        result = json.loads(receipt.read_text())
        if not (inputs / result['filename']).is_file():
            raise ValueError('The uploaded song was removed; start a new upload')
        return result

    def assemble(directory, metadata):
        descriptor, name = tempfile.mkstemp(prefix='.yue2-cover-', dir=inputs)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as target:
                for index in range(metadata['n']):
                    with (directory / f'{index}.part').open('rb') as source:
                        shutil.copyfileobj(source, target, CHUNK_BYTES)
            if temporary.stat().st_size != metadata['size']:
                raise ValueError('Upload size mismatch; completed pieces were retained')
            with publish_lock:
                destination = inputs / metadata['name']
                index = 2
                while destination.exists():
                    source_name = Path(metadata['name'])
                    destination = inputs / f'{source_name.stem} ({index}){source_name.suffix}'
                    index += 1
                temporary.replace(destination)
            result = {'ok': True, 'filename': destination.name, 'bytes': metadata['size']}
            receipt = directory / 'complete.writing'
            receipt.write_text(json.dumps(result))
            receipt.replace(directory / 'complete.json')
            for piece in directory.glob('*.part'):
                piece.unlink()
            return result
        finally:
            temporary.unlink(missing_ok=True)

    @server.routes.get('/yue2/cover-upload/status')
    async def status(request):
        try:
            directory, metadata = specification(request.rel_url.query)
            async with locks.setdefault(str(directory), asyncio.Lock()):
                prepare(directory, metadata)
                result = complete(directory)
                done = list(range(metadata['n'])) if result else [
                    i for i in range(metadata['n']) if (directory / f'{i}.part').is_file()]
            return web.json_response({'completed': done}, headers={'Cache-Control': 'no-store'})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @server.routes.post('/yue2/cover-upload/chunk')
    async def chunk(request):
        try:
            directory, metadata = specification(request.rel_url.query)
            index = int(request.rel_url.query.get('i', -1))
            if not 0 <= index < metadata['n']:
                raise ValueError('Invalid piece index')
            async with locks.setdefault(str(directory), asyncio.Lock()):
                prepare(directory, metadata)
                if complete(directory):
                    return web.json_response({'ok': True})
                expected = min(CHUNK_BYTES, metadata['size'] - index * CHUNK_BYTES)
                descriptor, name = tempfile.mkstemp(prefix='.writing-', dir=directory)
                temporary = Path(name)
                try:
                    written = 0
                    with os.fdopen(descriptor, 'wb') as target:
                        async for block in request.content.iter_chunked(256 * 1024):
                            written += len(block)
                            if written > expected:
                                raise ValueError('Upload piece exceeds expected size')
                            target.write(block)
                    if written != expected:
                        raise ValueError('Incomplete upload piece; retry this piece')
                    temporary.replace(directory / f'{index}.part')
                finally:
                    temporary.unlink(missing_ok=True)
            return web.json_response({'ok': True, 'got': index})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @server.routes.post('/yue2/cover-upload/finish')
    async def finish(request):
        try:
            directory, metadata = specification(await request.json())
            async with locks.setdefault(str(directory), asyncio.Lock()):
                prepare(directory, metadata)
                result = complete(directory)
                if result is None:
                    if not all((directory / f'{i}.part').is_file() for i in range(metadata['n'])):
                        return web.json_response({'error': 'Missing pieces; choose the same file to resume'}, status=409)
                    result = await asyncio.to_thread(assemble, directory, metadata)
            return web.json_response(result)
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
