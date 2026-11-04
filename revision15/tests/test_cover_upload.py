"""Real HTTP cover uploads plus the shipped JS handler; no GPU or model weights."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import uuid
import wave

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from native_yue2.cover_upload import CHUNK_BYTES, register_routes


class CoverUploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.inputs = self.root / 'input'
        self.routes = web.RouteTableDef()
        register_routes(SimpleNamespace(routes=self.routes), self.inputs, self.root / 'temp')
        self.app = web.Application(client_max_size=2 * CHUNK_BYTES)
        self.app.add_routes(self.routes)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def metadata(self, data, name='song.wav'):
        return dict(upload=uuid.uuid4().hex, name=name, size=len(data),
                    n=(len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES)

    async def upload(self, data, metadata):
        for index in range(metadata['n']):
            response = await self.client.post('/yue2/cover-upload/chunk',
                params={**metadata, 'i': index},
                data=io.BytesIO(data[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]))
            self.assertEqual(response.status, 200, await response.text())

    async def finish(self, metadata):
        response = await self.client.post('/yue2/cover-upload/finish', json=metadata)
        self.assertEqual(response.status, 200, await response.text())
        return await response.json()

    async def test_75_mib_wav_preserved_and_finish_retry_is_idempotent(self):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as target:
            target.setnchannels(2)
            target.setsampwidth(2)
            target.setframerate(48000)
            target.writeframes(b'\x01\x00\xff\xff' * (75 * 1024 ** 2 // 4))
        data = stream.getvalue()
        metadata = self.metadata(data)
        await self.upload(data, metadata)
        first = await self.finish(metadata)
        self.assertEqual(await self.finish(metadata), first)
        saved = (self.inputs / first['filename']).read_bytes()
        self.assertEqual(hashlib.sha256(saved).digest(), hashlib.sha256(data).digest())
        with wave.open(io.BytesIO(saved)) as audio:
            self.assertEqual(audio.getframerate(), 48000)

    async def test_partial_retry_preserves_good_piece_and_can_resume(self):
        data = b'a' * (CHUNK_BYTES + 7)
        metadata = self.metadata(data)
        await self.client.post('/yue2/cover-upload/chunk', params={**metadata, 'i': 0},
                               data=io.BytesIO(data[:CHUNK_BYTES]))
        response = await self.client.post('/yue2/cover-upload/chunk',
            params={**metadata, 'i': 0}, data=b'incomplete')
        self.assertEqual(response.status, 400)
        response = await self.client.get('/yue2/cover-upload/status', params=metadata)
        self.assertEqual((await response.json())['completed'], [0])
        response = await self.client.post('/yue2/cover-upload/finish', json=metadata)
        self.assertEqual(response.status, 409)
        await self.client.post('/yue2/cover-upload/chunk', params={**metadata, 'i': 1}, data=b'a' * 7)
        result = await self.finish(metadata)
        self.assertEqual((self.inputs / result['filename']).read_bytes(), data)

    async def test_parallel_same_name_uploads_preserve_both(self):
        one, two = b'one', b'two'
        m1, m2 = self.metadata(one), self.metadata(two)
        await asyncio.gather(self.upload(one, m1), self.upload(two, m2))
        a, b = await asyncio.gather(self.finish(m1), self.finish(m2))
        self.assertNotEqual(a['filename'], b['filename'])
        self.assertEqual((self.inputs / a['filename']).read_bytes(), one)
        self.assertEqual((self.inputs / b['filename']).read_bytes(), two)

    async def test_invalid_metadata_and_reused_id_cannot_replace_upload(self):
        metadata = self.metadata(b'abc')
        await self.upload(b'abc', metadata)
        response = await self.client.get('/yue2/cover-upload/status',
                                         params={**metadata, 'name': 'different.wav'})
        self.assertEqual(response.status, 400)
        for values in ([1], {**metadata, 'upload': '../outside'}, {**metadata, 'n': 99},
                       {**metadata, 'name': 'script.py'}):
            response = await self.client.post('/yue2/cover-upload/finish', json=values)
            self.assertEqual(response.status, 400)
        result = await self.finish(metadata)
        self.assertEqual((self.inputs / result['filename']).read_bytes(), b'abc')

    async def test_filename_cannot_escape_input_directory(self):
        metadata = self.metadata(b'original', '../../outside.wav')
        await self.upload(b'original', metadata)
        result = await self.finish(metadata)
        self.assertEqual(result['filename'], 'outside.wav')
        self.assertFalse((self.root / 'outside.wav').exists())

    async def test_shipped_frontend_retries_and_selects_the_uploaded_song(self):
        from native_yue2.cover_upload_web import COVER_UPLOAD_JS
        node = shutil.which('node') or '/opt/codex/runtimes/codex-primary-runtime/dependencies/node/bin/node'
        if not Path(node).is_file():
            self.skipTest('Node.js required for JS upload integration')
        failed_once = False
        @web.middleware
        async def transient_failure(request, handler):
            nonlocal failed_once
            if request.path.endswith('/chunk') and not failed_once:
                failed_once = True
                await request.read()
                return web.json_response({'error': 'temporary connection failure'}, status=503)
            return await handler(request)
        # A separate app shares the registered upload closures, adding a transient error.
        app = web.Application(client_max_size=2 * CHUNK_BYTES, middlewares=[transient_failure])
        app.add_routes(self.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        source = self.root / 'original.wav'
        source.write_bytes(b'RIFF' + b'\x03' * (CHUNK_BYTES + 17))
        base = str(client.make_url('/')).rstrip('/')
        script = '''import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {File} from 'node:buffer';
import {webcrypto} from 'node:crypto';
globalThis.crypto ??= webcrypto;
let extension;
const app = {registerExtension(e) {extension=e;}};
const api = {fetchApi(path, opts) {return fetch(BASE+path, opts);}};
const storage = new Map();
const sessionStorage = {getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)};
const window = {alert(message) {throw new Error(message);}};
let picker;
const document = {createElement() {picker={click(){},remove(){}};return picker;}};
'''.replace('BASE', json.dumps(base))
        script += '\n'.join(line for line in COVER_UPLOAD_JS.splitlines() if not line.startswith('import '))
        script += '''
const audio = {name:'audio',options:{values:[]},callback(value){this.selected=value;}};
let button;
const node = {type:'LoadAudio',widgets:[audio],graph:{setDirtyCanvas(){},change(){}},
 addWidget(type,name,value,callback,options) {button={name,callback,options};return button;}};
extension.nodeCreated(node);
assert.equal(button.options.serialize,false);
await button.callback();
picker.files=[new File([readFileSync(SOURCE)],'original.wav',{lastModified:1})];
await picker.onchange();
assert.equal(audio.value,'original.wav');
assert.equal(audio.selected,'original.wav');
assert.ok(audio.options.values.includes('original.wav'));
assert.ok(button.name.endsWith('ready'));
assert.equal(storage.size,0);
console.log('frontend upload and retry passed');
'''.replace('SOURCE', json.dumps(str(source)))
        script_path = self.root / 'frontend.mjs'
        script_path.write_text(script)
        process = await asyncio.create_subprocess_exec(node, str(script_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(process.communicate(), 30)
        self.assertEqual(process.returncode, 0, output.decode())
        self.assertTrue(failed_once)
        self.assertEqual((self.inputs / 'original.wav').read_bytes(), source.read_bytes())

    def test_generation_graph_and_native_engine_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        from native_yue2.comfy_canvas import build_song_canvas
        graph = json.dumps(build_song_canvas(), sort_keys=True)
        # Frozen from the working release f967707; also works outside a Git checkout.
        self.assertEqual(hashlib.sha256(graph.encode()).hexdigest(),
                         '1ff30815584ca96dedce8f9a4808b10099ba98d13bcce08efe101a0d65653bba')
        self.assertEqual(hashlib.sha256((root / 'native_yue2/engine.py').read_bytes()).hexdigest(),
                         'becb658e7e406c5377b7a4af16f2dc773b74c3bb9deb864a5460f2a75b0f320b')


if __name__ == '__main__':
    unittest.main()
