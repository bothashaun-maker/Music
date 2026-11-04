"""Real local HTTP transfer tests for the Timestamp Studio uploader; no model/GPU."""
import asyncio
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import wave

import numpy as np
import torch
import soundfile as sf

from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer

ROOT=Path(__file__).resolve().parents[1]

class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.inputs=self.root/'input';self.inputs.mkdir()
        fp=ModuleType('folder_paths');fp.get_input_directory=lambda:str(self.inputs)
        fp.get_temp_directory=lambda:str(self.root/'temp');fp.get_output_directory=lambda:str(self.root/'output')
        self.routes=web.RouteTableDef()
        server=ModuleType('server');server.PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=self.routes))
        self.mods=patch.dict(sys.modules,{'server':server,'folder_paths':fp})
        self.mods.start();self.addCleanup(self.mods.stop)
        pkg=self.root/'TestStudio';pkg.mkdir()
        shutil.copyfile(ROOT/'timestamp_studio/comfy_pack.py',pkg/'__init__.py')
        shutil.copyfile(ROOT/'timestamp_studio/engine.py',pkg/'sts_engine.py')
        spec=importlib.util.spec_from_file_location('R15Studio',pkg/'__init__.py',submodule_search_locations=[str(pkg)])
        self.pack=importlib.util.module_from_spec(spec);sys.modules['R15Studio']=self.pack
        self.addCleanup(lambda:[sys.modules.pop(k,None) for k in list(sys.modules) if k.startswith('R15Studio')])
        spec.loader.exec_module(self.pack)
        app=web.Application(client_max_size=64*1024**2)
        async def source(request):
            await asyncio.sleep(.02)
            return web.Response(body=b'cloud audio',content_type='audio/wav')
        app.router.add_get('/source.wav',source)
        app.add_routes(self.routes)
        self.client=TestClient(TestServer(app));await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def spec(self,data,name='song.wav',ident=None):
        return {'upload':ident or uuid.uuid4().hex,'name':name,'size':len(data),
                'n':(len(data)+self.pack.UPLOAD_CHUNK_BYTES-1)//self.pack.UPLOAD_CHUNK_BYTES}

    async def upload(self,data,values):
        for i in range(values['n']):
            block=data[i*self.pack.UPLOAD_CHUNK_BYTES:(i+1)*self.pack.UPLOAD_CHUNK_BYTES]
            response=await self.client.post('/sts/upload/chunk',params={**values,'i':i},data=io.BytesIO(block))
            self.assertEqual(response.status,200,await response.text())

    async def finish(self,values):
        response=await self.client.post('/sts/upload/finish',json=values)
        self.assertEqual(response.status,200,await response.text())
        return await response.json()

    async def test_75_mib_wav_arrives_byte_identical_and_finish_is_idempotent(self):
        stream=io.BytesIO()
        with wave.open(stream,'wb') as target:
            target.setnchannels(2);target.setsampwidth(2);target.setframerate(48000)
            target.writeframes(b'\x01\x00\xff\xff'*(75*1024**2//4))
        data=stream.getvalue();values=self.spec(data)
        await self.upload(data,values)
        one=await self.finish(values);two=await self.finish(values)
        self.assertEqual(one,two)
        status=await self.client.get("/sts/upload/status",params=values)
        self.assertEqual((await status.json())["completed"],list(range(values["n"])))
        saved=(self.inputs/one['filename']).read_bytes()
        self.assertEqual(hashlib.sha256(saved).digest(),hashlib.sha256(data).digest())
        self.assertEqual(len(list(self.inputs.iterdir())),1)
        with wave.open(io.BytesIO(saved)) as audio:self.assertEqual(audio.getframerate(),48000)

    async def test_interrupted_chunk_does_not_publish_a_partial_piece(self):
        data=b'x'*40;values=self.spec(data)
        response=await self.client.post('/sts/upload/chunk',params={**values,'i':0},data=data[:7])
        self.assertEqual(response.status,400)
        response=await self.client.get('/sts/upload/status',params=values)
        self.assertEqual((await response.json())['completed'],[])
        await self.upload(data,values)
        self.assertEqual((await self.finish(values))['bytes'],40)

    async def test_missing_piece_keeps_the_completed_piece_for_resume(self):
        data=b'x'*(self.pack.UPLOAD_CHUNK_BYTES+20);values=self.spec(data)
        response=await self.client.post('/sts/upload/chunk',params={**values,'i':0},data=data[:self.pack.UPLOAD_CHUNK_BYTES])
        self.assertEqual(response.status,200)
        response=await self.client.post('/sts/upload/finish',json=values)
        self.assertEqual(response.status,409)
        response=await self.client.get('/sts/upload/status',params=values)
        self.assertEqual((await response.json())['completed'],[0])

    async def test_two_same_named_uploads_keep_separate_files(self):
        a,b=b'a'*100,b'b'*100;one,two=self.spec(a),self.spec(b)
        await asyncio.gather(self.upload(a,one),self.upload(b,two))
        results=await asyncio.gather(self.finish(one),self.finish(two))
        self.assertNotEqual(results[0]['filename'],results[1]['filename'])
        self.assertEqual({p.read_bytes() for p in self.inputs.iterdir()},{a,b})

    async def test_invalid_metadata_fails_locally_without_creating_files(self):
        for values in [{'n':-1,'size':-1},{'n':'oops','size':1},{'n':1,'size':1,'upload':'../../outside'},
                       {'n':1,'size':2*1024**3+1,'upload':uuid.uuid4().hex}]:
            response=await self.client.post('/sts/upload/finish',json=values)
            self.assertEqual(response.status,400)
        self.assertEqual(list(self.inputs.iterdir()),[])

    async def test_missing_selected_song_does_not_silently_load_different_song(self):
        (self.inputs/'actual.wav').write_bytes(b'test')
        ok,_=self.pack.STSLoadSong.hidden_validate('missing.wav')
        self.assertFalse(ok)
        ok,path=self.pack.STSLoadSong.hidden_validate(self.pack.NO_SONG)
        self.assertTrue(ok);self.assertEqual(Path(path).name,'actual.wav')

    async def test_link_download_runs_as_a_polled_background_job(self):
        source=str(self.client.make_url('/source.wav'))
        response=await self.client.post('/sts/fetch_url/start',json={'url':source})
        self.assertEqual(response.status,200)
        job=(await response.json())['job']
        for _ in range(50):
            response=await self.client.get('/sts/fetch_url/status',params={'job':job})
            result=await response.json()
            if result['state']!='running':break
            await asyncio.sleep(.01)
        self.assertEqual(result['state'],'done',result)
        self.assertEqual((self.inputs/result['filename']).read_bytes(),b'cloud audio')

    async def test_link_failure_is_reported_without_a_broken_song(self):
        response=await self.client.post('/sts/fetch_url/start',json={'url':str(self.client.make_url('/absent.wav'))})
        job=(await response.json())['job']
        for _ in range(50):
            response=await self.client.get('/sts/fetch_url/status',params={'job':job})
            result=await response.json()
            if result['state']!='running':break
            await asyncio.sleep(.01)
        self.assertEqual(result['state'],'failed')
        self.assertEqual(list(self.inputs.iterdir()),[])

    async def test_changed_audio_file_invalidates_comfy_cache(self):
        path=self.inputs/'selected.wav';path.write_bytes(b'first')
        first=self.pack.STSLoadSong.IS_CHANGED('selected.wav')
        path.write_bytes(b'second')
        self.assertNotEqual(first,self.pack.STSLoadSong.IS_CHANGED('selected.wav'))

    async def test_audition_uses_nudged_result_on_canvas(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('sts_canvas_test',ROOT/'timestamp_studio/canvas.py')
        canvas=importlib.util.module_from_spec(spec);spec.loader.exec_module(canvas)
        graph=canvas.build_studio_canvas('Home')
        self.assertTrue(any(edge[1]==4 and edge[3]==6 for edge in graph['links']))

    async def test_alignment_temp_audio_is_unique_and_lossless(self):
        import torch,soundfile as sf
        waveform=torch.full((1,2,4800),.1234567)
        source={'waveform':waveform,'sample_rate':48000}
        first=self.pack._audio_to_temp_wav(source);second=self.pack._audio_to_temp_wav(source)
        self.addCleanup(lambda:os.unlink(first));self.addCleanup(lambda:os.unlink(second))
        self.assertNotEqual(first,second)
        self.assertEqual(sf.info(first).subtype,'FLOAT')
        self.assertEqual(float(sf.read(first,dtype='float32')[0][0]),float(waveform[0,0,0]))

if __name__=='__main__':unittest.main()
