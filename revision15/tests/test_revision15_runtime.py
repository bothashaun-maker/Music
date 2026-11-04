"""Execute the shipped node chain on CPU with a synthetic pipeline, no weights."""
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]

@dataclasses.dataclass
class Config:
    ode_steps:int=16
    ode_method:str='euler'

class Pipeline:
    runtime_sha256='synthetic'
    backend='torch'
    weights={'test':'synthetic'}
    last_acoustic_ranges=[(0,25)]
    def __init__(self):self.generation_config=Config()
    def plan(self,**kwargs):
        self.plan_args=kwargs
        def save(directory):
            (directory/'plan.json').write_text(json.dumps({'request':'preserve-me'}))
            (directory/'score.abc').write_text('X:1\nK:C\nCDEF|')
        return SimpleNamespace(prefix=[1]*10,abc_ids=[1,2],truncated=False,save=save,
            request=SimpleNamespace(guidance=kwargs['cfg_scale'],cot=kwargs['cot']))
    def generate_semantic(self,plan,sampling,**kwargs):
        kwargs['cancelled']()
        kwargs['on_token']('semantic',100)
        return SimpleNamespace(tokens=[0]*25,truncated=False,timing={'execution':'synthetic-cpu'})
    def synthesize(self,semantic,**kwargs):
        kwargs['cancelled']()
        return np.zeros((25,64),np.float32)
    def decode(self,latents):return np.full((48000,2),.125,np.float32)
    def close(self):self.closed=True

class NodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory();self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name)
        e=ModuleType('native_yue2.engine')
        e.pipe=None;e._cancelled=lambda:False;e.DURATION_CHOICES=[('Up to 8 min',12000)]
        e.DURATION_MAP=dict(e.DURATION_CHOICES);e.COVER_STATE={};e.ATTENTION_STATUS={};e.AR_DIAGNOSTICS={}
        e.yproto=SimpleNamespace(CONTEXT=24576,CODEC_OFFSET=100,CODEC_SIZE=20,MUSIC_END=9999)
        def initialize():e.pipe=Pipeline()
        e.initialize_pipeline=initialize
        self.engine=e
        comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management')
        class Interrupt(Exception):pass
        mm.InterruptProcessingException=Interrupt;mm.throw_exception_if_processing_interrupted=lambda:None
        comfy.model_management=mm;self.mm=mm
        fp=ModuleType('folder_paths');fp.get_output_directory=lambda:str(self.root)
        server=ModuleType('server');server.PromptServer=SimpleNamespace(instance=None)
        self.modules=patch.dict(sys.modules,{'native_yue2.engine':e,'comfy':comfy,'comfy.model_management':mm,'folder_paths':fp,'server':server})
        self.modules.start();self.addCleanup(self.modules.stop)
        # Python dotted import can reuse attributes from the existing parent.
        import native_yue2
        self.parent=patch.object(native_yue2,'engine',e,create=True)
        self.parent.start();self.addCleanup(self.parent.stop)
        spec=importlib.util.spec_from_file_location('r15_pack_test',ROOT/'native_yue2/comfy_pack.py')
        self.pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.pack)
        self.pack.OUTPUT_ROOT=self.root
        self.pack._acquire_exclusive_engine=lambda:None
        self.pack._require_gpu_room=lambda:None
        self.run=self.pack.YuE2RunContext().execute(17)[0]

    def prepare(self,keep=True,mode='Direct to audio'):
        loaded=self.pack.YuE2LoadEngine().execute(self.run,keep_loaded=keep)['result'][0]
        prompt=self.pack.YuE2StyleLyrics().execute('rock','[Verse]\nHome')[0]
        plan=self.pack.YuE2PlanScore().execute(self.run,loaded,prompt,mode,1.0)[0]
        tokens=self.pack.YuE2SongTokens().execute(self.run,loaded,prompt,'Up to 8 min',plan=plan)[0]
        return tokens

    def test_complete_chain_uses_real_engine_handle_and_writes_valid_masters(self):
        tokens=self.prepare()
        latents=self.pack.YuE2SynthesizeLatents().execute(self.run,tokens,16,'euler')[0]
        audio=self.pack.YuE2DecodeAudio().execute(self.run,latents)[0]
        result=self.pack.YuE2SaveSong().execute(self.run,tokens,audio)
        directory=self.run['directory']
        self.assertEqual(sf.info(directory/'audio.wav').subtype,'PCM_24')
        self.assertEqual(sf.info(directory/'audio.wav').duration,1)
        self.assertEqual(json.loads((directory/'status.json').read_text())['ending_reason'],'model_end_token')
        self.assertEqual(result['ui']['audio'][0]['subfolder'],directory.name)
        self.assertEqual(json.loads((directory/'plan.json').read_text())['request'],'preserve-me')

    def test_fixed_seed_still_gets_a_new_run_folder(self):
        other=self.pack.YuE2RunContext().execute(17)[0]
        self.assertNotEqual(self.run['directory'],other['directory'])
        self.assertTrue(math.isnan(self.pack.YuE2RunContext.IS_CHANGED(seed=17)))

    def test_full_planning_has_configurable_budget(self):
        loaded=self.pack.YuE2LoadEngine().execute(self.run)['result'][0]
        prompt=self.pack.YuE2StyleLyrics().execute('rock','Home')[0]
        self.pack.YuE2PlanScore().execute(self.run,loaded,prompt,'Full melody+chords',1.5,6144)
        self.assertEqual(self.engine.pipe.plan_args['abc_sampling'],{'max_tokens':6144})

    def test_reuse_returns_engine_object_and_updates_release_policy(self):
        one=self.pack.YuE2LoadEngine().execute(self.run)['result'][0]
        two=self.pack.YuE2LoadEngine().execute(self.run,keep_loaded=False)['result'][0]
        self.assertIs(one,self.engine);self.assertIs(two,self.engine)
        self.assertTrue(self.pack._STATE['release_after'])

    def test_release_after_save_really_releases_engine(self):
        tokens=self.prepare(keep=False);pipe=self.engine.pipe
        latents=self.pack.YuE2SynthesizeLatents().execute(self.run,tokens,16,'euler')[0]
        audio=self.pack.YuE2DecodeAudio().execute(self.run,latents)[0]
        self.pack.YuE2SaveSong().execute(self.run,tokens,audio)
        self.assertTrue(pipe.closed);self.assertIsNone(self.engine.pipe)

    def test_interrupt_reaches_acoustic_solver(self):
        tokens=self.prepare()
        def stop():raise self.mm.InterruptProcessingException('stop')
        self.mm.throw_exception_if_processing_interrupted=stop
        with self.assertRaises(self.mm.InterruptProcessingException):
            self.pack.YuE2SynthesizeLatents().execute(self.run,tokens,16,'euler')
        self.assertEqual(json.loads((self.run['directory']/'status.json').read_text())['status'],'cancelled')

    def test_cover_analysis_has_a_real_dependency_before_engine_load(self):
        from native_yue2.comfy_canvas import build_cover_canvas
        graph=build_cover_canvas()
        self.assertTrue(any(l[1]==3 and l[3]==5 for l in graph['links']))
        self.assertNotIn('OUTPUT_NODE',self.pack.YuE2LoadEngine.__dict__)

    def test_nonfinite_export_is_rejected_before_file_write(self):
        tokens=self.prepare()
        audio={'sample_rate':48000,'waveform':torch.full((1,2,48000),float('nan'))}
        with self.assertRaisesRegex(ValueError,'finite'):
            self.pack.YuE2SaveSong().execute(self.run,tokens,audio)
        self.assertFalse((self.run['directory']/'audio.wav').exists())

if __name__=='__main__':unittest.main()
