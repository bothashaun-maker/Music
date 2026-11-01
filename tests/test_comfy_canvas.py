"""CPU integration of visible stage nodes with a synthetic pipeline; no model weights."""
import ast
import base64
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import soundfile as sf
import torch
from comfy_yue2 import nodes

ROOT=Path(__file__).resolve().parents[1]

@dataclasses.dataclass
class Config:
    ode_steps:int=16
    ode_method:str='euler'

class Pipeline:
    generation_config=Config()
    weights={'test':'synthetic'}
    runtime_sha256='synthetic'
    last_acoustic_ranges=[(0,25)]
    def plan(self,**kw):
        self.plan_args=kw
        def save(directory):
            (directory/'score.abc').write_text('X:1\nK:C\nCDEF|')
            np.save(directory/'prefix.npy',np.arange(4))
        return SimpleNamespace(prefix=[1,2,3,4],abc_ids=[],abc='X:1\nK:C\nCDEF|',truncated=False,save=save)
    def generate_semantic(self,plan,sampling,**kw):
        self.budget=sampling['max_tokens']
        for _ in range(25):kw['on_token']('semantic',100)
        return SimpleNamespace(tokens=list(range(25)),truncated=False,timing={'execution':'cpu-test'})
    def synthesize(self,semantic,**kw):return np.zeros((25,64),np.float32)
    def decode(self,latents):return np.full((48000,2),.1,np.float32)
    def close(self):self.closed=True


def normalized_ast(value):
    if isinstance(value,ast.AST):
        return {'type':type(value).__name__, **{k:normalized_ast(v) for k,v in ast.iter_fields(value) if v is not None and v != []}}
    if isinstance(value,list):return [normalized_ast(v) for v in value]
    return value

class CanvasTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.engine=SimpleNamespace(pipe=Pipeline(),AR_OPTIONS={},release_cover_models=lambda:None,
            yproto=SimpleNamespace(CONTEXT=24576,CODEC_OFFSET=100,CODEC_SIZE=1000,MUSIC_END=9999))
        comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management');utils=ModuleType('comfy.utils')
        mm.throw_exception_if_processing_interrupted=lambda:None
        utils.ProgressBar=lambda _:SimpleNamespace(update_absolute=lambda *a:None)
        comfy.model_management=mm;comfy.utils=utils
        fp=ModuleType('folder_paths');fp.get_output_directory=lambda:str(self.root)
        self.modules=patch.dict(sys.modules,{'comfy':comfy,'comfy.utils':utils,'comfy.model_management':mm,'folder_paths':fp})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.env=patch.dict(os.environ,YUE2_OUTPUT_ROOT=str(self.root/'runs'),YUE2_COVER_ROOT=str(self.root/'covers'))
        self.env.start();self.addCleanup(self.env.stop)

    def stages(self,planning='off',abc=''):
        prompt=nodes.Prompt().prompt('rock','[Verse]\nHome',abc)[0]
        settings=nodes.Settings().settings(17,8,planning,16,'euler',1.0)[0]
        plan,score=nodes.Plan().plan(self.engine,prompt,settings)
        semantic=nodes.Semantic().sample(plan)[0]
        latent=nodes.Acoustic().synthesize(semantic)[0]
        audio,result=nodes.Decode().decode(latent)
        return audio,result

    def test_full_stage_chain_outputs_valid_stereo_masters_and_recovery(self):
        audio,result=self.stages()
        self.assertEqual(audio['waveform'].shape,(1,2,48000))
        output=nodes.SaveSong().save(audio,result)
        root=result['directory']
        for name in ['audio.wav','audio.flac','settings.json','prefix.npy','semantic.npy','latent.npy','latent_manifest.json','status.json']:
            self.assertTrue((root/name).is_file(),name)
        self.assertEqual(sf.info(root/'audio.wav').subtype,'PCM_24')
        self.assertEqual(json.loads((root/'status.json').read_text())['ending_reason'],'model_end_token')
        self.assertEqual(output['ui']['audio'][0]['subfolder'],str(root.relative_to(self.root)))
        self.assertEqual(self.engine.pipe.budget,12000)
        self.assertEqual(self.engine.pipe.plan_args['abc_sampling'],{'max_tokens':4096})
        self.assertEqual(self.engine.pipe.generation_config,Config(16,'euler'))

    def test_supplied_abc_promotes_direct_mode(self):
        self.stages(abc='X:1\nK:C\nCDEF|')
        self.assertEqual(self.engine.pipe.plan_args['cot'],'melody')

    def test_explicit_abc_budget_reaches_planner(self):
        prompt=nodes.Prompt().prompt('rock','[Verse]\nHome','')[0]
        settings=nodes.Settings().settings(17,8,'full',16,'euler',1.0,6144)[0]
        nodes.Plan().plan(self.engine,prompt,settings)
        self.assertEqual(self.engine.pipe.plan_args['abc_sampling'],{'max_tokens':6144})
        self.assertEqual(settings['max_tokens'],12000)

    def test_old_settings_without_abc_field_remain_supported(self):
        prompt=nodes.Prompt().prompt('rock','[Verse]\nHome','')[0]
        settings=nodes.Settings().settings(17,8,'melody',16,'euler',1.0)[0]
        settings.pop('abc_token_limit')
        nodes.Plan().plan(self.engine,prompt,settings)
        self.assertEqual(self.engine.pipe.plan_args['abc_sampling'],{'max_tokens':4096})

    def test_truncated_plan_keeps_score_and_stops_before_music(self):
        original=self.engine.pipe.plan
        def cutoff(**kw):
            plan=original(**kw);plan.truncated=True;return plan
        self.engine.pipe.plan=cutoff
        prompt=nodes.Prompt().prompt('rock','[Verse]\nHome','')[0]
        settings=nodes.Settings().settings(17,8,'melody',16,'euler',1.0)[0]
        with self.assertRaisesRegex(ValueError,'4096-token planning limit'):
            nodes.Plan().plan(self.engine,prompt,settings)
        self.assertTrue(list(self.root.rglob('score.abc')))
        status=json.loads(next(self.root.rglob('status.json')).read_text())
        self.assertEqual(status['status'],'abc_planning_cutoff')
        self.assertFalse(list(self.root.rglob('semantic.npy')))

    def test_cutoff_reported_separately(self):
        audio,result=self.stages()
        result['semantic'].truncated=True
        nodes.SaveSong().save(audio,result)
        self.assertEqual(json.loads((result['directory']/'status.json').read_text())['ending_reason'],'token_limit')

    def test_nonfinite_decode_fails_before_export(self):
        self.engine.pipe.decode=lambda _:np.full((48000,2),np.nan)
        with self.assertRaisesRegex(ValueError,'invalid'):
            self.stages()
        self.assertFalse(list(self.root.rglob('audio.wav')))

    def test_cancel_propagates_before_planning(self):
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted=lambda:(_ for _ in ()).throw(InterruptedError('stop'))
        with self.assertRaises(InterruptedError):self.stages()

    def test_wav_to_abc_uses_lossless_samples_and_removes_working_copy(self):
        original=torch.full((1,2,4800),.125,dtype=torch.float32)
        seen=[]
        def analyze(path,*args,**kwargs):
            samples,rate=sf.read(path,dtype='float32',always_2d=True)
            np.testing.assert_array_equal(samples,original.squeeze(0).numpy().T)
            self.assertEqual(rate,48000)
            seen.append(Path(path))
            return 'X:1\nK:C\nCDEF|','','melody','ready','analysis.zip'
        self.engine.analyze_cover_source=analyze
        output=nodes.CoverABC().analyze(self.engine,{'waveform':original,'sample_rate':48000})
        self.assertEqual(output['result'][0],'X:1\nK:C\nCDEF|')
        self.assertFalse(seen[0].exists())

    def test_headless_import_constructs_no_gradio_interface(self):
        import gradio as gr
        import yue2.pipeline as native
        fake=SimpleNamespace(device='cuda:0',vae_device='cuda:1')
        path=ROOT/'native_yue2/engine.py'
        scope={'__file__':str(path),'__name__':'headless_engine_test'}
        with patch.dict(os.environ,YUE2_HEADLESS='1',HF_HUB_CACHE=str(self.root)), \
             patch.object(native.YuE2Pipeline,'from_pretrained',return_value=fake), \
             patch.object(gr,'Blocks',side_effect=AssertionError('Gradio UI constructed')):
            exec(compile(path.read_text(),str(path),'exec'),scope)
        self.assertNotIn('demo',scope)
        self.assertTrue(callable(scope['generate_music']))
        self.assertTrue(callable(scope['analyze_cover_source']))

    def test_workflow_nodes_and_links_are_real_and_type_consistent(self):
        for name in ['song','cover']:
            graph=json.loads((ROOT/'workflows'/f'yue2_{name}.json').read_text())
            by_id={n['id']:n for n in graph['nodes']}
            for node in graph['nodes']:
                self.assertIn(node['type'],set(nodes.NODE_CLASS_MAPPINGS)|{'LoadAudio','PreviewAudio'})
                if node['type'] in nodes.NODE_CLASS_MAPPINGS:
                    schema=nodes.NODE_CLASS_MAPPINGS[node['type']].INPUT_TYPES()
                    for inp in node['inputs']:
                        self.assertIn(inp['name'],schema.get('required',{})|schema.get('optional',{}))
            for i,a,slot,b,inp,t in graph['links']:
                self.assertEqual(by_id[a]['outputs'][slot]['type'],t)
                self.assertEqual(by_id[b]['inputs'][inp]['type'],t)
                self.assertEqual(by_id[b]['inputs'][inp]['link'],i)
        self.assertGreater(len(json.loads((ROOT/'workflows/yue2_song.json').read_text())['nodes']),8)

    def test_standalone_comfy_notebook_embeds_all_current_files(self):
        nb=json.loads((ROOT/'YuE2_ComfyUI_Kaggle.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in node.targets):
                    data=json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(node.value))))
                    for name,text in data.items():self.assertEqual(text,(ROOT/name).read_text())
                    self.assertIn('comfy_yue2/web/canvas.js',data)
                    self.assertIn('workflows/yue2_song.json',data)
                    self.assertIn('native_yue2/comfy.py',data)

    def test_library_mode_excludes_ui_and_preserves_original_ast(self):
        tree=ast.parse((ROOT/'native_yue2/engine.py').read_text())
        self.assertIsInstance(tree.body[-1],ast.If)
        self.assertIn('YUE2_HEADLESS',ast.unparse(tree.body[-1].test))
        unwrapped=ast.Module(body=tree.body[:-1]+tree.body[-1].body,type_ignores=[])
        digest=hashlib.sha256(json.dumps(normalized_ast(unwrapped),sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest,'0d4e07d570e86cd0a187428fd7a1f79d88a567caa4b36456e1b3e821c60baead')

if __name__=='__main__':unittest.main()
