"""CPU checks of real UI/native contracts and synthetic audio exports; no GPU inference."""
import ast
import base64
from pathlib import Path
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import zipfile
import zlib

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import yue2.pipeline as native
import yue2.protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'native_yue2/engine.py'


class SyntheticPipe:
    device = 'cuda:0'
    vae_device = 'cuda:1'
    backend = 'torch'
    runtime_sha256 = 'synthetic-test-engine'
    weights = {'synthetic': True}

    def __init__(self):
        self.generation_config = protocol.GenerationConfig()
        self.frames = None
        self.plan_truncated = False
        self.fail_decode = False
        self.semantic_calls = 0
        self.prefix = [protocol.EOD] * 100

    def plan(self, style, lyrics, cot, abc, seed, cfg_scale, **kw):
        request = protocol.SongRequest(style, lyrics, cot=cot, seed=seed, cfg_scale=cfg_scale, abc=abc)
        return native.SymbolicPlan(request, abc, [], self.prefix, {}, self.plan_truncated)

    def generate_semantic(self, plan, sampling, **kw):
        self.semantic_calls += 1
        self.budget = sampling['max_tokens']
        frames = self.frames or self.budget
        if getattr(self, 'emit_tokens', False):
            for _ in range(frames):
                kw['on_token']('semantic', protocol.CODEC_OFFSET)
            if frames < self.budget:
                kw['on_token']('semantic', protocol.MUSIC_END)
        return native.SemanticResult(plan, [0] * frames,
            {'execution': 'synthetic_cpu_test', 'prefix_tokens': len(plan.prefix)},
            frames == self.budget)

    def synthesize(self, semantic, **kw):
        self.last_acoustic_ranges = protocol.chunk_ranges(len(semantic.tokens), len(semantic.plan.prefix))
        return np.zeros((len(semantic.tokens), 64), dtype=np.float32)

    def decode(self, latents):
        if self.fail_decode:
            raise RuntimeError('synthetic decoder failure')
        return np.full((len(latents) * 1920, 2), .125, dtype=np.float32)

    def close(self):
        self.closed = True


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix='native-yue2-check-')
        cls.previous_env = os.environ.copy()
        os.environ['YUE2_OUTPUT_ROOT'] = cls.scratch.name
        os.environ['YUE2_COVER_ROOT'] = str(Path(cls.scratch.name) / 'cover_sources')
        Path(os.environ['YUE2_COVER_ROOT']).mkdir()
        os.environ['HF_HUB_CACHE'] = str(Path(cls.scratch.name) / 'unused-cache')
        fake = SimpleNamespace(device='cuda:0', vae_device='cuda:1')
        cls.scope = {'__file__': str(ENGINE), '__name__': 'native_engine_check'}
        with patch.object(native.YuE2Pipeline, 'from_pretrained', return_value=fake) as loader, \
             patch.object(gr.Blocks, 'launch', return_value=None):
            exec(compile(ENGINE.read_text(), str(ENGINE), 'exec'), cls.scope)
        cls.load_kwargs = loader.call_args.kwargs

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls.previous_env)
        cls.scratch.cleanup()

    def setUp(self):
        self.pipe = SyntheticPipe()
        self.scope['pipe'] = self.pipe
        self.scope['COVER_STATE'].clear()
        for name in self.scope['COVER_MODELS']:
            self.scope['COVER_MODELS'][name] = None

    def generate(self, budget=12000, **kw):
        args = dict(style='rock', lyrics='[Verse]\nHome', cot_mode='off', duration_target=budget,
                    abc_score='', seed=17, ode_steps=16, ode_method='euler', cfg_scale=1.0)
        args.update(kw)
        return self.scope['generate_music'](**args)

    def test_real_ui_defaults_and_native_revision(self):
        self.assertEqual(self.scope['cot_selector'].value, 'off')
        self.assertEqual(self.scope['duration_selector'].value, 12000)
        self.assertEqual(self.scope['ode_steps_slider'].value, 16)
        self.assertEqual(self.scope['ode_method_selector'].value, 'euler')
        self.assertEqual(self.load_kwargs['backend'], 'torch')
        self.assertFalse(self.load_kwargs['offload_ar'])
        self.assertEqual(self.load_kwargs['revision'], '8f24312f187763b9854adeec87e437e2b03bbff1')
        self.assertIn(('Up to 7 min', 10500), self.scope['DURATION_CHOICES'])
        self.assertIn(('Up to 8 min', 12000), self.scope['DURATION_CHOICES'])

    def test_real_ui_http_config_and_api_schema(self):
        from fastapi.testclient import TestClient
        from gradio.routes import App
        with TestClient(App.create_app(self.scope['demo'])) as client:
            self.assertEqual(client.get('/config').status_code, 200)
            response = client.get('/gradio_api/info')
            self.assertEqual(response.status_code, 200)
            self.assertIn('named_endpoints', response.json())

    def test_native_long_budgets_and_chunk_context(self):
        for count in (10500, 12000):
            sample = protocol.Sampling(max_tokens=count)
            self.assertEqual(sample.max_tokens, count)
            ranges = protocol.chunk_ranges(count, 2000)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], count)
            for start, end in ranges:
                self.assertLessEqual(2000 + 2 * (end - start) + 3, protocol.CONTEXT)

    def test_eight_minute_export_and_zip_use_real_file_headers(self):
        preview, score, status, masters, archive = self.generate()
        self.assertEqual(self.pipe.budget, 12000)
        self.assertEqual(self.pipe.generation_config.ode_method, 'euler')
        self.assertEqual(self.pipe.generation_config.ode_steps, 16)
        for name in [preview, *masters]:
            info = sf.info(name)
            self.assertEqual(info.frames, 480 * 48000)
            self.assertEqual(info.samplerate, 48000)
            self.assertEqual(info.channels, 2)
        directory = Path(preview).parent
        report = json.loads((directory / 'status.json').read_text())
        self.assertEqual(report['actual_seconds'], 480)
        self.assertTrue(report['semantic_truncated'])
        self.assertEqual(report['ending_reason'], 'token_limit')
        self.assertIn('ending may be abrupt', status)
        self.assertTrue((directory / 'latent_manifest.json').exists())
        with zipfile.ZipFile(archive) as z:
            self.assertTrue({'audio.wav', 'audio.flac', 'settings.json', 'semantic.npy', 'latent.npy', 'status.json'} <= set(z.namelist()))

    def test_early_end_is_saved_honestly_and_same_seed_preserves_runs(self):
        self.pipe.frames = 25
        first = self.generate(10500)
        second = self.generate(10500)
        self.assertNotEqual(first[0], second[0])
        self.assertIn('ended below', first[2])
        report = json.loads((Path(first[0]).parent / 'status.json').read_text())
        self.assertEqual(report['actual_seconds'], 1)
        self.assertEqual(report['max_seconds'], 420)
        self.assertFalse(report['semantic_truncated'])
        self.assertEqual(report['ending_reason'], 'model_end_token')

    def test_unspecified_ceiling_uses_eight_minutes_and_unknown_label_is_rejected(self):
        self.pipe.frames = 25
        result = self.generate(budget=None)
        self.assertEqual(self.pipe.budget, 12000)
        self.assertIn('Natural model ending', result[2])
        self.assertEqual(self.pipe.semantic_calls, 1)
        with self.assertRaises(gr.Error):
            self.generate(budget='unrecognized duration')
        self.assertEqual(self.pipe.semantic_calls, 1)

    def test_competing_gpu_allocation_stops_before_token_generation(self):
        from unittest.mock import patch, Mock
        with patch.dict(self.scope, IS_MANAGED_ENGINE=True), \
             patch('native_yue2.process.check_gpu_room', side_effect=RuntimeError('GPU 0 occupied by another process')):
            result = self.generate()
        self.assertIsNone(result[0])
        self.assertIn('occupied by another process', result[2])
        self.assertEqual(self.pipe.semantic_calls, 0)

    def test_oom_saves_gpu_snapshot_and_closes_failed_pipeline(self):
        report = {'devices': [], 'applications': [{'pid':48,'gpu_uuid':'gpu0','used_mib':'7200'}]}
        self.pipe.frames = 25
        before = set(Path(self.scratch.name).iterdir())
        with patch.object(self.pipe,'decode',side_effect=torch.OutOfMemoryError('synthetic CUDA OOM')), \
             patch('native_yue2.process.gpu_report',return_value=report):
            result = self.generate()
        failed = next(p for p in set(Path(self.scratch.name).iterdir()) - before if p.is_dir())
        self.assertTrue(self.pipe.closed)
        self.assertTrue((failed/'latent.npy').exists())
        self.assertEqual(json.loads((failed/'gpu-memory-at-error.json').read_text()),report)
        self.assertIn('Do not retry',result[2])

    def test_failed_decode_keeps_latents_and_recovers_without_tokens(self):
        self.pipe.frames = 25
        self.pipe.fail_decode = True
        before = set(Path(self.scratch.name).iterdir())
        result = self.generate()
        self.assertIsNone(result[0])
        failed = next(p for p in set(Path(self.scratch.name).iterdir()) - before if p.is_dir())
        self.assertTrue((failed / 'latent_manifest.json').exists())
        self.assertEqual(json.loads((failed / 'status.json').read_text())['status'], 'failed')
        self.pipe.fail_decode = False
        calls = self.pipe.semantic_calls
        recovered = self.scope['recover_decode'](failed)
        self.assertEqual(self.pipe.semantic_calls, calls)
        self.assertEqual(sf.info(recovered / 'audio.wav').frames, 48000)
        with (failed / 'latent.npy').open('ab') as f:
            f.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.scope['recover_decode'](failed)

    def test_planner_cutoff_and_excessive_context_stop_before_music(self):
        self.pipe.plan_truncated = True
        result = self.generate(cot_mode='full')
        self.assertIsNone(result[0])
        self.assertIn('ABC score hit', result[2])
        self.assertEqual(self.pipe.semantic_calls, 0)
        self.pipe.plan_truncated = False
        self.pipe.prefix = [protocol.EOD] * 13000
        result = self.generate()
        self.assertIn('exceed', result[2])
        self.assertEqual(self.pipe.semantic_calls, 0)

    def test_nonfinite_acoustic_values_fail_instead_of_being_zeroed(self):
        solver = self.scope['patched_solve']
        engine = SimpleNamespace(chunk=SimpleNamespace(noise=torch.ones((2, 64))),
                                 device='cpu', dtype=torch.float32,
                                 velocity=lambda state, t: torch.full_like(state, float('nan')))
        with self.assertRaisesRegex(FloatingPointError, 'Non-finite'):
            solver(engine, steps=1)

    def test_live_progress_uses_real_tokens_and_saves_partial_checkpoint(self):
        self.pipe.frames = 25
        self.pipe.emit_tokens = True
        events = []
        result = self.generate(progress=lambda value, desc: events.append((value, desc)))
        directory = Path(result[0]).parent
        saved = np.load(directory / 'semantic.partial.npy', allow_pickle=False)
        self.assertEqual(saved.tolist(), [0] * 25)
        snapshot = json.loads((directory / 'semantic-progress.json').read_text())
        self.assertEqual(snapshot['content_tokens'], 25)
        self.assertEqual(snapshot['token_budget'], 12000)
        self.assertEqual(snapshot['represented_audio_seconds'], 1)
        self.assertIsNotNone(snapshot['decode_tokens_per_second'])
        self.assertIn(((25, 12000), unittest.mock.ANY), events)
        self.assertFalse(any(value in (.25, .6, .9) for value, _ in events))

    def test_failed_capture_resets_state_and_reports_actual_eager_execution(self):
        obj = SimpleNamespace(graph=object(), output=object(),
                              positions=torch.tensor([99]), initial_positions=torch.tensor([1]))
        with patch.dict(self.scope, _orig_graphar_capture=lambda obj: (_ for _ in ()).throw(RuntimeError('capture refused'))):
            self.scope['patched_graphar_capture'](obj)
        self.assertIsNone(obj.graph)
        self.assertEqual(obj.positions.tolist(), [1])
        self.assertEqual(self.scope['AR_DIAGNOSTICS']['execution'], 'eager')
        def wrongly_labeled_release_result():
            self.scope['AR_DIAGNOSTICS'].update(execution='eager', fallback_reason='capture refused')
            return [1], {'execution': 'cuda_graph'}, False
        with patch.dict(self.scope, _original_generate_tokens=wrongly_labeled_release_result):
            tokens, timing, cutoff = self.scope['observed_generate_tokens']()
        self.assertEqual(tokens, [1])
        self.assertFalse(cutoff)
        self.assertEqual(timing['execution'], 'eager')
        self.assertEqual(timing['graph_fallback_reason'], 'capture refused')

    def cover_fixture(self):
        source = Path(self.scratch.name) / 'cover-input.wav'
        sf.write(source, np.full((48000,2), .125, dtype=np.float32), 48000)
        score = "X:1\nM:4/4\nK:C\nC D E F |\n"
        model = SimpleNamespace(transcribe=Mock(return_value={'abc':score,'duration_seconds':1.0}),to=Mock())
        def load():
            self.scope['COVER_MODELS']['sheetsage'] = model
            return model
        return source, score, model, load

    def test_cover_ui_is_visible_and_shares_gpu_queue(self):
        self.assertTrue(self.scope['cover_panel'].visible)
        self.assertTrue(self.scope['cover_panel'].open)
        self.assertEqual(self.scope['cover_audio'].sources,['upload'])
        self.assertFalse(self.scope['cover_transcribe_lyrics'].value)
        functions = {fn.fn.__name__:fn for fn in self.scope['demo'].fns.values() if fn.fn is not None}
        self.assertEqual(functions['analyze_cover_source'].concurrency_id,'yue2_gpu')
        self.assertEqual(functions['generate_music'].concurrency_id,'yue2_gpu')

    def test_cover_preserves_original_lyrics_and_exports_score_without_asr(self):
        source, score, model, load = self.cover_fixture()
        words = '[Verse]\nMy original words'
        with patch.dict(self.scope,load_melody_transcriber=load,load_lyric_transcriber=Mock(side_effect=AssertionError('ASR must be opt-in'))):
            abc, lyrics, mode, status, archive = self.scope['analyze_cover_source'](str(source),'Auto-detect',words)
        self.assertEqual((abc,lyrics,mode),(score.strip(),words,'melody'))
        self.assertEqual(model.transcribe.call_args.kwargs['dtype'],'fp32')
        self.assertTrue(model.transcribe.call_args.kwargs['melody_only'])
        model.to.assert_called_with('cpu')
        self.assertTrue(all(v is None for v in self.scope['COVER_MODELS'].values()))
        with zipfile.ZipFile(archive) as z:
            self.assertTrue({'score.abc','source.json','lyrics.txt'} <= set(z.namelist()))
            self.assertEqual(z.read('lyrics.txt').decode(),words)
            self.assertEqual(json.loads(z.read('source.json'))['status'],'melody_ready')
        self.assertIn('Original lyrics kept',status)

    def test_cover_optional_asr_uses_language_and_real_resampling(self):
        source, score, model, load = self.cover_fixture()
        asr = Mock(return_value={'text':'Newly transcribed words'})
        def load_asr():
            self.scope['COVER_MODELS']['asr'] = asr
            return asr
        with patch.dict(self.scope,load_melody_transcriber=load,load_lyric_transcriber=load_asr):
            result = self.scope['analyze_cover_source'](str(source),'en','original',True)
        self.assertEqual(result[1],'Newly transcribed words')
        audio = asr.call_args.args[0]
        self.assertEqual(audio['sampling_rate'],16000)
        self.assertEqual(len(audio['array']),16000)
        self.assertEqual(asr.call_args.kwargs['generate_kwargs'],{'language':'en'})
        self.assertTrue(all(v is None for v in self.scope['COVER_MODELS'].values()))

    def test_cover_optional_asr_failure_keeps_melody_and_original_lyrics(self):
        source, score, model, load = self.cover_fixture()
        with patch.dict(self.scope,load_melody_transcriber=load,load_lyric_transcriber=Mock(side_effect=RuntimeError('ASR unavailable'))):
            result = self.scope['analyze_cover_source'](str(source),'Auto-detect','original',True)
        self.assertEqual(result[0],score.strip())
        self.assertEqual(result[1],'original')
        self.assertIn('lyric transcription failed',result[3])
        self.assertTrue(Path(result[4]).is_file())

    def test_cover_melody_failure_saves_error_and_releases_models(self):
        source, score, model, load = self.cover_fixture()
        model.transcribe.side_effect = RuntimeError('synthetic melody error')
        base = Path(os.environ['YUE2_COVER_ROOT'])
        before = set(base.iterdir())
        with patch.dict(self.scope,load_melody_transcriber=load):
            with self.assertRaisesRegex(gr.Error,'synthetic melody error'):
                self.scope['analyze_cover_source'](str(source),'Auto-detect','original')
        failed = next(p for p in set(base.iterdir())-before if p.is_dir())
        self.assertEqual(json.loads((failed/'source.json').read_text())['status'],'failed')
        self.assertTrue(all(v is None for v in self.scope['COVER_MODELS'].values()))

    def test_cover_cancellation_releases_models_and_preserves_partial_metadata(self):
        source, score, model, load = self.cover_fixture()
        def cancel(*args, **kwargs):
            self.scope['CANCEL']['stop'] = True
            kwargs['progress']({'stage':'encoding'})
        model.transcribe.side_effect = cancel
        with patch.dict(self.scope,load_melody_transcriber=load):
            with self.assertRaisesRegex(gr.Error,'stopped'):
                self.scope['analyze_cover_source'](str(source),'Auto-detect','original')
        self.assertTrue(all(v is None for v in self.scope['COVER_MODELS'].values()))

    def test_cover_score_generation_retains_provenance(self):
        self.pipe.frames = 25
        score = 'X:1\nK:C\nC D E F |'
        metadata = {'source_sha256':'synthetic-hash','status':'melody_ready'}
        self.scope['COVER_STATE'].update(abc=score,metadata=metadata)
        result = self.generate(cot_mode='melody',abc_score=score)
        directory = Path(result[0]).parent
        self.assertEqual(json.loads((directory/'cover_source.json').read_text()),metadata)
        with zipfile.ZipFile(result[4]) as z:
            self.assertIn('cover_source.json',z.namelist())

    def test_cover_stop_during_asr_is_observed_before_publishing_success(self):
        source, score, model, load = self.cover_fixture()
        def stop_after_asr(*args, **kwargs):
            self.scope['CANCEL']['stop'] = True
            return {'text':'words'}
        asr = Mock(side_effect=stop_after_asr)
        with patch.dict(self.scope,load_melody_transcriber=load,load_lyric_transcriber=lambda:asr):
            with self.assertRaisesRegex(gr.Error,'stopped'):
                self.scope['analyze_cover_source'](str(source),'Auto-detect','original',True)
        self.assertTrue(all(v is None for v in self.scope['COVER_MODELS'].values()))

    def test_cover_rejects_missing_upload_before_loading_models(self):
        with patch.dict(self.scope,load_melody_transcriber=Mock(side_effect=AssertionError('must not load'))):
            with self.assertRaisesRegex(gr.Error,'Upload'):
                self.scope['analyze_cover_source'](None,'Auto-detect','original')

    def test_notebook_embeds_exact_sources_and_no_automatic_generation(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from tools.build_fast_notebook import build
        nb = build()
        self.assertEqual(nb, json.loads((ROOT / 'yue2-3b-frontier-full-song-music-generation.ipynb').read_text()))
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
                self.assertFalse(c['outputs'])
        tree = ast.parse(''.join(nb['cells'][1]['source']))
        encoded = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        files = json.loads(zlib.decompress(base64.b64decode(encoded)))
        for name, value in files.items():
            self.assertEqual(value, (ROOT / name).read_text())
        self.assertNotIn('generate_music(', ''.join(nb['cells'][3]['source']))
