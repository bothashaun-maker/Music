"""Cached vocal analysis, synchronized acoustic views and honest onset review."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from timestamp_studio import acoustic, vocal

ROOT = Path(__file__).resolve().parents[1]


class VocalTests(unittest.TestCase):
    def test_separation_is_cached_without_modifying_source_samples(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'STS_OUTPUT_ROOT':tmp}):
            audio = np.full(16000, .1, np.float32)
            original = audio.copy()
            with patch.object(vocal, 'separate_samples', return_value=(audio*.5, 'cpu')) as infer:
                path, first = vocal.isolate(audio)
                again, second = vocal.isolate(audio)
            np.testing.assert_array_equal(audio, original)
            recovered, rate = sf.read(path, dtype='float32')
            np.testing.assert_array_equal(recovered, original*.5)
            self.assertEqual(rate, 16000)
            self.assertEqual(again, path)
            self.assertFalse(first['cache_reused'])
            self.assertTrue(second['cache_reused'])
            infer.assert_called_once()
            # A damaged cache must not be used or silently regenerated.
            Path(path).write_bytes(b'broken')
            with patch.object(vocal, 'separate_samples') as infer:
                with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
                    vocal.isolate(audio)
                infer.assert_not_called()

    def test_invalid_audio_stops_before_separation(self):
        for audio in (np.zeros(16000), np.full(10, np.nan), np.zeros((2,100))):
            with self.subTest(shape=audio.shape), patch.object(vocal, 'separate_samples') as infer:
                with self.assertRaises(ValueError):vocal.isolate(audio)
                infer.assert_not_called()

    def test_pinned_weights_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'STS_OUTPUT_ROOT':tmp}):
            root = Path(tmp)/'alignment_models/vocal-separator'
            root.mkdir(parents=True)
            data = b'model-fixture'
            digest = hashlib.sha256(data).hexdigest()
            (root/vocal.MODEL_FILE).write_bytes(data)
            (root/'studio-vocal-model.json').write_text(json.dumps({
                'model':vocal.MODEL_ID, 'revision':vocal.MODEL_REVISION, 'sha256':digest}))
            with patch.object(vocal, 'MODEL_SHA256', digest):
                self.assertEqual(vocal.model_directory(), root)
                (root/vocal.MODEL_FILE).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    vocal.model_directory()

    def test_acoustic_fusion_preserves_inputs_and_rejects_unsynchronized_frames(self):
        a = np.log(np.array([[.8,.2],[.1,.9]], np.float32))
        b = np.log(np.array([[.2,.8],[.4,.6]], np.float32))
        original_a, original_b = a.copy(), b.copy()
        times, vocab = np.array([.01,.03]), {'<pad>':0,'A':1}
        first = (a,times,vocab,{'audio_sha256':'mix','cache_reused':True})
        second = (b,times,vocab,{'audio_sha256':'vocal','cache_reused':True})
        result = acoustic.combine_views(first,second)
        expected = np.sqrt(np.exp(a)*np.exp(b));expected /= expected.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(np.exp(result[0]), expected, atol=1e-6)
        np.testing.assert_array_equal(a,original_a)
        np.testing.assert_array_equal(b,original_b)
        self.assertEqual(result[3]['audio_sha256'],'mix')
        self.assertEqual(result[3]['vocal_frames']['audio_sha256'],'vocal')
        with self.assertRaisesRegex(ValueError,'matching timestamps'):
            acoustic.combine_views(first,(b,times+.02,vocab,{}))
        with self.assertRaisesRegex(ValueError,'matching timestamps'):
            acoustic.combine_views(first,(b,times,{'A':0,'<pad>':1},{}))

    def test_high_line_average_does_not_hide_a_weak_detached_first_word(self):
        logp = np.full((200,6), -.001, np.float32)
        logp[1,1] = logp[2,2] = -4
        frames = (logp,np.arange(200)*.02,{'<pad>':0,'I':1,'N':2,'|':3,'E':4,'D':5},{'device':'cached'})
        with patch.object(acoustic,'acoustic_frames',return_value=frames), \
             patch.object(acoustic,'ctc_path',return_value=[[1],[2],[3],[150],[151],[152]]):
            _, meta = acoustic.align_lines(np.ones(64000,np.float32), ['IN END'])
        self.assertGreater(meta['review'][0]['acoustic_score'], .2)
        self.assertIn('Weak first word', meta['review'][0]['reason'])

    def test_requested_separation_failure_does_not_fall_back_to_unseparated_alignment(self):
        os.environ.setdefault('STS_AUTOLAUNCH','0')
        spec = importlib.util.spec_from_file_location('vocal_test_engine', ROOT/'timestamp_studio/engine.py')
        engine = importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        with patch.object(engine,'isolate_vocals',side_effect=RuntimeError('separator unavailable')), \
             patch.object(engine,'acoustic_align_lines') as align:
            with self.assertRaisesRegex(RuntimeError,'separator unavailable'):
                engine.run_alignment('song.wav','Hello',isolate_vocals_flag=True)
            align.assert_not_called()


if __name__ == '__main__':unittest.main()
