"""Reproduce the supplied VAD/CUDA failures without GPU time or model downloads."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('STS_AUTOLAUNCH', '0')
        spec = importlib.util.spec_from_file_location('recognition_test_engine', ROOT / 'timestamp_studio/engine.py')
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)
        self.audio = np.full(80000, .1, np.float32)
        self.load = patch.object(self.engine, 'load_audio_16k', return_value=self.audio)
        self.load.start()
        self.addCleanup(self.load.stop)

    def model(self):
        words = [SimpleNamespace(word='The', start=1., end=1.2),
                 SimpleNamespace(word='verdict', start=1.3, end=1.8),
                 SimpleNamespace(word='stands', start=1.9, end=2.4)]
        segment = SimpleNamespace(start=1., end=2.4, text='The verdict stands', words=words)
        model = Mock()
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(duration=5., language='en'))
        return model

    def test_music_is_not_filtered_out_and_real_word_anchors_are_preserved(self):
        model = self.model()
        with patch.object(self.engine, 'pick_device', return_value=('cpu', 'int8')), \
             patch.object(self.engine, 'get_model', return_value=model), \
             patch.object(self.engine, 'detect_beats', return_value=(120., [0., .5, 1.])):
            result = self.engine.run_alignment('song.wav', '', 'small', 'en')
        self.assertIs(model.transcribe.call_args.kwargs['vad_filter'], False)
        self.assertIsNone(result['match_rate'])
        line = [e for e in result['entries'] if e['kind'] == 'sing'][0]
        self.assertEqual((line['start'], line['end']), (1., 2.4))
        exported = json.loads(self.engine.fmt_json(result, 0))
        self.assertEqual(exported['transcription']['word_count'], 3)
        self.assertEqual(exported['transcription']['segments'][0]['text'], 'The verdict stands')

    def test_missing_cuda12_library_selects_cpu_even_with_a_visible_gpu(self):
        ctranslate2 = SimpleNamespace(get_cuda_device_count=lambda: 1)
        with patch.dict(sys.modules, {'ctranslate2': ctranslate2}), \
             patch('ctypes.CDLL', side_effect=OSError('libcublas.so.12 cannot be loaded')):
            self.assertEqual(self.engine.pick_device(), ('cpu', 'int8'))
        self.assertIn('libcublas.so.12', self.engine._CUDA_DISABLED_REASON)

    def test_lazy_cuda_library_error_retries_once_on_cpu_and_records_backend(self):
        def broken_generator():
            raise RuntimeError('Library libcublas.so.12 is not found or cannot be loaded')
            yield
        cuda_model = Mock()
        cuda_model.transcribe.return_value = (broken_generator(), SimpleNamespace(duration=5., language='en'))
        cpu_model = self.model()
        with patch.object(self.engine, 'pick_device', return_value=('cuda', 'float16')), \
             patch.object(self.engine, 'get_model', side_effect=[cuda_model, cpu_model]) as load:
            words, _, _, _, _ = self.engine.transcribe_words('song.wav', 'small', 'en')
        self.assertEqual(len(words), 3)
        self.assertEqual(load.call_args_list[1].args, ('small', 'cpu', 'int8'))
        self.assertEqual(self.engine._TRANSCRIPTION_DIAGNOSTICS['device'], 'cpu')
        self.assertIn('libcublas.so.12', self.engine._TRANSCRIPTION_DIAGNOSTICS['cuda_fallback_reason'])

    def test_other_gpu_failures_are_not_silently_retried(self):
        model = Mock()
        model.transcribe.side_effect = RuntimeError('CUDA out of memory')
        with patch.object(self.engine, 'pick_device', return_value=('cuda', 'float16')), \
             patch.object(self.engine, 'get_model', return_value=model) as load:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                self.engine.transcribe_words('song.wav', 'small', 'en')
        self.assertEqual(load.call_count, 1)

    def test_no_recognized_segments_fails_instead_of_spreading_lyrics(self):
        model = Mock()
        model.transcribe.return_value = (iter([]), SimpleNamespace(duration=5., language='en'))
        with patch.object(self.engine, 'pick_device', return_value=('cpu', 'int8')), \
             patch.object(self.engine, 'get_model', return_value=model):
            with self.assertRaisesRegex(ValueError, 'no vocal segments'):
                self.engine.run_alignment('song.wav', '', 'small', 'en')
        self.assertEqual(self.engine._TRANSCRIPTION_DIAGNOSTICS['word_count'], 0)

    def test_zero_word_match_is_not_exported_as_an_alignment(self):
        entries = self.engine.parse_lyrics('Hallelujah')
        with self.assertRaisesRegex(ValueError, 'No supplied lyric words matched'):
            self.engine.assign_word_times([], entries, 387.52)
        self.assertIsNone(entries[0]['words'][0]['start'])
        with self.assertRaisesRegex(ValueError, 'Export stopped'):
            self.engine.build_zip({'match_rate': 0, 'entries': []})

    def test_empty_lyric_box_still_supports_auto_transcription(self):
        with patch.object(self.engine, 'pick_device', return_value=('cpu', 'int8')), \
             patch.object(self.engine, 'get_model', return_value=self.model()), \
             patch.object(self.engine, 'detect_beats', return_value=(120., [0., .5, 1.])):
            result = self.engine.run_alignment('song.wav', '', 'small', 'en')
        self.assertIsNone(result['match_rate'])
        self.assertEqual(result['entries'][0]['text'], 'The verdict stands')
        self.assertEqual(result['entries'][0]['start'], 1.)

    def test_silent_recording_fails_before_model_loading(self):
        with patch.object(self.engine, 'pick_device', return_value=('cpu', 'int8')), \
             patch.object(self.engine, 'load_audio_16k', return_value=np.zeros(16000)), \
             patch.object(self.engine, 'get_model') as load:
            with self.assertRaisesRegex(ValueError, 'silent'):
                self.engine.transcribe_words('song.wav', 'small', 'en')
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
