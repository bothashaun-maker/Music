"""Evidence collection preserves samples and reuses existing caches only."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('studio_evidence', ROOT/'tools/collect_studio_evidence.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root/'output'
        self.comfy = self.root/'ComfyUI'
        self.input = self.comfy/'input'
        self.input.mkdir(parents=True)
        (self.out/'acoustic_cache').mkdir(parents=True)
        self.song = self.input/'song.wav'

    def cache(self, samples):
        digest = hashlib.sha256(np.ascontiguousarray(samples, dtype=np.float32).tobytes()).hexdigest()
        key = hashlib.sha256((digest+collector.MODEL_REVISION+':ctc-20s-context2-v1').encode()).hexdigest()
        np.savez_compressed(self.out/'acoustic_cache'/f'{key}.npz', key=key,
                            logp=np.zeros((10, 2)), times=np.arange(10)*.02,
                            vocab=json.dumps({'<pad>':0, 'A':1}))
        return digest

    def collect(self, **kwargs):
        return collector.collect(output_root=self.out, comfy_root=self.comfy, **kwargs)

    def test_exact_analysis_samples_and_existing_export_are_recovered_without_changing_song(self):
        source = np.random.default_rng(5).normal(0, .1, (48000, 2)).astype(np.float32)
        sf.write(self.song, source, 48000, subtype='FLOAT')
        original = self.song.read_bytes()
        expected = resample_poly(source.mean(axis=1), 1, 3).astype(np.float32)
        digest = self.cache(expected)
        result = self.out/'song_timestamps.json'
        result.write_text(json.dumps({'alignment':{'audio_sha256':digest}, 'entries':[]}))
        # An unrelated export must not be mistaken for this recording.
        (self.out/'wrong_timestamps.json').write_text(json.dumps({'alignment':{'audio_sha256':'other'}}))
        archives = self.collect()
        self.assertEqual(len(archives), 1)
        with zipfile.ZipFile(archives[0]) as bundle:
            self.assertEqual(bundle.testzip(), None)
            manifest = json.loads(bundle.read('evidence.json'))
            self.assertEqual(manifest['matching_export'], result.name)
            data = np.load(io.BytesIO(bundle.read('analysis_audio_000.npz')), allow_pickle=False)
            np.testing.assert_array_equal(data['samples'], expected)
            self.assertEqual(json.loads(bundle.read('timestamps.json'))['alignment']['audio_sha256'], digest)
        self.assertEqual(self.song.read_bytes(), original)

    def test_missing_exact_cache_stops_instead_of_starting_inference(self):
        sf.write(self.song, np.ones(16000, np.float32)*.1, 16000, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'No saved acoustic frames match'):
            self.collect()
        self.assertFalse(list(self.out.glob('studio_evidence_*.zip')))

    def test_full_length_audio_is_recoverable_in_order_including_the_last_partial_chunk(self):
        samples = (np.arange(387 * 16000 + 8320, dtype=np.float32) % 8000) / 8000
        sf.write(self.song, samples, 16000, subtype='FLOAT')
        self.cache(samples)
        archives = self.collect()
        chunks, manifest = {}, None
        for path in archives:
            with zipfile.ZipFile(path) as bundle:
                manifest = json.loads(bundle.read('evidence.json'))
                for name in bundle.namelist():
                    if name.startswith('analysis_audio_'):
                        chunks[name] = np.load(io.BytesIO(bundle.read(name)), allow_pickle=False)['samples']
        ordered = []
        cursor = 0
        for chunk in manifest['audio_chunks']:
            self.assertEqual(chunk['sample_start'], cursor)
            self.assertEqual(len(chunks[chunk['filename']]), chunk['sample_count'])
            ordered.append(chunks[chunk['filename']])
            cursor += chunk['sample_count']
        self.assertEqual(cursor, len(samples))
        np.testing.assert_array_equal(np.concatenate(ordered), samples)

    def test_multiple_songs_require_explicit_selection(self):
        for name in ('song.wav', 'other.wav'):
            sf.write(self.input/name, np.ones(100, np.float32), 16000, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'Set SONG to the exact original WAV path'):
            self.collect()

    def test_archive_size_limit_splits_files_without_truncating_samples(self):
        samples = np.random.default_rng(7).normal(0, .1, 4500).astype(np.float32)
        sf.write(self.song, samples, 16000, subtype='FLOAT')
        digest = self.cache(samples)
        (self.out/'song_timestamps.json').write_text(json.dumps({
            'alignment':{'audio_sha256':digest}, 'status':'x'*15000}))
        archives = self.collect(max_archive_bytes=28000)
        self.assertGreater(len(archives), 1)
        recovered = []
        for path in archives:
            self.assertLessEqual(path.stat().st_size, 28000)
            with zipfile.ZipFile(path) as bundle:
                self.assertEqual(json.loads(bundle.read('evidence.json'))['archive_count'], len(archives))
                if 'analysis_audio_000.npz' in bundle.namelist():
                    recovered.append(np.load(io.BytesIO(bundle.read('analysis_audio_000.npz')))['samples'])
        self.assertEqual(len(recovered), 1)
        np.testing.assert_array_equal(recovered[0], samples)


if __name__ == '__main__':
    unittest.main()
