"""CTC path, repeated passages, pipeline routing and precise-export regressions."""
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from timestamp_studio import acoustic

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('STS_AUTOLAUNCH', '0')


class AcousticTests(unittest.TestCase):
    def engine(self):
        spec = importlib.util.spec_from_file_location('ctc_test_engine', ROOT/'timestamp_studio/engine.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_ctc_tracks_two_occurrences_across_a_long_blank_gap(self):
        labels = np.zeros(500, dtype=int)
        labels[20:25]=1; labels[30:35]=2; labels[40:45]=3
        labels[400:405]=1; labels[410:415]=2
        logp = np.full((len(labels),4), -12., dtype=np.float32)
        logp[np.arange(len(labels)),labels]=-.001
        path = acoustic.ctc_path(logp,[1,2,3,1,2])
        self.assertEqual((min(path[0]),max(path[0])),(20,24))
        self.assertEqual((min(path[3]),max(path[3])),(400,404))
        self.assertGreater(min(path[3])-max(path[1]),300)

    def test_repeated_identical_letters_require_a_blank_between_them(self):
        logp=np.full((7,2),-12.,dtype=np.float32)
        for frame,label in enumerate([0,1,1,0,1,1,0]):logp[frame,label]=-.001
        path=acoustic.ctc_path(logp,[1,1])
        self.assertEqual((min(path[0]),max(path[0])),(1,2))
        self.assertEqual((min(path[1]),max(path[1])),(4,5))

    def test_impossible_or_corrupt_paths_fail_without_estimates(self):
        for logp,tokens in [(np.zeros((1,3)),[1,1]),(np.zeros((1,3)),[7]),
                             (np.full((3,3),np.nan),[1])]:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):acoustic.ctc_path(logp,tokens)

    def test_supplied_lyrics_use_acoustics_and_keep_repeated_line_order(self):
        engine=self.engine()
        words=[[{'text':'Hello','start':s,'end':s+.4,'matched':None,'acoustic_score':.8,
                 'timing_source':'ctc_acoustic'}] for s in [10.,15.,60.,65.]]
        metadata={'model':'test-ctc','device':'cpu','method':'ctc_acoustic','review':[]}
        with patch.object(engine,'load_audio_16k',return_value=np.ones(16000*100,np.float32)), \
             patch.object(engine,'acoustic_align_lines',return_value=(words,metadata)) as align, \
             patch.object(engine,'transcribe_words',side_effect=AssertionError('ASR matching must not run')), \
             patch.object(engine,'detect_beats',return_value=(120.,[0.,1.,2.])):
            result=engine.run_alignment('song.wav','[Intro]\nHello\nHello\n[Verse]\nHello\nHello','base','en')
        self.assertEqual(align.call_args.args[1],['Hello']*4)
        self.assertEqual([e['start'] for e in result['entries'] if e['kind']=='sing'],[10.,15.,60.,65.])
        self.assertIsNone(result['match_rate'])
        data=json.loads(engine.fmt_json(result,0))
        self.assertEqual(data['alignment_method'],'ctc_acoustic')
        self.assertIsNone(data['lyric_word_match_rate'])

    def test_stray_bracket_and_punctuation_do_not_become_lyric_lines(self):
        engine = self.engine()
        entries = engine.parse_lyrics('[Verse 3]\n[\nHello again!\n...\n]\n[Chorus]\nHello again!')
        self.assertEqual([(e['kind'], e['text']) for e in entries],
                         [('tag', 'Verse 3'), ('sing', 'Hello again!'),
                          ('tag', 'Chorus'), ('sing', 'Hello again!')])
        words = acoustic.prepare_lyric_words([e['text'] for e in entries if e['kind'] == 'sing'])
        self.assertEqual([w['line'] for w in words], [0, 0, 1, 1])

    def test_invalid_text_fails_before_acoustic_frames(self):
        for lines in (['Hello', '['], ['Hello', '123'], ['Hello', "'"], []):
            with self.subTest(lines=lines), \
                 patch.object(acoustic, 'acoustic_frames') as frames:
                with self.assertRaises(ValueError):
                    acoustic.align_lines(np.ones(16000, np.float32), lines)
                frames.assert_not_called()

    def test_invalid_lyrics_fail_before_isolation_or_audio_loading(self):
        engine = self.engine()
        for lyrics in ('[Verse]\n123', '[\n...'):
            with self.subTest(lyrics=lyrics), \
                 patch.object(engine, 'isolate_vocals') as isolate, \
                 patch.object(engine, 'load_audio_16k') as load, \
                 patch.object(engine, 'transcribe_words') as transcribe:
                with self.assertRaises(ValueError):
                    engine.run_alignment('song.wav', lyrics, isolate_vocals_flag=True)
                isolate.assert_not_called()
                load.assert_not_called()
                transcribe.assert_not_called()

    def test_section_follows_next_line_in_text_even_when_legacy_times_overlap(self):
        engine=self.engine()
        def line(text,start,end):
            return {'kind':'sing','text':text,'words':[{'text':text,'start':start,'end':end}]}
        entries=[line('before',10,20),{'kind':'tag','text':'Verse'},line('after',10,10.3),line('later',30,31)]
        engine.build_line_times(entries,40)
        self.assertEqual(entries[1]['start'],entries[2]['start'])
        self.assertEqual(entries[1]['start'],10)

    def test_cut_plan_has_no_start_or_section_gaps_and_csv_keeps_milliseconds(self):
        engine=self.engine()
        sections=[{'name':'Intro','start':0,'end':29.812},
                  {'name':'Verse 1','start':29.812,'end':60.017}]
        rows=engine.build_cut_plan(sections,list(np.arange(.448,61,.448)),133.9,60.017,4)
        self.assertEqual(rows[0]['start'],0)
        self.assertEqual(rows[-1]['end'],60.017)
        self.assertTrue(all(abs(a['end']-b['start'])<1e-9 for a,b in zip(rows,rows[1:])))
        self.assertEqual(sum(r['frames'] for r in rows),round(60.017*24))
        output=list(csv.DictReader(io.StringIO(engine.fmt_cutplan(rows))))
        self.assertTrue(any(r['start']=='00:00:29.812' for r in output))

    def test_downsampling_rejects_high_frequency_aliasing(self):
        engine=self.engine()
        x=np.sin(2*np.pi*12000*np.arange(48000)/48000).astype(np.float32)
        y=engine._resample_linear(x,48000,16000)
        self.assertLess(float(np.sqrt(np.mean(y[100:-100]**2))),.02)


if __name__=='__main__':unittest.main()
