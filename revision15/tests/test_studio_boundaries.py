"""Regression checks for word/phrase gap structure and local vocal attacks."""
import unittest
import copy
from unittest.mock import patch

import numpy as np

from timestamp_studio import acoustic


class BoundaryTests(unittest.TestCase):
    def test_internal_pause_prior_rejects_an_isolated_early_letter(self):
        # Two possible C's: a slightly stronger early spike, and a complete
        # contiguous CAT later. The unregularized path stretches C...AT.
        logp = np.full((150,4),-12.,np.float32)
        logp[:,0] = -.001
        logp[20,1] = -.01
        logp[90,1] = -.08
        logp[100:104,2] = -.01
        logp[110:114,3] = -.01
        tokens, owners = [1,2,3], [0,0,0]
        words = [{'line':0,'text':'CAT'}]
        old = acoustic.ctc_path(logp,tokens)
        costs = acoustic.context_penalties(tokens,owners,words,separator=4)
        new = acoustic.ctc_path(logp,tokens,state_penalties=costs)
        self.assertEqual(min(old[0]),20)
        self.assertEqual(min(new[0]),90)

    def test_line_boundaries_and_explicit_pauses_are_free(self):
        vocab = {'A':1,'B':2,'|':3}
        for line_text, free in [(['A B'],False),(['A','B'],True),
                                (['A... B'],True),(['A ... B'],True),(['A… B'],True)]:
            with self.subTest(text=line_text):
                words = acoustic.prepare_lyric_words(line_text)
                costs = acoustic.context_penalties([1,3,2],[0,None,1],words,3)
                np.testing.assert_allclose(costs[2:5],0 if free else .02)
                self.assertEqual(costs[0],0)
                self.assertEqual(costs[-1],0)

    def test_penalty_does_not_limit_a_sustained_sung_letter(self):
        logp = np.full((180,3),-12.,np.float32)
        logp[:,0] = -.001
        logp[20:120,1] = -.001
        logp[125:140,2] = -.001
        costs = acoustic.context_penalties([1,2],[0,0],[{'line':0,'text':'AB'}],separator=3)
        result = acoustic.ctc_path(logp,[1,2],state_penalties=costs)
        self.assertEqual((min(result[0]),max(result[0])),(20,119))
        self.assertEqual(costs[1],0)  # Label self-loop has no duration penalty.
        self.assertAlmostEqual(float(costs[2]),.1)

    def test_long_instrumental_gaps_between_lines_are_retained(self):
        logp = np.full((600,4),-12.,np.float32)
        logp[:,0] = -.001
        logp[20:25,1] = -.001
        logp[30:35,2] = -.001
        logp[100:105,3] = -.001
        logp[500:505,1] = -.001
        logp[510:515,2] = -.001
        targets,owners = [1,2,3,1,2],[0,0,None,1,1]
        words = [{'line':0,'text':'AB'},{'line':1,'text':'AB'}]
        costs = acoustic.context_penalties(targets,owners,words,3)
        result = acoustic.ctc_path(logp,targets,state_penalties=costs)
        self.assertEqual(min(result[0]),20)
        self.assertEqual(min(result[3]),500)

    def test_bad_penalties_fail_before_decoding(self):
        for costs in [np.zeros(2),np.array([0,-.1,0]),np.array([0,np.nan,0])]:
            with self.subTest(costs=costs), self.assertRaisesRegex(ValueError,'boundary penalties'):
                acoustic.ctc_path(np.zeros((10,2)),[1],state_penalties=costs)

    def word(self,start=1.02,score=.8,end=1.4):
        return {'text':'voice','start':start,'end':end,'acoustic_score':score,
                'timing_source':'ctc_acoustic'}

    def test_local_attack_refines_onset_without_changing_source_or_other_words(self):
        time = np.arange(32000)/16000
        audio = np.where(time>=.9,.1*np.sin(2*np.pi*250*time),0).astype(np.float32)
        original = audio.copy()
        first,other = self.word(),self.word(start=1.5,end=1.8)
        grouped = [[first,other]]
        result = acoustic.refine_vocal_onsets(audio,grouped)
        self.assertAlmostEqual(first['start'],.9,delta=.025)
        self.assertEqual(first['ctc_start'],1.02)
        self.assertEqual(first['timing_source'],'ctc_with_vocal_attack')
        self.assertEqual(other['start'],1.5)
        self.assertEqual(result['applied_lines'],1)
        np.testing.assert_array_equal(audio,original)

    def test_weak_words_and_continuous_energy_do_not_get_guessed_attacks(self):
        time = np.arange(32000)/16000
        attack = np.where(time>=.9,.1*np.sin(2*np.pi*250*time),0).astype(np.float32)
        for audio, score in [(attack,.04),(np.full(32000,.1,np.float32),.8),
                             (np.zeros(32000,np.float32),.8)]:
            with self.subTest(score=score):
                word = self.word(score=score)
                self.assertEqual(acoustic.refine_vocal_onsets(audio,[[word]])['applied_lines'],0)
                self.assertEqual(word['start'],1.02)

    def test_attack_refinement_cannot_cross_the_previous_line_end(self):
        time = np.arange(32000)/16000
        audio = np.where(time>=.9,.1*np.sin(2*np.pi*250*time),0).astype(np.float32)
        before,after = self.word(start=.4,end=.95),self.word()
        acoustic.refine_vocal_onsets(audio,[[before],[after]])
        self.assertGreaterEqual(after['start'],before['end'])

    def repeated_audio(self, noisy_query=False):
        from scipy.signal import chirp
        rate = 16000
        audio = np.zeros(13*rate,np.float32)
        phrase = np.zeros(3*rate,np.float32)
        for start,end,f0,f1 in [(.2,.5,350,650),(.7,1.3,1100,500),(1.5,2.,700,1500)]:
            time = np.arange(round((end-start)*rate))/rate
            signal = .1*chirp(time,f0,time[-1],f1)*np.sin(np.linspace(0,np.pi,len(time)))
            phrase[round(start*rate):round(start*rate)+len(signal)] = signal
        audio[rate:4*rate] = phrase
        audio[9*rate:12*rate] = (np.random.default_rng(2).normal(0,.1,len(phrase))
                                         if noisy_query else phrase)
        template = [dict(self.word(start=s,end=e),text=text,acoustic_score=.9)
                    for text,s,e in [('THE',1.2,1.5),('MUSIC',1.7,2.3),('STARTS',2.5,3.)]]
        query = [dict(self.word(start=s,end=e),text=text,acoustic_score=.1)
                 for text,s,e in [('THE',8.5,8.6),('MUSIC',9.7,10.3),('STARTS',10.5,11.)]]
        return audio,[template,query]

    def test_repeated_audio_repairs_a_detached_prefix_without_moving_the_suffix(self):
        audio,groups = self.repeated_audio()
        original_audio = audio.copy()
        original_template = copy.deepcopy(groups[0])
        suffix = copy.deepcopy(groups[1][1:])
        meta = acoustic.refine_repeated_cues(audio,groups)
        self.assertEqual(meta['applied_lines'],1)
        self.assertAlmostEqual(groups[1][0]['start'],9.2,delta=.04)
        self.assertEqual(groups[1][0]['timing_source'],'repeated_vocal_dtw')
        self.assertEqual(groups[0],original_template)
        self.assertEqual(groups[1][1:],suffix)
        np.testing.assert_array_equal(audio,original_audio)

    def test_repeat_repair_extends_prefix_when_needed_to_preserve_word_order(self):
        audio,groups = self.repeated_audio()
        groups[1][0].update(start=8.1,end=8.2)
        groups[1][1].update(start=8.3,end=8.5)
        suffix = copy.deepcopy(groups[1][2])
        meta = acoustic.refine_repeated_cues(audio,groups)
        self.assertEqual(meta['applied_lines'],1)
        self.assertEqual(meta['matches'][0]['prefix_words'],2)
        self.assertAlmostEqual(groups[1][0]['start'],9.2,delta=.04)
        self.assertAlmostEqual(groups[1][1]['start'],9.7,delta=.04)
        self.assertEqual(groups[1][2],suffix)
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(groups[1],groups[1][1:])))

    def test_repetition_pass_leaves_small_cue_differences_untouched(self):
        audio,groups = self.repeated_audio()
        groups[1][0].update(start=9.25,end=9.5)
        before = copy.deepcopy(groups)
        self.assertEqual(acoustic.refine_repeated_cues(audio,groups)['applied_lines'],0)
        self.assertEqual(groups,before)

    def test_repetition_pass_rejects_different_audio_and_single_word_loops(self):
        audio,groups = self.repeated_audio(noisy_query=True)
        before = copy.deepcopy(groups)
        self.assertEqual(acoustic.refine_repeated_cues(audio,groups)['applied_lines'],0)
        self.assertEqual(groups,before)
        with patch('librosa.feature.mfcc',side_effect=AssertionError('Short loops are ambiguous')):
            self.assertEqual(acoustic.refine_repeated_cues(audio,[[self.word()],[self.word()]])['applied_lines'],0)


if __name__ == '__main__':unittest.main()
