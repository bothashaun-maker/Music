"""Align supplied English lyrics to acoustic frames, including repeated passages.

This module never uses a text-similarity score to invent word positions. The
Viterbi path advances through the complete supplied text with CTC blank states
covering gaps. Model scores are diagnostic values, not timing-accuracy claims.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

import numpy as np

MODEL_ID = 'facebook/wav2vec2-base-960h'
MODEL_REVISION = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
STRIDE_SECONDS = .02
_MODELS = {}


def _working_root():
    return Path(os.environ.get('STS_OUTPUT_ROOT', str(Path(tempfile.gettempdir()) / 'timestamp_studio')))


def _model_directory():
    attached = Path('/kaggle/input')
    if attached.is_dir():
        for marker in attached.rglob('studio-alignment-model.json'):
            data = json.loads(marker.read_text())
            if data.get('model') == MODEL_ID and data.get('revision') == MODEL_REVISION:
                if not (marker.parent / 'model.safetensors').is_file():
                    raise ValueError('Attached Studio alignment model is incomplete: ' + str(marker.parent))
                return marker.parent
    destination = _working_root() / 'alignment_models' / 'english-ctc'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'studio-alignment-model.json'
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata != {'model': MODEL_ID, 'revision': MODEL_REVISION}:
            raise ValueError('Studio alignment model revision differs from this notebook')
    else:
        from huggingface_hub import snapshot_download
        print('[TimestampStudio] Preparing English acoustic alignment weights (~360 MB, on the server).', flush=True)
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
            allow_patterns=['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'vocab.json', 'model.safetensors'])
        temporary = destination / 'studio-alignment-model.writing'
        temporary.write_text(json.dumps({'model': MODEL_ID, 'revision': MODEL_REVISION}))
        temporary.replace(manifest)
    return destination


def _load_model():
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device not in _MODELS:
        location = _model_directory()
        processor = Wav2Vec2Processor.from_pretrained(str(location), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(location), local_files_only=True,
                                             use_safetensors=True).eval().to(device)
        _MODELS[device] = (processor, model)
    return (*_MODELS[device], device)


def acoustic_frames(audio, progress=None):
    """Bound inference to 24-second windows; reuse frame scores for lyric edits."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError('Audio is empty, silent or nonfinite')
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    key = hashlib.sha256((digest + MODEL_REVISION + ':ctc-20s-context2-v1').encode()).hexdigest()
    cache_root = _working_root() / 'acoustic_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / (key + '.npz')
    candidates = [cached]
    attached = Path('/kaggle/input')
    if attached.is_dir():
        candidates.extend(attached.rglob(key + '.npz'))
    for candidate in candidates:
        if candidate.is_file():
            with np.load(candidate, allow_pickle=False) as data:
                values = {name: data[name].copy() for name in data.files}
            if str(values['key']) != key:
                raise ValueError('Acoustic cache identity mismatch')
            print('[TimestampStudio] Reusing saved acoustic frames; no model inference needed.', flush=True)
            return values['logp'], values['times'], json.loads(str(values['vocab'])), {
                'cache_reused': True, 'audio_sha256': digest, 'cache_file': str(candidate), 'device': 'cached'}
    import torch
    processor, model, device = _load_model()
    chunks, timing = [], []
    step, context = 20 * 16000, 2 * 16000
    with torch.inference_mode():
        for start in range(0, len(audio), step):
            left, right = max(0, start-context), min(len(audio), start+step+context)
            batch = processor(audio[left:right], sampling_rate=16000, return_tensors='pt')
            logp = model(batch.input_values.to(device)).logits[0].float().log_softmax(-1).cpu().numpy()
            centers = left / 16000 + np.arange(len(logp)) * STRIDE_SECONDS + .0125
            keep = (centers >= start/16000) & (centers < min(len(audio), start+step)/16000)
            chunks.append(logp[keep])
            timing.extend(centers[keep])
            if progress:
                progress({'stage': 'acoustic_frames', 'seconds': min(len(audio), start+step)/16000,
                          'duration': len(audio)/16000, 'device': device})
    logp, times = np.concatenate(chunks), np.asarray(timing)
    vocab = processor.tokenizer.get_vocab()
    descriptor, temporary = tempfile.mkstemp(prefix='.frames-', suffix='.npz', dir=cache_root)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, logp=logp, times=times, vocab=json.dumps(vocab), key=key)
        Path(temporary).replace(cached)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return logp, times, vocab, {'cache_reused': False, 'audio_sha256': digest,
                               'cache_file': str(cached), 'device': device}


def ctc_path(logp, tokens, blank=0, state_penalties=None):
    """Full-sequence CTC Viterbi. Blank/label self-loops retain long gaps/notes."""
    logp = np.asarray(logp, dtype=np.float32)
    tokens = np.asarray(tokens, dtype=np.int64)
    if logp.ndim != 2 or not len(tokens) or len(tokens) > len(logp):
        raise ValueError('The recording cannot accommodate this lyric sequence')
    if not np.isfinite(logp).all() or tokens.min() < 0 or tokens.max() >= logp.shape[1]:
        raise ValueError('Invalid acoustic frame scores or target tokens')
    states = np.full(2*len(tokens)+1, blank, dtype=np.int64)
    states[1::2] = tokens
    if state_penalties is not None:
        state_penalties = np.asarray(state_penalties, dtype=np.float32)
        if state_penalties.shape != states.shape or not np.isfinite(state_penalties).all() or np.any(state_penalties < 0):
            raise ValueError('Invalid CTC boundary penalties')
    if len(states)*len(logp) > 512*1024**2:
        raise ValueError('Lyric alignment exceeds the 512 MiB path budget; split this recording')
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    score = np.full(len(states), -np.inf, dtype=np.float32)
    score[0] = 0
    back = np.zeros((len(logp), len(states)), dtype=np.uint8)
    for frame_index, frame in enumerate(logp):
        one = np.concatenate(([-np.inf], score[:-1])).astype(np.float32)
        two = np.concatenate(([-np.inf, -np.inf], score[:-2])).astype(np.float32)
        two[~skip] = -np.inf
        choices = np.stack((score, one, two))
        back[frame_index] = choices.argmax(axis=0)
        score = choices.max(axis=0) + frame[states]
        if state_penalties is not None:
            score -= state_penalties
    state = len(states)-1 if score[-1] >= score[-2] else len(states)-2
    if not np.isfinite(score[state]):
        raise ValueError('No complete acoustic alignment path exists; no estimated times were substituted')
    frames = [[] for _ in tokens]
    for frame_index in range(len(logp)-1, -1, -1):
        if state % 2:
            frames[state//2].append(frame_index)
        state -= int(back[frame_index, state])
    if any(not indices for indices in frames):
        raise ValueError('Some lyric characters were not aligned; no estimated times were substituted')
    return frames


def prepare_lyric_words(lines):
    """Validate all text before model loading, isolation or acoustic inference."""
    words = []
    for line_index, line in enumerate(lines):
        found = False
        for original in line.split():
            text = unicodedata.normalize('NFKD', original.replace('’', "'")).upper()
            text = ''.join(c for c in text if c.isascii() and (c.isalpha() or c == "'"))
            if not any(c.isalpha() for c in text):
                if any(c.isalnum() for c in original):
                    raise ValueError('English acoustic alignment cannot encode ' + repr(original) + '; spell numbers and use English/romanized lyrics')
                if words and words[-1]['line'] == line_index and ('...' in original or '…' in original):
                    words[-1]['pause_after'] = True
                continue
            words.append({'text': original, 'line': line_index, 'encoded': text,
                          'pause_after': '...' in original or '…' in original})
            found = True
        if not found:
            raise ValueError('A lyric line has no alignable English words: ' + repr(line))
    if not words:
        raise ValueError('No alignable English lyric words were supplied')
    return words


def combine_views(first, second):
    """Equal geometric acoustic fusion; retain independent, reusable frame caches."""
    logp, times, vocab, diagnostics = first
    vocal_logp, vocal_times, vocal_vocab, vocal_diagnostics = second
    if logp.shape != vocal_logp.shape or not np.array_equal(times, vocal_times) or vocab != vocal_vocab:
        raise ValueError('Mix and vocal acoustic frames do not have matching timestamps and vocabulary')
    fused = (logp+vocal_logp)*.5
    # Frame-wise normalization does not change the Viterbi optimum, but scores
    # must describe the normalized joint distribution, not unnormalized products.
    fused -= np.logaddexp.reduce(fused, axis=1, keepdims=True)
    device = diagnostics.get('device')
    if device == 'cached':
        device = vocal_diagnostics.get('device', device)
    return fused, times, vocab, dict(diagnostics, variant='mix_vocal_ensemble',
        fusion='equal_geometric', mix_frames=dict(diagnostics), vocal_frames=vocal_diagnostics,
        device=device, cache_reused=bool(diagnostics.get('cache_reused') and vocal_diagnostics.get('cache_reused')))


def context_penalties(targets, owners, words, separator):
    """Soft gap priors: within-word blanks cost more; line/ellipsis pauses stay free.

    Letter self-loops are untouched so a supported sung vowel can sustain. No
    maximum word duration or song-specific times are imposed.
    """
    penalties = np.zeros(2*len(targets)+1, dtype=np.float32)
    for index in range(1, len(targets)):
        if targets[index-1] != separator and targets[index] != separator:
            penalties[2*index] = .1
    for index, token in enumerate(targets):
        if token != separator:
            continue
        before, after = words[owners[index-1]], words[owners[index+1]]
        if before['line'] == after['line'] and not before.get('pause_after', False):
            penalties[2*index:2*index+3] = .02
    return penalties


def refine_vocal_onsets(audio, grouped):
    """Backtrack to a local vocal attack, only with usable acoustic/energy evidence."""
    from scipy.ndimage import uniform_filter1d
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Invalid vocal samples for onset refinement')
    envelope = np.sqrt(np.maximum(0, uniform_filter1d(audio.astype(np.float64)**2, 320, mode='nearest')))[::80]
    applied = 0
    previous_end = 0.
    for group in grouped:
        word = group[0]
        start = word['start']
        word['ctc_start'] = start
        if word['acoustic_score'] >= .05:
            lower = max(previous_end, start-.4, 0.)
            ids = np.arange(max(0, int(lower/.005)), min(len(envelope), int((start+.12)/.005)+1))
            times, energy = ids*.005, envelope[ids]
            before, after = energy[times < start], energy[times >= start]
            if len(before) >= 6 and len(after):
                floor, peak = np.quantile(before,.1), np.quantile(after,.9)
                if peak > 2*floor and peak >= 1e-5:
                    threshold = floor+.2*(peak-floor)
                    candidates = [float(times[j]) for j in range(4,len(energy)-1)
                                  if lower <= times[j] < start and energy[j] >= threshold
                                  and energy[j-1] < threshold and np.mean(energy[j-4:j] < threshold) >= .75]
                    if candidates:
                        word['start'] = candidates[-1]
                        word['timing_source'] = 'ctc_with_vocal_attack'
                        applied += 1
        previous_end = group[-1]['end']
    return {'method':'bounded_local_vocal_attack', 'applied_lines':applied,
            'lookback_seconds':.4, 'rms_window_seconds':.02, 'stride_seconds':.005,
            'minimum_word_score':.05, 'relative_attack_threshold':.2,
            'scores_describe':'CTC frames, not measured boundary accuracy'}


def refine_repeated_cues(audio, grouped):
    """Repair uncertain leading cues by matching a clearer repeated vocal phrase.

    Only major cue shifts are considered. Matching is bounded to each existing
    occurrence, and at most three leading words can change; the suffix and all
    other lyric occurrences remain independently aligned.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for index, words in enumerate(grouped):
        key = tuple(''.join(c for c in unicodedata.normalize('NFKD',w['text'].replace('’',"'")).upper()
                            if c.isascii() and (c.isalpha() or c == "'")) for w in words)
        if len(key) >= 3:
            groups[key].append(index)
    candidates = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        template = max(indices, key=lambda i: np.mean([w['acoustic_score'] for w in grouped[i]]))
        if grouped[template][0]['acoustic_score'] < .25:
            continue
        for query in indices:
            if query != template and grouped[query][0]['acoustic_score'] < .2:
                candidates.append((template,query))
    details = {'method':'bounded_repeated_vocal_dtw', 'applied_lines':0, 'matches':[],
               'minimum_cue_shift':.35, 'maximum_cue_shift':3., 'maximum_cosine_distance':.55,
               'minimum_phrase_words':3, 'maximum_changed_prefix_words':3}
    if not candidates:
        return details
    import librosa
    audio = np.asarray(audio, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=audio,sr=16000,n_mfcc=13,n_mels=40,n_fft=512,
                                hop_length=160,fmin=80,fmax=7600)[1:]
    features = np.concatenate((mfcc,librosa.feature.delta(mfcc)))
    features = (features-features.mean(axis=1,keepdims=True))/(features.std(axis=1,keepdims=True)+1e-5)
    for template_index, query_index in candidates:
        template,query = grouped[template_index],grouped[query_index]
        tl = max(0,int((template[0]['start']-.5)*100))
        th = min(features.shape[1],int((template[-1]['end']+.3)*100))
        ql = max(0,int((query[0]['start']-2)*100))
        qh = min(features.shape[1],int((query[-1]['end']+1)*100))
        if not 4 <= th-tl <= 1500 or not 4 <= qh-ql <= 2500:
            continue
        first,second = features[:,tl:th],features[:,ql:qh]
        if np.any(np.linalg.norm(first,axis=0)<1e-8) or np.any(np.linalg.norm(second,axis=0)<1e-8):
            continue
        try:
            _,path = librosa.sequence.dtw(X=first,Y=second,metric='cosine',subseq=True,
                step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),weights_mul=np.array([2.,3.,3.]))
        except ValueError:
            continue  # No usable acoustic path: retain the CTC result.
        path = path[::-1]
        x = np.unique(path[:,0])
        y = np.array([np.median(path[path[:,0]==i,1]) for i in x])
        def mapped(time):
            return float((np.interp(time*100-tl,x,y)+ql)/100)
        cue = mapped(template[0]['start'])
        shift = cue-query[0]['start']
        norms = np.linalg.norm(first[:,path[:,0]],axis=0)*np.linalg.norm(second[:,path[:,1]],axis=0)
        distance = float(np.mean(1-np.sum(first[:,path[:,0]]*second[:,path[:,1]],axis=0)/(norms+1e-8)))
        if not .35 <= abs(shift) <= 3 or not np.isfinite(distance) or distance > .55:
            continue
        previous_end = grouped[query_index-1][-1]['end'] if query_index else 0.
        if cue < previous_end:
            continue
        # Extend the changed prefix only when a mapped word would otherwise
        # overlap the old next word. Never compress a whole word to a few ms.
        count = 1
        while count < len(query) and (query[count]['start'] <= cue or
                                      mapped(template[count-1]['end']) > query[count]['start']):
            count += 1
        if count > 3 or count >= len(query):
            continue
        proposal = [(mapped(w['start']),mapped(w['end'])) for w in template[:count]]
        if any(not 0 <= a < b <= len(audio)/16000 for a,b in proposal):
            continue
        if any(a[1] > b[0] for a,b in zip(proposal,proposal[1:])) or proposal[-1][1] > query[count]['start']:
            continue
        before = query[0]['start']
        for word,(start,end) in zip(query,proposal):
            word.setdefault('ctc_start',word['start'])
            word['ctc_end'] = word['end']
            word.update(start=start,end=end,timing_source='repeated_vocal_dtw',
                        repeat_template_line=template_index+1,repeat_distance=round(distance,4))
        details['applied_lines'] += 1
        details['matches'].append({'line':query_index+1,'template_line':template_index+1,
            'previous_cue':before,'cue':cue,'cosine_distance':round(distance,4),'prefix_words':count})
    return details


def align_lines(audio, lines, progress=None, vocal_audio=None):
    """Return per-line acoustic word times with explicit diagnostic scores."""
    words = prepare_lyric_words(lines)
    first = acoustic_frames(audio, progress)
    if vocal_audio is not None:
        if np.shape(vocal_audio) != np.shape(audio):
            raise ValueError('Mix and vocal sample counts differ')
        vocal = acoustic_frames(vocal_audio, progress)
        metadata = combine_views(first, vocal)[3]
        metadata.update(variant='vocal_context_ctc', fusion='none_vocal_primary',
                        reference_mix='retained_for_diagnostics')
        first = (*vocal[:3], metadata)
    logp, times, vocab, diagnostics = first
    targets, owners = [], []
    for word_index, word in enumerate(words):
        text = word['encoded']
        if any(c not in vocab for c in text):
            raise ValueError('Unsupported lyric character in ' + repr(word['text']))
        if targets:
            targets.append(vocab['|'])
            owners.append(None)
        targets.extend(vocab[c] for c in text)
        owners.extend([word_index]*len(text))
    if vocal_audio is not None:
        penalties = context_penalties(targets, owners, words, vocab['|'])
        path = ctc_path(logp, targets, vocab['<pad>'], state_penalties=penalties)
        diagnostics['decoder'] = {'method':'context_gap_prior', 'within_word_blank_cost':.1,
                                  'within_phrase_gap_cost':.02, 'line_and_ellipsis_pauses':'free',
                                  'sustained_label_states':'unpenalized'}
    else:
        path = ctc_path(logp, targets, vocab['<pad>'])
    word_frames = [[] for _ in words]
    for token_index, owner in enumerate(owners):
        if owner is not None:
            word_frames[owner].extend((frame, targets[token_index]) for frame in path[token_index])
    grouped = [[] for _ in lines]
    warnings = []
    for word, positions in zip(words, word_frames):
        start = min(times[i] for i,_ in positions)
        end = min(len(audio)/16000, max(times[i] for i,_ in positions)+STRIDE_SECONDS)
        confidence = float(np.exp(np.mean([logp[i,t] for i,t in positions])))
        item = {'text': word['text'], 'start': float(start), 'end': float(end),
                'matched': None, 'acoustic_score': round(confidence, 4), 'timing_source': 'ctc_acoustic'}
        grouped[word['line']].append(item)
    if vocal_audio is not None:
        diagnostics['onset_refinement'] = refine_vocal_onsets(vocal_audio, grouped)
        diagnostics['repetition_refinement'] = refine_repeated_cues(vocal_audio, grouped)
    for index, group in enumerate(grouped):
        mean = float(np.mean([w['acoustic_score'] for w in group]))
        span = group[-1]['end']-group[0]['start']
        reasons = []
        if mean < .2:
            reasons.append('Low acoustic score')
        if span/max(len(group),1) < .12:
            reasons.append('Unusually dense timing')
        if any(w['end']-w['start'] > 8 for w in group):
            reasons.append('A word spans more than eight seconds')
        if len(group) > 1 and group[0]['acoustic_score'] < .2 and group[1]['start']-group[0]['end'] > 1:
            reasons.append('Weak first word separated from the next word by more than one second')
        if reasons:
            warnings.append({'line': index+1, 'text': lines[index], 'start': group[0]['start'],
                             'end': group[-1]['end'], 'acoustic_score': round(mean,4),
                             'reason': '; '.join(reasons) + '; listen and review'})
    diagnostics.update(method='ctc_acoustic', model=MODEL_ID, revision=MODEL_REVISION,
                       text_source='supplied_lyrics', review=warnings,
                       word_count=len(words), language='en', language_source='alignment_model')
    return grouped, diagnostics
