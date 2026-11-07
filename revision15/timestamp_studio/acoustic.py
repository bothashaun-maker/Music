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


def ctc_path(logp, tokens, blank=0):
    """Full-sequence CTC Viterbi. Blank/label self-loops retain long gaps/notes."""
    logp = np.asarray(logp, dtype=np.float32)
    tokens = np.asarray(tokens, dtype=np.int64)
    if logp.ndim != 2 or not len(tokens) or len(tokens) > len(logp):
        raise ValueError('The recording cannot accommodate this lyric sequence')
    if not np.isfinite(logp).all() or tokens.min() < 0 or tokens.max() >= logp.shape[1]:
        raise ValueError('Invalid acoustic frame scores or target tokens')
    states = np.full(2*len(tokens)+1, blank, dtype=np.int64)
    states[1::2] = tokens
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
                continue
            words.append({'text': original, 'line': line_index, 'encoded': text})
            found = True
        if not found:
            raise ValueError('A lyric line has no alignable English words: ' + repr(line))
    if not words:
        raise ValueError('No alignable English lyric words were supplied')
    return words


def align_lines(audio, lines, progress=None):
    """Return per-line acoustic word times with explicit diagnostic scores."""
    words = prepare_lyric_words(lines)
    logp, times, vocab, diagnostics = acoustic_frames(audio, progress)
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
    for index, group in enumerate(grouped):
        mean = float(np.mean([w['acoustic_score'] for w in group]))
        span = group[-1]['end']-group[0]['start']
        if mean < .2 or span/max(len(group),1) < .12 or any(w['end']-w['start'] > 8 for w in group):
            warnings.append({'line': index+1, 'text': lines[index], 'start': group[0]['start'],
                             'end': group[-1]['end'], 'acoustic_score': round(mean,4),
                             'reason': 'Low acoustic score or unusually dense timing; listen and review'})
    diagnostics.update(method='ctc_acoustic', model=MODEL_ID, revision=MODEL_REVISION,
                       text_source='supplied_lyrics', review=warnings,
                       word_count=len(words), language='en', language_source='alignment_model')
    return grouped, diagnostics
