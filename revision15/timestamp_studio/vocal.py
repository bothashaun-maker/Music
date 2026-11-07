"""Deterministic, cached analysis-only vocals with pinned safe model weights."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

MODEL_ID = 'adefossez/HTDemucs'
MODEL_REVISION = 'cbc8a9b1a87023b7fd74e7b3412e6321c0eab003'
MODEL_FILE = '955717e8.safetensors'
MODEL_SHA256 = 'd9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd'
PIPELINE = 'mono16k-htdemucs-shifts0-overlap025-float32-v1'


def working_root():
    return Path(os.environ.get('STS_OUTPUT_ROOT', str(Path(tempfile.gettempdir())/'timestamp_studio')))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def model_directory():
    expected = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'sha256': MODEL_SHA256}
    attached = Path('/kaggle/input')
    candidates = list(attached.rglob('studio-vocal-model.json')) if attached.is_dir() else []
    destination = working_root()/'alignment_models'/'vocal-separator'
    candidates.append(destination/'studio-vocal-model.json')
    for marker in candidates:
        if marker.is_file() and json.loads(marker.read_text()) == expected:
            if not (marker.parent/MODEL_FILE).is_file() or file_sha256(marker.parent/MODEL_FILE) != MODEL_SHA256:
                raise ValueError('Saved Studio vocal model is incomplete or has a checksum mismatch: ' + str(marker.parent))
            return marker.parent
    from huggingface_hub import hf_hub_download
    destination.mkdir(parents=True, exist_ok=True)
    print('[TimestampStudio] Preparing pinned vocal separator (~80 MB, on the server).', flush=True)
    path = hf_hub_download(MODEL_ID, MODEL_FILE, revision=MODEL_REVISION, local_dir=str(destination))
    if file_sha256(path) != MODEL_SHA256:
        raise ValueError('Studio vocal separator checksum mismatch')
    marker = destination/'studio-vocal-model.json'
    marker.write_text(json.dumps(expected))
    return destination


def separate_samples(audio, progress=None):
    """Mirror Demucs Separator's normalization; keep all full-track buffers on CPU."""
    import torch
    import julius
    from demucs.apply import apply_model
    from demucs.hf import load_safetensors_model
    from scipy.signal import resample_poly

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = load_safetensors_model(model_directory()/MODEL_FILE).eval()
    waveform = torch.from_numpy(audio).unsqueeze(0).repeat(2, 1)
    waveform = julius.resample_frac(waveform, 16000, model.samplerate)
    reference = waveform.mean(0)
    mean, std = reference.mean(), reference.std()+1e-8
    def callback(info):
        if progress and info['state'] == 'end':
            progress({'stage': 'vocal_isolation', 'seconds': round(info['segment_offset']/model.samplerate, 1),
                      'duration': len(audio)/16000, 'device': device})
    try:
        with torch.inference_mode():
            stems = apply_model(model, ((waveform-mean)/std)[None], device=device,
                                shifts=0, overlap=.25, split=True, num_workers=0,
                                progress=progress is None, callback=callback)
            vocals = stems[0, model.sources.index('vocals')].mean(0)*std+mean
            vocals = vocals.cpu().numpy()
            vocals = resample_poly(vocals, 160, 441).astype(np.float32)
    finally:
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    if vocals.shape != audio.shape or not np.isfinite(vocals).all():
        raise ValueError('Vocal isolation changed the analysis sample count or produced invalid samples')
    return vocals, device


def isolate(audio, progress=None):
    """Return a persistent float WAV; original media and its samples are read only."""
    import soundfile as sf
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError('Audio for vocal isolation must be finite, non-silent mono/16-kHz samples')
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    key = hashlib.sha256((digest+MODEL_REVISION+PIPELINE).encode()).hexdigest()
    folder = working_root()/'vocal_cache'/key
    attached = Path('/kaggle/input')
    candidates = [folder/'vocals.json']
    if attached.is_dir():
        candidates.extend(attached.rglob(key+'/vocals.json'))
    for marker in candidates:
        if marker.is_file():
            metadata = json.loads(marker.read_text())
            path = marker.parent/'vocals.wav'
            if metadata.get('key') != key or metadata.get('source_audio_sha256') != digest:
                raise ValueError('Vocal cache identity mismatch')
            if not path.is_file() or file_sha256(path) != metadata.get('wav_sha256'):
                raise ValueError('Saved vocal analysis is incomplete or changed')
            info = sf.info(path)
            if info.frames != len(audio) or info.samplerate != 16000 or info.channels != 1:
                raise ValueError('Saved vocal analysis does not match the source duration')
            print('[TimestampStudio] Reusing saved vocals; no separation inference needed.', flush=True)
            return str(path), dict(metadata, cache_reused=True)
    print('[TimestampStudio] Isolating analysis vocals; the original song is unchanged.', flush=True)
    vocals, device = separate_samples(audio, progress)
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.vocals-', suffix='.wav', dir=folder)
    os.close(descriptor)
    metadata = {'key': key, 'source_audio_sha256': digest, 'model': MODEL_ID,
                'revision': MODEL_REVISION, 'pipeline': PIPELINE, 'device': device}
    try:
        sf.write(temporary, vocals, 16000, subtype='FLOAT')
        metadata['wav_sha256'] = file_sha256(temporary)
        Path(temporary).replace(folder/'vocals.wav')
        (folder/'vocals.json').write_text(json.dumps(metadata, indent=2))
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(folder/'vocals.wav'), dict(metadata, cache_reused=False)
