"""Collect an existing Studio run without loading models or running alignment.

Run in the SAME Kaggle session after stopping the Studio setup cell. The source
WAV is read only. Audio arrays contain the exact float32 mono/16-kHz samples used
by the current engine; this is an analysis copy, not a replacement for the song.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import uuid
import zipfile


MODEL_REVISION = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'


def collect(audio_path=None, output_root='/kaggle/working/timestamp_studio',
            comfy_root='/kaggle/temp/timestamp_studio/ComfyUI',
            max_archive_bytes=30 * 1024**2):
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    output = Path(output_root)
    if not output.is_dir():
        raise ValueError('Studio outputs are missing. Use the original Kaggle session; do not start a new alignment')
    if audio_path is None:
        songs = sorted(p for p in (Path(comfy_root) / 'input').glob('*')
                       if p.is_file() and p.suffix.lower() in {'.wav', '.flac', '.aiff', '.aif'})
        if len(songs) != 1:
            raise ValueError('Set SONG to the exact original WAV path. Found: ' + repr([str(p) for p in songs]))
        audio_path = songs[0]
    audio_path = Path(audio_path)
    audio, rate = sf.read(audio_path, dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Invalid analysis audio')
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    key = hashlib.sha256((digest + MODEL_REVISION + ':ctc-20s-context2-v1').encode()).hexdigest()
    cache = output / 'acoustic_cache' / (key + '.npz')
    if not cache.is_file():
        raise ValueError('No saved acoustic frames match these exact audio samples. '
                         'Nothing was aligned or exported. Keep the original session and report this message')
    with np.load(cache, allow_pickle=False) as frames:
        if str(frames['key']) != key:
            raise ValueError('Acoustic cache identity mismatch')

    matching = []
    for candidate in output.rglob('*_timestamps.json'):
        data = json.loads(candidate.read_text())
        if data.get('alignment', {}).get('audio_sha256') == digest:
            matching.append(candidate)
    latest = max(matching, key=lambda p: p.stat().st_mtime) if matching else None
    manifest = {
        'purpose': 'Timing investigation; no model inference or alignment was performed',
        'source_filename': audio_path.name, 'source_sample_rate': rate,
        'analysis_sample_rate': 16000, 'analysis_dtype': 'float32',
        'analysis_samples': len(audio), 'duration_seconds': len(audio)/16000,
        'analysis_audio_sha256': digest, 'cache_key': key,
        'model_revision': MODEL_REVISION,
        'matching_export': latest.name if latest else None,
        'audio_chunks': [],
        'notes': 'The source song was not modified. Concatenate chunks by sample_start. '
                 'Scores and synthetic tests do not establish timing accuracy. '
                 'If no matching export exists, attach the latest timestamp ZIP separately.',
    }
    for name in ('sts_engine.py', 'acoustic.py'):
        source = Path(comfy_root) / 'custom_nodes' / 'TimestampStudio' / name
        if source.is_file():
            manifest[name + '_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory(prefix='evidence_', dir=output) as scratch:
        scratch = Path(scratch)
        files = [(cache, 'acoustic_frames.npz')]
        if latest:
            files.append((latest, 'timestamps.json'))
        # Small lossless array chunks allow independent archives below the file
        # transfer limit, including for seven/eight-minute recordings.
        for index, start in enumerate(range(0, len(audio), 120 * 16000)):
            chunk = audio[start:start + 120 * 16000]
            name = f'analysis_audio_{index:03d}.npz'
            path = scratch / name
            np.savez_compressed(path, samples=chunk)
            manifest['audio_chunks'].append({'filename': name, 'sample_start': start,
                                              'sample_count': len(chunk)})
            files.append((path, name))
        reserve = len(json.dumps(manifest).encode()) + 8192
        groups, current, size = [], [], reserve
        for path, name in files:
            needed = path.stat().st_size + len(name.encode()) * 2 + 256
            if needed + reserve > max_archive_bytes:
                raise ValueError('One evidence file exceeds the archive limit; no partial audio was substituted')
            if current and size + needed > max_archive_bytes:
                groups.append(current)
                current, size = [], reserve
            current.append((path, name))
            size += needed
        if current:
            groups.append(current)
        manifest['archive_count'] = len(groups)
        attempt = uuid.uuid4().hex[:10]
        archives = []
        for index, group in enumerate(groups):
            destination = output / f'studio_evidence_{attempt}_{index+1:02d}.zip'
            staged = scratch / destination.name
            with zipfile.ZipFile(staged, 'w', zipfile.ZIP_STORED) as archive:
                archive.writestr('evidence.json', json.dumps(dict(manifest, archive_number=index+1), indent=2))
                for path, name in group:
                    archive.write(path, name)
            if staged.stat().st_size > max_archive_bytes:
                raise ValueError('Evidence archive exceeds the size limit')
            archives.append((staged, destination))
        for staged, destination in archives:
            staged.replace(destination)
    return [destination for _, destination in archives]


if __name__ == '__main__':
    # Leave empty when only one song is in Studio's input folder.
    SONG = ''
    paths = collect(SONG or None)
    print('Evidence collected. No alignment or model inference was run. Original song unchanged.')
    for path in paths:
        print(path)
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(path)))
        except ImportError:
            pass
