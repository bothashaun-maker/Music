# ============================================================================
#  Song Timestamp Studio  —  single-file engine + web app
#  Turns a song + your known lyrics into every timestamp a music video needs:
#  synced lyrics (LRC / SRT / VTT), YouTube chapters, REAPER markers CSV,
#  a beat grid and a beat-snapped Cut Plan.
#
#  Core idea: forced alignment. Whisper listens to the song, then the KNOWN
#  lyric words are fuzzy-matched onto the words Whisper heard, so lyric text
#  is always correct and only timing has to be estimated.
# ============================================================================
import io
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
import zipfile
from difflib import SequenceMatcher

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
APP_TITLE = "Song Timestamp Studio"
VIDEO_FPS = 24  # frame rate used by the Cut Plan (matches MiniMax H3 renders)
DEFAULT_BEATS_PER_CLIP = 4
MIN_LINE_DUR = 0.35  # seconds — a sung line is never shorter than this
MATCH_SIM_THRESHOLD = 0.40  # min word similarity to accept a DP match

LANG_CHOICES = ["auto", "en", "zh", "es", "fr", "de", "pt", "it", "nl", "ru", "ja", "ko"]
MODEL_CHOICES = ["auto", "tiny", "base", "small", "medium"]

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty",
}

EXAMPLE_LYRICS = """[Intro]
[Verse 1]
Walking through the city lights at midnight
Neon shadows falling on the street
I can hear the distant engines humming
Feel the rhythm moving through my feet
[Chorus]
We are the dreamers in the night
Holding on until the morning light"""

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def normalize_word(w):
    """Lowercase, strip accents/punctuation, expand tiny numbers — for matching only."""
    w = unicodedata.normalize("NFKD", str(w))
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    w = w.lower().strip()
    w = re.sub(r"[^a-z0-9' ]", "", w)
    w = w.replace("'", "")
    if w in DIGIT_WORDS:
        w = DIGIT_WORDS[w]
    return w


TAG_RE = re.compile(r"^\[([^\[\]]+)\]\s*[:\-–—]?\s*$")


def parse_lyrics(text):
    """Lyric sheet -> list of entries: {'kind': 'tag'|'sing', 'text', 'words'}."""
    entries = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not any(c.isalnum() for c in line):
            print("[TimestampStudio] Ignoring punctuation-only lyric line: " + repr(line), flush=True)
            continue
        m = TAG_RE.match(line)
        if m:
            entries.append({"kind": "tag", "text": m.group(1).strip(), "words": []})
        else:
            entries.append(
                {"kind": "sing", "text": line, "words": [w for w in line.split() if w]}
            )
    return entries


def fmt_clock(t, comma=False, force_hours=False):
    """MM:SS.mmm (or H:MM:SS.mmm) — integer-ms math so 59.9996 never rounds to '60.000'."""
    total_ms = int(round(max(0.0, float(t)) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    if h or force_hours:
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
    return f"{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_yt(t):
    total_s = int(round(max(0.0, float(t))))
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_lrc_stamp(t):
    """[MM:SS.xx] — centisecond math so 59.996 s becomes [01:00.00], never [00:60.00]."""
    total_cs = int(round(max(0.0, float(t)) * 100))
    m, rem = divmod(total_cs, 6000)
    s, cs = divmod(rem, 100)
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


def fmt_plain(t):
    return fmt_yt(t)


# ----------------------------------------------------------------------------
# device / model handling
# ----------------------------------------------------------------------------

_CUDA_DISABLED_REASON = None
_TRANSCRIPTION_DIAGNOSTICS = {}


def pick_device():
    global _CUDA_DISABLED_REASON
    if _CUDA_DISABLED_REASON:
        return "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            # CTranslate2's CUDA 12 runtime is independent of PyTorch's CUDA 13
            # wheels. A visible T4 does not establish that these libraries load.
            import ctypes
            try:
                ctypes.CDLL("libcublas.so.12")
                ctypes.CDLL("libcudnn.so.9")
            except OSError as exc:
                _CUDA_DISABLED_REASON = str(exc)
                print("[TimestampStudio] Whisper CUDA unavailable; using CPU INT8: " + str(exc), flush=True)
                return "cpu", "int8"
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def resolve_model_size(choice):
    device, _ = pick_device()
    if choice and choice != "auto":
        return choice
    return "small" if device == "cuda" else "base"


_MODEL_CACHE = {}


def get_model(size, device, compute_type):
    key = (size, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        _MODEL_CACHE[key] = WhisperModel(
            size, device=device, compute_type=compute_type, cpu_threads=4, num_workers=1
        )
    return _MODEL_CACHE[key]


# ----------------------------------------------------------------------------
# transcription
# ----------------------------------------------------------------------------


def _resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    divisor = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out)//divisor, int(sr_in)//divisor).astype(np.float32)


def load_audio_16k(path):
    """Decode any common audio format to float32 mono 16 kHz.

    soundfile handles WAV/FLAC/MP3/OGG; plain av.open handles M4A/AAC and
    anything else (no keyword args, so it works on both old and new PyAV).
    """
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = _resample_linear(data, sr, 16000)
        return np.asarray(data, dtype=np.float32)
    except Exception:
        pass
    try:
        import av

        chunks = []
        with av.open(path, mode="r") as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise RuntimeError("no audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(streams[0]):
                out = resampler.resample(frame)
                if out is None:
                    continue
                frames = out if isinstance(out, (list, tuple)) else [out]
                for rf in frames:
                    chunks.append(np.asarray(rf.to_ndarray()).reshape(-1))
        if not chunks:
            raise RuntimeError("no decoded frames")
        return (np.concatenate(chunks).astype(np.float32) / 32768.0).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"could not decode audio {path!r}: {exc}")


def transcribe_words(audio_path, model_size, language):
    """Return (words, segments, duration, detected_language).

    words: [{'word','start','end'}] with word timestamps from faster-whisper.
    """
    global _CUDA_DISABLED_REASON
    _TRANSCRIPTION_DIAGNOSTICS.clear()
    device, compute_type = pick_device()
    size = model_size if model_size and model_size != "auto" else ("small" if device == "cuda" else "base")
    audio = load_audio_16k(audio_path)
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("The decoded recording is empty, silent or nonfinite; check the source audio")
    _TRANSCRIPTION_DIAGNOSTICS.update(
        model=size, device=device, compute_type=compute_type, vad_filter=False,
        cuda_fallback_reason=_CUDA_DISABLED_REASON,
        audio_seconds=float(len(audio) / 16000),
        audio_peak=float(np.max(np.abs(audio))),
        audio_rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
    lang = None if (not language or language == "auto") else language
    def recognize(selected_device, selected_compute):
        print(f"[TimestampStudio] Transcribing music: {size}, {selected_device}/{selected_compute}, speech filter OFF", flush=True)
        model = get_model(size, selected_device, selected_compute)
        segments_iter, info = model.transcribe(
            audio, language=lang, word_timestamps=True,
            condition_on_previous_text=False,
            # Speech-only VAD removed this user's entire sung recording.
            vad_filter=False, no_speech_threshold=None, beam_size=5)
        words, segs = [], []
        _TRANSCRIPTION_DIAGNOSTICS.update(words=words, segments=segs)
        # CUDA failures can occur only when this lazy iterator is consumed.
        for seg in segments_iter:
            segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
            for w in seg.words or []:
                token = (w.word or "").strip()
                if token:
                    words.append({"word": token, "start": float(w.start), "end": float(w.end)})
            _TRANSCRIPTION_DIAGNOSTICS.update(word_count=len(words), segment_count=len(segs))
            print(f"[TimestampStudio] Whisper: {seg.end:.1f}s decoded, {len(words)} words", flush=True)
        return words, segs, info
    try:
        words, segs, info = recognize(device, compute_type)
    except (RuntimeError, OSError) as exc:
        message = str(exc).lower()
        library_failure = any(term in message for term in
                              ("libcublas", "libcudnn", "libcudart", "cublas_status", "cudnn_status"))
        if device != "cuda" or not library_failure:
            raise
        _CUDA_DISABLED_REASON = str(exc)
        failed_model = _MODEL_CACHE.pop((size, device, compute_type), None)
        del failed_model
        print("[TimestampStudio] Whisper CUDA library failure; retrying this transcription once on CPU INT8: " + str(exc), flush=True)
        device, compute_type = "cpu", "int8"
        _TRANSCRIPTION_DIAGNOSTICS.update(device=device, compute_type=compute_type,
                                          cuda_fallback_reason=str(exc))
        words, segs, info = recognize(device, compute_type)
    _TRANSCRIPTION_DIAGNOSTICS.update(words=words, segments=segs,
                                      word_count=len(words), segment_count=len(segs))
    print(f"[TimestampStudio] Heard {len(words)} timestamped words in {len(segs)} segments", flush=True)
    if not segs:
        raise ValueError("Whisper recognized no vocal segments. Alignment stopped; inspect the saved transcription diagnostics before retrying")
    duration = float(getattr(info, "duration", 0.0)) or float(len(audio) / 16000.0)
    detected = getattr(info, "language", None)
    return words, segs, duration, detected, size


# ----------------------------------------------------------------------------
# fuzzy alignment of known lyrics onto heard words
# ----------------------------------------------------------------------------


def word_sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def dp_match(whisper_words, lyric_words):
    """Monotonic alignment: lyric word j -> whisper word index or None.

    Costs: match = 2 - sim (only if sim >= threshold); skip a heard word = 1.0;
    leave a lyric word unmatched = 1.15.
    """
    n, m = len(whisper_words), len(lyric_words)
    if n == 0 or m == 0:
        return [None] * m
    W = [normalize_word(w["word"]) for w in whisper_words]
    L = [normalize_word(w) for w in lyric_words]
    INF = float("inf")
    SKIP_W, DEL_L = 1.0, 1.15
    sim_ok = [[word_sim(W[i], L[j]) for j in range(m)] for i in range(n)]

    # dp[i][j] = best cost of aligning W[i:] with L[j:]
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[n][m] = 0.0
    for i in range(n - 1, -1, -1):
        dp[i][m] = dp[i + 1][m] + SKIP_W
    for j in range(m - 1, -1, -1):
        dp[n][j] = dp[n][j + 1] + DEL_L
    for i in range(n - 1, -1, -1):
        row, nxt = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            best = nxt[j] + SKIP_W  # whisper word not used
            if sim_ok[i][j] >= MATCH_SIM_THRESHOLD:
                c = nxt[j + 1] + (2.0 - sim_ok[i][j])
                if c < best:
                    best = c
            c2 = row[j + 1] + DEL_L  # lyric word left unmatched
            if c2 < best:
                best = c2
            row[j] = best

    mapping = [None] * m
    i = j = 0
    while i < n and j < m:
        choices = []
        if sim_ok[i][j] >= MATCH_SIM_THRESHOLD:
            choices.append((dp[i + 1][j + 1] + (2.0 - sim_ok[i][j]), "match"))
        choices.append((dp[i + 1][j] + SKIP_W, "skip"))
        choices.append((dp[i][j + 1] + DEL_L, "del"))
        choices.sort(key=lambda x: x[0])
        _, choice = choices[0]
        if choice == "match":
            mapping[j] = i
            i += 1
            j += 1
        elif choice == "skip":
            i += 1
        else:
            j += 1
    return mapping


def assign_word_times(whisper_words, sing_entries, duration):
    """Fill start/end for every word of every sing entry.

    Model: matched words carry their Whisper times; unmatched words cling to
    their OWN line's matched words (gaps live between lines, not inside a
    line). Fully unheard lines are spread between the nearest heard anchors.
    """
    lyric_words = []
    line_ranges = []
    for e in sing_entries:
        lo = len(lyric_words)
        e["words"] = [{"text": w, "start": None, "end": None, "matched": False} for w in e["words"]]
        lyric_words.extend(e["words"])
        line_ranges.append((lo, len(lyric_words) - 1))
    mapping = dp_match(whisper_words, [w["text"] for w in lyric_words])

    total = len(lyric_words)
    starts = [None] * total
    ends = [None] * total
    match_count = 0
    for j, i in enumerate(mapping):
        if i is not None:
            starts[j] = whisper_words[i]["start"]
            ends[j] = whisper_words[i]["end"]
            match_count += 1

    chars = [len(w["text"]) for w in lyric_words]
    cum = [0.0]
    for c in chars:
        cum.append(cum[-1] + c + 1)
    matched_js = [j for j in range(total) if starts[j] is not None]

    if not matched_js:
        raise ValueError("No supplied lyric words matched the recording. Alignment stopped without generating estimated timestamps. Check the saved heard transcript and that the lyrics match this audio")
    else:
        # global median char rate as fallback
        rates = []
        for a, b in zip(matched_js, matched_js[1:]):
            if cum[b] > cum[a]:
                rates.append((starts[b] - starts[a]) / (cum[b] - cum[a]))
        g_rate = min(0.25, max(0.04, sorted(rates)[len(rates) // 2])) if rates else 0.09

        def line_rate(mjs):
            if len(mjs) >= 2:
                r = (starts[mjs[-1]] - starts[mjs[0]]) / max(cum[mjs[-1]] - cum[mjs[0]], 1e-6)
                return min(0.30, max(0.03, r))
            return g_rate

        for lo, hi in line_ranges:
            mjs = [j for j in range(lo, hi + 1) if starts[j] is not None]
            if not mjs:
                # whole line unheard: spread between nearest anchors outside it
                prev = [j for j in matched_js if j < lo]
                nxt = [j for j in matched_js if j > hi]
                if prev and nxt:
                    t0, t1 = ends[prev[-1]], starts[nxt[0]]
                    span = max(0.0, t1 - t0)
                    c0, c1 = cum[lo], cum[hi + 1]
                    for j in range(lo, hi + 1):
                        frac = (cum[j] - c0) / max(c1 - c0, 1e-6)
                        starts[j] = t0 + span * frac
                        ends[j] = starts[j] + chars[j] * g_rate
                elif nxt:
                    t = max(0.02, starts[nxt[0]] - (cum[nxt[0]] - cum[lo]) * g_rate)
                    for j in range(lo, hi + 1):
                        starts[j] = t
                        ends[j] = t + chars[j] * g_rate
                        t = ends[j] + 0.03
                else:
                    t = ends[prev[-1]] + 0.05
                    for j in range(lo, hi + 1):
                        starts[j] = t
                        ends[j] = t + chars[j] * g_rate
                        t = ends[j] + 0.03
                continue
            r = line_rate(mjs)
            first_m, last_m = mjs[0], mjs[-1]
            # leading unmatched words cling backwards to the first heard word
            for j in range(first_m - 1, lo - 1, -1):
                end_j = starts[j + 1] - 0.04
                start_j = end_j - chars[j] * r
                starts[j], ends[j] = start_j, max(start_j + 0.1, end_j)
            # middle unmatched: char-weighted between same-line anchors
            for a, b in zip(mjs, mjs[1:]):
                if b - a <= 1:
                    continue
                base, span = ends[a], max(0.0, starts[b] - ends[a])
                c0 = cum[a] + chars[a]
                for j in range(a + 1, b):
                    frac = (cum[j] + chars[j] / 2.0 - c0) / max(cum[b] - c0, 1e-6)
                    frac = min(1.0, max(0.0, frac))
                    center = base + span * frac
                    half = chars[j] * r / 2.0
                    starts[j], ends[j] = center - half, center + half
            # trailing unmatched words cling forwards from the last heard word
            prev_end = ends[last_m]
            nxt_anchor = [j for j in matched_js if j > hi]
            cap = starts[nxt_anchor[0]] - 0.05 if nxt_anchor else duration
            for j in range(last_m + 1, hi + 1):
                s = prev_end + 0.04
                e = s + chars[j] * r
                if e > cap:
                    e = cap
                    s = e - chars[j] * r
                starts[j], ends[j] = s, max(e, s + 0.1)
                prev_end = ends[j]

    # sanity: finite + monotonic
    for j in range(total):
        if not np.isfinite(starts[j]):
            starts[j] = 0.5
        if not np.isfinite(ends[j]) or ends[j] <= starts[j]:
            ends[j] = starts[j] + 0.3
    for j in range(1, total):
        if starts[j] < starts[j - 1]:
            starts[j] = starts[j - 1]
            ends[j] = max(ends[j], starts[j] + 0.2)

    k = 0
    for e in sing_entries:
        for w in e["words"]:
            w["start"], w["end"], w["matched"] = starts[k], ends[k], mapping[k] is not None
            k += 1
    return match_count, total


def build_line_times(entries, duration, acoustic=False):
    """Line start = first word start, end = last word end (bounded by next line)."""
    sing = [e for e in entries if e["kind"] == "sing"]
    for e in sing:
        if not e["words"]:
            continue
        e["start"] = e["words"][0]["start"]
        e["end"] = e["words"][-1]["end"] if acoustic else max(e["words"][-1]["end"], e["start"] + MIN_LINE_DUR)
    if acoustic:
        for previous, following in zip(sing, sing[1:]):
            if previous["end"] > following["start"] + .001:
                raise ValueError("Acoustic lyric lines overlap or are out of order; export stopped")
    # Legacy line estimates may require clamping; acoustic onsets remain untouched.
    for a in range(0 if acoustic else len(sing) - 1):
        if sing[a + 1]["start"] > sing[a]["start"]:
            sing[a]["end"] = min(sing[a]["end"], sing[a + 1]["start"] - 0.05)
        sing[a]["end"] = max(sing[a]["end"], sing[a]["start"] + 0.3)
    if sing and not acoustic:
        sing[-1]["end"] = min(max(sing[-1]["end"], sing[-1]["start"] + 0.3), max(duration, sing[-1]["end"]))
    # section tags take the time of the next sung line (or previous end)
    last_end = 0.0
    for index, e in enumerate(entries):
        if e["kind"] == "sing":
            last_end = e["end"]
        else:
            nxt = next((s["start"] for s in entries[index+1:] if s["kind"] == "sing"), None)
            e["start"] = nxt if nxt is not None else last_end
            e["end"] = e["start"]


# ----------------------------------------------------------------------------
# beat detection + cut plan
# ----------------------------------------------------------------------------


def detect_beats(audio_16k):
    try:
        import librosa

        tempo, beats = librosa.beat.beat_track(
            y=audio_16k, sr=16000, units="time", trim=False
        )
        bpm = float(np.atleast_1d(tempo)[0])
        beats = [float(b) for b in np.atleast_1d(beats)]
        return round(bpm, 1), beats
    except Exception:
        return None, []


def build_cut_plan(sections, beats, bpm, duration, beats_per_clip):
    """Keep exact section edges; snap interior cuts to beats without gaps."""
    rows = []
    bpm = bpm or 120.0
    grid = sorted(set(float(t) for t in beats if 0 <= t <= duration)) if beats else list(np.arange(0, duration, 60.0/bpm))
    nominal = beats_per_clip*60.0/bpm
    for section in sections:
        start, end = max(0., section["start"]), min(section["end"], duration)
        if end <= start:
            continue
        inside = [t for t in grid if start < t < end]
        boundaries = [start] + inside[beats_per_clip-1::beats_per_clip] + [end]
        if len(boundaries) > 2 and boundaries[-1]-boundaries[-2] < nominal*.6:
            boundaries.pop(-2)
        for a,b in zip(boundaries,boundaries[1:]):
            rows.append({"clip":len(rows)+1, "section":section["name"], "start":a, "end":b,
                         "duration":round(b-a,6),
                         "frames":int(round(b*VIDEO_FPS))-int(round(a*VIDEO_FPS))})
    return rows


def _collect_tags(entries, duration):
    """Tag entries as [(name, time)], with a proper 0:00 intro chapter."""
    tags = [(e["text"], float(e["start"])) for e in entries if e["kind"] == "tag"]
    if tags and "intro" in tags[0][0].strip().lower():
        tags[0] = (tags[0][0], 0.0)
    elif tags and tags[0][1] > 1.0:
        tags.insert(0, ("Intro", 0.0))
    return tags


def build_sections(entries, duration):
    """Sections from tag entries (YouTube chapters drive the spans)."""
    tags = _collect_tags(entries, duration)
    if not tags:
        return [{"name": "Full song", "start": 0.0, "end": duration}]
    sections = []
    for k, (name, t0) in enumerate(tags):
        t_end = tags[k + 1][1] if k + 1 < len(tags) else duration
        sections.append({"name": name.title(), "start": min(t0, duration), "end": max(t_end, min(t0, duration) + 1.0)})
    return sections


def build_chapters(entries, duration):
    tags = _collect_tags(entries, duration)
    if not tags:
        return [], "no [Verse]/[Chorus] tags found — add tags like [Chorus] to your lyrics to get YouTube chapters"
    chapters = []
    for k, (name, t) in enumerate(tags):
        nxt = tags[k + 1][1] if k + 1 < len(tags) else duration
        chapters.append({"name": name.title(), "start": t, "end": nxt})
    notes = []
    if len(chapters) < 3:
        notes.append("YouTube needs at least 3 chapters")
    if chapters and chapters[0]["start"] > 0.5:
        notes.append("first chapter must start at 0:00 for YouTube")
    short = [c["name"] for c in chapters if c["end"] - c["start"] < 10.0]
    if short:
        notes.append("chapters under 10 s: " + ", ".join(short))
    return chapters, ("; ".join(notes) if notes else "valid for YouTube")


# ----------------------------------------------------------------------------
# output formatters (times already offset-corrected when rendered)
# ----------------------------------------------------------------------------


def shifted_entries(entries, offset):
    out = []
    for e in entries:
        c = dict(e)
        c["start"] = e.get("start", 0.0) + offset
        c["end"] = e.get("end", c["start"]) + offset
        c["words"] = [dict(w, start=w["start"] + offset, end=w["end"] + offset) for w in e.get("words", [])]
        out.append(c)
    return out


def fmt_lrc(entries):
    lines = []
    for e in entries:
        if e["kind"] == "tag":
            lines.append(f"{fmt_lrc_stamp(e['start'])}[{e['text']}]")
        else:
            lines.append(f"{fmt_lrc_stamp(e['start'])}{e['text']}")
    return "\n".join(lines)


def fmt_srt(entries):
    sing = [e for e in entries if e["kind"] == "sing"]
    blocks = []
    for i, e in enumerate(sing, 1):
        # SRT requires strict H:MM:SS,mmm — CapCut/Resolve/Premiere reject MM:SS,mmm
        blocks.append(
            f"{i}\n{fmt_clock(e['start'], comma=True, force_hours=True)} --> "
            f"{fmt_clock(e['end'], comma=True, force_hours=True)}\n{e['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def fmt_vtt(entries):
    sing = [e for e in entries if e["kind"] == "sing"]
    body = ["WEBVTT", ""]
    for i, e in enumerate(sing, 1):
        body.append(f"{i}")
        body.append(f"{fmt_clock(e['start'])} --> {fmt_clock(e['end'])}")
        body.append(e["text"])
        body.append("")
    return "\n".join(body)


def fmt_chapters(chapters):
    return "\n".join(f"{fmt_yt(c['start'])} {c['name']}" for c in chapters)


def _csv_field(text):
    """Quote + escape a name for the REAPER marker CSV (doubling inner quotes)."""
    return '"' + str(text).replace('"', '""') + '"'


def fmt_reaper(entries, duration):
    """REAPER marker/region CSV: lyric lines -> markers, section tags -> regions.

    Header columns are matched by name on import (REAPER 7.66+; drag the CSV into
    the Region/Marker Manager). Times are H:MM:SS.mmm, one of the formats REAPER
    parses natively.
    """
    rows = ["#,Name,Start,End,Length,Color"]
    m_id = r_id = 0
    # Use the same section boundaries as chapters/cut plans. A tag's lyric time
    # is the first vocal onset; an opening Intro region covers the audio from 0.
    sec = build_sections(entries, duration) if any(e["kind"] == "tag" for e in entries) else []
    sing = [e for e in entries if e["kind"] == "sing"]
    for e in sing:
        m_id += 1
        rows.append(
            f"M{m_id},{_csv_field(e['text'])},{fmt_clock(e['start'], force_hours=True)},,,"
        )
    for e in sec:
        r_id += 1
        end = e["end"]
        rows.append(
            f"R{r_id},{_csv_field(e['name'])},{fmt_clock(e['start'], force_hours=True)},"
            f"{fmt_clock(end, force_hours=True)},{fmt_clock(end - e['start'], force_hours=True)},"
        )
    return "\n".join(rows)


def fmt_cutplan(rows):
    out = ["clip,section,start,end,duration_seconds,frames_at_24fps"]
    for r in rows:
        out.append(
            f'{r["clip"]},{_csv_field(r["section"])},{fmt_clock(r["start"], force_hours=True)},{fmt_clock(r["end"], force_hours=True)},'
            f'{r["end"]-r["start"]:.6f},{r["frames"]}'
        )
    return "\n".join(out)


def fmt_beats(bpm, beats):
    if not beats:
        return "no beat grid detected for this audio"
    out = [f"approx {bpm} BPM — {len(beats)} beats"]
    out += [f"{fmt_plain(b)}" for b in beats]
    return "\n".join(out)


def fmt_json(result, offset):
    data = {
        "song": result.get("song"),
        "language_detected": result.get("detected_language"),
        "model": result.get("model_used"),
        "device": result.get("device"),
        "status": result.get("status"),
        "duration_seconds": result.get("duration"),
        "transcription": result.get("transcription", {}),
        "alignment_method": result.get("alignment_method"),
        "alignment": result.get("alignment", {}),
        "lyric_word_match_rate": result.get("match_rate"),
        "offset_seconds": offset,
        "bpm": result.get("bpm"),
        "entries": [
            {
                "kind": e["kind"],
                "text": e["text"],
                "start": round(e.get("start", 0.0), 3),
                "end": round(e.get("end", 0.0), 3),
                "words": [
                    {
                        "w": w["text"],
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                        "matched": w.get("matched", False),
                        "timing_source": w.get("timing_source", "recognizer_or_estimate"),
                        "acoustic_score": w.get("acoustic_score"),
                    }
                    for w in e.get("words", [])
                ],
            }
            for e in shifted_entries(result["entries"], offset)
        ],
        "chapters": [
            {"name": c["name"], "start": round(c["start"] + offset, 3)} for c in result.get("chapters", [])
        ],
        "beats": [round(b + offset, 3) for b in result.get("beats", [])],
        "cut_plan": [
            {
                "clip": r["clip"],
                "section": r["section"],
                "start": round(r["start"] + offset, 3),
                "end": round(r["end"] + offset, 3),
                "duration": r["duration"],
                "frames": r["frames"],
            }
            for r in result.get("cut_plan", [])
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# vocal isolation (optional)
# ----------------------------------------------------------------------------


def isolate_vocals(audio_path):
    out_dir = tempfile.mkdtemp(prefix="sts_demucs_")
    import subprocess

    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", out_dir, audio_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for root, _dirs, files in os.walk(out_dir):
        if "vocals.wav" in files:
            return os.path.join(root, "vocals.wav")
    raise RuntimeError((proc.stderr or proc.stdout or "demucs produced no vocals")[-400:])


# ----------------------------------------------------------------------------
# main pipeline
# ----------------------------------------------------------------------------


def validate_acoustic_bundle():
    """Load the adjacent helper independently of ComfyUI's custom module name."""
    import hashlib
    import importlib.util
    from pathlib import Path
    source = Path(__file__).with_name('acoustic.py').resolve()
    if not source.is_file():
        raise RuntimeError('Timestamp Studio installation is incomplete: missing ' + str(source))
    name = '_timestamp_studio_acoustic_' + hashlib.sha256(str(source).encode()).hexdigest()[:20]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module


def acoustic_align_lines(audio, lines, progress=None):
    return validate_acoustic_bundle().align_lines(audio, lines, progress)


def run_alignment(audio_path, lyrics_text, model_size="auto", language="auto", isolate_vocals_flag=False):
    result = {
        "song": os.path.splitext(os.path.basename(audio_path))[0],
        "entries": [],
        "chapters": [],
        "chapter_note": "",
        "beats": [],
        "bpm": None,
        "cut_plan": [],
        "match_rate": None,
        "status": "",
        "audio_16k": None,
        "duration": 0.0,
    }
    if not audio_path:
        result["status"] = "⚠️ **Upload a song first** (step 1)."
        return result

    entries = parse_lyrics(lyrics_text)
    sing = [e for e in entries if e["kind"] == "sing"]
    if (lyrics_text or "").strip() and not sing:
        raise ValueError("The lyric sheet contains no sung lines, only section tags or punctuation. Add the actual sung lines")

    if sing:
        if language not in ("auto", "en", "", None):
            raise ValueError("Supplied-lyric acoustic alignment currently supports English/romanized lyrics. Multilingual automatic transcription remains available with an empty lyric box")
        validate_acoustic_bundle().prepare_lyric_words([e["text"] for e in sing])

    notes = []
    src = audio_path
    if isolate_vocals_flag:
        try:
            src = isolate_vocals(audio_path)
            notes.append("🎛️ vocal isolation applied")
        except Exception as exc:  # graceful fallback
            notes.append(f"⚠️ vocal isolation unavailable ({exc} — continuing with the full mix)")

    if sing:
        audio16 = load_audio_16k(src)
        duration = len(audio16) / 16000
        _TRANSCRIPTION_DIAGNOSTICS.clear()
        _TRANSCRIPTION_DIAGNOSTICS.update(method="ctc_acoustic", text_source="supplied_lyrics")
        def progress(fields):
            print("[TimestampStudio] " + str(fields), flush=True)
        grouped, metadata = acoustic_align_lines(audio16, [e["text"] for e in sing], progress)
        if len(grouped) != len(sing):
            raise ValueError("The acoustic aligner did not return every lyric line; export stopped")
        for entry, word_times in zip(sing, grouped):
            if not word_times or any(not (0 <= w["start"] < w["end"] <= duration + .001) for w in word_times):
                raise ValueError("A lyric line has invalid acoustic timestamps; export stopped")
            entry["words"] = word_times
        build_line_times(entries, duration, acoustic=True)
        _TRANSCRIPTION_DIAGNOSTICS.update(metadata)
        result.update(entries=entries, duration=duration, detected_language=None,
                      model_used=metadata["model"], device=metadata["device"],
                      alignment_method="ctc_acoustic", alignment=metadata, match_rate=None)
        review = len(metadata.get("review", []))
        result["status"] = (
            f"Acoustic alignment computed for {len(sing)} supplied lyric lines, including repeats. "
            f"English model on {metadata['device']}. {review} lines flagged for listening review. "
            "Acoustic scores are model diagnostics, not measured timestamp accuracy."
        )
    else:
        words, segs, duration, detected, size_used = transcribe_words(src, model_size, language)
        entries = []
        for segment in segs:
            text = (segment["text"] or "").strip()
            if not text:
                continue
            heard = [w for w in words if segment["start"]-.001 <= w["start"] < segment["end"]]
            entries.append({"kind":"sing", "text":text, "start":segment["start"],
                            "end":segment["end"], "words":[
                {"text":w["word"], "start":w["start"], "end":w["end"],
                 "matched":None, "timing_source":"whisper_transcription"} for w in heard]})
        if not entries:
            raise ValueError("Whisper returned no usable lyric text; inspect the transcription diagnostics")
        result.update(entries=entries, duration=duration, detected_language=detected,
                      model_used=size_used, device=_TRANSCRIPTION_DIAGNOSTICS.get("device") or pick_device()[0],
                      transcription=dict(_TRANSCRIPTION_DIAGNOSTICS), alignment_method="whisper_transcription")
        result["status"] = (
            f"Auto-transcribed {len(entries)} segments with Whisper ({size_used} on {result['device']}). "
            "Text is what the recognizer heard. For supplied English lyrics, the Studio uses direct acoustic alignment instead."
        )
    if notes:
        result["status"] += "\n" + " · ".join(notes)

    # structure: chapters + sections + beats + cut plan
    chapters, chapter_note = build_chapters(result["entries"], duration)
    result["chapters"], result["chapter_note"] = chapters, chapter_note
    sections = build_sections(result["entries"], duration)
    audio16 = load_audio_16k(audio_path)
    result["audio_16k"] = audio16
    bpm, beats = detect_beats(audio16)
    result["bpm"], result["beats"] = bpm, beats
    result["cut_plan"] = build_cut_plan(sections, beats, bpm, duration, DEFAULT_BEATS_PER_CLIP)
    result["sections"] = sections

    tail = []
    if chapters:
        tail.append(f"📌 chapters: {chapter_note}")
    elif chapter_note:
        tail.append(f"📌 {chapter_note}")
    if bpm:
        tail.append(f"🥁 detected ≈ **{bpm} BPM**, {len(beats)} beats → Cut Plan uses 4-beat clips (~{4 * 60 / bpm:.1f} s)")
    else:
        tail.append("🥁 no beat grid detected — Cut Plan fell back to 2-second slots")
    result["status"] += "\n\n" + " · ".join(tail)
    return result


# ----------------------------------------------------------------------------
# rendering + zip export
# ----------------------------------------------------------------------------


def render_outputs(result, offset=0.0):
    entries = shifted_entries(result["entries"], offset)
    chapters = [{"name": c["name"], "start": c["start"] + offset, "end": c["end"] + offset} for c in result.get("chapters", [])]
    cut_rows = [
        {**r, "start": r["start"] + offset, "end": r["end"] + offset}
        for r in result.get("cut_plan", [])
    ]
    preview = []
    for i, e in enumerate(entries, 1):
        mark = "🏷️" if e["kind"] == "tag" else "🎵"
        words = e.get("words") or []
        if e["kind"] == "tag":
            conf = "— section"
        elif result.get("alignment_method") == "ctc_acoustic":
            score = float(np.mean([w.get("acoustic_score", 0) for w in words])) if words else 0
            conf = f"acoustic score {score:.2f} — audition"
        elif not words:
            conf = "auto"
        else:
            hit = sum(1 for w in words if w.get("matched"))
            pct = int(round(100.0 * hit / len(words)))
            conf = f"{pct}% — solid" if pct >= 85 else (f"{pct}% — ok" if pct >= 60 else f"{pct}% — check")
        preview.append([i, fmt_plain(e["start"]), fmt_plain(e.get("end", e["start"])), f'{mark} {e["text"]}', conf])
    outs = {
        "preview": preview,
        "lrc": fmt_lrc(entries),
        "srt": fmt_srt(entries),
        "vtt": fmt_vtt(entries),
        "chapters": (fmt_chapters(chapters) + ("\n\n_(" + result.get("chapter_note", "") + ")_" if result.get("chapter_note") and chapters else "")),
        "reaper": fmt_reaper(entries, result.get("duration", 0.0) + offset),
        "cutplan": fmt_cutplan(cut_rows),
        "beats": fmt_beats(result.get("bpm"), [b + offset for b in result.get("beats", [])]),
        "json": fmt_json(result, offset),
    }
    return outs


def build_zip(result, offset=0.0):
    if result.get("match_rate") == 0:
        raise ValueError("Export stopped: no lyric words matched the recording")
    outs = render_outputs(result, offset)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", result.get("song") or "song") or "song"
    files = {
        f"{stem}.lrc": outs["lrc"],
        f"{stem}.srt": outs["srt"],
        f"{stem}.vtt": outs["vtt"],
        f"{stem}_youtube_chapters.txt": outs["chapters"],
        f"{stem}_reaper_markers.csv": outs["reaper"],
        f"{stem}_cut_plan.csv": outs["cutplan"],
        f"{stem}_beat_grid.txt": outs["beats"],
        f"{stem}_timestamps.json": outs["json"],
    }
    readme = (
        "Song Timestamp Studio export\n"
        "=============================\n"
        f"Song: {result.get('song')}\n"
        f"Offset applied: {offset:+.2f} s\n\n"
        f"Alignment status: {result.get('status', 'not recorded')}\n\n"
        f"{stem}.lrc                  synced lyrics, line level (LRC players / lyric-video tools)\n"
        f"{stem}.srt                  subtitles for CapCut / DaVinci Resolve / Premiere\n"
        f"{stem}.vtt                  captions for YouTube upload\n"
        f"{stem}_youtube_chapters.txt paste into the video description (first 0:00, 3+, 10s+)\n"
        f"{stem}_reaper_markers.csv  drag onto the REAPER timeline (7.66+); lyrics = markers, sections = regions\n"
        f"{stem}_cut_plan.csv        beat-snapped clip slots with exact seconds + frames @ 24 fps\n"
        f"{stem}_beat_grid.txt       every detected beat time (for manual cut planning)\n"
        f"{stem}_timestamps.json     full data incl. per-word times, for other tools\n"
    )
    if result.get("alignment_method") == "ctc_acoustic":
        review = ["line,start,end,acoustic_score,reason,lyric"]
        for issue in result.get("alignment", {}).get("review", []):
            review.append(f"{issue['line']},{fmt_clock(issue['start']+offset, force_hours=True)},{fmt_clock(issue['end']+offset, force_hours=True)},{issue['acoustic_score']},{_csv_field(issue['reason'])},{_csv_field(issue['text'])}")
        files[f"{stem}_alignment_review.csv"] = "\n".join(review)
        readme += "\nAcoustic scores are model diagnostics, not timing accuracy. Review flagged lines in _alignment_review.csv.\n"
    files["README.txt"] = readme
    descriptor, zip_path = tempfile.mkstemp(prefix=f"{stem}_timestamps_", suffix=".zip")
    os.close(descriptor)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


# ----------------------------------------------------------------------------
# gradio UI
# ----------------------------------------------------------------------------


def _launch_theme():
    """Gradio 6 wants theme= at launch() time; gradio 4/5 want it in Blocks()."""
    try:
        import gradio as gr

        if int(gr.__version__.split(".")[0]) >= 6:
            return {"theme": gr.themes.Soft(primary_hue="violet")}
    except Exception:
        pass
    return {}


CF_VERSION = "2026.10.3"
CF_SHA256 = "053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0"
CF_URL = (f"https://github.com/cloudflare/cloudflared/releases/download/"
          f"{CF_VERSION}/cloudflared-linux-amd64")


def start_cloudflare(port, timeout=90):
    """Optional Cloudflare link (…trycloudflare.com) — same style as the YuE2 / H3 apps.

    Downloads the checksum-pinned cloudflared binary, tunnels the local Gradio
    port and returns the public URL. Any failure raises; the caller falls back
    to the gradio.live share link, which works on its own.
    """
    import hashlib
    import platform
    import shutil
    import subprocess
    import time
    import urllib.request
    from pathlib import Path

    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("needs Linux x86_64 (Kaggle notebooks are fine)")
    root = Path(tempfile.gettempdir()) / "sts_cloudflared"
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "cloudflared"

    def _digest(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    if not binary.is_file() or _digest(binary) != CF_SHA256:
        tmp = root / "cloudflared.download"
        print("  Fetching the Cloudflare helper (~25 MB, one-time per session)…", flush=True)
        with urllib.request.urlopen(CF_URL, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if _digest(tmp) != CF_SHA256:
            raise RuntimeError("cloudflared checksum mismatch")
        tmp.chmod(0o700)
        os.replace(tmp, binary)
    binary.chmod(0o700)

    log = root / "cloudflared.log"
    # Parent-death watcher: cloudflared exits automatically when the notebook
    # kernel dies or restarts (same trick as the YuE2 / H3 notebooks).
    watcher = root / "cf_watch.py"
    watcher.write_text(
        "import ctypes, os, signal, sys\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "libc.prctl(1, signal.SIGTERM, 0, 0, 0)\n"  # PR_SET_PDEATHSIG
        "if os.getppid() != int(sys.argv[1]):\n"
        "    raise SystemExit(1)\n"
        "os.execv(sys.argv[2], [sys.argv[2], 'tunnel', '--no-autoupdate',\n"
        "    '--protocol', 'http2', '--url', 'http://127.0.0.1:' + sys.argv[3]])\n"
    )
    with open(log, "w") as out:
        proc = subprocess.Popen(
            [sys.executable, str(watcher), str(os.getpid()), str(binary), str(int(port))],
            stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("cloudflared exited: " + log.read_text(errors="replace")[-800:])
            text = log.read_text(errors="replace")
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com\b", text)
            if m and "Registered tunnel connection" in text:
                return m.group(0)
            time.sleep(0.5)
        raise TimeoutError(f"cloudflared did not connect in {timeout}s; see {log}")
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
        raise


def build_ui():
    import gradio as gr

    def TB(**kwargs):
        """Textbox with a copy button on gradio 4/5; plain (copy built-in) on 6+."""
        try:
            return gr.Textbox(**kwargs, show_copy_button=True)
        except TypeError:
            return gr.Textbox(**kwargs)


    blocks_kwargs = {"title": APP_TITLE}
    try:
        if int(gr.__version__.split(".")[0]) < 6:
            blocks_kwargs["theme"] = gr.themes.Soft(primary_hue="violet")
    except Exception:
        pass
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            f"# 🎛️ {APP_TITLE}\n"
            "Drop in a song + the lyrics **you already wrote** → get synced lyrics, YouTube chapters, "
            "REAPER markers, a beat grid and a beat-snapped cut plan. The AI pins your known words onto "
            "the exact seconds (forced alignment), so the text is always correct — only timing is measured."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1 · Load your song")
                audio_in = gr.Audio(
                    sources=["upload"], type="filepath", label="Song — WAV / MP3 / FLAC / M4A (e.g. the WAV from your YuE2 run ZIP)"
                )
                with gr.Row():
                    model_dd = gr.Dropdown(MODEL_CHOICES, value="auto", label="Whisper model (auto = small on GPU, base on CPU)")
                    lang_dd = gr.Dropdown(LANG_CHOICES, value="en", label="Language")
                isolate_cb = gr.Checkbox(
                    False, label="Vocal isolation (demucs) — slower, helps on dense mixes"
                )
            with gr.Column():
                gr.Markdown("### 2 · Paste your lyrics")
                lyrics_tb = gr.Textbox(
                    lines=12,
                    value="",
                    placeholder=EXAMPLE_LYRICS,
                    label="One line per row · [Verse]/[Chorus] tags become chapters & REAPER regions · leave empty to auto-transcribe",
                )
        run_btn = gr.Button("⚡ Get timestamps", variant="primary", size="lg")
        status_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Preview"):
                preview_df = gr.Dataframe(
                    headers=["#", "start", "end", "entry", "confidence"], interactive=False, wrap=True
                )
            with gr.Tab("LRC"):
                lrc_tb = TB(lines=14, interactive=False, label="LRC (lyric-video tools / players)")
            with gr.Tab("SRT"):
                srt_tb = TB(lines=14, interactive=False, label="SRT (CapCut / Resolve / Premiere)")
            with gr.Tab("WebVTT"):
                vtt_tb = TB(lines=14, interactive=False, label="VTT (YouTube captions)")
            with gr.Tab("YouTube chapters"):
                chap_tb = TB(lines=8, interactive=False, label="Paste into the video description")
            with gr.Tab("REAPER CSV"):
                reaper_tb = TB(lines=14, interactive=False, label="Drag this file onto the REAPER timeline")
            with gr.Tab("Cut plan"):
                cut_tb = TB(lines=14, interactive=False, label="Beat-snapped clip slots (seconds + frames @ 24 fps)")
            with gr.Tab("Beat grid"):
                beats_tb = TB(lines=14, interactive=False, label="Every detected beat")
            with gr.Tab("JSON"):
                json_tb = TB(lines=14, interactive=False, label="Full data for other tools")

        with gr.Row():
            offset_num = gr.Number(
                value=0.0, step=0.05, minimum=-60, maximum=60,
                label="Nudge — shift all timestamps (seconds; + later / − earlier)",
            )
            apply_btn = gr.Button("🔁 Apply nudge / refresh downloads", variant="secondary")
        with gr.Row():
            line_pick = gr.Number(value=1, precision=0, minimum=1, label="Audition line #")
            audition_btn = gr.Button("▶️ Audition line", variant="secondary")
            audition_audio = gr.Audio(label="Plays ~1.5 s before the line, for 8 s")

        zip_file = gr.File(label="📦 Download all formats (.zip)")

        gr.Markdown(
            "### Where each file goes\n"
            "| File | Use it in |\n|---|---|\n"
            "| `.srt` | CapCut Desktop / DaVinci Resolve / Premiere — captions on the timeline |\n"
            "| `.lrc` | lyric-video makers and players with synced lyrics |\n"
            "| `.vtt` | YouTube caption upload |\n"
            "| `_youtube_chapters.txt` | video description — clickable chapters |\n"
            "| `_reaper_markers.csv` | drag onto REAPER timeline to audit timings against the waveform |\n"
            "| `_cut_plan.csv` | how many H3 clips to queue, and how long each must be |\n\n"
            "*First run downloads the Whisper model (~150 MB) — that's normal. Timings are line-level; "
            "audit in REAPER or with ▶️ Audition, then nudge globally if needed.*"
        )

        OUTS = [status_md, preview_df, lrc_tb, srt_tb, vtt_tb, chap_tb, reaper_tb, cut_tb, beats_tb, json_tb, zip_file]

        def _do_run(audio_path, lyrics, model_size, language, isolate):
            result = run_alignment(audio_path, lyrics, model_size, language, isolate)
            global _LAST
            _LAST = result
            outs = render_outputs(result, 0.0)
            zip_path = build_zip(result, 0.0) if result.get("entries") else None
            return [result["status"], outs["preview"], outs["lrc"], outs["srt"], outs["vtt"],
                    outs["chapters"], outs["reaper"], outs["cutplan"], outs["beats"], outs["json"], zip_path]

        def _do_apply(offset):
            global _LAST
            if not _LAST or not _LAST.get("entries"):
                return ["⚠️ Run the app on a song first."] + [gr.update() for _ in range(10)]
            outs = render_outputs(_LAST, float(offset or 0.0))
            zip_path = build_zip(_LAST, float(offset or 0.0))
            return [_LAST["status"], outs["preview"], outs["lrc"], outs["srt"], outs["vtt"],
                    outs["chapters"], outs["reaper"], outs["cutplan"], outs["beats"], outs["json"], zip_path]

        def _do_audition(line_no, offset):
            global _LAST
            audio = _LAST.get("audio_16k") if _LAST else None
            if audio is None or len(audio) == 0:
                return None
            sing = [e for e in _LAST["entries"] if e["kind"] == "sing"]
            idx = int(max(1, min(line_no or 1, len(sing)))) - 1
            t0 = max(0.0, sing[idx]["start"] + float(offset or 0.0) - 1.5)
            seg = _LAST["audio_16k"][int(t0 * 16000): int((t0 + 8.0) * 16000)]
            path = os.path.join(tempfile.gettempdir(), f"sts_audition_{idx + 1}.wav")
            import soundfile as sf

            sf.write(path, seg, 16000)
            return path

        run_btn.click(_do_run, [audio_in, lyrics_tb, model_dd, lang_dd, isolate_cb], OUTS)
        apply_btn.click(_do_apply, [offset_num], OUTS)
        audition_btn.click(_do_audition, [line_pick, offset_num], audition_audio)
    return demo


_LAST = None

if os.environ.get("STS_AUTOLAUNCH", "1") == "1":
    demo = build_ui()
    print("\n" + "=" * 62)
    print("  SONG TIMESTAMP STUDIO is starting — watch for the links below.")
    print("=" * 62 + "\n")
    # Launch in the background so the optional Cloudflare twin link can be
    # opened too; block_thread() keeps the notebook cell alive afterwards.
    demo.launch(share=True, show_error=True, prevent_thread_lock=True,
                **_launch_theme())
    try:
        port = int(getattr(demo, "server_port", None) or 7860)
    except Exception:
        port = 7860
    cf_url = None
    try:
        cf_url = start_cloudflare(port)
    except Exception as exc:  # Cloudflare is a bonus, never a blocker
        print(f"  (Optional Cloudflare link unavailable: {exc})")
        print("   The app still works through the gradio.live link above.")
    share_url = getattr(demo, "share_url", None)
    print("\n" + "-" * 62)
    print("  🎛️  Open the app with EITHER link:")
    if cf_url:
        print(f"  ☁️  Cloudflare  :  {cf_url}")
    else:
        print("  ☁️  Cloudflare  :  (unavailable this session — use the Gradio link)")
    if share_url:
        print(f"  🟣 Gradio backup:  {share_url}")
    print("-" * 62 + "\n")
    demo.block_thread()
