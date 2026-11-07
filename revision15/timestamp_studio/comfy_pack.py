"""TimestampStudio — ComfyUI node pack for Song Timestamp Studio.

A visible, connected canvas of real, individually executed nodes:

    STS Load Song ──▶ STS Align ◀── STS Lyrics & Options
                     │
        ┌────────────┼──────────────────┐
        ▼            ▼                  ▼
   STS Nudge   STS Audition ──AUDIO──▶ PreviewAudio
        │
        ▼
   STS Export All  (LRC / SRT / VTT / chapters / REAPER / beats / Cut Plan / JSON / ZIP)

The installed engine aligns supplied English lyrics with acoustic CTC frames,
vocal gap context, local attacks and repeated-phrase audio matching. Empty lyrics use
Whisper transcription. Beat grid and Cut Plan come from librosa; the Cut Plan
speaks seconds AND frames @ 24 fps. Scores do not establish timing accuracy.
"""
import os

os.environ.setdefault("STS_AUTOLAUNCH", "0")  # never auto-start the Gradio route here

import asyncio
import json
import math
import uuid
import hashlib
import re
import shutil
import tempfile
import threading
import urllib.parse
import zipfile

import numpy as np

from . import sts_engine as engine
engine.validate_acoustic_bundle()  # Fail registration before printing a usable canvas link.
engine._adjacent_helper('vocal.py')

CATEGORY = "Song Timestamp Studio"
_PREF = "[TimestampStudio] "

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
              ".webm", ".aiff", ".aif", ".wma"}
MAX_SONG_BYTES = 2 * 1024 ** 3  # 2 GB — a whole album is fine
NO_SONG = "(no songs yet — open the 📤 upload link from Kaggle)"


def _safe_stem(name):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "song")) or "song"
    return stem[:80]


def _audio_to_temp_wav(audio):
    """ComfyUI AUDIO dict -> mono WAV file path the engine can read."""
    import soundfile as sf

    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 44100))
    data = waveform.detach().cpu().numpy()
    if data.ndim == 3:  # [batch, channels, samples]
        data = data[0]
    if data.ndim == 2 and data.shape[0] > 1:  # stereo -> mono
        data = data.mean(axis=0)
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    descriptor, path = tempfile.mkstemp(prefix="sts_input_", suffix=".wav")
    os.close(descriptor)
    sf.write(path, data, sample_rate, subtype="FLOAT")
    return path


def _result_summary(result):
    lines = len([e for e in result.get("entries", []) if e.get("kind") == "sing"])
    rate = result.get("match_rate")
    if result.get("alignment_method") == "ctc_acoustic":
        review = len(result.get("alignment", {}).get("review", []))
        return f"Acoustic alignment: {lines} lyric lines · {review} lines to review · {result.get('duration',0):.1f}s. Listen to verify sync."
    rate_text = f"{int(round(100 * rate))}% of lyric words located" if rate is not None else "auto-transcribed"
    bpm = result.get("bpm")
    bpm_text = f", ~{bpm:.1f} BPM" if bpm else ""
    indicator = "✅" if rate is None or rate >= 0.75 else "⚠️"
    return (f"{indicator} {lines} sung lines · {rate_text} · "
            f"{(result.get('duration') or 0):.1f}s{bpm_text}")


def _input_dir():
    try:
        import folder_paths

        d = folder_paths.get_input_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "input")
    os.makedirs(d, exist_ok=True)
    return d


def _temp_dir():
    try:
        import folder_paths

        d = folder_paths.get_temp_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "temp")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_song_name(name):
    name = os.path.basename(str(name or "")).strip()
    name = re.sub(r"[^A-Za-z0-9 _.\-()\[\]]+", "_", name)
    return (name or "song.wav")[:120]


def _is_audio_name(name):
    return os.path.splitext(name)[1].lower() in AUDIO_EXTS


def _unique_path(dest):
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(dest)
    i = 2
    while os.path.exists(f"{base} ({i}){ext}"):
        i += 1
    return f"{base} ({i}){ext}"


def _list_input_songs():
    d = _input_dir()
    try:
        return sorted(f for f in os.listdir(d)
                      if _is_audio_name(f) and not f.startswith("."))
    except Exception:
        return []


def _scan_roots():
    roots = [r for r in os.environ.get(
        "STS_SCAN_ROOTS", "/kaggle/working:/kaggle/input").split(":") if r]
    out_root = os.environ.get("STS_OUTPUT_ROOT")
    if out_root:
        roots.append(out_root)
    return [r for r in roots if os.path.isdir(r)]


def _load_audio_file(path):
    """(waveform [channels, samples] float32, sample_rate) — mirrors LoadAudio."""
    last = None
    try:
        from comfy_extras.nodes_audio import load as _comfy_load

        return _comfy_load(path)
    except Exception as exc:
        last = exc
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        return wav, int(sr)
    except Exception as exc:
        last = exc
    try:
        import soundfile as sf
        import torch

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), int(sr)
    except Exception as exc:
        last = exc
    raise RuntimeError(f"Could not read '{os.path.basename(path)}': {last}")


class STSLoadSong:
    """Your song — pick it from the list, or add it with the 📤 uploader link.

    The old single-shot upload button could die halfway ("Fetch timeout") when a
    big song crossed a slow tunnel. Songs now arrive through the chunked uploader
    (small pieces, auto-retry, progress bar) or straight from a link — both land
    in ComfyUI's input folder and appear in this list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_songs()
        return {"required": {"audio": (files or [NO_SONG],)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Pick your song from the list. To add a song use the 📤 upload link "
                   "printed by the setup cell (chunked, big-file safe, progress bar) or "
                   "the link box on that page.")

    @classmethod
    def hidden_validate(cls, audio):
        base = os.path.realpath(_input_dir())
        p = os.path.realpath(os.path.join(base, str(audio)))
        if os.path.commonpath([base, p]) == base and os.path.isfile(p):
            return True, p
        files = _list_input_songs()
        if str(audio) == NO_SONG and len(files) == 1:  # Only auto-select from the placeholder.
            return True, os.path.join(_input_dir(), files[0])
        return False, (f"'{audio}' is not on the studio server yet. Open the 📤 upload link "
                       "printed by the setup cell, add the song there, then pick it from "
                       "this list (reload the canvas page if the list looks old).")

    @classmethod
    def validate_inputs(cls, audio):
        ok, info = cls.hidden_validate(audio)
        return True if ok else info

    @classmethod
    def fingerprint_inputs(cls, audio):
        ok, p = cls.hidden_validate(audio)
        if not ok:
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    @classmethod
    def IS_CHANGED(cls, audio):
        return cls.fingerprint_inputs(audio)

    def execute(self, audio):
        ok, path = self.hidden_validate(audio)
        if not ok:
            raise ValueError(str(path))
        waveform, sample_rate = _load_audio_file(path)
        print(_PREF + f"loaded '{os.path.basename(path)}' "
              f"({waveform.shape[-1] / max(sample_rate, 1):.1f}s @ {sample_rate} Hz)", flush=True)
        import torch

        return {"ui": {}, "result": ({"waveform": waveform.unsqueeze(0),
                                      "sample_rate": int(sample_rate)},)}


class STSLyricsOptions:
    """Lyrics you already wrote + engine choices. Empty lyrics = auto-transcribe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "song_name": ("STRING", {"default": "my_song"}),
            "lyrics": ("STRING", {"default": engine.EXAMPLE_LYRICS, "multiline": True,
                                  "placeholder": "[Verse]\n...one line per row...\n\n[Chorus]\n..."}),
            "model": (engine.MODEL_CHOICES, {"default": "auto"}),
            "language": (engine.LANG_CHOICES, {"default": "auto"}),
            "isolate_vocals": ("BOOLEAN", {"default": True,
                                           "label_on": "on — recommended for music",
                                           "label_off": "off — clear mixes"}),
        }}

    RETURN_TYPES = ("STS_OPTS",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Paste English lyrics exactly as sung, including every repeat, one line per row. "
                   "Supplied lyrics use acoustic CTC alignment. The model menu selects Whisper only "
                   "when lyrics are empty. Section tags become regions; an opening Intro starts at zero. "
                   "Empty lyrics enable multilingual automatic transcription.")

    def execute(self, song_name, lyrics, model, language, isolate_vocals):
        return ({"song_name": (song_name or "song").strip() or "song",
                 "lyrics": (lyrics or "").strip(),
                 "model": model, "language": language,
                 "isolate_vocals": bool(isolate_vocals)},)


class STSAlign:
    """Forced alignment: pins your known lyric lines onto exact seconds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "opts": ("STS_OPTS",)}}

    RETURN_TYPES = ("STS_RESULT",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Aligns supplied English lyrics to acoustic frames. With isolation ON, "
                   "uses vocal context, local attacks and repeated-phrase audio matching; "
                   "models, vocals and frames are cached. "
                   "Weak or stretched words need review. The Whisper menu affects empty-lyric "
                   "transcription only; a bigger Whisper setting does not change CTC alignment.")

    def execute(self, audio, opts):
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
        path = _audio_to_temp_wav(audio)
        try:
            import folder_paths
            output_root = folder_paths.get_output_directory()
        except Exception:
            output_root = os.environ.get("STS_OUTPUT_ROOT") or os.path.join(os.getcwd(), "output")
        diagnostics_root = os.path.join(output_root, "diagnostics")
        os.makedirs(diagnostics_root, exist_ok=True)
        diagnostic_dir = tempfile.mkdtemp(prefix=_safe_stem(opts["song_name"]) + "_", dir=diagnostics_root)
        diagnostic_path = os.path.join(diagnostic_dir, "alignment.json")
        details = {"song": opts["song_name"], "options": dict(opts), "status": "running"}
        engine._TRANSCRIPTION_DIAGNOSTICS.clear()
        print(_PREF + "Alignment diagnostics: " + diagnostic_path, flush=True)
        try:
            result = engine.run_alignment(
                path, opts["lyrics"], model_size=opts["model"], language=opts["language"],
                isolate_vocals_flag=opts["isolate_vocals"])
            if not result.get("entries"):
                raise ValueError(result.get("status") or "No aligned lyric lines were produced")
            details.update(status="finished", match_rate=result.get("match_rate"),
                           alignment_status=result.get("status"))
        except Exception as exc:
            details.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            details["transcription"] = dict(engine._TRANSCRIPTION_DIAGNOSTICS)
            with open(diagnostic_path, "w") as target:
                json.dump(details, target, indent=2, ensure_ascii=False)
            os.remove(path)
        result["song"] = opts["song_name"]
        result["nudge"] = 0.0
        print(_PREF + _result_summary(result), flush=True)
        return {"ui": {"text": [_result_summary(result), result.get("status", ""),
                                  "Diagnostics: " + diagnostic_path]}, "result": (result,)}


class STSNudge:
    """Shift every timestamp at once (early/late by a constant)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("STS_RESULT",),
            "offset_seconds": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0,
                                         "step": 0.05}),
        }}

    RETURN_TYPES = ("STS_RESULT",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("If everything lands a constant amount early or late (long intro, slight "
                   "latency), nudge all timestamps at once. Audition and Export apply it too.")

    def execute(self, result, offset_seconds):
        shifted = dict(result)
        shifted["nudge"] = float(result.get("nudge", 0.0)) + float(offset_seconds or 0.0)
        print(_PREF + f"nudge now {shifted['nudge']:+.2f}s", flush=True)
        return (shifted,)


class STSAudition:
    """Play ~8 s of the song around one lyric line for a quick ear check."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("STS_RESULT",),
            "line_number": ("INT", {"default": 1, "min": 1, "max": 999}),
            "lead_in_seconds": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Wire into PreviewAudio. Plays from a moment before the chosen line so you "
                   "hear whether the vocal enters when the line fires. Nudged times apply.")

    def execute(self, result, line_number, lead_in_seconds):
        sing = [e for e in result.get("entries", []) if e.get("kind") == "sing"]
        if not sing:
            raise ValueError("No sung lines to audition — add lyrics or auto-transcribe first.")
        idx = int(max(1, min(int(line_number), len(sing)))) - 1
        t0 = max(0.0, sing[idx]["start"] + float(result.get("nudge", 0.0)) - float(lead_in_seconds))
        segment = np.asarray(result["audio_16k"][int(t0 * 16000): int((t0 + 8.0) * 16000)],
                             dtype=np.float32)
        import torch

        audio = {"waveform": torch.from_numpy(segment[None, None, :]),
                 "sample_rate": 16000}
        print(_PREF + f"audition line {idx + 1}/{len(sing)} from {t0:.2f}s", flush=True)
        return (audio,)


class STSExportAll:
    """Write every timestamp format into the ComfyUI output folder + one ZIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("STS_RESULT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Writes synced lyrics (LRC/SRT/VTT), YouTube chapters, REAPER markers CSV, "
                   "beat grid, beat-snapped Cut Plan (seconds + frames @ 24 fps), full JSON "
                   "and a README — plus one ZIP with everything, into the output folder.")

    def execute(self, result):
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
        if result.get("match_rate") == 0:
            raise ValueError("Export stopped: no lyric words matched the recording. These timestamps are not an alignment")
        try:
            import folder_paths

            out_root = folder_paths.get_output_directory()
        except Exception:
            out_root = os.environ.get("STS_OUTPUT_ROOT") or os.path.join(os.getcwd(), "output")
        stem = _safe_stem(result.get("song") or "song")
        nudge = float(result.get("nudge", 0.0))
        os.makedirs(out_root, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix=f"{stem}_timestamps_", dir=out_root)

        zip_path = engine.build_zip(result, nudge)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            zf.extractall(out_dir)
        dest_zip = os.path.join(out_dir, os.path.basename(zip_path))
        if os.path.abspath(dest_zip) != os.path.abspath(zip_path):
            shutil.copyfile(zip_path, dest_zip)

        report = [f"✅ {len(names)} files → {out_dir}"]
        for name in names:
            report.append("   · " + name)
        report.append(f"   · {os.path.basename(dest_zip)} (everything in one ZIP)")
        if abs(nudge) >= 1e-6:
            report.append(f"nudge applied: {nudge:+.2f}s")
        preview = engine.render_outputs(result, nudge)["preview"]
        report.append("line check: " + " | ".join(
            f"{row[0]}:{row[4]}" for row in preview[:8]))
        report.append("REAPER audit: drag the _reaper_markers.csv into the Region/Marker "
                      "Manager (View menu) over your waveform.")
        text = "\n".join(report)
        print(_PREF + text, flush=True)
        return {"ui": {"text": [text], "sts_download": [{
                    "filename": os.path.basename(dest_zip),
                    "subfolder": os.path.relpath(out_dir, out_root), "type": "output"}]}, "result": ()}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    STSLoadSong, STSLyricsOptions, STSAlign, STSNudge, STSAudition, STSExportAll)}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STSLoadSong": "Your Song (upload / pick / link)",
    "STSLyricsOptions": "Lyrics & Options",
    "STSAlign": "Align Lyrics (forced alignment)",
    "STSNudge": "Nudge All Timestamps",
    "STSAudition": "Audition One Line",
    "STSExportAll": "Export All Formats + ZIP",
}

WEB_DIRECTORY = "./web"

# ---------------------------------------------------------------------------
# Big-file-safe song delivery: chunked upload, link download, Kaggle import.
# Songs land in ComfyUI's input folder and appear in the STSLoadSong list.
# ---------------------------------------------------------------------------

_UPLOADER_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>📤 Add a song — Song Timestamp Studio</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#171717;color:#e6e6e6;font:16px/1.55 system-ui,'Segoe UI',Roboto,sans-serif;padding:24px}
main{max-width:760px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:26px 0 8px;color:#9ad}
.sub{color:#aaa;margin:0 0 18px}
.card{background:#222;border:1px solid #383838;border-radius:12px;padding:18px;margin:14px 0}
#drop{border:2px dashed #5a7;border-radius:12px;padding:26px;text-align:center;cursor:pointer;transition:background .15s}
#drop.hover{background:#1c2a20}
button{background:#2f6f4f;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:16px;cursor:pointer}
button:hover{background:#3a8a62}
button:disabled{opacity:.5;cursor:default}
#bar-wrap{background:#111;border-radius:8px;height:22px;overflow:hidden;margin:14px 0 6px;display:none}
#bar{height:100%;width:0%;background:linear-gradient(90deg,#3a8a62,#7fd4a8);transition:width .2s}
#bar.fail{background:#c0392b}
#msg{min-height:24px;color:#bbb;white-space:pre-wrap}
ol{margin:8px 0 0 20px;padding:0}
li{margin:6px 0}
.urlrow{display:flex;gap:8px}
input[type=text]{flex:1;background:#111;border:1px solid #444;color:#eee;border-radius:8px;padding:10px;font-size:15px}
ul{list-style:none;margin:8px 0 0;padding:0}
li.song{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 10px;background:#1c1c1c;border-radius:8px;margin:6px 0;flex-wrap:wrap}
li.song b{word-break:break-all}
.tag{font-size:12px;padding:2px 8px;border-radius:99px;background:#2f6f4f}
.tag.src{background:#3a5a7f}
.mini{font-size:13px;color:#888}
.err{color:#ff9c93}
.ok{color:#8fe3b0}
</style></head><body><main>
<h1>📤 Add a song to the studio</h1>
<p class="sub">Songs travel in small chunks with automatic retries, so big files work even on a
slow or shaky connection. <b>Keep this tab open until it says 100%.</b></p>

<div class="card">
<ol>
<li>Add the song here (file, or a link).</li>
<li>Go back to the <b>canvas tab</b> — the song appears in the <b>1 · Your song</b> node list by itself.</li>
<li>Pick it and press <b>Run</b>.</li>
</ol>
</div>

<div class="card">
<div id="drop">📄 <b>Drop a song here</b> or click to choose<br>
<span class="mini">WAV · FLAC · MP3 · OGG · M4A — up to 2 GB</span>
<input type="file" id="file" accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,.aiff,.aif,.wma,audio/*" hidden>
</div>
<div id="bar-wrap"><div id="bar"></div></div>
<div id="msg">Ready.</div>
</div>

<div class="card">
<h2 style="margin-top:0">…or grab it from a link</h2>
<div class="urlrow">
<input type="text" id="url" placeholder="https://…/my-song.mp3" spellcheck="false">
<button id="fetchbtn">⬇️ Download</button>
</div>
<p class="mini">The studio server downloads it at full speed — nothing travels through your own connection.</p>
</div>

<div class="card">
<h2 style="margin-top:0">Songs already on the studio server</h2>
<p class="mini">Green = ready to pick on the canvas. Blue = found in your Kaggle files — press Import once.</p>
<ul id="songs"></ul>
</div>

<script>
const CHUNK = 8 * 1024 * 1024;            // 8 MB per piece
const CHUNK_TIMEOUT_MS = 300000;          // 5 min per piece
const RETRIES = 5;
let uploadId = '';
const esc = value => { const e = document.createElement('span'); e.textContent = String(value); return e.innerHTML.replaceAll('"', '&quot;').replaceAll("'", '&#39;'); };
const drop = document.getElementById('drop'), fileEl = document.getElementById('file');
const barWrap = document.getElementById('bar-wrap'), bar = document.getElementById('bar');
const msg = document.getElementById('msg'), songsEl = document.getElementById('songs');
let busy = false;

function setBar(pct, fail){ barWrap.style.display='block'; bar.style.width=(pct*100).toFixed(1)+'%';
  bar.classList.toggle('fail', !!fail); }
function say(text, cls){ msg.innerHTML = text; msg.className = cls || ''; }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function upChunk(i, n, name, size, blob){
  for (let t = 1; ; t++){
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort('timeout'), CHUNK_TIMEOUT_MS);
    try {
      const q = `i=${i}&n=${n}&name=${encodeURIComponent(name)}&size=${size}&upload=${uploadId}`;
      const r = await fetch('/sts/upload/chunk?' + q, {method:'POST', body: blob,
        headers:{'Content-Type':'application/octet-stream'}, signal: ctl.signal});
      clearTimeout(timer);
      if (!r.ok){ const e = await r.json().catch(() => ({})); throw new Error(e.error || ('HTTP ' + r.status)); }
      return;
    } catch (err) {
      clearTimeout(timer);
      if (t > RETRIES) throw err;
      say(`piece ${i+1} of ${n} wobbled (${esc(err.message)}) — retry ${t}/${RETRIES}…`);
      await sleep(1200 * t);
    }
  }
}

async function uploadFile(file){
  if (busy) return; busy = true;
  bar.classList.remove('fail'); setBar(0); say(`Uploading <b>${esc(file.name)}</b> (${(file.size/1048576).toFixed(1)} MB)…`);
  try {
    if (!file.size || file.size > 2 * 1024 ** 3) throw new Error('Choose a nonempty file under 2 GB');
    const key = `sts-upload:${file.name}:${file.size}:${file.lastModified}`;
    uploadId = sessionStorage.getItem(key) || crypto.randomUUID().replaceAll('-', '');
    sessionStorage.setItem(key, uploadId);
    const n = Math.ceil(file.size / CHUNK);
    const status = await fetch(`/sts/upload/status?upload=${uploadId}&name=${encodeURIComponent(file.name)}&n=${n}&size=${file.size}`);
    if (!status.ok) throw new Error('Cannot check saved upload pieces');
    const completed = new Set((await status.json()).completed || []);
    let done = 0;
    for (let i = 0; i < n; i++){
      const blob = file.slice(i * CHUNK, Math.min(file.size, (i + 1) * CHUNK));
      if (!completed.has(i)) await upChunk(i, n, file.name, file.size, blob);
      done += blob.size; setBar(done / file.size);
      say(`Uploading <b>${esc(file.name)}</b>… ${Math.round(100 * done / file.size)}% (piece ${i+1}/${n})`);
    }
    const r = await fetch('/sts/upload/finish', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: file.name, n, size: file.size, upload: uploadId})});
    const fin = await r.json();
    if (!fin.ok) throw new Error(fin.error || ('HTTP ' + r.status));
    setBar(1); say(`✅ <b>${esc(fin.filename)}</b> is on the studio server. Go to the canvas tab and pick it in the <b>1 · Your song</b> node.`, 'ok');
    sessionStorage.removeItem(key);
    announce(fin.filename); await loadSongs();
  } catch (err) {
    setBar(1, true);
    say(`❌ Upload failed: ${esc(err.message)}<br>Check your internet, then choose the file again — finished pieces are kept, so it continues where it stopped.`, 'err');
  } finally { busy = false; }
}

async function fetchUrl(){
  const url = document.getElementById('url').value.trim();
  if (!url || busy) return; busy = true;
  setBar(0.02); say('Server is downloading from the link…');
  try {
    const r = await fetch('/sts/fetch_url/start', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})});
    const started = await r.json();
    if (!r.ok || !started.job) throw new Error(started.error || 'Could not start download');
    let j;
    while (true) {
      await sleep(1000);
      const status = await fetch('/sts/fetch_url/status?job=' + started.job, {cache:'no-store'});
      j = await status.json();
      if (j.state === 'done') break;
      if (j.state === 'failed' || !status.ok) throw new Error(j.error || 'Download failed');
      say('Server download is running. Keep this tab open for the result…');
    }
    if (!j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    setBar(1); say(`✅ <b>${esc(j.filename)}</b> downloaded to the studio server. Pick it on the canvas.`, 'ok');
    announce(j.filename); await loadSongs();
  } catch (err) {
    setBar(1, true); say('❌ ' + esc(err.message), 'err');
  } finally { busy = false; }
}

function announce(name){ try { new BroadcastChannel('sts-uploads').postMessage({name}); } catch {} }

async function importSong(path, btn){
  btn.disabled = true; btn.textContent = 'importing…';
  try {
    const r = await fetch('/sts/import', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({path})});
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    announce(j.filename); await loadSongs();
  } catch (err) { alert('Import failed: ' + err.message); btn.disabled = false; btn.textContent = 'Import'; }
}

async function loadSongs(){
  songsEl.innerHTML = '<li class="mini">loading…</li>';
  try {
    const r = await fetch('/sts/songs', {cache:'no-store'});
    const data = await r.json();
    let html = '';
    for (const name of data.input || [])
      html += `<li class="song"><b>${esc(name)}</b><span class="tag">ready</span></li>`;
    for (const path of data.kaggle || [])
      html += `<li class="song"><b>${esc(path.split('/').pop())}</b><span class="mini">${esc(path)}</span>` +
              `<button onclick="importSong(this.dataset.p, this)" data-p="${esc(path)}">Import</button><span class="tag src">Kaggle</span></li>`;
    songsEl.innerHTML = html || '<li class="mini">No songs yet — add one above.</li>';
  } catch (err) { songsEl.innerHTML = `<li class="err">Could not load the song list: ${esc(err.message)}</li>`; }
}

drop.addEventListener('click', () => fileEl.click());
fileEl.addEventListener('change', () => fileEl.files[0] && uploadFile(fileEl.files[0]));
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]));
document.getElementById('fetchbtn').addEventListener('click', fetchUrl);
document.getElementById('url').addEventListener('keydown', e => e.key === 'Enter' && fetchUrl());
loadSongs();
</script></main></body></html>"""


UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_FETCH_JOBS = {}
_UPLOAD_LOCKS = {}
_PUBLISH_LOCK = threading.Lock()


def _upload_spec(values):
    try:
        total, size = int(values.get("n", 0)), int(values.get("size", 0))
    except (TypeError, ValueError):
        raise ValueError("Invalid upload size or piece count")
    if not 0 < size <= MAX_SONG_BYTES or total != math.ceil(size / UPLOAD_CHUNK_BYTES):
        raise ValueError("Invalid upload size or piece count")
    ident = str(values.get("upload", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise ValueError("Upload ID is missing or invalid; reload the upload page")
    name = _safe_song_name(values.get("name"))
    if not _is_audio_name(name):
        raise ValueError("Choose a supported audio file")
    parts = os.path.join(_temp_dir(), "sts_parts", ident + f".{total}.{size}.{name}")
    return parts, name, total, size


def _assemble_upload(parts, name, total, size):
    descriptor, temporary = tempfile.mkstemp(prefix=".sts-assembling-", dir=_input_dir())
    try:
        with os.fdopen(descriptor, "wb") as out:
            for i in range(total):
                with open(os.path.join(parts, f"{i:05d}.part"), "rb") as source:
                    shutil.copyfileobj(source, out, 1 << 20)
        if os.path.getsize(temporary) != size:
            raise ValueError("Size check failed; completed pieces were retained for retry")
        with _PUBLISH_LOCK:
            dest = _unique_path(os.path.join(_input_dir(), name))
            os.replace(temporary, dest)
        result = {"ok": True, "filename": os.path.basename(dest), "bytes": size}
        with open(os.path.join(parts, "complete.json"), "w") as receipt:
            json.dump(result, receipt)
        for i in range(total): os.remove(os.path.join(parts, f"{i:05d}.part"))
        return result
    finally:
        if os.path.exists(temporary): os.remove(temporary)


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # engine-only import (tests) — routes need the running server

    routes = PromptServer.instance.routes

    @routes.get("/sts/health")
    async def sts_health(request):  # noqa: ANN202
        return web.json_response({"ok": True, "pack": "TimestampStudio"})

    @routes.get("/sts_uploader")
    async def sts_uploader(request):  # noqa: ANN202
        return web.Response(text=_UPLOADER_PAGE, content_type="text/html")

    @routes.get("/sts/songs")
    async def sts_songs(request):  # noqa: ANN202
        kaggle = []
        for root_dir in _scan_roots():
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for fn in filenames:
                    if _is_audio_name(fn) and not fn.startswith("."):
                        kaggle.append(os.path.join(dirpath, fn))
        return web.json_response({"input": _list_input_songs(),
                                  "kaggle": sorted(set(kaggle))[:300]})

    @routes.get("/sts/upload/status")
    async def sts_upload_status(request):
        try:
            parts, name, total, size = _upload_spec(request.rel_url.query)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if os.path.isfile(os.path.join(parts, "complete.json")):
            return web.json_response({"completed": list(range(total))})
        completed = [i for i in range(total) if os.path.isfile(os.path.join(parts, f"{i:05d}.part"))]
        return web.json_response({"completed": completed})

    @routes.post("/sts/upload/chunk")
    async def sts_chunk(request):
        try:
            parts, name, total, size = _upload_spec(request.rel_url.query)
            idx = int(request.rel_url.query["i"])
            if not 0 <= idx < total:
                raise ValueError("piece index out of range")
        except (ValueError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        async with _UPLOAD_LOCKS.setdefault(parts, asyncio.Lock()):
            os.makedirs(parts, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=parts)
            written = 0
            expected = min(UPLOAD_CHUNK_BYTES, size - idx * UPLOAD_CHUNK_BYTES)
            try:
                with os.fdopen(descriptor, "wb") as out:
                    async for block in request.content.iter_chunked(1 << 20):
                        written += len(block)
                        if written > expected:
                            raise ValueError("piece exceeds its expected size")
                        out.write(block)
                if written != expected:
                    raise ValueError(f"incomplete piece: {written} vs {expected} bytes")
                os.replace(temporary, os.path.join(parts, f"{idx:05d}.part"))
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=400)
            finally:
                if os.path.exists(temporary): os.remove(temporary)
        return web.json_response({"ok": True, "got": idx, "bytes": written})

    @routes.post("/sts/upload/finish")
    async def sts_finish(request):
        try:
            parts, name, total, size = _upload_spec(await request.json())
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        async with _UPLOAD_LOCKS.setdefault(parts, asyncio.Lock()):
            receipt = os.path.join(parts, "complete.json")
            if os.path.isfile(receipt):
                with open(receipt) as source:
                    return web.json_response(json.load(source))
            if not all(os.path.isfile(os.path.join(parts, f"{i:05d}.part")) for i in range(total)):
                return web.json_response({"error": "missing upload pieces; retry the same file"}, status=409)
            try:
                result = await asyncio.to_thread(_assemble_upload, parts, name, total, size)
            except (ValueError, OSError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(result)

    @routes.post("/sts/fetch_url")
    async def sts_fetch_url(request):  # noqa: ANN202
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        url = str(body.get("url") or "").strip()
        if not re.match(r"^https?://", url, re.I):
            return web.json_response({"error": "only http(s) links work"}, status=400)
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=1800, sock_read=180)
        dest = tmp = None
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "TimestampStudio/2.0"}) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"error": f"that link returned HTTP {resp.status}"}, status=400)
                    ctype = resp.headers.get("Content-Type", "").lower()
                    if "text/html" in ctype:
                        return web.json_response(
                            {"error": "that link is a web page, not a song file"}, status=400)
                    url_path = urllib.parse.urlparse(url).path
                    raw = os.path.basename(urllib.parse.unquote(url_path)) or body.get("name") or "song"
                    name = _safe_song_name(raw)
                    if not _is_audio_name(name):
                        ext = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/vnd.wave": ".wav",
                               "audio/mpeg": ".mp3", "audio/flac": ".flac", "audio/x-flac": ".flac",
                               "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/aac": ".aac",
                               "audio/opus": ".opus", "audio/webm": ".webm"}.get(
                                   ctype.split(";")[0].strip(), "")
                        if not ext:
                            return web.json_response(
                                {"error": f"don't recognise that file type ({ctype or 'unknown'})"},
                                status=400)
                        name += ext
                    dest = _unique_path(os.path.join(_input_dir(), name))
                    tmp = dest + ".part"
                    written = 0
                    with open(tmp, "wb") as out:
                        async for block in resp.content.iter_chunked(1 << 20):
                            written += len(block)
                            if written > MAX_SONG_BYTES:
                                raise ValueError("file is over the 2 GB limit")
                            out.write(block)
            os.replace(tmp, dest)
            tmp = None
        except asyncio.TimeoutError:
            return web.json_response({"error": "the link was too slow / timed out"}, status=504)
        except Exception as exc:
            return web.json_response({"error": f"download failed: {exc}"}, status=502)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        print(_PREF + f"song downloaded from link: {os.path.basename(dest)}", flush=True)
        return web.json_response({"ok": True, "filename": os.path.basename(dest)})

    @routes.post("/sts/fetch_url/start")
    async def sts_fetch_start(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        if not re.match(r"^https?://", str(body.get("url", "")), re.I):
            return web.json_response({"error": "only http(s) links work"}, status=400)
        if sum(not job["task"].done() for job in _FETCH_JOBS.values()) >= 3:
            return web.json_response({"error": "Three downloads are already running; wait for one to finish"}, status=429)
        ident = uuid.uuid4().hex
        job = {"state": "running"}
        async def run_download():
            async def payload(): return body
            from types import SimpleNamespace
            try:
                response = await sts_fetch_url(SimpleNamespace(json=payload))
                job.update(json.loads(response.text))
                job["state"] = "done" if response.status == 200 else "failed"
            except Exception as exc:
                job.update(state="failed", error=str(exc))
        job["task"] = asyncio.create_task(run_download())
        _FETCH_JOBS[ident] = job
        return web.json_response({"job": ident})

    @routes.get("/sts/fetch_url/status")
    async def sts_fetch_status(request):
        job = _FETCH_JOBS.get(request.rel_url.query.get("job"))
        if job is None:
            return web.json_response({"error": "download job not found"}, status=404)
        return web.json_response({k:v for k,v in job.items() if k != "task"})

    application = getattr(PromptServer.instance, "app", None)
    if application is not None:
        async def cancel_downloads(app):
            tasks = [job["task"] for job in _FETCH_JOBS.values() if not job["task"].done()]
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        application.on_cleanup.append(cancel_downloads)

    @routes.post("/sts/import")
    async def sts_import(request):  # noqa: ANN202
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        src = os.path.realpath(str(body.get("path") or ""))
        allowed = tuple(os.path.realpath(r) + os.sep for r in _scan_roots())
        if not (src.startswith(allowed) and os.path.isfile(src) and _is_audio_name(src)):
            return web.json_response({"error": "that path is not an importable song"}, status=400)
        if os.path.getsize(src) > MAX_SONG_BYTES:
            return web.json_response({"error": "file is over the 2 GB limit"}, status=413)
        dest = _unique_path(os.path.join(_input_dir(), _safe_song_name(os.path.basename(src))))
        shutil.copyfile(src, dest)
        return web.json_response({"ok": True, "filename": os.path.basename(dest)})


_register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
