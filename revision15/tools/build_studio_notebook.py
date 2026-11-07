"""Assemble Song-Timestamp-Studio-Canvas.ipynb from the Music repository sources.

Movies-style delivery: ONE executable cell — pinned tools -> TimestampStudio node
pack (engine + nodes + web extension + canvas) -> pinned ComfyUI checkout ->
persistent server -> Cloudflare link that opens the full canvas directly.

Embedded sources (single source of truth, validated by tests/test_studio.py):
  - timestamp_studio/engine.py       the alignment engine (37-check suite)
  - timestamp_studio/comfy_pack.py   the ComfyUI node pack
  - timestamp_studio/web.py          the ?sts_workflow= browser extension
  - timestamp_studio/canvas.py       the visible studio canvas graph
"""
import base64
import json
import os
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = os.environ.get("STS_OUT_DIR", str(ROOT))

COMFYUI_COMMIT = "423cbd20511becb1f16122ab518e92e58bfa03d9"  # v0.43.2, same pin as YuE2


def build_payload():
    engine_src = (ROOT / "timestamp_studio/engine.py").read_text()
    pack_src = (ROOT / "timestamp_studio/comfy_pack.py").read_text()
    scope = {}
    exec((ROOT / "timestamp_studio/web.py").read_text(), scope)
    web_src = scope["STS_OPEN_WORKFLOW_JS"]
    ns = {}
    exec(compile((ROOT / "timestamp_studio/canvas.py").read_text(), "canvas.py", "exec"), ns)
    # The canvas defaults carry the engine's own example lyrics.
    example = engine_src.split('EXAMPLE_LYRICS = """', 1)[1].split('"""', 1)[0]
    graph_builder = ns["build_canvases"]
    canvases = {name: json.dumps(graph) for name, graph in graph_builder(example.strip("\n") + "\n").items()}
    files = {
        "TimestampStudio/__init__.py": pack_src,
        "TimestampStudio/sts_engine.py": engine_src,
        "TimestampStudio/acoustic.py": (ROOT / "timestamp_studio/acoustic.py").read_text(),
        "TimestampStudio/web/open_workflow.js": web_src,
    }
    for name, graph_text in canvases.items():
        files[f"TimestampStudio/web/sts-{name}.json"] = graph_text
    for rel, src in files.items():
        if rel.endswith(".py"):
            compile(src, rel, "exec")  # Fail the build, never the browser session.
    return base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()


MD_TOP = """# 🎛️ Song Timestamp Studio — acoustic alignment (revision 16.2)

**One cell. One link. Lyric timestamps you can check against the recording.**

**Supplied English lyrics:** the Studio aligns the complete lyric sequence to
acoustic frames, including repeated passages. It no longer fuzzy-matches an
imperfect transcription and spreads missed words into estimated times. Include
every performed repeat in the lyrics. The Whisper model menu applies only when
the lyric box is empty (multilingual automatic transcription).

The acoustic model (~360 MB) downloads on Kaggle on first use and is saved in
`timestamp_studio/alignment_models/`. Acoustic frames are also cached, so changing
lyrics for the same recording reuses the analysis. Save the Studio outputs and
attach them with **Add Input** in a future session to reuse models and cached
frames. You do not have to download model files to your PC.

After Export completes, use **Download ZIP** on the Export node. Scores are
diagnostics, not timing guarantees; the ZIP includes an alignment review CSV.
Runtime logs and per-attempt diagnostics remain in Kaggle Output.

Works exactly like your Movies / H3 canvases: attach nothing, select **Accelerator: None**
(CPU supported — no GPU hours burned) and **Internet ON**, run the single cell below, then
click the printed ☁️ **Cloudflare** link — the full node graph opens directly, ready to run.
The canvas is also under ComfyUI's **Workflows** menu.

| Output | Use it in |
|---|---|
| Synced lyrics (`.lrc` / `.srt` / `.vtt`) | lyric videos, CapCut, DaVinci, Premiere, YouTube captions |
| YouTube chapters | paste into the video description |
| REAPER markers CSV | drag into the Region/Marker Manager — your accuracy audit |
| Beat grid + **Cut Plan CSV** | every beat + clip slots in seconds **and frames @ 24 fps** (MiniMax H3 shot budget) |
| Full JSON | per-word times for any other tool |

**The canvas:** `Your Song (STSLoadSong) → Align Lyrics ← Lyrics & Options`, with `Nudge` and
`Export All Formats + ZIP` on one branch and `Audition One Line → PreviewAudio`
(ear check) on the other. Every node is a real stage — nothing is collapsed.
"""

CODE_CELL = '''
# Song Timestamp Studio — single-cell setup (needs Internet ON; Accelerator: None is fine)
import base64, json, os, shutil, socket, subprocess, sys, threading, time, urllib.request, zlib
from pathlib import Path

ROOT = Path("/kaggle/temp/timestamp_studio")
OUT = Path("/kaggle/working/timestamp_studio")
LOGS = OUT / "logs"
SUPPORT = ROOT / "support"
COMFY_PORT = 8188
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_COMMIT = "@COMFYUI_COMMIT"
for p in (ROOT, OUT, SUPPORT, LOGS):
    p.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": str(ROOT / "huggingface"), "HF_HUB_CACHE": str(ROOT / "huggingface/hub"),
    "XDG_CACHE_HOME": str(ROOT / "cache"), "TMPDIR": str(ROOT / "tmp"),
    "STS_OUTPUT_ROOT": str(OUT), "PIP_NO_CACHE_DIR": "1",
})
(ROOT / "tmp").mkdir(exist_ok=True)

# --- 1 · tools (Kaggle's own torch/numpy/numba stay pinned) -------------------
import importlib.metadata

constraints = []
for name in ("torch", "torchaudio", "torchvision", "numpy", "numba", "llvmlite", "scipy"):
    try:
        constraints.append(f"{name}=={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        pass
constraint_file = ROOT / "kaggle-constraints.txt"
constraint_file.write_text("\\n".join(constraints) + "\\n")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                       "-c", str(constraint_file),
                       "faster-whisper==1.2.1", "librosa==0.10.2.post1", "soundfile==0.13.1",
                       "transformers==4.57.6", "huggingface-hub==0.36.2", "safetensors>=0.7.0"])
print("✅ tools installed", flush=True)

# --- 2 · unpack the tested engine + node pack --------------------------------
payload = "@PAYLOAD"
for rel, text in json.loads(zlib.decompress(base64.b64decode(payload))).items():
    target = SUPPORT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
for source in (SUPPORT / "TimestampStudio").rglob("*.py"):
    compile(source.read_text(), str(source), "exec")
print("✅ TimestampStudio node pack unpacked (engine + nodes + canvas)", flush=True)

# --- 3 · pinned ComfyUI checkout (session-only) -------------------------------
comfy = ROOT / "ComfyUI"
def _sh(args, cwd=None):
    subprocess.check_call(args, cwd=cwd)
if not comfy.exists():
    comfy.mkdir(parents=True)
    _sh(["git", "init", str(comfy)])
    _sh(["git", "-C", str(comfy), "remote", "add", "origin", COMFYUI_REPO])
head = subprocess.run(["git", "-C", str(comfy), "rev-parse", "HEAD"], capture_output=True, text=True)
if head.returncode != 0 or head.stdout.strip() != COMFYUI_COMMIT:
    _sh(["git", "-C", str(comfy), "fetch", "--depth", "1", "origin", COMFYUI_COMMIT])
    _sh(["git", "-C", str(comfy), "checkout", "--detach", "FETCH_HEAD"])
head = subprocess.run(["git", "-C", str(comfy), "rev-parse", "HEAD"], capture_output=True, text=True)
if head.stdout.strip() != COMFYUI_COMMIT:
    raise RuntimeError("ComfyUI checkout does not match the pinned commit")
def install_studio_pack(support, comfy_root):
    source_root = Path(support) / "TimestampStudio"
    destination = Path(comfy_root) / "custom_nodes" / "TimestampStudio"
    required = ("__init__.py", "sts_engine.py", "acoustic.py", "web/open_workflow.js", "web/sts-studio.json")
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError("Timestamp Studio payload is incomplete: " + ", ".join(missing))
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        relative = source.relative_to(source_root)
        if source.suffix == ".py":
            compile(source.read_text(), str(source), "exec")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print("   wrote", relative, flush=True)
    workflows = Path(comfy_root) / "user" / "default" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_root / "web/sts-studio.json", workflows / "Song Timestamp Studio.json")
    return destination

pack = install_studio_pack(SUPPORT, comfy)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                       "-c", str(constraint_file), "-r", str(comfy / "requirements.txt")])
print(f"✅ ComfyUI {COMFYUI_COMMIT[:12]} + TimestampStudio pack installed", flush=True)

# --- 4 · pre-download the Whisper models (one-time per session) ---------------
try:
    from huggingface_hub import snapshot_download

    for repo in ("Systran/faster-whisper-base", "Systran/faster-whisper-small"):
        snapshot_download(repo)
        print("   cached", repo, flush=True)
    print("✅ Whisper files cached; alignment still takes processing time", flush=True)
except Exception as exc:
    print(f"(model cache skipped: {exc} — the first canvas Run downloads instead)", flush=True)

# --- 5 · start the persistent ComfyUI server ----------------------------------
def comfy_ready(port=COMFY_PORT):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

if "comfy_process" in globals() and comfy_process.poll() is None:
    comfy_process.terminate()
    try:
        comfy_process.wait(timeout=10)
    except Exception:
        comfy_process.kill()
if comfy_ready():
    raise RuntimeError("A server is already on port 8188. Stop its owning notebook cell before starting Timestamp Studio.")
else:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", COMFY_PORT)) == 0:
            raise RuntimeError(f"Port {COMFY_PORT} is occupied by another notebook.")
    log_path = LOGS / "comfyui.log"
    comfy_log = log_path.open("a")
    # CPU-only machines (Kaggle "Accelerator: None") need --cpu; T4 sessions use the GPU.
    launch_flags = ["--listen", "127.0.0.1", "--port", str(COMFY_PORT),
                    "--output-directory", str(OUT), "--disable-auto-launch",
                    "--preview-method", "none", "--max-upload-size", "2048"]
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            launch_flags.append("--cpu")
    except Exception:
        launch_flags.append("--cpu")
    comfy_process = subprocess.Popen(
        [sys.executable, "-u", str(comfy / "main.py"), *launch_flags],
        cwd=str(comfy), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _drain(stream, log):
        for line in stream:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        log.close()

    threading.Thread(target=_drain, args=(comfy_process.stdout, comfy_log), daemon=True).start()
    for _ in range(150):
        if comfy_process.poll() is not None:
            raise RuntimeError("ComfyUI exited during startup; see " + str(log_path))
        if comfy_ready():
            break
        time.sleep(2)
    else:
        raise TimeoutError("ComfyUI did not become ready; see " + str(log_path))
object_info = json.loads(urllib.request.urlopen(
    f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=20).read().decode())
missing = [n for n in ("STSLyricsOptions", "STSAlign", "STSNudge", "STSAudition", "STSExportAll")
           if n not in object_info]
if missing:
    raise RuntimeError(f"TimestampStudio nodes missing from the server: {missing}")
print("✅ ComfyUI server ready — all 5 TimestampStudio nodes registered", flush=True)

# --- 6 · Cloudflare link (same checksummed helper as the YuE2 / H3 apps) ------
def start_cloudflare(port, timeout=90):
    global cf_process
    import hashlib, platform

    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("needs Linux x86_64 (Kaggle is fine)")
    cf_root = ROOT / "cloudflared"
    cf_root.mkdir(parents=True, exist_ok=True)
    binary = cf_root / "cloudflared"
    version = "2026.10.3"
    sha = "053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0"
    url = f"https://github.com/cloudflare/cloudflared/releases/download/{version}/cloudflared-linux-amd64"

    def digest(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    if not binary.exists() or digest(binary) != sha:
        tmp = cf_root / "cloudflared.download"
        print("   fetching the Cloudflare helper (~25 MB, one-time per session)…", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if digest(tmp) != sha:
            raise RuntimeError("cloudflared checksum mismatch")
        os.chmod(tmp, 0o700)
        os.replace(tmp, binary)
    os.chmod(binary, 0o700)
    log = cf_root / "cloudflared.log"
    watcher = cf_root / "cf_watch.py"
    watcher.write_text(
        "import ctypes, os, signal, sys\\n"
        "libc = ctypes.CDLL(None, use_errno=True)\\n"
        "libc.prctl(1, signal.SIGTERM, 0, 0, 0)\\n"
        "if os.getppid() != int(sys.argv[1]): raise SystemExit(1)\\n"
        "os.execv(sys.argv[2], [sys.argv[2], 'tunnel', '--no-autoupdate', '--protocol', 'http2',"
        " '--url', 'http://127.0.0.1:' + sys.argv[3]])\\n")
    with open(log, "w") as out:
        proc = subprocess.Popen([sys.executable, str(watcher), str(os.getpid()), str(binary), str(int(port))],
                                stdout=out, stderr=subprocess.STDOUT)
    cf_process = proc
    import re as _re

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("cloudflared exited: " + log.read_text(errors="replace")[-600:])
        text = log.read_text(errors="replace")
        m = _re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com\\b", text)
        if m and "Registered tunnel connection" in text:
            return m.group(0)
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise TimeoutError("cloudflared did not connect; see " + str(log))

cf_link = None
try:
    cf_link = start_cloudflare(COMFY_PORT)
except Exception as exc:
    print(f"(Cloudflare unavailable: {exc})", flush=True)
print("", flush=True)
print("=" * 64, flush=True)
print("  🎛️  SONG TIMESTAMP STUDIO is ready", flush=True)
if cf_link:
    canvas_link = cf_link + "/?sts_workflow=studio"
    print("  ☁️  Canvas (opens the graph directly):", flush=True)
    print("      " + canvas_link, flush=True)
    print("  📤 Upload your WAV: " + cf_link + "/sts_uploader", flush=True)
    print("  ☁️  Plain ComfyUI: " + cf_link, flush=True)
else:
    print("  No public link was established; check the tunnel error above.", flush=True)
print("  Workflows menu → Song Timestamp Studio", flush=True)
print("  Runtime log: " + str(LOGS / "comfyui.log"), flush=True)
print("=" * 64, flush=True)

# --- 7 · keep the session alive while you use the canvas ----------------------
try:
    while comfy_process.poll() is None:
        time.sleep(5)
    raise RuntimeError("ComfyUI server stopped; see " + str(ROOT / "comfyui.log"))
except KeyboardInterrupt:
    print("Interrupted — stopping the ComfyUI server.", flush=True)
    comfy_process.terminate()
finally:
    if comfy_process.poll() is None:
        comfy_process.terminate()
        try:
            comfy_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            comfy_process.kill(); comfy_process.wait(timeout=5)
    if globals().get("cf_process") is not None and cf_process.poll() is None:
        cf_process.terminate()
'''.replace("@COMFYUI_COMMIT", COMFYUI_COMMIT)

MD_BOTTOM = """## 🖱️ Using the canvas (30 seconds)

1. Click the ☁️ link — the graph opens directly (also under **Workflows**).
2. **Your Song (STSLoadSong)** — pick from the list, paste a link, or 📤 upload
   (the WAV from a YuE2 run ZIP works; big files go through the upload page).
3. **Lyrics & Options** — paste the lyrics exactly as sung, one line per row. Keep your
   `[Verse]` / `[Chorus]` tags (they become chapters + REAPER regions). Leave it EMPTY to
   auto-transcribe instead. Pick model / language / vocal isolation if needed.
4. Press **Run** (Queue Prompt). **Align Lyrics** reports the match rate; **Export All**
   lists every written file with a per-line confidence check.
5. Not happy? Change the **Nudge** offset (or the model / isolation) and press **Run** again
   — everything after Nudge re-runs. **Audition One Line → PreviewAudio** plays ~8 s around
   any line for a quick ear check.

## 📕 Where your files go

Everything lands in `…/timestamp_studio/<song>_timestamps/` under **Kaggle Output**
(ComfyUI output folder): exports, alignment review and one ZIP. Use **Download ZIP**
on the Export node before the session ends. REAPER audit: drag `_reaper_markers.csv` into the
**Region/Marker Manager** (View menu) — lyrics become markers, sections become regions.

## ✅ How to trust the timings

- **Acoustic scores** for supplied lyrics describe model evidence, not measured
  timing accuracy. Audition the repeated passages and flagged lines. An impossible
  alignment fails instead of interpolating made-up positions.
- **▶️ Audition** any line — the vocal should enter when the line fires.
- **REAPER** — eyeball every flag against the waveform; you make the final call.
- **Nudge** — if everything feels early/late by the same amount, shift once, re-run.

The alignment fix has CPU/mock regression checks for the reported filtering and
CUDA failures. Those checks do not measure accuracy on your recording. There is
no guaranteed line-level error bound; check the exports by listening. Word-level
karaoke timing is not included.

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| First Run is slow | Whisper models are pre-cached by the setup cell; if the cache step was skipped, the first Run downloads once (~150–500 MB). |
| CUDA library unavailable | Studio reports CPU INT8 fallback; no CUDA packages are installed over Kaggle's PyTorch stack. |
| No words / zero matches | Export stops. Check `diagnostics/<attempt>/alignment.json` for the heard text, audio levels and backend before another attempt. |
| Supplied lyrics | English/romanized lyrics use a separate acoustic model; the Whisper base/small menu applies to empty-lyrics transcription only. |
| Low acoustic score | Listen to the flagged lines; confirm every repeat appears in the lyric box. Optional isolation may need additional software. |
| All times slightly early/late | **Nudge** node → set the offset → Run again → re-download. |
| No YouTube chapters | Add `[Verse]` / `[Chorus]` style tags in the lyrics box and run again. |
| REAPER CSV won't import | Drag into the **Region/Marker Manager** window (not the timeline); REAPER 7.66+. |
| Link stopped working | Both links die with the Kaggle session — re-run the cell for fresh ones. |
| Port 8188 busy | Only one notebook setup at a time per session. |

*Timestamp Studio canvas v1.0 — Movies-style delivery, CPU-friendly by design.
Works on any song, not just YuE2 renders.*
"""


def cell(kind, source):
    source = textwrap.dedent(source).strip("\n") + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    payload = build_payload()
    code = CODE_CELL.replace("@PAYLOAD", payload)
    return {
        "cells": [cell("markdown", MD_TOP), cell("code", code), cell("markdown", MD_BOTTOM)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "Song-Timestamp-Studio-Canvas.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1, ensure_ascii=False)
    print("wrote:", path, f"{os.path.getsize(path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
