"""Build the standalone notebook around the original native FP16 Kaggle engine."""
from pathlib import Path
import base64
import json
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    files = {}
    for directory in ('kaggle_yue2', 'native_yue2'):
        for p in sorted((ROOT / directory).glob('*.py')):
            files[str(p.relative_to(ROOT))] = p.read_text()
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    cells = [
        cell('markdown', r'''
# YuE2 on Kaggle — restored native FP16 engine (revision 12)

**First time:** run `yue2-build-kaggle-assets.ipynb` once on Kaggle with
Internet and **no GPU**, then save its outputs as a private dataset.
Add that dataset to this notebook with **Add Input**. You can also attach
the saved builder notebook's output directly. Nothing large needs to be
downloaded to your PC or uploaded from your PC.

**Every music session:** attach that saved input, select **GPU T4 ×2**
and enable Internet for the Gradio and optional Cloudflare links. Setup reads weights from
`/kaggle/input` and installs dependencies from saved local wheels.
Missing/incompatible assets cause an early error, never a replacement
download. Keep the same Kaggle Python/image version as the builder;
upgrades can require rebuilding the dependency bundle.

This restores the original
`yue2_infer 0.1.5` engine: FP16 transformer on GPU 0, CUDA-graph AR decoding
with SDPA, and tiled FP32 VAE on GPU 1. Default generation uses **Direct to
audio**, **16 Euler steps**, CFG **1.0**, and an **8-minute safety ceiling**.
The model chooses a natural ending from your prompt and lyrics. It does
not have to fill the ceiling. A token-limit cutoff is reported separately.

**No diagnostic song runs automatically.** Run setup and open the UI,
enter your full song lyrics and style, leave or adjust the safety ceiling, then click
Generate Song. Optional score planning remains off unless you select it.
**Cover your own song:** open **Cover / restyle my existing song** in the
UI, upload its WAV/FLAC/MP3, paste your original lyrics, and click
**Analyze melody for cover**. The score fills Custom ABC and selects
Melody score mode. Enter the new style, then Generate Song. Optional
**Transcribe lyrics too** uses Whisper when needed. Extracted-score
regeneration can change timing, voice and arrangement; it is not exact
audio-to-audio copying. Melody analysis reads SheetSage2 and its MERT
parent from the saved dataset; optional Whisper is included there too. Download the
cover-analysis ZIP to retain the editable score. Transcription models
are unloaded after analysis and GPU jobs share one queue.

**T4 attention change:** equivalent grouped-head views make PyTorch's
memory-efficient kernel eligible, instead of the native T4 GQA math path.
The first supported CUDA dispatch is verified and logged. Unsupported
shapes retain native attention and are reported. CPU equivalence checks
passed; actual T4 speed improvement has not been measured.

Durations are maximums. A natural early ending or token cutoff is reported.
Long acoustic generations may span multiple chunks; listen at the reported
boundaries. Seven/eight-minute performance and continuity remain unverified.

The small installed environment uses session-only `/kaggle/temp/yue2_native`.
Large models stay in the attached read-only Kaggle input dataset.
Song audio, exact tokens and recovery latents use
`/kaggle/working/yue2_native/runs`. The paths may share underlying storage;
setup prints actual free space rather than assuming a separate 50 GB quota.
Old notebook caches and files are preserved.

**Process protection:** setup and launch stop only verified older engines
using this script/output root, including survivors of a kernel restart.
A process lock prevents two managed engines from loading together. GPU
memory is checked before loading and before generation. An unrecognized
GPU process is reported and left running; setup stops before generation.
CUDA OOM saves GPU process details and releases the failed engine's model.

**After generation, download the complete run ZIP in the UI, or Save
Version with outputs in Kaggle before ending the session.** A saved draft
or a notebook download containing logs does not preserve audio files.
'''.replace('PAYLOAD', payload)),
        cell('markdown', r'''
**Start here:** attach `yue2-music-assets`, select GPU T4 x2 and Internet ON,
then run the next code cell. It performs setup and opens the generator in
one operation. Keep it running while using either UI link. Nothing
generates until you press Generate in the UI.

Setup/engine work runs in separate processes. A kernel restart does not
require reusing variables from another cell. Full startup details are saved
to `/kaggle/working/yue2_native/runtime/startup.log`; `startup-status.json`
records setup, engine startup, ready or failure. The independent file-list
cell below starts no setup or GPU job.
'''.replace('PAYLOAD', payload)),
        cell('code', r'''
import base64, json, os, signal, subprocess, sys, zlib
from pathlib import Path
SUPPORT = Path("/kaggle/temp/yue2_native_support")
ROOT = "/kaggle/temp/yue2_native"
OUTPUTS = "/kaggle/working/yue2_native/runs"
ASSET_DATASET = "auto"
ENABLE_CLOUDFLARE = True
SUPPORT.mkdir(parents=True, exist_ok=True)
payload = "PAYLOAD"
for relative, contents in json.loads(zlib.decompress(base64.b64decode(payload))).items():
    path = SUPPORT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
env = dict(os.environ)
env["PYTHONPATH"] = str(SUPPORT) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUNBUFFERED"] = "1"
arguments = [sys.executable, "-u", "-m", "native_yue2.launch",
             "--root", ROOT, "--outputs", OUTPUTS, "--assets", ASSET_DATASET]
if not ENABLE_CLOUDFLARE:
    arguments.append("--no-cloudflare")
if globals().get("RECOVER_RUN"):
    arguments.extend(["--recover", str(RECOVER_RUN)])
log_path = Path(OUTPUTS).parent / "runtime/startup.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
worker = subprocess.Popen(arguments, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, start_new_session=True)
try:
    with log_path.open("a") as log:
        for line in worker.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
    code = worker.wait()
    if code:
        raise RuntimeError(f"Startup stopped (exit {code}). Full details: {log_path}")
finally:
    if worker.poll() is None:
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
'''.replace('PAYLOAD', payload)),
        cell('markdown', r'''
**Find and preserve your song.** Each attempt has its own folder. After a
successful generation the UI offers WAV, FLAC, and a complete ZIP containing
settings, tokens, timings and recovery latents. Download it before stopping.
The folder also appears under Kaggle Output at `yue2_native/runs`.
`status.json` records the measured audio length and any cutoff; it does not
establish musical completeness. Listen to the ending and chunk boundaries.

Run the next cell after stopping the UI to list all saved attempts. It
starts no model or generation. Files still need to be downloaded or saved
with a Kaggle output version before the session is discarded.
'''.replace('PAYLOAD', payload)),
        cell('code', r'''
from pathlib import Path
import json
OUTPUTS = "/kaggle/working/yue2_native/runs"
from IPython.display import FileLink, display
for directory in sorted(Path(OUTPUTS).glob("*")):
    if directory.is_dir():
        status = directory / "status.json"
        print(directory.name, status.read_text() if status.exists() else "Partial run")
        for filename in ("audio.wav", "audio.flac"):
            if (directory / filename).exists():
                display(FileLink(str(directory / filename)))
for archive in sorted(Path(OUTPUTS).glob("*.zip")):
    display(FileLink(str(archive)))

if not Path(OUTPUTS).exists():
    print("No saved song folders in this session yet.")
'''.replace('PAYLOAD', payload)),
        cell('markdown', r'''
Optional recovery: set RECOVER_RUN below, then rerun the startup cell. Leave None for normal music creation.
'''.replace('PAYLOAD', payload)),
        cell('code', r'''
# Optional: only after stopping the generator.
RECOVER_RUN = None
if RECOVER_RUN:
    print("Recovery selected. Run the startup cell above; it will decode these saved latents.")
'''.replace('PAYLOAD', payload)),
        cell('markdown', r'''
Based on the original AIQUEST Academy notebook in Movies commit
`fcf4d45dc5dbca57b39e54311807410cdf41f737`.
Sources: [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
[YuE2 VAE](https://huggingface.co/m-a-p/YuE2-Vae).
The native inference wheel and model revisions are pinned; wheel and
model checksums are verified. Upstream model/license terms apply.
Source and checks are included in the Music repository for review.
'''.replace('PAYLOAD', payload)),
    ]
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': True},
    }, 'nbformat': 4, 'nbformat_minor': 4}


def build_asset_notebook():
    files = {}
    for directory in ('kaggle_yue2', 'native_yue2'):
        for p in sorted((ROOT / directory).glob('*.py')):
            files[str(p.relative_to(ROOT))] = p.read_text()
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    bootstrap = """
    import base64, json, os, sys, zlib, subprocess
    from pathlib import Path
    support = Path("/kaggle/temp/yue2_asset_builder_support")
    support.mkdir(parents=True, exist_ok=True)
    payload = "PAYLOAD"
    for relative, contents in json.loads(zlib.decompress(base64.b64decode(payload))).items():
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    for name in list(sys.modules):
        if name.split(".")[0] in {"native_yue2", "kaggle_yue2"}:
            del sys.modules[name]
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from native_yue2.setup import install
    BUILDER_ROOT = "/kaggle/temp/yue2_asset_builder"
    ASSET_OUTPUT = "/kaggle/working/yue2-assets"
    native_python, native_env = install(
        BUILDER_ROOT, "/kaggle/temp/yue2_asset_builder_records/runs",
        assets=None, require_gpu=False)
    subprocess.run([native_python, "-u", "-m", "native_yue2.assets",
                    "--build", ASSET_OUTPUT, "--runtime-root", BUILDER_ROOT],
                   env=native_env, check=True)
    """.replace('PAYLOAD', payload)
    cells = [cell('markdown', """
        # YuE2 — save Kaggle assets once (revision 9)

        **Internet ON; accelerator None.** No GPU model or song is run.
        Downloads go directly to Kaggle, not through your home connection.
        This saves the pinned YuE2/VAE, SheetSage2/MERT, optional Whisper,
        Python overlay wheels, bootstrap pip and Gradio tunnel executable.
        Weights are saved once without a duplicate Hub blobs tree.

        Keep this notebook **private**. Use **Save Version → Save & Run All**
        with outputs to build and preserve `/kaggle/working/yue2-assets` in one
        saved run. Wait for **COMPLETE** in its logs. No test song follows.
        You do not need to run it interactively first and repeat the download.
        In your music notebook choose **Add Input → Notebook Output Files**
        and select the completed saved builder version. That is already reusable
        Kaggle storage; you do not need to download/re-upload any files.

        If you specifically want a named **Dataset**, use the optional publish
        cell below from this Kaggle session. It uploads directly from Kaggle to
        your account and creates a **private** dataset. The normal generator
        accepts either the dataset or saved notebook output automatically.

        Keep the same Python minor version and preinstalled scientific package
        versions in builder and generator. If Kaggle changes the image, the
        generator explains the mismatch before any model work. Rebuilding on
        the new image may be necessary. This is not a portable copy of all CUDA.
        Enough output storage for all models is required; an interrupted build
        has no complete assets.json and cannot be mistaken for a finished bundle.
        Model licenses/cards are retained; upstream terms still apply.
        """), cell('code', bootstrap), cell('markdown', """
        **Optional: create a named private dataset, directly from Kaggle.**
        Set DATASET_HANDLE to `your-kaggle-username/yue2-music-assets` and run
        the next cell. Leave it as None if using saved notebook outputs instead.
        KaggleHub needs your Kaggle account authentication. If it requests a
        token, configure Kaggle authentication privately; never paste credentials
        into this notebook. New datasets are private by default in this API;
        choose a new unused slug to avoid updating an older public dataset.
        After upload, attach it using **Add Input → Your Datasets**. Keep the
        dataset private. No model files travel through your home computer.
        """), cell('code', """
        DATASET_HANDLE = None  # Example: "janpierrrewiet/yue2-music-assets"
        if DATASET_HANDLE:
            from native_yue2.assets import validate_assets
            validate_assets(ASSET_OUTPUT, full=True)
            import kagglehub
            if not hasattr(kagglehub, "dataset_upload"):
                raise RuntimeError("This Kaggle image needs a newer KaggleHub for dataset_upload. "
                                   "Use saved notebook outputs now, or update KaggleHub in this builder only.")
            kagglehub.dataset_upload(
                DATASET_HANDLE, ASSET_OUTPUT,
                version_notes="Pinned YuE2 music + cover models and offline dependency wheels")
            print(f"Saved: https://www.kaggle.com/datasets/{DATASET_HANDLE}. Keep this dataset private.")
        """)]
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'isGpuEnabled': False, 'isInternetEnabled': True},
    }, 'nbformat': 4, 'nbformat_minor': 4}


if __name__ == '__main__':
    output = ROOT / 'yue2-3b-frontier-full-song-music-generation.ipynb'
    output.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + '\n')
    print(output)
    assets = ROOT / 'yue2-build-kaggle-assets.ipynb'
    assets.write_text(json.dumps(build_asset_notebook(), indent=1, ensure_ascii=False) + '\n')
    print(assets)
