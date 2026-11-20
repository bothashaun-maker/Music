# Music Studio on Kaggle

[Music-Studio.ipynb](../../Music-Studio.ipynb) ·
[Automatic Dataset builder](../../Music-Studio-Dataset.ipynb) ·
[Complete tool guide PDF](../../Music-Studio-Guide.pdf) ·
[Editable guide](GUIDE.md) · [Validation](VALIDATION.md)

This is a separate full browser studio based on the pinned asb-42 Linux fork
of timoncool's YuE2 Studio. Existing FP16 music generation and Song Timestamp
Studio notebooks and datasets remain usable.

## First time: save one private Dataset

1. Import **Music-Studio-Dataset.ipynb** into Kaggle.
2. Select **Accelerator None**, **Internet ON**. Enable the private Kaggle
   Secret **KAGGLE_API_TOKEN** for this notebook. Do not paste its value into code.
3. Run the single code cell. Leave `DATASET_OWNER=''` unless username detection
   needs your Kaggle username. The builder checks authentication before doing
   expensive work.
4. It compiles the x86-64/T4 runtime once, verifies models, and automatically
   submits **one private Music Studio Dataset** using the established native
   `dataset_create_new(public=False)` upload method. There is no manual PC
   download/re-upload or separate dataset-save cell.
5. Wait for **PRIVATE MUSIC STUDIO DATASET SUBMITTED**, then allow Kaggle to
   process the Dataset. Its title is **Music Studio Dataset <identity>**.

The selected 18 assets total **24,770,652,419 bytes** (about 24.77 GB), plus
the compiled runtime. The first compilation needs at least **7 GiB measured
free disk**. Model upload needs room for only the largest missing file
(about 8.71 GB) plus reserve, after compiler intermediates are removed. The
SDK uploads one verified file at a time in one Dataset creation request.
Successfully uploaded, newly downloaded cache files are released; attached
inputs and existing source files are preserved.

The first build/download can take substantial time. It uses CPU and compiles
CUDA device code for SM75 without requiring a GPU. A pinned NVIDIA driver-dev
stub is used only while linking; no driver is installed or shipped. Normal
GPU studio sessions reuse the saved runtime and do not compile it.

There is no custom `datasets.get/status` prerequisite. The SDK retains its
normal internal behavior. Existing Datasets are never versioned or overwritten.
An interrupted final creation request has an uncertain outcome: inspect the
Dataset page before retrying. A local receipt confirms submission, not final
Kaggle processing.

## Every studio session

1. Import **Music-Studio.ipynb**. Add that ONE **Music Studio Dataset** under
   **Add Input → Your Datasets**.
2. Choose **GPU T4 ×2**, **Internet ON**. Stop another startup cell using the
   same runtime's GPU or ports.
3. Run the single code cell. Models are checksum-verified from the input;
   the native CUDA engine must become ready. No automatic song is generated.
4. Open **OPEN MUSIC STUDIO**, including its private key. Keep the cell running.
5. In the browser choose Create, Library or Studio tools and run the job yourself.

The CUDA 13 runtime requires NVIDIA driver 580 or newer, matching the CUDA 13
T4 environment recorded in this repository. Startup reports an older driver
before model loading; it cannot install or upgrade Kaggle's host driver.

Generation uses the Q8 GGUF model on **GPU 0**. The second T4's memory is not
pooled. Stems and Parakeet/Whisper recognition are explicitly configured for
CPU. The bundled local Gemma writer uses CPU llama.cpp; the MIDI and MOSS
caption helpers are native CPU builds too. Actual performance needs a Kaggle run.

The current Kaggle image must load the saved Linux binaries; rebuild on the
appropriate image if loader/GLIBC compatibility changes. The launcher checks
the architecture, free GPU memory and loader errors instead of declaring a
broken installation ready.

## Included tools and boundaries

The browser studio includes song generation, ABC score planning/editing,
MIDI editing/import, replay/re-render, covers, library/playlists, six-stem
HT-Demucs separation, MuScriptor Medium audio-to-MIDI, Parakeet/Whisper
karaoke, audio processing/reference mastering, lyric video controls, artwork,
local/external writing providers, LoRA application, MOSS style description,
player/visualizer/Webamp and authenticated MCP.

Use **Upload large WAV** for resumable original WAV/MP3 imports up to 2 GiB.
Each 4-MiB piece is hashed and previous pieces are checked before resume.
Finished imports are idempotent and retain source SHA256/length. The original
small-file import remains available. No audio conversion is required.

The pinned Linux fork's feature set differs from current Windows Studio.
Windows VST hosting, native desktop integration, AB2/chord-variety and newer
extension/continuation controls are not provisioned. **LoRA training is not
provisioned for T4**; training-base weights are deliberately excluded.
The PDF covers every available panel plus these boundaries. It does not
present unavailable controls as completed features.

## Preserve and restore songs

Generated audio, the SQLite library, settings, installed adapters, resumable
uploads and logs live in **/kaggle/working/music_studio**. Temporary models
and runtime state live under **/kaggle/temp/music_studio_runtime**; they do
not become a second model pack in your saved song output.

Stop the startup cell before a final library snapshot. **Save Version with
output files**, or download the library/media you want. A draft and an .ipynb
containing logs do not preserve audio. The model Dataset does not save new songs.

For a later session, attach the saved output alongside the model Dataset.
Startup restores one saved `music_studio` library into empty working output.
Use `RESTORE_LIBRARY_PATH` for an explicit folder if several are attached;
use `'none'` to start a fresh library. An existing working library is not
overwritten. The SQLite backup path accounts for a saved WAL file.

Keep source WAVs and manual corrections. Code-publication permission does not
publish your recordings or private project files.

## Logs and repairs

- Studio: `music_studio/logs/studio.log`
- Cloudflare: `music_studio/logs/cloudflare.log`
- Startup: `music_studio/startup-status.json`
- GPU/runtime identity: `music_studio/runtime-identity.json`
- Model/engine ready check: `music_studio/ready-check.json`
- Dataset: `music_studio_dataset/dataset-upload.json`, `dataset-error.json`
- One-time compiler log: `/kaggle/temp/music_studio_build/runtime-build.log`

Stop the prior cell for a port/VRAM conflict. The launcher does not kill
unrecognized processes. A CUDA startup failure is not converted into CPU
generation. Interrupted, owned extraction staging can be retried automatically.
An old Cloudflare URL/key stops being valid after restart.

## Rebuild and sources

```bash
python tools/build_music_studio_notebooks.py
python tools/build_music_studio_pdf.py
python -m unittest discover -s tests/music_studio -v
```

Packaging/lifecycle sources: `music_studio/`. Exact upstream commits and runtime
download hashes: [sources.json](../../music_studio/sources.json). Immutable
model URLs, sizes, hashes and terms: [catalog.json](../../music_studio/catalog.json).
The PDF uses the fonts, page dimensions, palette, numbered sections and link
conventions of the repository's existing Music Production System guide.

Read [component notices](../../music_studio/NOTICES.md) before commercial use.
YuE2's individual-creator permission does not automatically cover the companion,
SheetSage2 or MuScriptor. The standard Studio path uses the companion.
