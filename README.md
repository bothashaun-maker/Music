# YuE2 music generation on Kaggle — native FP16 engine

[Download the notebook](yue2-3b-frontier-full-song-music-generation.ipynb) using
GitHub's **Download raw file**, then import it into Kaggle. Look for
**restored native FP16 engine (revision 9)** at the top.

## Save model downloads once — revision 9

Use [the asset-builder notebook](yue2-build-kaggle-assets.ipynb) **once on Kaggle**,
with Internet enabled and accelerator **None**. It downloads the music/VAE,
SheetSage2/MERT and Whisper weights, dependency wheels, pip and the Gradio tunnel
binary into `/kaggle/working/yue2-assets`. It does not generate a test song or load
a GPU model. The large transfers are between Kaggle and the model hosts, not
through your PC.

**Save Version with outputs.** Attach that saved notebook output to the music
notebook using **Add Input → Notebook Output Files**. For a named private Dataset,
the builder includes an optional `kagglehub.dataset_upload` cell: set your own
`username/dataset-slug` and authenticate your Kaggle account privately. Uploads
run directly on Kaggle. Keep the source notebook and dataset private.

The generator automatically discovers one attached bundle. It installs its small
Python overlay from saved wheels and reads large weights directly from
`/kaggle/input`; it does not copy weights into temporary storage. Hub/Transformers
are forced into offline mode, pip uses `--no-index`, and the share executable is
copied locally for execution. Missing, corrupted or incompatible assets stop
setup; **there is no silent download fallback**. Generation defaults, quality,
precision and token ceilings are unchanged.

Read [KAGGLE_DATASET.md](KAGGLE_DATASET.md) for exact first-run/reuse steps,
image compatibility and validation limits. The Gradio share link still needs
Internet; source-song uploads and final audio downloads still use normal browser
bandwidth. Kaggle image/dependency changes or future model revisions may require
one rebuild, so “never download again” is not an unlimited compatibility promise.

This revision returns to the original notebook's `yue2_infer` engine. The previous
ComfyUI implementation changed the runtime and defaults and was substantially
slower in the uploaded Kaggle run. The native version retains the original FP16 and
CUDA-graph path, restores direct generation, and adds seven/eight-minute options.
The original engine's reported three-minute speed is not a measured speed claim
for this revision or for eight-minute songs.

## Cover / restyle a song you already created (revision 8)

The **Cover / restyle my existing song** panel is restored and visible in the UI.
Its previous hidden state and disabled handler were implementation omissions.

1. Upload your existing song as WAV, FLAC or MP3.
2. Paste your original words into **Song Lyrics** below the panel. By default,
   they are kept unchanged. Enable **Transcribe lyrics too** only if you need ASR.
3. Click **Analyze melody for cover**. Wait for the extracted score to fill
   **Custom ABC** and select **Melody score, then audio**. Download the analysis ZIP.
4. Enter the new genre, instruments or vocal direction in **Music Style**, check
   the lyrics and score, then click **Generate Song**.

YuE2's native API does not accept a source audio recording directly. This workflow
extracts a melody-only ABC score using SheetSage2 and regenerates from that score,
your lyrics and the new style. Melody extraction is approximate: exact voice,
timing, instrumentation and arrangement are not guaranteed to match the recording.
It is not a voice-cloning or waveform-editing feature.

**Revision 9 loads SheetSage2, its pinned MERT parent and optional Whisper from
the attached saved dataset.** Setup installs only their small
support dependencies while preserving Kaggle's Torch/TorchAudio/scientific stack.
FFmpeg and the image-matched TorchAudio are checked. The pinned SheetSage2 API
normally defaults to BF16; this version explicitly loads/transcribes in FP32 for
T4. Whisper uses FP16 on T4. The upstream long-song transcription uses overlapping
windows; no manual five-minute crop is applied by this notebook.

Cover analysis and song generation share one GPU queue. Melody and ASR models
are released sequentially and after analysis, and a cached VAE is moved to CPU
before analysis. Generated-song work releases any remaining transcribers first.
The default workflow avoids a Whisper analysis when you already have
the words. Stop requests are observed at transcription progress boundaries.

Analysis files are under `/kaggle/working/yue2_native/cover_sources/<id>` with a
ZIP beside the folder. They include the source filename/checksum, model revisions,
score, lyrics, status and any upstream MIDI/event/window exports. The uploaded
waveform itself is not duplicated in this ZIP. Matching unchanged cover-score
provenance is copied into the generated run as `cover_source.json`. Save/download
the analysis ZIP and resulting song ZIP before ending the session. To reuse an
extracted melody without repeating transcription, paste its `score.abc` into
Custom ABC and select Melody score mode.

Nine new CPU cover-workflow tests pass for the visible upload controls/shared
queue, original words, optional ASR and resampling, partial failure/cancellation,
model release, analysis ZIP and generated-run provenance. The pinned remote
SheetSage2 class was also imported and constructed with a tiny random-weight
configuration on Python 3.13.5/Transformers 4.57.6, without downloading its model
weights. These checks do **not** validate full GPU transcription, fidelity,
long-song accuracy, runtime or memory peaks.

SheetSage2 revision: `eab522a8168e8b8b8c4856bf8609cd86198f01fe`.
Its config pins MERT parent `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` and the
remote loader checks parent code/weight hashes. Whisper revision:
`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. SheetSage2's upstream model license is
CC-BY-NC-4.0; consult its and MERT's model cards for applicable terms.

## Duplicate-engine and GPU-memory protection (revision 7)

The reported CUDA OOM showed approximately 7.03 GiB held by another process and
7.52 GiB held by the failing process, leaving only about 11 MiB free on GPU 0.
That is GPU-memory competition; the screenshot does not identify the other
process's application. It is not evidence that the audio model ran on a local PC.
The second T4's memory is separate and cannot cover GPU 0's allocations.

Before loading another model, setup and launch close only verified prior native
engines using the same exact script and output-root environment. Verification
includes UID, PID start time and boot identity, with PID handles where available.
Older revisions without an owner file are also found by their exact script and
output root. An original `YuE2Pipeline` still in this notebook kernel is closed,
along with its original UI. Other models and unverified processes are not killed.
A lifetime process lock prevents two managed engines loading concurrently.

A GPU-memory/process preflight runs before setup/launch and each generation.
It requires a conservative 12 GiB available/reusable on GPU 0 and 4 GiB on GPU 1;
this is a startup guard, not proof every long-song memory peak fits. The current
engine's resident allocation is counted as reusable for subsequent songs.
Unrecognized GPU users are reported and startup/generation stops early.
A CUDA OOM records `gpu-memory-at-error.json` and closes this failed engine's
models. These changes do not reduce quality, token ceilings or precision.

**For the currently failed attempt:** interrupt its generator/UI cell to stop
that failed process. Keep the saved run folder. Use the updated setup in the
same retained session before another generation; inspect any reported unknown
GPU process rather than retrying Generate. This update cannot clear memory in
your remote session until you execute it. No outputs or saved stages are deleted.
Decode recovery still requires completed `latent.npy` and its manifest; a failure
before those files exist cannot be recovered by the decode cell.

Eight real CPU child-process checks passed for surviving/legacy engines, lock
contention, PID identity, preserving unrelated processes/files and low-memory
startup guards. Two additional synthetic native tests cover pre-generation memory
failure and OOM diagnostics/model release. No Kaggle GPU generation was used.

## Natural endings and safety ceilings (revision 6)

The model should end the song from its prompt and lyrics; it does not need to
fill a selected length. A natural four-minute ending under a six-minute ceiling
is normal behavior. Hitting the ceiling without an end token is a different
condition: a cutoff that may interrupt the song.

The UI now calls this setting **Maximum song length (safety ceiling)**, with
**Up to 8 min** selected by default. Clear/reset and unspecified API duration
also use that ceiling. Unknown labels are rejected instead of silently selecting
a shorter budget. Earlier 1/2/3-minute ceilings remain available only when selected.
Natural end tokens remain enabled; minimum duration, padding and looping were
not added. Export status and `status.json` explicitly distinguish `model_end_token`
from `token_limit`; a cutoff is not labeled successful song completion.

## T4 attention candidate fix (revision 5)

The uploaded native run measured 715.42 seconds for 5,977 AR outputs (8.35 tokens/s),
about 277 seconds for 16 acoustic steps, and about 7.5 seconds for VAE decoding.
There was no recorded capture-failure message. It generated 5,976 music tokens,
representing 239.04 seconds, before a natural ending despite the six-minute maximum.

Source inspection identified a slow-kernel restriction: the release's CUDA
`enable_gqa=True` path cannot use memory-efficient attention. PyTorch 2.11 requires
SM80+ for FlashAttention/cuDNN SDPA, while T4 is SM75, leaving the math path.
This identifies a backend restriction; there is no per-kernel GPU profile here.

Revision 5 offers mathematically equivalent grouped-head views to ordinary SDPA,
which can use the memory-efficient kernel. It checks eligibility and a real first
kernel call, records confirmed selection, and retains native attention for
unsupported shapes. It changes neither weights/precision, token budgets nor
solver settings. No global Torch function is patched. Backend counters are
process totals, included in live reports and completed-run status.

**The performance benefit on T4 has not been measured.** Numerical equivalence
passed CPU checks for decode, CFG, causal prefill, acoustic chunks and masks.
Dispatch eligibility/cache/refusal tests use emulated CPU control flow, not CUDA.
Different attention kernels may change floating-point results and seeded songs.
Set `native_env["YUE2_NATIVE_ATTENTION"] = "native"` before launch to retain the
original attention implementation. See [PERFORMANCE.md](PERFORMANCE.md).

## Live progress correction (revision 4)

Earlier native UI progress used fixed stage markers: 25% meant semantic token
sampling had started, not that one quarter of the song had been generated.
Revision 4 replaces that marker with actual token counts, token rate, time to
first token, and prompt-token count. Acoustic synthesis and VAE decoding display
stage labels instead of invented overall percentages.

The inference package also labeled a surviving GraphAR object `cuda_graph` after
capture failed. Execution reporting now records actual capture success or eager
fallback and its reason. Live reports are saved to `semantic-progress.json`;
`semantic.partial.npy` checkpoints partial codec tokens every five seconds.
Partial files are diagnostic artifacts and cannot be used by decode recovery.

**These reporting changes do not establish a speed improvement.** They do not
update an already-running process. Diagnose slow runs using the existing engine
log's token rate, prompt/capacity, and capture/fallback lines before spending more
GPU time on a new run.

## Use it

1. Select **GPU T4 ×2** and enable **Internet** in Kaggle.
2. Run setup. It creates an isolated environment and preserves Kaggle's
   PyTorch/CUDA and scientific packages. It bypasses `ensurepip` using the
   previously checked pip bootstrap, and verifies native/UI imports.
3. Run **Open the generator**, then open the printed Gradio share link.
4. Enter your style and full, section-tagged lyrics. Leave **Up to 8 min** as
   the safety ceiling, or select a different ceiling, then click **Generate Song**.
5. Download the WAV/FLAC masters or **complete run ZIP** in the UI immediately,
   or **Save Version with outputs** in Kaggle before discarding the session.

**No diagnostic song starts automatically and no repeated short test is required.**
The default is an eight-minute safety ceiling, not a target duration. The model
chooses its ending from the prompt and lyrics and may naturally finish earlier.
A lower selected ceiling can cut the song off if reached.
Keep the UI cell running while generating. Its transformer can stay loaded for
additional songs. Interrupting the UI cell stops that process; the UI Stop button
requests cancellation at a generation boundary.

A browser/computer restart does not itself establish whether the remote Kaggle
session survived. If a session is discarded, unsaved working files may be lost.
A downloaded notebook containing logs is not a backup of its generated audio.

## Restored generation settings

- Transformer: original `m-a-p/YuE2-3B`, FP16 on GPU 0, retained between songs.
- AR backend: native CUDA graphs, SDPA on T4, with the original eager fallback.
  Actual execution and throughput appear in the status panel.
- Decoder: tiled FP32 `m-a-p/YuE2-Vae` on GPU 1.
- Default planning: **Direct to audio**, with no ABC score generation.
- Default acoustic solver: **16 Euler steps**; CFG **1.0**.
- Optional score planning/editing and upload-based cover transcription are available.
  Original lyrics are preserved unless optional lyric transcription is selected.
- The native context stays at 24,576 tokens. It is not enlarged to fit a song.

At 25 music tokens per second, seven minutes requests 10,500 tokens and eight
minutes requests 12,000. These are **maximum lengths**. The model can finish early,
or reach the limit with an abrupt ending. The status panel and `status.json`
report the actual duration and cutoff. Audio is never padded or looped to meet
an advertised length. Long songs may require multiple acoustic chunks; their
boundary times are reported for listening. Full-song continuity requires listening.

## Storage and saved stages

Runtime, dependencies, caches and model files use session-only
`/kaggle/temp/yue2_native`; support code is under `/kaggle/temp/yue2_native_support`.
Setup prints actual free space and requires at least 16 GiB for runtime and 2 GiB
for output storage. Paths can share a filesystem; these checks do not establish
a separate 50 GB quota. Existing files from older revisions are preserved.

Every attempt gets a unique directory under `/kaggle/working/yue2_native/runs`:

- `settings.json`: prompts, seeds, token budget, solver and engine/weight identity.
- `plan.json`, `plan_manifest.json`, `prefix.npy`, `abc_tokens.npy`, and optional
  `score.abc`: the exact saved plan, including direct generation's empty score.
- `semantic.npy`, `semantic.json`: completed music tokens, cutoff flag and AR timing.
- `latent.npy`, `latent_manifest.json`: completed acoustic latents and checksum,
  saved before VAE decoding.
- `audio.wav`, `audio.flac`: checked 48 kHz stereo PCM-24 masters.
- `playback.wav`: PCM-16 playback copy.
- `status.json`: measured duration, stage timings, chunk boundaries, or failure.
- `gpu-memory-at-error.json`: GPU/process snapshot if a CUDA OOM occurs.

Files appear as stages complete. Failed/cancelled attempts retain earlier files.
Non-finite acoustic/audio values cause an explicit failure instead of being
silently replaced. Exports are checked against the generated token duration and
read back through their actual file headers. Successful runs also get a ZIP
beside the directory; if space prevents ZIP creation, original files remain.
The UI provides direct master-file and ZIP downloads. Package/hardware records
and the engine log are retained under `/kaggle/working/yue2_native/runtime`.

## Decode recovery

If a run failed after `latent.npy` and `latent_manifest.json` were written, stop
the UI cell and use the notebook's optional decode-recovery cell. It verifies
latent checksums and matching model weights, then writes into a new run folder.
It does not rerun planning, music-token generation or acoustic synthesis.

For a new Kaggle session, attach the saved ZIP as an input and extract it first;
setup/model assets are still needed. Revision 2 ComfyUI artifacts are a different
format and cannot be fed to this native recovery cell. Token-stage recovery after
an acoustic failure remains a future addition; those tokens are saved for review.

## Revisions and checks

- Inference wheel: `yue2_infer-0.1.5-py3-none-any.whl`, SHA256
  `8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641`.
- Transformer revision: `8f24312f187763b9854adeec87e437e2b03bbff1`.
- VAE revision: `152733a19ad43aa67e367f9b5503ef8075bb5126`.
- Key compatibility pins: Transformers 4.57.6, Hub 0.36.2, Gradio 5.49.1,
  Pillow 11.3.0, mir_eval 0.8.2, pretty_midi 0.2.10 and mido 1.3.3. Full installed versions are recorded; this is not an entire
  Kaggle image lock. The inference wheel is installed without replacing the
  release's development CUDA stack over Kaggle's matched binary packages.
- Original source: AIQUEST Academy notebook in Movies commit
  `fcf4d45dc5dbca57b39e54311807410cdf41f737`. Attribution is retained in the source.
  Upstream model, inference-package and component license terms apply.

Thirty-nine CPU checks passed on Python 3.13.5 and PyTorch 2.11.0 CPU, using the real
native package, Gradio 5.49.1 UI and HTTP API schema. Checks cover long token/context
budgets, eight-minute WAV/FLAC/ZIP exports with **synthetic audio**, early endings,
unique run preservation, planner cutoff, non-finite failure handling, decode
recovery/tampering, live token checkpoints, truthful capture-fallback labels,
real CPU process ownership/locking and synthetic GPU-memory guard/OOM handling,
cover controls, lyric preservation/ASR paths, analysis cleanup/ZIP and provenance,
equivalent grouped attention and its eligibility/cache/fallback control,
and embedded source consistency. They did not run the full
model or validate CUDA graphs, eight-minute GPU memory, speed, or musical quality.
No Kaggle GPU time was used for these local checks.

Source is in `native_yue2/`; `tools/build_fast_notebook.py` rebuilds the standalone
notebook. `kaggle_yue2/` supplies the existing checked environment-bootstrap helpers;
its ComfyUI installer and generation runtime are not invoked by this notebook.
For contributors with the native dependencies installed:
`python -m unittest discover -s tests -p test_native.py -v` and
`python -m unittest discover -s tests -p test_attention.py -v` and
`python -m unittest discover -s tests -p test_process.py -v`.
