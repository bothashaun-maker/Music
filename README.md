# YuE2 music generation on Kaggle — native FP16 engine

[Download the notebook](yue2-3b-frontier-full-song-music-generation.ipynb) using
GitHub's **Download raw file**, then import it into Kaggle. Look for
**restored native FP16 engine (revision 6)** at the top.

This revision returns to the original notebook's `yue2_infer` engine. The previous
ComfyUI implementation changed the runtime and defaults and was substantially
slower in the uploaded Kaggle run. The native version retains the original FP16 and
CUDA-graph path, restores direct generation, and adds seven/eight-minute options.
The original engine's reported three-minute speed is not a measured speed claim
for this revision or for eight-minute songs.

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
- Optional score planning/editing remains available; cover transcription is disabled.
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
  Pillow 11.3.0. Full installed versions are recorded; this is not an entire
  Kaggle image lock. The inference wheel is installed without replacing the
  release's development CUDA stack over Kaggle's matched binary packages.
- Original source: AIQUEST Academy notebook in Movies commit
  `fcf4d45dc5dbca57b39e54311807410cdf41f737`. Attribution is retained in the source.
  Upstream model, inference-package and component license terms apply.

Twenty CPU checks passed on Python 3.13.5 and PyTorch 2.11.0 CPU, using the real
native package, Gradio 5.49.1 UI and HTTP API schema. Checks cover long token/context
budgets, eight-minute WAV/FLAC/ZIP exports with **synthetic audio**, early endings,
unique run preservation, planner cutoff, non-finite failure handling, decode
recovery/tampering, live token checkpoints, truthful capture-fallback labels,
equivalent grouped attention and its eligibility/cache/fallback control,
and embedded source consistency. They did not run the full
model or validate CUDA graphs, eight-minute GPU memory, speed, or musical quality.
No Kaggle GPU time was used for these local checks.

Source is in `native_yue2/`; `tools/build_fast_notebook.py` rebuilds the standalone
notebook. `kaggle_yue2/` supplies the existing checked environment-bootstrap helpers;
its ComfyUI installer and generation runtime are not invoked by this notebook.
For contributors with the native dependencies installed:
`python -m unittest discover -s tests -p test_native.py -v` and
`python -m unittest discover -s tests -p test_attention.py -v`.
