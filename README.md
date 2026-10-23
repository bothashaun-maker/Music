# YuE2 music generation on Kaggle

Download [yue2-3b-frontier-full-song-music-generation.ipynb](yue2-3b-frontier-full-song-music-generation.ipynb) using GitHub’s **Download raw file** button, then import it into a new Kaggle notebook. It is
self-contained: no repository clone or local GPU is needed on your computer.
Select **GPU T4 ×2** and enable Internet. Run the notebook from the top.

Start with the default **30-second diagnostic**. After successful audio generation,
change `MAX_DURATION_SECONDS` to `180`, `420`, or `480` and supply full lyrics.
These values are maximum lengths: the model may end early or hit a cutoff.
Seven/eight-minute quality, continuity, runtime and T4 memory usage are **not yet
GPU-validated**. The notebook does not pad or loop audio to meet a target.

The initial implementation uses native ComfyUI with the official INT8 checkpoint.
T4 text generation uses native FP32, with staged offloading and tiled FP32 VAE
decoding. It uses the explicitly selected GPU; two cards are not pooled or claimed
to be accelerated together. Cover transcription is not included in this initial
text-to-song baseline. No public tunnel or Gradio server is required.

## What is saved

Each attempt gets a unique folder under `/kaggle/working/yue2/runs/`:

- `request.json`, `api-prompt.json`: prompts, three seeds, settings and runtime identity.
- `score.abc`, `abc_tokens.json`: score and exact token artifact when planning is used.
- `semantic_tokens.json`: music tokens and native ending/truncation status.
- `*_effective.json`: actual token budgets, prefixes and sampling arguments.
- `latents.safetensors`, `latents_manifest.json`: checked acoustic latents before decoding.
- `audio.flac`, `audio.wav`: 48 kHz stereo PCM-24 exports after finite/silence checks.
- `stages.json`, `summary.json`: actual duration, cutoff state, acoustic boundaries,
  observed dtype, GPU memory and any export gain.
- `history.json`, `client-error.json`, `gpu.csv`, `server-log.txt`: execution diagnostics
  when available (the per-run server log retains up to the final 20 MiB).

The runtime root also retains package versions, hardware, node schemas, and
`server.log`. A file only appears after its stage completes. Failed/cancelled runs
retain earlier artifacts. A header check does not establish musical completeness;
listen to the full song, especially the outro and reported chunk boundaries.

**Download the run ZIP or save a Kaggle version with outputs before stopping your
session.** A working directory alone is not durable across all Kaggle session resets.

## Recovery

Use the notebook's recovery cell with the directory containing `request.json`:

- `decode`: reuse saved latents; optionally reduce the VAE tile size.
- `tokens`: reuse matching completed token stages and recompute acoustic synthesis.

Recovery writes into a new run folder and checks identities/checksums. Token
recovery requires the same effective prompt, seeds, budgets, package identity and
checkpoint. It intentionally refuses to reuse mismatched or corrupt artifacts.
Decode recovery verifies the checkpoint and latent checksum. For a new Kaggle
session, attach saved outputs as an input or extract the saved ZIP into working storage.

Interrupting the generation cell asks the managed server to cancel at a native
processing boundary. It does not promise immediate interruption of an active CUDA
kernel. `app.stop()` terminates only this notebook's managed server. Wait for idle
or restart before submitting another run.

## Reproducibility and limitations

- ComfyUI: `bc5f880dbdd43e426077f42ccec53bb577ffee9f`.
- Weights: `Comfy-Org/YuE2` revision `2f76ca75e6ee094169de899cc7fc99d6887e2196`.
- Checkpoint: `yue2_3b_int8_convrot.safetensors`, SHA256
  `96fe199377309001ed8cd26a944baeee8cc31a20ba7c36d1d3c0a7e1f4149db6`.
- Installation uses an isolated virtual environment inheriting Kaggle's binary
  stack. Constraints preserve PyTorch/TorchVision/TorchAudio and the scientific
  packages already installed. Transformers, Hub and SoundFile are pinned; other
  dependencies follow the pinned ComfyUI requirements. Installed versions are
  recorded; this is not a complete cross-image dependency lock.
- Use a current Kaggle image with PyTorch >=2.8. Dependency/import/kernel failures
  stop setup and need diagnosis; verification is never disabled.
- Native ComfyUI controls precision, sampling, attention, acoustic chunking and
  its decoder's loudness adjustment. The managed nodes observe ending status,
  save recovery artifacts and validate exports. No numerical failure is replaced
  with zero-filled audio.
- Seven/eight-minute songs can require separate acoustic chunks. Their boundaries
  are reported for listening; a larger duration setting does not establish continuity.
- Models and generated outputs remain subject to their upstream license terms.

## Validation status

Before publication, 25 local tests passed for orchestration, graph wiring,
truncation reporting, recovery, tamper detection, and API error/cancellation handling.
Checks against the pinned ComfyUI source also passed. These checks used no GPU
and do not establish successful model inference or audio quality.

The next validation must happen in Kaggle: installation, the default short
generation, rerun, decode recovery, three minutes, then seven/eight minutes
with memory measurements and listening checks at the outro and chunk boundaries.
