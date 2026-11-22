# Music Studio — Model Pack Builder

This folder contains the Kaggle notebook that builds and publishes the **private model
dataset** required by the Music Studio / Stem Splitter ComfyUI canvas.

The notebook downloads every model listed in [`NOTICES.md`](NOTICES.md) directly from
Hugging Face into Kaggle working storage, verifies each file with SHA256, assembles the
standard `ComfyUI/models/` directory tree, writes a `manifest.json`, and publishes the
whole thing as a **private** Kaggle Dataset under the configured account.

No model data passes through a local PC. Everything happens inside the Kaggle notebook.

---

## Files

| File | Purpose |
|---|---|
| `Music_Studio_Model_Pack_Builder.ipynb` | Downloads, verifies, and publishes the model pack |
| `NOTICES.md` | Source list, original licenses, and attribution for every component |

---

## Requirements

- A Kaggle account with **Internet ON** and **Accelerator: None** for the builder notebook.
- `kagglehub` (installed automatically inside the notebook if missing).
- The target Kaggle Dataset must be **private** (default in the notebook).
- If the notebook hits an authentication error, enable the private Kaggle Secret
  **`KAGGLE_API_TOKEN`** in *Add-ons → Secrets*. Never paste tokens into code or chat.

---

## Workflow

### 1. Build the pack (once, or when models change)

1. Open `Music_Studio_Model_Pack_Builder.ipynb` on Kaggle.
2. Set **Accelerator: None**, **Internet: ON**.
3. Run **Cell 1** (builder). It downloads ~2–3 GiB and prints per-file status.
4. Run **Cell 2** (publisher). It uploads to the private dataset handle configured at the
   top of Cell 1 (`DATASET_HANDLE`, default `janpierrrewiet/music-studio-model-pack`).
5. Open the dataset link Kaggle prints and wait for processing to finish.

**Idempotent:** re-running Cell 1 skips any file that already exists and only re-downloads
what's missing or failed. Safe to re-run after a transient network error.

### 2. Attach the pack to ComfyUI

1. Open the ComfyUI notebook (`Stem_Splitter_ComfyUI_Kaggle.ipynb` or equivalent).
2. **Add Input → Your Datasets →** select the model pack.
3. Set **Accelerator: T4 ×2** (or your preferred GPU) and run startup.

ComfyUI scans `/kaggle/input/music-studio-model-pack/ComfyUI/models/` automatically and
picks up every model without extra configuration.

---

## What gets downloaded

| Component | ComfyUI subfolder | Source |
|---|---|---|
| YuE2-3B | `diffusion_models/` | `m-a-p/YuE2-3B` |
| YuE2-Vae | `vae/` | `m-a-p/YuE2-Vae` |
| SheetSage2 | `checkpoints/` | `m-a-p/SheetSage2` |
| Mothersuperior tokenizer | `checkpoints/` | `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` |
| MuScriptor medium | `checkpoints/` | `cocktailpeanut/muscriptor-medium` |
| Gemma 4 E4B (QAT q4_0 GGUF) | `text_encoders/` | `google/gemma-4-E4B-it-qat-q4_0-gguf` |
| Parakeet TDT 0.6B v3 | `checkpoints/` | `nvidia/parakeet-tdt-0.6b-v3` |
| Demucs 6-stem (ONNX) | `audio_separator/` | `StemSplitio/htdemucs-6s-onnx` |
| MOSS-Music 8B Instruct | `checkpoints/` | `OpenMOSS-Team/MOSS-Music-8B-Instruct` |

The canonical list is `MODEL_SPECS` at the top of Cell 1. To add or swap a model, edit
that list — the rest of the notebook adapts automatically.

---

## Outputs

On Kaggle working storage (`/kaggle/working/music-studio-pack/`):
