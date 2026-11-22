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

## Configuration

Two flags at the top of **Cell 1** control the build:

| Flag | Default | Meaning |
|---|---|---|
| `INCLUDE_MOSS` | `True` | Include the ~16–18 GiB MOSS-Music 8B Instruct shards. Set `False` to skip. |
| `SEED_FROM` | `/kaggle/input/music-studio-model-pack` | If the previous dataset is attached as an input, files are copied from here instead of re-downloaded. Empty path disables seeding. |

`DATASET_HANDLE` (default `janpierrrewiet/music-studio-model-pack`) is the dataset the
builder publishes to. Publishing to an existing handle **creates a new version** — same
URL, existing attachments keep working. **Never delete the published dataset** to update it.

---

## Workflow

### 1. Build or update the pack

1. Open `Music_Studio_Model_Pack_Builder.ipynb` on Kaggle.
2. Set **Accelerator: None**, **Internet: ON**.
3. *(Recommended when updating)* **Add Input → Datasets →** attach the existing
   `janpierrrewiet/music-studio-model-pack`. This lets the builder seed from
   `/kaggle/input/…` instead of re-downloading ~16 GiB.
4. Run **Cell 1** (builder). It prints per-file status: `✓ Already present`,
   `⤵ seeded from previous dataset`, `↓ Downloading`, or `✗ FAILED`.
5. Run **Cell 2** (publisher). It uploads as a **new version** of the same dataset.
6. Open the dataset link Kaggle prints and wait for processing to finish.

**Idempotent:** re-running Cell 1 skips any file that already exists (or can be seeded)
and only fetches what's missing or previously failed. Safe to re-run after a transient
network error.

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
| YuE2-3B | `diffusion_models/YuE2-3B/` | `m-a-p/YuE2-3B` |
| YuE2-Vae | `vae/YuE2-Vae/` | `m-a-p/YuE2-Vae` |
| SheetSage2 | `checkpoints/SheetSage2/` | `m-a-p/SheetSage2` |
| Mothersuperior tokenizer v4 | `checkpoints/YuE2-Mothersuperior/` | `drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui` (safetensors conversion of `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`) |
| MuScriptor medium | `checkpoints/MuScriptor/` | `cocktailpeanut/muscriptor-medium` |
| Gemma 4 E4B (QAT q4_0 GGUF) | `text_encoders/` | `google/gemma-4-E4B-it-qat-q4_0-gguf` |
| Parakeet TDT 0.6B v3 | `checkpoints/parakeet-tdt-0.6b-v3/` | `nvidia/parakeet-tdt-0.6b-v3` |
| Demucs 6-stem (ONNX) | `audio_separator/` | `StemSplitio/htdemucs-6s-onnx` |
| MOSS-Music 8B Instruct | `checkpoints/MOSS-Music-8B-Instruct/` | `OpenMOSS-Team/MOSS-Music-8B-Instruct` (4 bf16 shards + index + config) |

The canonical list is `MODEL_SPECS` at the top of Cell 1. To add or swap a model, edit
that list — the rest of the notebook adapts automatically.

**Note on Mothersuperior:** the upstream repo only publishes `.pt` pickle files, which
ComfyUI cannot load safely. The pack uses a verbatim safetensors conversion (repo
`drbaph/yue2-mothersuperior-realaudio-tokenizer-comfyui`) pinned to the same upstream
revision. `NOTICES.md` cites the canonical Mothersuperior source; the conversion is
noted as the delivery mechanism.

---

## Outputs

On Kaggle working storage (`/kaggle/working/music-studio-pack/`):
