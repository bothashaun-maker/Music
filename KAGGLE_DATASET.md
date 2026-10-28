# Save YuE2 assets once on Kaggle

The generator uses a saved input bundle for every session. Model downloads already
happen on Kaggle's connection; they should not be repeated each time a session
restarts. You do not need to download the model files to your computer.

## First time: save the assets

1. Import [yue2-build-kaggle-assets.ipynb](yue2-build-kaggle-assets.ipynb) into Kaggle
   using GitHub's **Download raw file**. Keep the notebook private.
2. Enable Internet and select accelerator **None**. Choose **Save Version →
   Save & Run All**, with outputs enabled, to build and save in one run. This
   fetches assets without creating a song or loading models onto a GPU. You do
   not need to run the build interactively first and download it twice.
3. Wait for **COMPLETE** in the saved run's logs. The finished version must retain
   the `yue2-assets` directory, including `assets.json`; a partial folder or a
   draft containing only logs is not a valid bundle. If you prefer an interactive
   build, publish its dataset with the optional cell before ending that session;
   starting a fresh Save & Run All session would fetch assets again.
4. You can use that version immediately as persistent Kaggle storage: in your
   music notebook use **Add Input → Notebook Output Files** and select it.
   No browser download of weights is needed.

To create a specifically named **private Dataset**, run the builder's optional
publish cell in Kaggle with `DATASET_HANDLE = "your-username/yue2-music-assets"`.
It uses KaggleHub's dataset upload API directly from the Kaggle machine. Your
Kaggle authentication is required; keep tokens in private Kaggle authentication
or Secrets, never in notebook source. Select a new unused slug. The API creates
new datasets privately; updating an existing dataset retains that dataset's
existing visibility. Attach the result with **Add Input → Your Datasets**.
If your image has an older KaggleHub without `dataset_upload`, use saved notebook
outputs or update KaggleHub in the builder session. No credentials or remote
Kaggle session were available to this engineer to create that dataset for you.

## Every music session

1. Import the updated [music notebook](yue2-3b-frontier-full-song-music-generation.ipynb),
   revision **9**. Attach your saved dataset or builder output using **Add Input**.
2. Select **GPU T4 ×2**. Enable Internet for the existing Gradio share link.
3. Run setup. It prints **Using saved Kaggle assets**. Open the generator and
   create or cover your song as before. There is no test song.
4. Preserve generated songs with Save Version including outputs, or download
   only the audio/run ZIP you want. The asset dataset does not save new songs.

`ASSET_DATASET = "auto"` discovers the single `assets.json` bundle anywhere under
`/kaggle/input`, including newer nested mount layouts. If multiple versions are
attached, set it to the exact folder containing the desired `assets.json`.
If none is attached, setup stops. It never silently redownloads missing assets.
Only the separate builder is intended for online asset acquisition.

## Contents and storage

- YuE2-3B and YuE2-VAE, at the generator's pinned revisions.
- SheetSage2 and its pinned MERT-v2-FullSong parent for covers.
- Whisper large-v3-turbo for optional lyric transcription.
- Exact locally installed overlay-package wheels, verified bootstrap pip, and
  the verified Gradio Linux share executable.
- Kaggle image identity, overlay requirements, original model cards/licenses,
  file sizes/checksums and completion manifest.

The builder preserves HF `models--*/snapshots/<revision>` paths. Snapshot files
are ordinary files, not symlinks that could break during Kaggle output packaging.
It removes duplicate Hub blob storage without allocating a second weight copy.
The generator reads weights from the input mount. A small writable environment,
remote-code import cache and share executable live in `/kaggle/temp`; generated
songs remain in `/kaggle/working/yue2_native`.

The model cache and pip index are offline during reuse even though Internet is
on for the UI. Missing cached files produce errors. Small manifests, wheels,
code and the executable are hashed at setup. Before downloading models, the
builder checks that its saved wheelhouse resolves dependencies in a fresh
overlay with the pip index disabled. Native music/VAE loaders separately
check model-weight hashes; MERT's upstream loader checks parent hashes. Call
`validate_assets(path, full=True)` for a complete dataset hash check without
using a GPU, including optional model files not loaded by the current request.

## Compatibility and limits

The overlay inherits Kaggle's existing Torch/CUDA/scientific stack. It does not
ship another multi-gigabyte Torch/CUDA installation or transplant a whole virtual
environment with broken absolute paths. Builder and generator must have the same
Python minor/platform and public versions of Torch, TorchAudio, TorchVision,
NumPy, SciPy, Numba and llvmlite. CPU/CUDA local Torch build suffixes are ignored
so a matching CPU builder can serve a GPU session. The generator still checks
real CUDA/T4 availability. Other inherited dependency changes can also cause an
offline pip/import failure; these do not trigger downloads.

If CPU and GPU Kaggle images differ, choose matching image versions where
available, or run the builder in the same GPU image once. That uses accelerator
session time but performs no GPU inference. Rebuild the bundle when moving to an
incompatible image or deliberately updating models/packages. Normal reuse needs
local dependency installation and model loading time, not another network fetch.
A small share-link connection, page assets, source-song uploads and your final
song downloads still use network traffic.

CPU checks cover nested input discovery, missing/ambiguous bundles, corruption,
image mismatch, offline setup arguments/environment, read-only input reuse,
shared-blob materialization, complete/partial builds and exact source embedding
in both notebooks. A fresh-process check of the real Hub 0.36.2 resolves
synthetic cached snapshots and files with socket connections blocked and no
cache mutations. Synthetic model files are used; no full dataset download,
Kaggle upload or GPU generation was run in this development environment.
