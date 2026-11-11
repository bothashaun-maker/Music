# Music Studio v2 validation

Checked 11 November 2026. The user requested that extended verification stop
to conserve credits. Unfinished builds were stopped; they are not reported as passed.

## Passed

- Patched official Studio 3.8.5 browser production build and TypeScript check.
- 15 packaging/recovery tests, including the normal private Kaggle create/upload
  methods with mocked network, interrupted copy/commit recovery, preservation of
  new work, tunnel registration gating, and real Linux parent-death cleanup.
- Native CPU ace-caption and Linux VST3 host built, staged and loaded. The VST3
  host scanned successfully with no plugins installed. Actual plugin processing
  and MOSS inference were not run.
- Pinned audio.cpp and RoFormer CPU archives verified and loaded. Real audio.cpp
  HT-Demucs inference on a synthetic one-second recording returned all six stems
  at 44.1 kHz stereo. This was a direct helper check, not a completed integrated
  Studio browser/inference test. RoFormer inference was not run.
- The pinned Studio source patch applies to the clean source index and is
  idempotent against the patched checkout.
- Both notebooks regenerate as independent one-cell notebooks with embedded
  support and no saved outputs. Python syntax checks passed.
- The 27-page PDF rebuilt in the repository's existing format; page bounds,
  links/bookmarks and sample pages were inspected.

## Not completed

The new Rust service release build and full CUDA engine/trainer builds were
stopped before completion. The local build exposed a desktop-only Wayland
dependency; the patch removes the native file-dialog dependency and uses
browser uploads or explicit Kaggle folder paths. That final Rust build has
not completed. The updated browser integration test is supplied but was not
run against a completed new service. Previous v1 compilation/browser results
do not establish that v2 passes.

No live Kaggle credentials or GPU were available. Authenticated Dataset creation,
Kaggle processing, CUDA generation, SheetSage2/MuScriptor/Gemma/MOSS inference,
T4 training, VRAM use and live Cloudflare access remain unverified. The official
3.8.5 source includes the Turing training fallback; this is source support,
not proof of a successful training run here.

## Scope

Music Studio v2 pins official 3.8.5, updates its audio.cpp/model dependencies,
and removes recognition controls, routes, MCP actions and weights. Musical
score/MIDI transcription and external LRC import remain. The launcher waits
for tunnel registration, binds to its notebook parent, recovers interrupted
library restores, and persists MIDI/training/plugin work. Existing Timestamp,
FP16 and ComfyUI notebooks are unchanged.

Rebuild the separate Music Studio Dataset with Music-Studio-Dataset.ipynb
before using the updated Music-Studio.ipynb. A v1 pack does not match this
new source/model fingerprint. Generated songs still need Kaggle output saves.

Reproduce the inexpensive checks:

```bash
python tools/build_music_studio_notebooks.py
python tools/build_music_studio_pdf.py
python -m unittest discover -s tests/music_studio -v
```

Source pins and runtime hashes: [sources.json](../../music_studio/sources.json).
Model URLs, sizes and hashes: [catalog.json](../../music_studio/catalog.json).
