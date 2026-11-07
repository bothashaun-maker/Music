# Revision 15 repair context

This directory is based on the supplied revision 14 bundle, rather than the
older root-level implementation. Concrete fixes and verification limits live
in REPAIR_REPORT.md. Run tools/build_release.py to rebuild both delivered
notebooks and the ZIP. Keep native FP16 generation and existing datasets.
The prior branch notes below are retained for continuity; their historical
validation claims do not establish a new live GPU or Cloudflare run.

Publish the music creation notebook separately; exclude it from the release ZIP.
The user confirmed full-song creation works. Preserve its graph and engine.
Cover uploads now add a resumable button to the existing LoadAudio node via
native_yue2/cover_upload{,_web}.py; preserve the original upload control too.

Studio follow-up: real Kaggle logs showed speech VAD removed all music and
CTranslate2 could not load libcublas.so.12 despite PyTorch CUDA 13 seeing T4s.
Keep music recognition VAD off, report CPU INT8 fallback for CUDA library errors,
stop zero-match/empty recognition before export, retain heard-word diagnostics,
and stream logs to the notebook plus /kaggle/working/timestamp_studio/logs/.
Never describe matched-word percentage as a measured timing accuracy bound.
REAPER export follow-up: derive section regions from build_sections, as chapters
do, so an opening Intro begins at project zero even if vocals begin later.
Never shift lyric marker onsets just to repair the Intro boundary. REAPER's
Start column can display bars/beats; verify units before claiming a seconds offset.

Studio revision 16 bypasses fuzzy transcription matching for supplied English
lyrics. timestamp_studio/acoustic.py runs a pinned Wav2Vec2 CTC acoustic model,
bounded inference windows and a full-target Viterbi path with blank states for
gaps. Model/frame caches live in working outputs and can be reattached. Automatic
transcription remains Whisper-based. Do not present acoustic scores or synthetic
speech results as verified accuracy on the user's recording. The full WAV upload
could not be transferred here because of the 32 MiB tool limit. Music creation
must remain unchanged. See STUDIO_ACOUSTIC_REPAIR.md for current limits.

Revision 16.1 fixes an installed-bundle failure: the hardcoded installer omitted
acoustic.py despite embedding it. Copy the complete payload, validate required
files, and load the adjacent acoustic helper by file path. Preserve the exact
ComfyUI path-named package regression in tests/test_studio_installed_bundle.py;
tests of source imports alone do not cover the deployed node layout.

# User preferences and continuity

- Preserve working behavior. Prefer scoped, additive improvements; do not remove
  working access routes or change model/generation defaults incidentally.
- User explicitly wants original WAV uploads preserved, without requiring file
  conversion. Cloudflare is additional to the existing Gradio share route.
- Existing Kaggle asset datasets must remain usable for this access-only update.
- Publish authorized notebook changes to bothashaun-maker/Music, including sources
  and regenerated notebooks, after relevant local checks. Report GPU/network
  validation limits honestly. Do not push music changes to Movies.
- Source: native_yue2/; builder: tools/build_fast_notebook.py.
- Give concise instructions, one step at a time when guiding the user's screen.

## Movies-style delivery for ALL music workflows (2026-11-03)

- The user instruction: "I want all my music workflows to work exactly the same
  as the movies. The music creation and the time stamp studio." They call the
  old multi-cell layout "the AIQUEST wrapper" and reject it.
- Movies style means: ONE executable cell → setup → weight verification →
  persistent ComfyUI → Cloudflare link that opens the full node graph directly →
  canvases pre-installed in the Workflows menu → the user presses Run. No Gradio
  front door, no separate launcher/repair cells, no auto-render, no download
  prompts (outputs via Kaggle Output panel).
- Revision 12 = the YuE2 notebook rebuilt this way (GRADIO_UI=True keeps the
  original generator reachable inside the single cell; tests assert the shape).
- Song Timestamp Studio is delivered the same way: the Gradio web app is
  superseded by `Song-Timestamp-Studio-Canvas.ipynb` (TimestampStudio node pack:
  STSLyricsOptions / STSAlign / STSNudge / STSAudition / STSExportAll over the
  same tested engine). Sources in `timestamp_studio/`; builder
  `tools/build_studio_notebook.py`; checks `tests/test_studio.py`.
- Validated: 77/77 repo checks (incl. 5 STS structure/format tests); the STS
  canvas was queued end-to-end through real pinned ComfyUI on the known-truth
  song (per-line error max 0.46 s, identical to the standalone engine; nudge
  exact); browser audit auto-loaded the 7-node graph from ?sts_workflow=studio
  with no page errors. Not yet run on Kaggle GPUs by the user.

## ComfyUI canvas wrapper (2026-10-30)

- The user's YuE2 asset dataset is built, working, and must not be rebuilt for
  wrapper work. Model weights always come from the dataset ("as-is").
- The user asked for the Cloudflare link to open a normal ComfyUI canvas with
  the YuE2 workflow — nodes, models, everything — exactly like the Movies H3
  canvases: persistent ComfyUI, full visible node graph, workflows pre-installed
  in the menu, nothing collapsed into one wrapper node, no auto-render.
- Decision (user-selected): wrap the proven native FP16 engine as real
  ComfyUI nodes (KaggleYuE2 pack), NOT the official ComfyUI YuE2 nodes; keep
  the Gradio wrapper as a second route; canvases = Full Song + Cover/Restyle.
- Both wrappers share one engine lock; each hands over to the other
  (verified stop / /yue2/release). Same saved-run artifact schema.
- ComfyUI pinned to v0.43.2 (423cbd20511becb1f16122ab518e92e58bfa03d9),
  session-only, requirements under the Kaggle constraints file.
- Revision 11 = this canvas wrapper. Delivery of revision 11 was a full-repo
  ZIP for the user to upload via GitHub's web UI (no push credentials).

## Wheelhouse self-heal (2026-11-04, revision 13)

- Real Kaggle failure: with assets attached, `PIP_NO_INDEX=1` made the ComfyUI
  v0.43.2 requirements install crash on `comfyui-frontend-package==1.57.4`
  (plus workflow-templates/embedded-docs/comfy-aimdo/comfy-kitchen) — wheels
  the YuE2 wheelhouse predates. CalledProcessError in install_comfyui.
- Fix: `kaggle_yue2.runtime.pip_install` — offline wheelhouse attempt first,
  one automatic PyPI retry (PIP_INDEX_URL set, PIP_NO_INDEX dropped, wheelhouse
  kept on find-links, Kaggle constraints kept in args), actionable RuntimeError
  if both fail. Used by the overlay install, install_comfyui (wheelhouse from
  env YUE2_ASSET_DATASET), and the legacy route install. The asset dataset is
  NOT rebuilt and stays fully usable; models remain offline (HF_HUB_OFFLINE).
- Real-CLI proof: sandbox test reproduced the exact "No matching distribution
  found for comfyui-frontend-package==1.57.4" offline wall, then pip_install
  installed that wheel from PyPI; a complete wheelhouse still installs fully
  offline. Unit tests: tests/test_assets.py::PipFallbackTests.
