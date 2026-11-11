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

- Revision 12 uses one independent startup cell and native_yue2/launch.py. Keep
  file-list and optional-recovery controls usable from empty notebook globals.
  Preserve startup.log and startup-status.json for diagnosing interrupted setup.

- Latest user request: the standard ComfyUI canvas, with no AIQUEST Gradio wrapper.
  Use YuE2_ComfyUI_Kaggle.ipynb and tools/build_comfy_notebook.py for that interface.
  Preserve the existing native FP16 engine and current model dataset; do not switch
  back to the earlier Comfy INT8/FP32 generation runtime. The old UI remains only
  as a fallback. Native stage nodes live in comfy_yue2/; never flatten the graph
  into one wrapper node. A headless guard skips Gradio construction in this mode.

- Latest task (2026-11-04): repair the supplied yue2-music-revision-14.zip.
  The repaired branch is in revision15/ so the prior root implementation remains
  available. Current delivery: YuE2_Music_Revision15.ipynb and
  Song-Timestamp-Studio-Canvas.ipynb. Rebuild via
  python revision15/tools/build_release.py; audit: revision15/REPAIR_REPORT.md.
  Use that branch's sources/tests for future revision-15 fixes, preserve the
  existing asset schema/native engine, and distinguish local/CPU checks from
  live Kaggle or Cloudflare runs. No new asset dataset is required by this repair.

- 2026-11-07: after being told this repository is public and the project ZIP
  includes project data/lyrics, the user explicitly authorized publishing the
  exact `Studio17_REAPER_Repair.zip` here because chat downloads were blocked.
  This permission covers that archive only, not other private projects or audio.


## Music Studio browser notebook (11 November 2026)

- User explicitly requested a separate Music-Studio notebook, a separate automatic
  save Dataset notebook named Music Studio Dataset, a PDF covering all tools in
  the format of the existing music PDF, and publication to this Music repository.
- Deliverables: Music-Studio.ipynb, Music-Studio-Dataset.ipynb,
  Music-Studio-Guide.pdf. Sources: music_studio/. Rebuild with
  tools/build_music_studio_notebooks.py and tools/build_music_studio_pdf.py.
  Editable PDF content: docs/music-studio/guide.json. Match the existing dark,
  coral/cyan, numbered-section PDF design; keep the original PDF unchanged.
- Preserve one private model Dataset. Use the existing working native direct
  dataset_create_new(public=False) method; no custom datasets.get/status gate,
  no manual upload/save cell, no overwrite/version of an existing Dataset.
  Upload one verified file at a time and release only owned temporary downloads.
- This is an additional full Linux browser interface, not a replacement for the
  native FP16/ComfyUI or corrected Timestamp Studio branches. Do not change their
  engines, datasets, graphs or defaults during Music Studio work.
- Pin sources/models/toolchain in music_studio/sources.json and catalog.json.
  CPU builder compiles x86-64 CUDA 13 SM75 runtime once; normal GPU startup uses
  the saved runtime, verifies weights, checks CUDA readiness and opens an
  authenticated Cloudflare link. Never auto-generate a diagnostic song.
- Keep library/media under /kaggle/working/music_studio; model links/runtime are
  temporary so output saves do not duplicate all weights. Preserve originals,
  resumable upload checks, exact byte checks and idempotent imports.
- Preserve native resumable WAV/MP3 upload and YUE_LIBRARY_ROOT patches. Startup
  may restore one saved library only into empty working output; never overwrite
  current work. Keep private user audio, lyrics and projects out of GitHub.
- The Linux fork has a different feature set from newer Windows Studio. Do not
  claim T4 LoRA training, Windows VSTs, AB2/chord-variety or newer continuation
  controls are provisioned. Explain actual tool/model license boundaries.
- Distinguish local compilation/browser/CPU checks and mocked Dataset uploads
  from live Kaggle GPU inference or authenticated Dataset creation. See
  docs/music-studio/VALIDATION.md for evidence and remaining limits.
