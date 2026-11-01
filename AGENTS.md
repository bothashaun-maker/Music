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
