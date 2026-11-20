# Music Studio v1 validation

Checked on 11 November 2026. This is an experimental Kaggle integration of the
pinned Linux browser fork. Local build and functional results below do not
constitute a live Kaggle GPU run.

## Results

| Check | Result and scope |
| --- | --- |
| Rust service | Patched `music-server` compiled successfully in release mode on Linux x86-64, Rust 1.99.1, with the upstream lockfile. |
| Browser application | The patched Vite production application built successfully. Chromium opened the actual Library and Studio tools screens without JavaScript page errors. |
| Music engine, CPU build | Pinned native `yue-server` compiled successfully. This checks the source/build path; the delivered Dataset requires its CUDA build. |
| Music engine, CUDA 13 / SM75 | Pinned native `yue-server` and its CUDA backend compiled and linked successfully for `75-real`. Its staged binary was checked for x86-64 and local shared-library resolution. No GPU inference was available on this validation host. |
| MIDI and style-description binaries | Pinned native `ace-midi` and `ace-caption` CPU targets compiled, and their command-line help executed. The large model inference paths were not run here. |
| Whisper binary | Pinned `whisper-cli` compiled on CPU; its staged executable loaded its local shared libraries and displayed help. Recognition inference was not run here. |
| Local writer binary | Verified and unpacked the pinned llama.cpp b11236 x86-64 archive. Its staged executable loaded and reported the expected version after including its required `libmtmd` companion library. Gemma inference was not run here. |
| Original WAV upload | The actual patched Rust service and browser imported a 78,643,244-byte WAV. A deliberately lost acknowledgement after a server write was retried successfully. Repeating the import produced no duplicate song. Original library bytes matched SHA256 `f601222f891d1f8f7f31ce21c9062876769a89772b768799cf14854c57fb7ceb`. |
| Authentication | Forwarded requests without the network key were rejected; authenticated requests succeeded. The native MCP endpoint listed 180 tools. Listing does not mean every tool's inference path was executed. |
| Integrated stem inference | The actual Studio separation route ran the pinned HT-Demucs six-stem ONNX model with ONNX Runtime 1.30.0 on CPU. A synthetic one-second WAV produced drums, bass, other, vocals, guitar and piano, all registered in the library. Each exported WAV contained 44,100 frames at 44.1 kHz, stereo PCM16. This was a functional check, not a music-quality or speed benchmark. |
| Packaging regressions | Nine tests passed: actual native Kaggle SDK create/upload methods with mocked network transport, one private create request, source-file preservation, temporary-cache release, rejected-request receipts, manifest/checksum changes, ARM rejection, archive escapes, valid internal library links, saved-library restoration, credential redaction and self-contained notebook syntax. |
| PDF | 26 pages, 540 × 765.12 pt, 31 links, 26 bookmarks, no text spans outside page bounds. Cover and sample tool pages were visually inspected. The original Music Production System guide supplied the fonts, colors and layout conventions. |
| Notebook generation | Both notebooks contain one independent executable code cell, embedded support sources and no saved outputs. Python syntax compiled. They were not executed inside Kaggle during this check. |

The Create screen was inspected using a browser-only readiness fixture. No
diagnostic song was generated, and that UI check is not evidence of successful
generation. Stem inference used the real model and service without that fixture.
All test audio was synthetic and remains outside the published repository.

## Build decisions

- The fork's released Linux archive was inspected and contained ARM/aarch64
  executables despite its x64 name. This package rejects ARM and builds the
  pinned x86-64 sources instead of shipping that archive.
- CUDA 12.8 failed against this host's newer glibc math headers. CUDA 13.0.2's
  unmodified official toolkit compiled those headers successfully. The notebook
  uses pinned CUDA 13 components and requires driver 580 or newer at startup.
  The small extracted driver-dev stub is for linking only; it is neither
  installed as a host driver nor shipped in the runtime.
- Each native helper keeps its own GGML libraries and an `$ORIGIN` runtime path.
  MIDI, MOSS captioning, the local writer, stems and recognition use CPU paths.
  The music engine uses CUDA; its failure never becomes silent CPU generation.
- Runtime compilation happens in the CPU Dataset notebook. Models are uploaded
  one at a time through the SDK's established direct private-create method.
  The runtime archive uses an opaque `.bundle` extension so Kaggle does not
  automatically unpack a `.tar.gz` and remove the verified archive file.

## Not yet verified on Kaggle

No Kaggle credentials or GPU were available to this validation run. Consequently
the following remain unverified:

- Authenticated Dataset submission, Kaggle processing, actual quota behavior,
  and a full first-time builder execution in the current Kaggle CPU image.
- Loading the saved runtime in the current Kaggle GPU image, CUDA generation,
  SheetSage2 inference, VRAM use, speed, long-song completion and musical quality.
- Full Gemma, MuScriptor, MOSS, Parakeet and Whisper inference; LoRA application;
  browser MIDI devices, artwork providers and video export.
- A live Cloudflare link from Kaggle. A real quick-tunnel request on this
  development host failed while connecting to `api.trycloudflare.com:443`
  (`connection refused`). That result does not establish Kaggle's network behavior.

Start with a short song manually after startup reports both Studio and CUDA
engine ready. Check the output before committing to a longer session. Use the
saved logs and readiness receipt described in [setup](README.md) for failures.

## Reproduce the local checks

```bash
python tools/build_music_studio_notebooks.py
python tools/build_music_studio_pdf.py
python -m unittest discover -s tests/music_studio -v
```

`tools/check_music_studio_browser.py` exercises a running patched native Studio
with Playwright/Chromium. It creates a synthetic large WAV and test library
entries; use an isolated test library. It does not download generation weights
or claim to test CUDA inference. Exact source/toolchain revisions and download
hashes are recorded in [sources.json](../../music_studio/sources.json); model
sizes, destinations, immutable URLs and hashes are in
[catalog.json](../../music_studio/catalog.json).

## Final CUDA build

The final native CUDA 13.0.2 / SM75 executable compiled and linked successfully
with GNU 14.2.0 on Linux x86-64. The build used the pinned source and official,
unmodified NVIDIA toolkit files. CUDA kernel compilation and the server's final
link completed; staging with the actual runtime-copy function preserved its
`$ORIGIN` library path. Dynamic dependency inspection found only `libcuda.so.1`
missing on this GPU-less host, as expected. Kaggle's NVIDIA host driver supplies
that library; the build stub is not packaged.

The final link initially exposed missing cuBLAS SONAME aliases and a transitive
CUDA library search path. The builder now reconstructs SONAME aliases from the
verified ELF libraries, including link-only stub aliases, and supplies a
build-time `-rpath-link` to the toolkit libraries/stubs. These fixes were applied
before the successful final link. Stub directories are not embedded in the
runtime search path.

Compilation is not a CUDA inference test. The live Kaggle limits above still
apply.
