# Music Studio component notices

Music Studio v2 packages official YuE2 Studio 3.8.5 for Linux browsers. The existing
FP16 ComfyUI and Song Timestamp Studio notebooks are separate products.

- YuE2 Studio: Ilya Timonin / timoncool, MIT.
  https://github.com/timoncool/YuE2-Studio
- yue2.cpp: ServeurpersoCom and timoncool, MIT. GGML retains its MIT notice.
  https://github.com/ServeurpersoCom/yue2.cpp
- HOT-Step native MIDI and caption tools: MIT; upstream dependencies retain
  their notices. https://github.com/timoncool/HOT-Step-CPP
- audio.cpp / BSRoformer.cpp: retain MIT and their dependency notices.
  https://github.com/timoncool/audio.cpp
  https://github.com/timoncool/BSRoformer.cpp
- ONNX Runtime (Beat This! tempo): MIT.
  https://github.com/microsoft/onnxruntime
- llama.cpp: MIT. Cloudflared: Apache-2.0.
  https://github.com/ggml-org/llama.cpp
  https://github.com/cloudflare/cloudflared
- NVIDIA CUDA/cuBLAS redistributables remain subject to NVIDIA's CUDA Toolkit
  EULA and redistribution conditions. This builder includes the shared runtime
  libraries, their notices, and no NVIDIA driver.
  https://docs.nvidia.com/cuda/eula/index.html
- The browser MIDI editor uses Signal (MIT) and A320U SoundFonts (GPL-2.0).
  The visualizer includes audioMotion-analyzer (AGPL-3.0). Original source and
  licensing links remain in the studio interface and runtime notices.

## Models are separately licensed

The exact immutable URLs, byte sizes and SHA256 hashes are in catalog.json
and in the private Dataset's music-studio-manifest.json. No software license
relicenses a model or the user's input material.

- YuE2-3B and YuE2 VAE: CC BY-NC 4.0 plus M-A-P's individual-creator permission.
  That permission allows qualifying individuals to monetize outputs under its
  conditions. It does not automatically extend to other bundled models.
  https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE
- SheetSage2 and the Mothersuperior decoder companion: CC BY-NC 4.0. The Studio
  offers Auto/On/Off for the companion (Auto enables it for applicable trained
  adapters). Do not treat this
  complete bundle as unrestricted commercial software/model use.
  https://huggingface.co/m-a-p/SheetSage2
  https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
- The INT8 ConvRot training base, VAE encoder, tokenizer and training SheetSage2
  retain their source model terms. Training data needs its own authorization.
  https://huggingface.co/scragnog/YuE2-GGUF
- Mel-Band RoFormer voc_fv6 retains its model author terms.
  https://huggingface.co/chenmozhijin/BSRoformer-GGUF
- MuScriptor: CC BY-NC 4.0.
  https://huggingface.co/cocktailpeanut/muscriptor-medium
- Gemma: Google's Gemma terms and prohibited-use policy; preserve its notices.
  https://ai.google.dev/gemma/terms
  https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf
- HT-Demucs 6 stems and Beat This!: MIT.
  https://huggingface.co/audio-cpp/audio.cpp-gguf
  https://github.com/CPJKU/beat_this
- MOSS-Music: Apache-2.0.
  https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct

The Dataset is private by default, with metadata license "other" because this
is a mixed-license collection. Keep original component terms and attribution.
Training recordings, covers, artwork, plugins and externally supplied adapters
need their own rights. Your songs and private recordings are never part of the
model Dataset or the public GitHub deliverables.

## Build and feature scope

The runtime is built from the immutable source revisions in sources.json for
Linux x86-64, CUDA 13.0, and T4 compute capability 7.5. Rust 1.99.1 is pinned.
The CPU Dataset notebook compiles it once. The GPU Studio notebook reuses it.
The Studio storage patch introduces YUE_LIBRARY_ROOT for durable output. A
native resumable WAV/MP3 upload module and browser page preserve original audio
over the tunnel. These patches do not replace model inference or audio DSP.

This profile includes AB2/chord-variety, score continuation, song extension and
an INT8 ConvRot CUDA trainer with the upstream Turing fallback. Musical score
and MIDI transcription remain available. Timestamp recognition, speech-to-text,
automatic karaoke timing and MMS lyric alignment are excluded, including model
downloads and UI/MCP actions. The video editor accepts external LRC.
Native Linux VST3 offline processing is included; Windows VSTs and remote desktop
plugin editor windows are outside Kaggle's browser workflow.
