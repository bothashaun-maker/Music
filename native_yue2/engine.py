COVER_MELODY_READY = False
COVER_LYRICS_READY = False
# Based on the original AIQUEST Academy Kaggle notebook in Movies commit fcf4d45.
# Restores its native FP16/CUDA-graph engine; extends duration and output handling.
# === Cell 2: Runtime Engine, Model Load & AIQUEST Gradio UI ===
# -----------------------------------------------------------------------------
# 0. Defensive Compatibility Shim for huggingface_hub & transformers
# -----------------------------------------------------------------------------
import huggingface_hub
if not hasattr(huggingface_hub, "is_offline_mode"):
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE
        huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)
    except Exception:
        huggingface_hub.is_offline_mode = lambda: False

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import soundfile as sf

print("Initializing YuE2-3B engine: MoT on cuda:0 (CUDA graphs), VAE on cuda:1...")

# Hardware Acceleration Flags for Turing SM 7.5 (GPU T4 x2)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("medium")

# -----------------------------------------------------------------------------
# 1. Programmatic Runtime Patching for Active Dual-GPU Sharding (GPU T4 x2)
# -----------------------------------------------------------------------------
import yue2.pipeline as ypipe
import yue2.nar as ynar
import yue2.cuda_graph as ycgraph
import yue2.sampling as ysampling
import yue2.modeling_yue2 as ymodel
import yue2.protocol as yproto
import yue2.progress as yprogress

# T4 cannot use the native GQA FlashAttention path. Offer an equivalent layout
# to the efficient backend, retaining the original path if unsupported.
from native_yue2.attention import FunctionalProxy, STATUS as ATTENTION_STATUS
if os.environ.get("YUE2_NATIVE_ATTENTION", "efficient_grouped") == "efficient_grouped":
    ycgraph.F = FunctionalProxy()
    ynar.F = FunctionalProxy()
    ymodel.F = FunctionalProxy()
# Defensive tqdm & Gradio monkeypatch protection
import tqdm.std
import tqdm.notebook
tqdm.std.tqdm._progress = None
if hasattr(tqdm.notebook, "tqdm_notebook"):
    tqdm.notebook.tqdm_notebook._progress = None
if hasattr(tqdm.notebook, "tqdm"):
    tqdm.notebook.tqdm._progress = None
try:
    from tqdm.auto import tqdm
    tqdm._progress = None
except Exception:
    from tqdm import tqdm
    tqdm._progress = None

# Protect against gradio close_tqdm AttributeError
try:
    import gradio.helpers
    orig_close_tqdm = getattr(gradio.helpers, "close_tqdm", None)
    if orig_close_tqdm is not None:
        def safe_close_tqdm(self, *args, **kwargs):
            if not hasattr(self, "_progress"):
                self._progress = None
            try:
                return orig_close_tqdm(self, *args, **kwargs)
            except Exception:
                pass
        gradio.helpers.close_tqdm = safe_close_tqdm
except Exception:
    pass

# Patch 0: Clean in-place tqdm progress bar replacing 5-second polling text dumps
class TqdmStage:
    def __init__(self, owner, label, total=None, unit=None):
        self._owner = owner
        self.label = str(label)
        self.unit = str(unit) if unit else "it"
        self.total = int(total) if total is not None else None
        self.completed = 0
        self._finished = False
        self._pbar = None

    def __enter__(self):
        self._pbar = tqdm(
            total=self.total,
            desc=self.label,
            unit=self.unit,
            dynamic_ncols=True,
            leave=True,
        )
        self._pbar._progress = None
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(status="failed" if exc_type else "completed")
        return False

    def update(self, completed, total=None):
        if self._finished:
            return
        if total is not None and self._pbar is not None:
            self.total = int(total)
            self._pbar.total = int(total)
        delta = int(completed) - self.completed
        if delta > 0 and self._pbar is not None:
            self._pbar.update(delta)
        self.completed = int(completed)

    def set_total(self, total):
        if self._pbar is not None and total is not None:
            self.total = int(total)
            self._pbar.total = int(total)
            self._pbar.refresh()

    def advance(self, count=1):
        self.update(self.completed + int(count))

    def token(self, phase, token):
        self.advance(1)

    def finish(self, status="completed"):
        if not self._finished:
            self._finished = True
            if self._pbar is not None:
                try:
                    self._pbar._progress = None
                    self._pbar.close()
                except Exception:
                    pass

class TqdmProgress:
    def __init__(self, enabled=True, stream=None, refresh_interval=0.25):
        self.enabled = bool(enabled)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stage(self, label, total=None, unit=None):
        return TqdmStage(self, label, total=total, unit=unit)

    def close(self, status="completed"):
        pass

    def complete(self, audio_seconds, elapsed, *, truncated=False):
        status = "Finished" if truncated else "Completed"
        print(f"\n[YuE2] {status}: {audio_seconds:.1f}s audio in {elapsed:.1f}s\n")

yprogress.Progress = TqdmProgress
yprogress._Stage = TqdmStage
ypipe.Progress = TqdmProgress

# Patch 0b: Allow both midpoint and euler ODE methods in GenerationConfig
orig_config_post_init = yproto.GenerationConfig.__post_init__

def patched_config_post_init(self):
    if self.context != yproto.CONTEXT or self.ode_method not in {"midpoint", "euler"} or type(self.ode_steps) is not int or self.ode_steps < 1:
        raise ValueError("Require context=24576, positive integer steps, and ode_method in {'midpoint', 'euler'}")

yproto.GenerationConfig.__post_init__ = patched_config_post_init

# -----------------------------------------------------------------------------
# Device plan, from the checkpoint config
# -----------------------------------------------------------------------------
#   MoT weights   : 3.63B params x 2 bytes (FP16)             = 7.26 GB
#   AR KV cache   : 28 layers x 2 x 8 kv-heads x 128 dim x 2  = 112 KB / token
#                   -> 1.15 GB even at the 10k-token maximum
#   NAR prefix KV : same 112 KB / token                       -> ~1.2 GB worst case
#   Peak on cuda:0 stays near 10 GB of the ~14.5 GB usable on one T4.
#
# The whole MoT fits on a single T4, so it is kept intact on cuda:0 and cuda:1
# is handed entirely to the 48 kHz VAE. Keeping the model on one device is what
# allows the CUDA-graph AR decoder to be used at all: a model split across two
# cards cannot be captured, which would force backend="torch-eager", and in
# yue2/pipeline.py that flag is exactly what disables the graph decoder:
#     use_cuda_graph=self.backend != "torch-eager"
# -----------------------------------------------------------------------------

# Patch 1: Pipeline init - bypass the BF16 gatekeeper, load FP16, pick devices
def patched_pipeline_init(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=14,
                          backend="torch", generation_config=None, verify_hashes=True,
                          vae_core_frames=512, quantization="none", offload_ar=False, progress=True):
    self.progress = progress
    self.backend = backend
    self.quantization = quantization
    self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
    self.offload_ar = offload_ar
    self.vae_core_frames = vae_core_frames

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.vae_device = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else self.device

    self.memory_budget_gib = float(memory_budget_gib)
    self.generation_config = generation_config or yproto.GenerationConfig()
    self.tokenizer = ypipe.YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")

    # Reduced-precision FP16 accumulation is a free win on Turing; upstream only
    # disables it to keep BF16 runs bit-exact on Ampere.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # cudnn.benchmark must stay off. The transformer has no convolutions, so it
    # has nothing to tune there, while the VAE decoder is ~40 conv layers fed
    # tiles of three different lengths (the first has no left halo, the last is
    # a remainder). Benchmark mode re-runs its exhaustive algorithm search on
    # every unseen shape, which costs far more than the kernels it picks save.
    torch.backends.cudnn.benchmark = False

    with self._status("Verifying model files"):
        self.weights = {
            "mot": ypipe.model_identity(self.model_dir, verify_hashes),
            "vae": ypipe.model_identity(self.vae_dir, verify_hashes),
        }
    import hashlib
    self.runtime_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    self._model, self._vae = None, None
    self.load_timing = {}

ypipe.YuE2Pipeline.__init__ = patched_pipeline_init

# Patch 2: Load the MoT whole, in FP16, on cuda:0
def patched_load_model(self, for_nar=False):
    loading = self._model is None or next(self._model.parameters()).device != self.device
    with self._status(f"Loading YuE2-3B (FP16) on {self.device}") if loading else ypipe.nullcontext():
        if self._model is None:
            start = time.perf_counter()
            from yue2.modeling_yue2 import YuE2ForCausalLM
            self._model = YuE2ForCausalLM.from_pretrained(
                self.model_dir,
                local_files_only=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).eval()
            self.load_timing["mot_load_seconds"] = time.perf_counter() - start
        # decode() parks the MoT on the CPU when a single GPU has to serve both
        # the transformer and the VAE, so always confirm placement here.
        if next(self._model.parameters()).device != self.device:
            self._model.to(self.device)
            if torch.cuda.is_available():
                resident = torch.cuda.memory_allocated(self.device) / 2 ** 30
                print(f"MoT resident on {self.device}: {resident:.2f} GB "
                      f"| VAE device: {self.vae_device}")
    return self._model

ypipe.YuE2Pipeline._load_model = patched_load_model

# Runtime AR switches, read by patched_graphar_init below and set from the UI.
AR_OPTIONS = {"fuse_projections": False}
AR_DIAGNOSTICS = {"execution": "not_started", "fallback_reason": None}

# Patch 3: Make the CUDA-graph AR decoder safe on Turing.
# GraphAR probes for Ampere variable-length FlashAttention and would select it
# on a T4, where that kernel does not exist. Pin plain SDPA below SM 8.0.
_orig_graphar_init = ycgraph.GraphAR.__init__

def patched_graphar_init(self, model, prefixes, max_tokens, *, capture=True,
                         attention_backend="auto", fuse_projections=False):
    AR_DIAGNOSTICS.clear()
    AR_DIAGNOSTICS.update(execution="prefill", fallback_reason=None,
                          prefix_tokens=max(map(len, prefixes)), max_tokens=max_tokens,
                          cfg_branches=len(prefixes))
    if attention_backend == "auto" and torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] < 8:
            attention_backend = "sdpa"
    # Fusing q/k/v into one GEMM and gate/up into another cuts seven matmuls per
    # layer down to five and gives cuBLAS wider, better-shaped work at batch 1.
    # It duplicates those weights (~1.9 GB) and routes through a less-travelled
    # branch of the library, so it stays opt-in from the UI and additionally
    # requires the card to have room to spare.
    if not fuse_projections and AR_OPTIONS.get("fuse_projections") and torch.cuda.is_available():
        try:
            free_bytes = torch.cuda.mem_get_info(model.model.embed_tokens.weight.device.index or 0)[0]
            fuse_projections = free_bytes > 4 * 2 ** 30
        except Exception:
            fuse_projections = False
    _orig_graphar_init(self, model, prefixes, max_tokens, capture=capture,
                       attention_backend=attention_backend,
                       fuse_projections=fuse_projections)
    if not capture or max_tokens <= 1:
        AR_DIAGNOSTICS["execution"] = "eager"
    print(f"[YuE2] AR decoder: attention={self.attention_backend}, "
          f"fused_projections={bool(self.fused_weights)}, "
          f"capture_requested={capture}, capacity={self.capacity}", flush=True)

ycgraph.GraphAR.__init__ = patched_graphar_init

# Patch 3b: If graph capture is refused, drop to eager decode on the same GPU
# rather than failing the request. step() already falls back to _decode() when
# self.graph is None, so restoring the starting positions is all that is needed.
_orig_graphar_capture = ycgraph.GraphAR._capture

def patched_graphar_capture(self):
    try:
        _orig_graphar_capture(self)
        AR_DIAGNOSTICS["execution"] = "cuda_graph" if self.graph is not None else "eager"
        print(f"[YuE2] Actual AR execution: {AR_DIAGNOSTICS['execution']}", flush=True)
    except Exception as exc:
        print(f"[YuE2] CUDA graph capture unavailable ({exc}); falling back to eager decode.")
        self.graph = None
        self.output = None
        self.positions.copy_(self.initial_positions)
        AR_DIAGNOSTICS.update(execution="eager", fallback_reason=str(exc))

ycgraph.GraphAR._capture = patched_graphar_capture

# The release calls any surviving GraphAR object 'cuda_graph', including an
# object whose capture failed. Preserve its output and correct only that label.
_original_generate_tokens = ypipe.generate_tokens
def observed_generate_tokens(*args, **kwargs):
    AR_DIAGNOSTICS.clear()
    AR_DIAGNOSTICS.update(execution="eager", fallback_reason=None)
    tokens, timing, truncated = _original_generate_tokens(*args, **kwargs)
    timing = dict(timing)
    timing["execution"] = AR_DIAGNOSTICS["execution"]
    if AR_DIAGNOSTICS.get("fallback_reason"):
        timing["graph_fallback_reason"] = AR_DIAGNOSTICS["fallback_reason"]
    return tokens, timing, truncated
ypipe.generate_tokens = observed_generate_tokens

# Patch 4: Length-aware attention tiling for the NAR flow-matching pass.
# T4 has no FlashAttention, so SDPA can materialize the full [heads, Q, K] score
# matrix. A fixed 1536-token tile is fine for a 1-minute song but allocates
# almost 1 GB for a 6-minute one. Size the tile from the actual key length so
# the score tile stays near 384 MB whatever the song length.
_SCORE_TILE_BYTES = 384 * 2 ** 20

def patched_cached_nar_attention(self, q, k, v, causal=False):
    chunk = getattr(self, "query_chunk_size", None)
    if not chunk:
        per_query = max(1, len(k) * q.shape[1] * 2)
        chunk = max(256, min(len(q), _SCORE_TILE_BYTES // per_query))
    return ynar.attention(q, k, v, causal=causal, backend=self.backend, query_chunk_size=int(chunk))

ynar.CachedNAR._attention = patched_cached_nar_attention

# Patch 5: Euler / midpoint ODE solver with an FP32 integration accumulator.
# Euler evaluates the velocity field once per step instead of twice, halving
# synthesis time. FP32 state keeps the integration stable while the velocity
# itself is still evaluated in the model's FP16.
@torch.inference_mode()
def patched_solve(self, steps=16, method="euler", cancelled=None, on_progress=None):
    state = self.chunk.noise.to(device=self.device, dtype=torch.float32)
    dt = 1.0 / steps
    for step in range(steps):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during acoustic flow matching")
        t = 1.0 - step * dt
        raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
        first = self.velocity(state.to(dtype=self.dtype), raw).to(dtype=torch.float32)

        if method == "euler":
            state = state - first * dt
        else:
            mid = state - first * (dt / 2.0)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            raw_mid = torch.logit(torch.tensor(t - dt / 2.0, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            second = self.velocity(mid.to(dtype=self.dtype), raw_mid).to(dtype=torch.float32)
            state = state - second * dt

        if on_progress is not None:
            on_progress(step + 1, int(steps))

    result = state.float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite acoustic latents; refusing to replace them with fabricated values")
    return result

ynar.CachedNAR.solve = patched_solve

# Patch 5b: synthesize dispatcher carrying the chosen ODE method and step count
@torch.inference_mode()
def patched_synthesize(model, prefix, codec, seed, steps=16, method="euler",
                       context=yproto.CONTEXT, attention="sdpa", offload_ar=False,
                       cancelled=None, query_chunk_size=None, on_progress=None):
    chunks = ynar.song_chunks(prefix, codec, seed, context)
    pipe.last_acoustic_ranges = yproto.chunk_ranges(len(codec), len(prefix), context)
    print("Acoustic chunk boundaries (seconds):", [b / 25 for _, b in pipe.last_acoustic_ranges[:-1]])
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = ynar.CachedNAR(model, chunk, attention, query_chunk_size)
        try:
            progress = None
            if on_progress is not None:
                def progress(completed, total, _index=chunk_index, _count=len(chunks)):
                    on_progress(_index * total + completed, total * _count)
            output.append(engine.solve(steps, method=method, cancelled=cancelled, on_progress=progress))
        finally:
            engine.close()
        del engine
    return torch.cat(output, dim=0)

ynar.synthesize = patched_synthesize

# Patch 5c: pipeline hook passing the configured ode_steps / ode_method through.
# query_chunk_size stays None so Patch 4 can size the tile from the song length.
@torch.inference_mode()
def patched_pipeline_synthesize(self, semantic, *, cancelled=None):
    model = self._load_model(for_nar=True)
    steps = getattr(self.generation_config, "ode_steps", 16)
    method = getattr(self.generation_config, "ode_method", "euler")
    with self._status("Synthesizing audio", total=steps, unit="steps") as status:
        report = (lambda completed, total: status.update(completed, total=total)) if self.progress else None
        result = ynar.synthesize(model, semantic.plan.prefix, semantic.tokens,
                                 semantic.plan.request.seed, steps=steps,
                                 method=method, context=self.generation_config.context,
                                 offload_ar=False, cancelled=cancelled,
                                 query_chunk_size=None, on_progress=report)
        return result.detach().float().cpu().numpy()

ypipe.YuE2Pipeline.synthesize = patched_pipeline_synthesize

# Patch 6: Tiled 48 kHz VAE decode on the dedicated VAE GPU.
# The latent stays on the CPU on purpose. decode_tiled() slices it and calls
# decode(), which moves each tile to the GPU itself; handing it a latent that is
# already resident on the GPU keeps the whole song plus every intermediate alive
# at once and pushes the transposed convolutions into cuDNN's slow fallback
# (a measured 163s for a single chunk, against ~0.7s per 512-frame tile).
def patched_decode(self, latents, *, full=False, vae=None):
    from yue2.modeling_vae import YuE2VAE
    vae_device = getattr(self, "vae_device", self.device)

    # Only when one GPU has to serve both models does the MoT need to step aside.
    if vae_device == self.device and self._model is not None:
        self._model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with self._status(f"Loading 48 kHz Oobleck VAE decoder on {vae_device}"):
        if self._vae is None:
            self._vae = YuE2VAE.from_pretrained(
                self.vae_dir, decoder_only=True, device="cpu", local_files_only=True
            )
        model = self._vae.to(vae_device)

    z = torch.as_tensor(latents, dtype=torch.float32)
    if z.ndim == 2 and z.shape[1] == 64:
        z = z.T.unsqueeze(0)
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
        raise ValueError("Expected latents [T,64] or [1,64,T]")
    try:
        core_frames = getattr(self, "vae_core_frames", 512) or 512
        tiles = (z.shape[-1] + core_frames - 1) // core_frames
        with self._status("Decoding 48 kHz stereo audio", total=tiles, unit="chunks") as status:
            report = (lambda completed, total: status.update(completed, total=total)) if self.progress else None
            with torch.inference_mode():
                audio = model.decode_tiled(
                    z,
                    core_frames=core_frames,
                    halo_frames=16,
                    output_device="cpu",
                    on_progress=report
                )
            if not torch.isfinite(audio).all():
                raise FloatingPointError("Non-finite VAE audio; saved latents remain available for recovery")
            return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

ypipe.YuE2Pipeline.decode = patched_decode

print("All Turing T4 runtime patches applied successfully.")

# -----------------------------------------------------------------------------
# 2. Pipeline Initialization
# -----------------------------------------------------------------------------
# backend="torch" (not "torch-eager") is what enables the CUDA-graph AR decoder
# in yue2/pipeline.py::_generate. Patch 3 keeps that path safe on Turing.
print("\nInitializing YuE2 Pipeline from Hugging Face Hub (m-a-p/YuE2-3B)...")
try:
    pipe = ypipe.YuE2Pipeline.from_pretrained(
        "m-a-p/YuE2-3B",
        vae="m-a-p/YuE2-Vae",
        revision="8f24312f187763b9854adeec87e437e2b03bbff1",
        vae_revision="152733a19ad43aa67e367f9b5503ef8075bb5126",
        cache_dir=os.environ["HF_HUB_CACHE"],
        backend="torch",
        offload_ar=False,
        progress=True
    )
    print("YuE2-3B Pipeline initialized: MoT on cuda:0 (CUDA graphs on), "
          f"VAE on {pipe.vae_device}.")
except Exception as e:
    print(f"Pipeline initialization error: {e}")
    raise e

# Decode recovery uses saved latents and never regenerates music tokens.
def recover_decode(source):
    import shutil, uuid
    from yue2.storage import sha256_file, write_json
    source = Path(source).resolve()
    manifest = json.loads((source / "latent_manifest.json").read_text())
    if manifest["model_weights"] != pipe.weights:
        raise ValueError("Recovery weights differ from the original run")
    if (source / "latent.npy").is_symlink() or sha256_file(source / "latent.npy") != manifest["sha256"]:
        raise ValueError("Saved latent checksum mismatch")
    latents = np.load(source / "latent.npy", allow_pickle=False)
    if list(latents.shape) != manifest["shape"] or latents.ndim != 2 or latents.shape[1] != 64 or not np.isfinite(latents).all():
        raise ValueError("Invalid saved latent shape or values")
    target = Path(os.environ["YUE2_OUTPUT_ROOT"]) / ("recovered-" + uuid.uuid4().hex[:12])
    target.mkdir(parents=True, exist_ok=False)
    for name in ("settings.json", "plan.json", "plan_manifest.json", "score.abc", "prefix.npy",
                 "abc_tokens.npy", "semantic.npy", "semantic.json", "latent.npy", "latent_manifest.json"):
        original = source / name
        if original.is_file():
            if original.is_symlink():
                raise ValueError("Recovery artifacts must be regular files")
            shutil.copyfile(original, target / name)
    audio = pipe.decode(latents)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("Invalid recovered audio")
    if not np.isclose(len(audio) / 48000, len(latents) / 25, atol=0.1):
        raise ValueError("Recovered audio duration mismatch")
    for name in ("audio.wav", "audio.flac"):
        sf.write(str(target / name), audio, 48000, subtype="PCM_24")
        info = sf.info(str(target / name))
        if info.channels != 2 or info.samplerate != 48000 or not np.isclose(info.duration, len(audio) / 48000, atol=0.001):
            raise ValueError("Recovered export header mismatch")
    write_json(target / "status.json", {"status": "decode_recovered", "source": str(source),
                                        "actual_seconds": len(audio) / 48000})
    print("Recovered audio:", target, flush=True)
    return target

if os.environ.get("YUE2_RECOVER_RUN"):
    recover_decode(os.environ["YUE2_RECOVER_RUN"])
    raise SystemExit(0)

# -----------------------------------------------------------------------------
# 3. Preset Library & Examples
# -----------------------------------------------------------------------------
PRESETS = {
    "Cyber Metal (High-Energy Heavy Metal)": {
        "style": "Cyber Metal, high-energy heavy metal, aggressive electric guitar distortion, driving double-bass drums, soaring passionate male rock vocal, 140 BPM, high production value",
        "lyrics": """[Verse 1]
Circuits fire in the deep neon light
Mechanical shadows claim the dark night
Steel and thunder tearing through the code
A digital heart on an endless road

[Chorus]
Unchain the voltage, hear the engine roar
Breaking the limits forevermore
In the chrome storm, we reignite the fire
Pushing the frequency higher and higher!

[Guitar Solo]

[Outro]
Fading into the neon glow
Only the echoes remain below"""
    },
    "Tonight Awake (Funk / Nu-Disco)": {
        "style": "Mandarin funk, nu-disco, infectious slap bass, lush Rhodes piano, crisp rhythm guitar, silky smooth vocal, modern dance groove, 118 BPM",
        "lyrics": """[Verse 1]
夜色降临 街灯渐次亮起
微风带走白日的喧嚣气息
随着节奏 迈开随性的步履
今晚的旋律 只属于我和你

[Chorus]
今晚不眠 尽情沉醉在光影之间
跳动的节拍 跨越所有的界线
举起酒杯 庆祝这未完的诗篇
让音乐伴随 直到破晓的黎明前

[Outro]
直到破晓的黎明前"""
    },
    "Passion (Modern English Rock)": {
        "style": "Modern English rock, emotive male lead vocal, melodic overdrive guitars, punchy drums, atmospheric reverb, uplifting anthem, 125 BPM",
        "lyrics": """[Verse 1]
Standing on the edge of tomorrow's dream
Nothing is as quiet as it used to seem
We carried our hopes through the pouring rain
Turning the struggle right into gain

[Chorus]
This is our passion, this is our time
Reaching the summit, starting the climb
Voices echoing out in the clear
Everything we fought for is finally here!

[Outro]
Finally here...
Yeah, we are here."""
    },
    "Auld Lang Syne (Jazz-Funk Cover)": {
        "style": "Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight brass section, brushed drums, sophisticated harmony",
        "lyrics": """[Verse]
Should old acquaintance be forgot
And never brought to mind?
Should old acquaintance be forgot
And days of auld lang syne?

[Chorus]
For auld lang syne, my dear
For auld lang syne
We will take a cup of kindness yet
For days of auld lang syne!"""
    },
    "Cinematic Epic Orchestral (Symphonic Film Score)": {
        "style": "Cinematic epic orchestral, grand choir, thundering taiko drums, sweeping string section, French horns, heroic brass, Hans Zimmer style, dramatic climax",
        "lyrics": """[Verse]
Across the valleys of forgotten kings
The wind carries what the dawn light brings
Armies gather where the shadows grow
Steel reflecting the morning glow

[Chorus]
Rise for the honor, stand for the land
United together, side by side we stand
From the ashes of the battleground
A new horizon will be found!"""
    }
}

# -----------------------------------------------------------------------------
# 4. Music Generation Engine Function
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Cover support: turning a source song into symbolic inputs
# -----------------------------------------------------------------------------
# YuE2 accepts no audio anywhere in its API, so a "cover" is produced by
# transcribing the source into the two symbolic inputs it does accept:
#   SheetSage2   -> the melody, as an ABC score   (0.21 GB)
#   Whisper      -> the lyrics, as text           (1.6 GB, via transformers)
# Generation is then the ordinary abc= path: pipe.plan(abc=<score>, cot="melody")
# with your own style prompt, which is what makes it a cover rather than a copy.
#
# Each loads lazily and independently onto cuda:1, which holds only the VAE and
# has room to spare, and stays resident so a second analysis is instant. The
# melody half is the one a cover needs; lyrics can always be typed by hand.
COVER_MODELS = {"sheetsage": None, "asr": None}

COVER_LANGUAGES = ["Auto-detect", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "it", "pt"]

# m-a-p/SheetSage2 main as of this notebook. Set to None to track main instead.
SHEETSAGE2_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"


def _analysis_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:1") if torch.cuda.device_count() >= 2 else torch.device("cuda:0")


def load_melody_transcriber():
    """SheetSage2. Its remote modeling code imports mir_eval and pretty_midi."""
    if COVER_MODELS["sheetsage"] is None:
        from transformers import AutoModel
        device = _analysis_device()
        print(f"[Cover] Loading SheetSage2 melody transcriber on {device}...")
        # Pinned revision: this repo ships executable remote code, so pin what
        # runs rather than tracking main. Update deliberately, not silently.
        COVER_MODELS["sheetsage"] = AutoModel.from_pretrained(
            "m-a-p/SheetSage2",
            trust_remote_code=True,
            revision=SHEETSAGE2_REVISION,
        ).eval().to(device)
    return COVER_MODELS["sheetsage"]


def load_lyric_transcriber():
    """Whisper via transformers, in FP16: Turing has no native bfloat16.

    large-v3-turbo is ~1.6 GB in FP16 and strongly multilingual, which suits
    song lyrics. transformers is already required by YuE2, so this adds no
    dependency that could conflict with the image's numpy/numba stack.
    """
    if COVER_MODELS["asr"] is None:
        from transformers import pipeline
        device = _analysis_device()
        print(f"[Cover] Loading Whisper large-v3-turbo on {device}...")
        COVER_MODELS["asr"] = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=device,
            chunk_length_s=30,
        )
    return COVER_MODELS["asr"]


def _load_audio_16k(path):
    """Read any soundfile-readable audio as mono 16 kHz float32.

    Doing this here avoids librosa entirely, and avoids handing the pipeline a
    path (which would route it through an ffmpeg subprocess instead).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), 16000)
        mono = resample_poly(mono, 16000 // g, int(sr) // g).astype(np.float32)
    return mono


def analyze_cover_source(audio_path, language, current_lyrics):
    """Transcribe an uploaded song into a melody-only ABC score plus its lyrics.

    The melody is what a cover actually needs, so a lyric failure is reported
    rather than raised: the existing lyrics are kept and you can type your own.
    """
    raise gr.Error("Cover transcription is disabled in this generation-only version.")
    if not audio_path:
        raise gr.Error("Upload a source song first.")

    # --- Melody: required ---
    try:
        sheetsage = load_melody_transcriber()
        # melody_only keeps the vocal and instrumental lines but drops chords,
        # which is the form YuE2's planner expects as an external score.
        result = sheetsage.transcribe(audio_path, melody_only=True)
        abc_score = (result or {}).get("abc")
    except ImportError as exc:
        raise gr.Error(
            "Melody transcription is not installed. See Cell 1's output: it reports "
            f"each half of cover support separately. ({exc})"
        ) from exc
    except Exception as exc:
        raise gr.Error(f"Melody transcription failed: {exc}") from exc

    if not abc_score:
        raise gr.Error("SheetSage2 produced no melody for this audio. "
                       "Try a cleaner recording or a longer excerpt.")

    # --- Lyrics: optional ---
    transcript = None
    lyric_note = ""
    try:
        asr = load_lyric_transcriber()
        generate_kwargs = {} if language == "Auto-detect" else {"language": language}
        try:
            audio_input = _load_audio_16k(audio_path)
        except Exception:
            # soundfile cannot read every container; let the pipeline try ffmpeg.
            audio_input = str(audio_path)
        asr_result = asr(audio_input, generate_kwargs=generate_kwargs)
        transcript = (asr_result.get("text") or "").strip()
        if not transcript:
            lyric_note = " No lyrics were detected, so the Lyrics box is unchanged."
    except ImportError:
        lyric_note = (" Lyric transcription is not installed, so the Lyrics box is "
                      "unchanged: type or paste the words yourself.")
    except Exception as exc:
        lyric_note = f" Lyric transcription failed ({exc}); the Lyrics box is unchanged."

    lyrics_out = transcript if transcript else (current_lyrics or "")
    status = ("Melody extracted." + lyric_note +
              " Add section tags such as [Verse] and [Chorus] to the lyrics, write the "
              "style you want the cover to take, then Generate.")
    # Planning must be on for an external score to attach to.
    return abc_score, lyrics_out, "melody", status


# Latent frames run at 48000 / 1920 = 25 frames per second, so the token budget
# maps directly onto audio seconds: 1500 tokens is a full minute, not 30-45s.
FRAMES_PER_SECOND = 48000 // 1920

# Short display text: the long "(~1 min, 1500 tokens)" form overflowed the
# dropdown in a narrow column and collided with the caret glyph.
DURATION_CHOICES = [
    ("Quick Preview - 1 min", 1500),
    ("Standard Song - 2 min", 3000),
    ("Full Song - 3 min", 4500),
    ("Full Song - 6 min", 9000),
    ("Full Song - 7 min", 10500),
    ("Full Song - 8 min", 12000),
]
DURATION_MAP = dict(DURATION_CHOICES)
DEFAULT_DURATION = 4500

# Gradio cannot interrupt a running Python callback, so the Stop button sets a
# flag that the pipeline polls between steps via its `cancelled` hook.
CANCEL = {"stop": False}


def request_stop():
    CANCEL["stop"] = True
    return "Stop requested. Generation aborts at the next step boundary."


def _cancelled():
    return CANCEL["stop"]


def generate_music(style, lyrics, cot_mode, duration_target, abc_score, seed, ode_steps, ode_method, cfg_scale, fuse_projections=False, score_only=False, progress=gr.Progress(track_tqdm=False)):
    import uuid
    output_dir = Path(os.environ["YUE2_OUTPUT_ROOT"]) / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10])
    output_dir.mkdir(parents=True, exist_ok=False)
    from yue2.storage import write_json
    write_json(output_dir / "status.json", {"status": "started"})
    archive_path = None
    CANCEL["stop"] = False
    AR_OPTIONS["fuse_projections"] = bool(fuse_projections)

    # SongRequest requires 0 <= seed < 2**63; -1 means "surprise me" and any
    # other negative value would otherwise raise deep inside the pipeline.
    if seed is None or int(seed) == -1:
        current_seed = int(np.random.randint(0, 2 ** 31 - 1))
    else:
        current_seed = abs(int(seed)) % (2 ** 31 - 1)

    cleaned_style = style.strip()
    cleaned_lyrics = lyrics.strip()
    cleaned_abc = abc_score.strip() if abc_score and abc_score.strip() else None

    if not cleaned_style:
        raise gr.Error("Please enter a style or genre prompt.")
    if not cleaned_lyrics:
        raise gr.Error("Please provide song lyrics with section tags (e.g. [Verse], [Chorus]).")

    # An external ABC score is only meaningful when a planning stage exists;
    # upstream rejects the pairing outright, so promote 'off' to 'melody'.
    notes = []
    if cleaned_abc and cot_mode == "off":
        cot_mode = "melody"
        notes.append("Custom ABC score supplied, so planning mode was switched from 'off' to 'melody'.")
    # "Score only" needs a planning stage to stop after; 'off' produces no score.
    if score_only and cot_mode == "off":
        cot_mode = "melody"
        notes.append("Score only requires a planning stage, so the mode was switched from 'off' to 'melody score'.")

    if isinstance(duration_target, str):
        target_max_tokens = DURATION_MAP.get(duration_target, 3000)
    else:
        target_max_tokens = int(duration_target or 3000)
    if target_max_tokens not in {1500, 3000, 4500, 9000, 10500, 12000}:
        raise gr.Error("Select a supported song duration (1, 2, 3, 6, 7 or 8 minutes)")
    method_clean = "euler" if "euler" in str(ode_method).lower() else "midpoint"
    write_json(output_dir / "settings.json", {
        "style": cleaned_style, "lyrics": cleaned_lyrics, "abc": cleaned_abc,
        "mode": cot_mode, "seed": current_seed, "max_tokens": target_max_tokens,
        "max_seconds": target_max_tokens / 25, "ode_steps": int(ode_steps),
        "ode_method": method_clean, "cfg_scale": float(cfg_scale),
        "engine_sha256": pipe.runtime_sha256, "backend": pipe.backend,
        "dtype": "float16", "vae_dtype": "float32", "fused_projections": bool(fuse_projections),
        "model_weights": pipe.weights,
    })

    start_time = time.perf_counter()
    status_msg = (f"YuE2-3B on {pipe.device} (CUDA graphs) | Seed {current_seed} | "
                  f"CoT {cot_mode} | {method_clean.title()} x {int(ode_steps)}\n")
    for note in notes:
        status_msg += f"Note: {note}\n"

    try:
        with torch.inference_mode():
            import dataclasses
            pipe.generation_config = dataclasses.replace(
                pipe.generation_config,
                ode_steps=int(ode_steps),
                ode_method=method_clean
            )

            # Cap the planner in every mode. Left uncapped, 'melody' inherits the
            # 4096-token upstream default and has been observed running away to
            # ~3800 tokens, which then bloats the prefix the semantic stage must
            # attend over and drags its throughput down with it.
            abc_sampling_kwargs = {"max_tokens": 1200 if cot_mode == "full" else 800}

            print(f"\n[YuE2] Step 1/4: Symbolic score plan (CoT: {cot_mode})...")
            progress(None, desc="Preparing prompt / optional score (not audio generation progress)")
            t0 = time.perf_counter()
            plan = pipe.plan(
                style=cleaned_style,
                lyrics=cleaned_lyrics,
                cot=cot_mode,
                abc=cleaned_abc,
                seed=current_seed,
                abc_sampling=abc_sampling_kwargs,
                cfg_scale=float(cfg_scale) if cfg_scale is not None else 1.0,
                cancelled=_cancelled
            )
            plan_seconds = time.perf_counter() - t0
            plan.save(output_dir)
            if plan.truncated:
                raise ValueError("ABC score hit its token limit. Saved the score; choose Direct to audio or supply a complete score.")
            if len(plan.prefix) + target_max_tokens > yproto.CONTEXT:
                raise ValueError("Lyrics/score plus song budget exceed the 24,576-token context; shorten the prompt.")

            abc_output = plan.abc if plan.abc else "(No symbolic score generated; CoT was set to 'off')"
            status_msg += f"Step 1 - score plan: {len(plan.abc_ids)} tokens in {plan_seconds:.1f}s\n"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if score_only:
                status_msg += ("\nScore only: stopped after planning, no audio rendered.\n"
                               "Edit the score above if you like, then untick 'Score only' "
                               "and regenerate with the same seed to hear it.")
                progress(1.0, desc="Score generated.")
                write_json(output_dir / "status.json", {"status": "score_saved"})
                return None, abc_output, status_msg, None, None

            print(f"\n[YuE2] Step 2/4: Acoustic semantic tokens (limit {target_max_tokens})...")
            progress(None, desc=f"AR load/prefill: {len(plan.prefix)} prompt tokens; budget {target_max_tokens}")
            print(f"[YuE2] AR preflight: prefix={len(plan.prefix)}, budget={target_max_tokens}, "
                  f"context={len(plan.prefix) + target_max_tokens}, CFG={plan.request.guidance}", flush=True)
            t0 = time.perf_counter()
            live_tokens = []
            live = {"last_report": t0, "first_token_at": None}
            def report_music_token(phase, token):
                if phase != "semantic":
                    return
                now = time.perf_counter()
                if live["first_token_at"] is None:
                    live["first_token_at"] = now
                if yproto.CODEC_OFFSET <= token < yproto.CODEC_OFFSET + yproto.CODEC_SIZE:
                    live_tokens.append(int(token) - yproto.CODEC_OFFSET)
                if len(live_tokens) != 1 and now - live["last_report"] < 5 and token != yproto.MUSIC_END:
                    return
                elapsed = now - t0
                decode_time = now - live["first_token_at"]
                rate = (len(live_tokens) - 1) / decode_time if decode_time > 0 and len(live_tokens) > 1 else None
                snapshot = {
                    "status": "partial_tokens_only", "content_tokens": len(live_tokens),
                    "token_budget": target_max_tokens, "represented_audio_seconds": len(live_tokens) / 25,
                    "elapsed_seconds": elapsed, "decode_tokens_per_second": rate,
                    "time_to_first_token_seconds": live["first_token_at"] - t0,
                    "prefix_tokens": len(plan.prefix), "ar": dict(AR_DIAGNOSTICS),
                    "attention_dispatch": dict(ATTENTION_STATUS),
                }
                write_json(output_dir / "semantic-progress.json", snapshot)
                partial = output_dir / "semantic.partial.writing"
                with partial.open("wb") as stream:
                    np.save(stream, np.asarray(live_tokens, dtype=np.int32), allow_pickle=False)
                os.replace(partial, output_dir / "semantic.partial.npy")
                speed = f"{rate:.1f} token/s" if rate is not None else "measuring token rate"
                label = (f"Music tokens {len(live_tokens)}/{target_max_tokens} (budget); "
                         f"{len(live_tokens)/25:.1f}s audio represented; {speed}; "
                         f"execution={AR_DIAGNOSTICS.get('execution')}")
                progress((len(live_tokens), target_max_tokens), desc=label)
                print(f"[YuE2 live] {label}; elapsed={elapsed:.1f}s", flush=True)
                live["last_report"] = now
            semantic = pipe.generate_semantic(
                plan,
                sampling={"max_tokens": target_max_tokens},
                cancelled=_cancelled, on_token=report_music_token
            )
            semantic_seconds = time.perf_counter() - t0
            np.save(output_dir / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32), allow_pickle=False)
            write_json(output_dir / "semantic.json", {"truncated": semantic.truncated, "timing": semantic.timing,
                                                       "token_count": len(semantic.tokens)})
            tps = len(semantic.tokens) / semantic_seconds if semantic_seconds > 0 else 0.0
            status_msg += (f"Step 2 - semantic tokens: {len(semantic.tokens)} in "
                           f"{semantic_seconds:.1f}s ({tps:.1f} tok/s)\n")
            # generate_tokens() records which decode path actually ran. Print it
            # rather than assuming: "cuda_graph" means the graph was captured and
            # replayed, "eager" means it silently fell back.
            t = semantic.timing
            status_msg += (f"         execution={t.get('execution')} "
                           f"attention={t.get('attention')} "
                           f"cfg_branches={t.get('cfg_branches')} "
                           f"prefix={t.get('prefix_tokens')} "
                           f"prefill={t.get('prefill_seconds', 0):.1f}s\n")
            print(f"[YuE2] AR timing: {t}")
            status_msg += f"         attention dispatch (process totals): {dict(ATTENTION_STATUS)}\n"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n[YuE2] Step 3/4: Acoustic flow matching ({ode_steps} {method_clean} steps)...")
            progress(None, desc=f"Music tokens completed; acoustic synthesis ({ode_steps} {method_clean} steps)")
            t0 = time.perf_counter()
            latents = pipe.synthesize(semantic, cancelled=_cancelled)
            synth_seconds = time.perf_counter() - t0
            np.save(output_dir / "latent.npy", latents, allow_pickle=False)
            from yue2.storage import sha256_file
            write_json(output_dir / "latent_manifest.json", {
                "sha256": sha256_file(output_dir / "latent.npy"),
                "shape": list(latents.shape), "model_weights": pipe.weights,
            })
            status_msg += f"Step 3 - flow matching: {latents.shape[0]} latent frames in {synth_seconds:.1f}s\n"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n[YuE2] Step 4/4: Decoding 48 kHz stereo audio on {pipe.vae_device}...")
            progress(None, desc="Acoustic synthesis completed; decoding stereo audio")
            t0 = time.perf_counter()
            audio = pipe.decode(latents)
            decode_seconds = time.perf_counter() - t0
            status_msg += f"Step 4 - VAE decode: {decode_seconds:.1f}s\n"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_time = time.perf_counter() - start_time
            audio_duration = len(audio) / 48000.0
            if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
                raise ValueError("Expected finite stereo audio")
            if not np.isclose(audio_duration, len(semantic.tokens) / 25, atol=0.1):
                raise ValueError("Audio length does not match generated tokens")
            if not np.any(audio):
                raise ValueError("Decoded audio is entirely silent")

            # Hand the player a plain 16-bit buffer and let Gradio write the
            # file: browsers can misreport the duration of 24-bit audio, showing
            # half the true length while still playing it in full. The 24-bit
            # masters are written to disk below and are unaffected.
            playback = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

            wav_path = output_dir / "audio.wav"
            flac_path = output_dir / "audio.flac"
            sf.write(str(wav_path), audio, 48000, subtype="PCM_24")
            sf.write(str(flac_path), audio, 48000, subtype="PCM_24")
            for saved in (wav_path, flac_path):
                info = sf.info(str(saved))
                if info.channels != 2 or info.samplerate != 48000 or not np.isclose(info.duration, audio_duration, atol=0.001):
                    raise ValueError(f"Export validation failed: {saved}")

            preview_path = output_dir / "playback.wav"
            sf.write(str(preview_path), playback, 48000, subtype="PCM_16")

            # Read the duration back out of the written header. If this says
            # 120.0s then the files are correct and any wrong number in the
            # player is purely a browser/Gradio rendering issue, not the audio.
            try:
                info = sf.info(str(wav_path))
                header_note = (f"header: {info.duration:.1f}s, {info.samplerate} Hz, "
                               f"{info.channels} ch, {info.subtype}")
            except Exception as info_err:
                header_note = f"header check unavailable ({info_err})"

            from yue2.storage import write_json
            report = {
                "status": "audio_saved_with_cutoff" if semantic.truncated else "audio_saved_natural_end",
                "max_seconds": target_max_tokens / 25, "actual_seconds": audio_duration,
                "semantic_truncated": semantic.truncated, "abc_truncated": plan.truncated,
                "timing": {"planning": plan_seconds, "semantic": semantic_seconds,
                           "acoustic": synth_seconds, "vae": decode_seconds, "total": total_time},
                "ar_timing": semantic.timing, "attention_dispatch": dict(ATTENTION_STATUS),
                "acoustic_ranges": pipe.last_acoustic_ranges,
                "boundary_seconds": [b / 25 for _, b in pipe.last_acoustic_ranges[:-1]],
                "files": ["audio.wav", "audio.flac", "playback.wav"],
            }
            write_json(output_dir / "status.json", report)
            if semantic.truncated:
                status_msg += "\nDuration/token limit reached; ending may be abrupt."
            if audio_duration < target_max_tokens / 25 - 1:
                status_msg += "\nThe model ended below the selected maximum. No padding or looping was added."
            if report["boundary_seconds"]:
                status_msg += f"\nListen for seams at these seconds: {report['boundary_seconds']}"
            import shutil
            try:
                archive_bytes = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file())
                if shutil.disk_usage(output_dir).free < archive_bytes + 64 * 2**20:
                    raise OSError("Not enough space for the additional ZIP; original files are saved")
                archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
            except OSError as archive_error:
                status_msg += f"\nZIP unavailable: {archive_error}"
            realtime = audio_duration / total_time if total_time > 0 else 0.0
            status_msg += (f"\nDone: {audio_duration:.1f}s of 48 kHz stereo audio in "
                           f"{total_time:.1f}s ({realtime:.2f}x realtime).")
            status_msg += f"\nSaved: {wav_path}"
            status_msg += f"\n       {flac_path}"
            status_msg += f"\n{header_note}"

            progress(1.0, desc="Generation completed successfully!")
            status_msg += "\nDownload the ZIP now, or Save Version with outputs before ending the Kaggle session."
            return str(preview_path), abc_output, status_msg, [str(wav_path), str(flac_path)], archive_path

    except InterruptedError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(output_dir / "status.json", {"status": "cancelled", "message": "Saved stages retained"})
        return None, "", status_msg + f"\nStopped. Saved stages: {output_dir}", None, None
    except Exception as err:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        err_msg = f"Generation error: {str(err)}"
        print(err_msg)
        write_json(output_dir / "status.json", {"status": "failed", "error": str(err)})
        return None, "", err_msg + f"\nSaved stages: {output_dir}", None, None

# -----------------------------------------------------------------------------
# 5. AIQUEST Academy Modern Gradio UI
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

.brand-header {
    text-align: center;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
    padding: 32px 24px;
    border-radius: 20px;
    margin-bottom: 24px;
    box-shadow: 0 12px 32px rgba(79, 70, 229, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.brand-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}

.brand-title {
    color: #ffffff !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    font-size: 2.3em !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 8px 0 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.25);
}

.brand-subtitle {
    color: #e0e7ff !important;
    font-size: 1.05em !important;
    font-weight: 500 !important;
    margin-bottom: 14px !important;
    text-align: center;
}

.brand-badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

.badge-pill {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    color: #ffffff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82em;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.25);
}

.social-buttons {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}

.social-btn {
    padding: 9px 22px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 13.5px;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #ffffff !important;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

.social-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}

.youtube-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.x-btn {
    background: linear-gradient(135deg, #18181b 0%, #27272a 100%);
}

/* Glassmorphism Card Containers */
.card-group {
    /* No backdrop-filter here. It establishes a new stacking context and
       containing block, which clips the absolutely-positioned list that
       gr.Dropdown opens, leaving the control looking dead on click. */
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(229, 231, 235, 0.9) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 16px !important;
}

.dark .card-group {
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

/* Modern Gradient Action Buttons */
#gen-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 14px !important;
    padding: 14px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

#gen-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(99, 102, 241, 0.5) !important;
}

#stop-btn {
    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(244, 63, 94, 0.25) !important;
    transition: all 0.2s ease !important;
}

#stop-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(244, 63, 94, 0.35) !important;
}

#clear-btn {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(100, 116, 139, 0.2) !important;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(100, 116, 139, 0.3) !important;
}

/* Keep long option text from colliding with the dropdown caret. */
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap {
    min-width: 0 !important;
}

.footer {
    text-align: center;
    padding: 24px;
    margin-top: 36px;
    border-top: 1px solid rgba(229, 231, 235, 0.8);
    color: #64748b;
    font-size: 0.9em;
}

.dark .footer {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #94a3b8;
}
"""

# Cleanly close any lingering Gradio instances from previous runs in this kernel
try:
    gr.close_all()
except Exception:
    pass

# gr.themes.Soft fills block labels with the primary hue in dark mode
# (block_label_background_fill_dark), which turns every field label into a solid
# purple slab. Flatten those tokens back to plain text. Theme tokens are the
# supported way to do this; blanket CSS over Gradio's internal label markup
# risks breaking the widgets themselves.
modern_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)
try:
    modern_theme = modern_theme.set(
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_200",
        block_label_text_weight="600",
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        block_title_text_color="*neutral_700",
        block_title_text_color_dark="*neutral_200",
        block_title_text_weight="600",
    )
except Exception as theme_err:
    print(f"Theme token override skipped ({theme_err}); CSS fallback still applies.")

# Seamless cross-version Gradio 4/5/6 theme & CSS compatibility
import inspect
launch_params = inspect.signature(gr.Blocks.launch).parameters
blocks_params = inspect.signature(gr.Blocks.__init__).parameters

blocks_kwargs = {"title": "YuE2-3B Music Generator - AIQUEST Academy"}
if "theme" in blocks_params and "theme" not in launch_params:
    blocks_kwargs["theme"] = modern_theme
if "css" in blocks_params and "css" not in launch_params:
    blocks_kwargs["css"] = CUSTOM_CSS

with gr.Blocks(**blocks_kwargs) as demo:
    # Header Banner
    gr.HTML("""
    <div class='brand-header'>
        <h1 class='brand-title'>🎵 YuE2-3B - Frontier Full-Song Music Generator</h1>
        <p class='brand-subtitle'>Kaggle Dual T4 Edition (GPU T4 x2) - Engineered by <strong>AIQUEST Academy</strong></p>
        <div class='brand-badges'>
            <span class='badge-pill'>⚡ GPU T4 x2 Sharded</span>
            <span class='badge-pill'>🎼 FP16 Native</span>
            <span class='badge-pill'>🎧 48 kHz Stereo FLAC</span>
            <span class='badge-pill'>🚀 Euler Accelerated</span>
        </div>
        <div class='social-buttons'>
            <a href='https://www.youtube.com/@aiquestacademy?sub_confirmation=1' target='_blank' class='social-btn youtube-btn'>
                📺 Subscribe on YouTube
            </a>
            <a href='https://x.com/aiquestacademy' target='_blank' class='social-btn x-btn'>
                🐦 Follow on X
            </a>
        </div>
    </div>
    """)

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            label="✨ Quick Preset Selection (Optional: Choose a ready-to-play song)",
            choices=list(PRESETS.keys()),
            value=None,
            interactive=True
        )

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["card-group"]):
                with gr.Accordion("🎤 Cover an existing song (optional)", open=False, visible=False):
                    gr.Markdown(
                        "YuE2 takes no audio directly. Upload a source song and it is "
                        "transcribed into a **melody score** (SheetSage2) and **lyrics** "
                        "(Whisper). Then write the style you want the cover to take and "
                        "generate.\n\n"
                        "*First run downloads the transcription models onto `cuda:1`: "
                        "0.2 GB for melody, 1.6 GB for lyrics.*"
                    )
                    with gr.Row():
                        cover_audio = gr.Audio(
                            label="Source song",
                            type="filepath",
                            sources=["upload"]
                        )
                        with gr.Column():
                            cover_language = gr.Dropdown(
                                label="Lyric language",
                                choices=COVER_LANGUAGES,
                                value="Auto-detect"
                            )
                            analyze_btn = gr.Button("🔍 Analyze source song", variant="secondary")
                    cover_status = gr.Textbox(
                        label="Analysis status",
                        lines=2,
                        interactive=False
                    )

                style_input = gr.Textbox(
                    label="🎸 Music Style & Instrumentation Prompt",
                    placeholder="e.g. Cyber Metal, driving heavy rock, distorted electric guitars, fast drums, passionate vocal, 135 BPM",
                    lines=3,
                    value=PRESETS["Cyber Metal (High-Energy Heavy Metal)"]["style"]
                )
                lyrics_input = gr.Textbox(
                    label="📝 Song Lyrics (Use section tags: [Verse], [Chorus], [Bridge], [Outro])",
                    placeholder="[Verse 1]\nEnter your lyrics here...\n\n[Chorus]\nEnter your chorus here...",
                    lines=10,
                    value=PRESETS["Cyber Metal (High-Energy Heavy Metal)"]["lyrics"]
                )

                # The radio spans the full width on its own: sharing a row with
                # the controls below leaves it too narrow and wraps every option
                # onto three or four lines.
                cot_selector = gr.Radio(
                    label="🎼 Symbolic Planning (what the model composes first)",
                    choices=[
                        ("Direct to audio (no score)", "off"),
                        ("Melody score, then audio", "melody"),
                        ("Melody + chords score, then audio", "full")
                    ],
                    value="off"
                )
                with gr.Row():
                    duration_selector = gr.Dropdown(
                        label="⏱️ Target Song Duration",
                        choices=DURATION_CHOICES,
                        value=DEFAULT_DURATION,
                        interactive=True,
                        scale=2
                    )
                    seed_input = gr.Number(
                        label="🎲 Seed (-1 = random)",
                        value=-1,
                        precision=0,
                        scale=1
                    )
                    score_only_checkbox = gr.Checkbox(
                        label="📜 Score only (skip audio)",
                        value=False,
                        scale=1
                    )

                with gr.Accordion("⚙️ Advanced Flow Matching Settings", open=False):
                    with gr.Row():
                        ode_steps_slider = gr.Slider(
                            label="ODE Flow Matching Steps",
                            minimum=8,
                            maximum=32,
                            value=16,
                            step=2,
                            info="Default: 16 steps (Fast & high-fidelity on GPU T4 x2)"
                        )
                        ode_method_selector = gr.Radio(
                            label="ODE Solver Method",
                            choices=[
                                ("Euler (Fast - 1 pass/step, Recommended)", "euler"),
                                ("Midpoint (High Quality - 2 passes/step)", "midpoint")
                            ],
                            value="euler",
                            info="Euler evaluates the velocity field once per step"
                        )
                        cfg_scale_slider = gr.Slider(
                            label="CFG Guidance Scale",
                            minimum=1.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            info="Default: 1.0 (Text guidance)"
                        )
                    fuse_checkbox = gr.Checkbox(
                        label="Fused QKV / gate-up projections (experimental)",
                        value=False,
                        info="Fewer, wider matmuls per layer. Costs ~1.9 GB VRAM. "
                             "Listen to one song before trusting it."
                    )
                    abc_input = gr.Textbox(
                        label="Custom ABC Score (Optional: supply your own melody/chords)",
                        placeholder="X: 1\nM: 4/4\nL: 1/8\n...",
                        lines=3
                    )

                # 3 Standard Action Buttons
                with gr.Row():
                    gen_btn = gr.Button("🎵 Generate Song", variant="primary", size="lg", elem_id="gen-btn")
                    stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg", elem_id="stop-btn")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", elem_id="clear-btn")

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["card-group"]):
                audio_output = gr.Audio(
                    label="🎧 Generated 48 kHz Stereo Song",
                    interactive=False
                )
                status_output = gr.Textbox(
                    label="📊 Generation Status & Performance Metrics",
                    lines=5,
                    interactive=False
                )
                score_output = gr.Code(
                    label="📜 Generated ABC Musical Score Plan",
                    language="markdown",
                    interactive=False,
                    lines=8
                )
                master_download = gr.File(label="Download WAV / FLAC masters", file_count="multiple")
                archive_download = gr.File(label="Download complete run ZIP")
                gr.Markdown("**Download the ZIP or Save Version with outputs before ending the Kaggle session.**")
                edit_btn = gr.Button(
                    "✏️ Send score to the ABC editor (edit and regenerate)",
                    variant="secondary",
                    size="sm"
                )

    # Preset Selection Callback
    def apply_preset(choice):
        if choice and choice in PRESETS:
            return PRESETS[choice]["style"], PRESETS[choice]["lyrics"]
        return gr.update(), gr.update()

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[style_input, lyrics_input]
    )

    # Cover: fills the ABC score, the lyrics and the planning mode from the
    # uploaded song, leaving the style prompt for the user to write.
    analyze_btn.click(
        fn=analyze_cover_source,
        inputs=[cover_audio, cover_language, lyrics_input],
        outputs=[abc_input, lyrics_input, cot_selector, cover_status]
    )

    # Editing loop: yue2_infer 0.1.5 has no audio input, so the supported way to
    # revise a song is symbolic. Copy the score the planner produced into the ABC
    # box, edit the notes there, and regenerate with the same seed: the acoustic
    # stages are then re-run against your edited score instead of a fresh plan.
    def send_score_to_editor(score, cot_mode):
        if not score or score.strip().startswith("(No symbolic score"):
            raise gr.Error("Generate a song with planning mode 'Melody Only' or 'Full' first, "
                           "so there is a score to edit.")
        # An external score needs a planning stage to attach to.
        return score, ("melody" if cot_mode == "off" else cot_mode)

    edit_btn.click(
        fn=send_score_to_editor,
        inputs=[score_output, cot_selector],
        outputs=[abc_input, cot_selector]
    )

    # Clear Callback
    def clear_all():
        return "", "", "off", DEFAULT_DURATION, "", -1, None, "", "", None, None

    clear_btn.click(
        fn=clear_all,
        outputs=[style_input, lyrics_input, cot_selector, duration_selector, abc_input, seed_input, audio_output, status_output, score_output, master_download, archive_download]
    )

    # Generation Event Binding
    gen_event = gen_btn.click(
        fn=generate_music,
        inputs=[style_input, lyrics_input, cot_selector, duration_selector, abc_input, seed_input, ode_steps_slider, ode_method_selector, cfg_scale_slider, fuse_checkbox, score_only_checkbox],
        outputs=[audio_output, score_output, status_output, master_download, archive_download]
    )
    # queue=False so the stop request is served immediately, rather than queueing
    # behind the generation it is meant to interrupt.
    stop_btn.click(fn=request_stop, outputs=[status_output], queue=False, cancels=[gen_event])

    # Branded Footer
    gr.HTML("""
    <div class='footer'>
        <p style='text-align: center; margin: 0 0 6px 0;'>
            <strong>AIQUEST Academy</strong> - Empowering the AI Community with Free & Open Source Solutions
        </p>
        <p style='text-align: center; margin: 0; color: #888; font-size: 0.85em;'>
            Optimized for Kaggle Dual NVIDIA Tesla T4 (GPU T4 x2) | Powered by YuE2-3B (M·A·P)
        </p>
    </div>
    """)


# Launch only one sharing mechanism; all audio and recoverable stages remain local.
launch_kwargs = {
    "server_name": "127.0.0.1", "share": True, "inline": False,
    "debug": True, "show_error": True,
    "allowed_paths": [os.environ["YUE2_OUTPUT_ROOT"]],
}
if "theme" in launch_params:
    launch_kwargs["theme"] = modern_theme
if "css" in launch_params:
    launch_kwargs["css"] = CUSTOM_CSS
print("Open the Gradio share link below. No generation is started automatically.", flush=True)
demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
