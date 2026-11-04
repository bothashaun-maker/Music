"""SOURCE TEMPLATE for custom_nodes/KaggleYuE2/__init__.py.

native_yue2.setup.write_comfy_pack() copies this file verbatim into the ComfyUI
custom-node directory at setup time. The pack wraps the proven native FP16 YuE2
engine (native_yue2.engine, yue2_infer 0.1.5) as real, individually executed
ComfyUI stages: load engine -> plan score -> song tokens -> acoustic latents ->
VAE decode -> save song. Weights, precision, attention layout, token budgets,
solver defaults and saved-run artifacts are identical to the managed engine.
"""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import numpy as np
import torch
import soundfile as sf

import folder_paths
import comfy.model_management

# --- native engine bootstrap -------------------------------------------------
# The notebook launch cell passes these; standalone reuse falls back to the
# documented Kaggle paths. Import native_yue2 from the extracted support tree.
SUPPORT = Path(os.environ.get("YUE2_SUPPORT_DIR", "/kaggle/temp/yue2_native_support"))
if str(SUPPORT) not in sys.path and SUPPORT.is_dir():
    sys.path.insert(0, str(SUPPORT))
RUNTIME_ROOT = Path(os.environ.get("YUE2_RUNTIME_ROOT", "/kaggle/temp/yue2_native"))
OUTPUT_ROOT = Path(os.environ.get("YUE2_OUTPUT_ROOT", "/kaggle/working/yue2_native/runs"))
ENGINE_SCRIPT = Path(os.environ.get("YUE2_ENGINE_SCRIPT", SUPPORT / "native_yue2/engine.py"))

import native_yue2.engine as engine  # Applies the T4 patches; loads nothing.
import native_yue2.process as proc

try:  # ComfyUI interrupt type moved between versions.
    from comfy.model_management import InterruptProcessingException
except ImportError:  # pragma: no cover - validated against the pinned checkout
    from comfy_execution.interrupts import InterruptProcessingException

_STATE = {"lock": None, "run": None}
STAGE_TIMES = {}

CATEGORY = "YuE2/Kaggle"
COT_MODES = ["Direct to audio", "Melody score", "Full melody+chords"]
COT_MAP = {"Direct to audio": "off", "Melody score": "melody", "Full melody+chords": "full"}
DURATION_LABELS = [label for label, _ in engine.DURATION_CHOICES]
METHODS = ["euler", "midpoint"]


def _pipe():
    if engine.pipe is None:
        raise RuntimeError("Run the Load YuE2 Engine node first; no engine is loaded.")
    return engine.pipe


def _status_text():
    attention = dict(getattr(engine, "ATTENTION_STATUS", {}))
    ar = dict(getattr(engine, "AR_DIAGNOSTICS", {}))
    return f"attention={attention} ar={ar}"


def _ensure_run():
    if _STATE["run"] is None:
        raise RuntimeError("Add the YuE2 Run Context node; every song needs one run folder.")
    return _STATE["run"]


def _new_run(seed):
    directory = OUTPUT_ROOT / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True, exist_ok=False)
    from yue2.storage import write_json
    write_json(directory / "status.json", {"status": "started"})
    run = {"directory": directory, "seed": seed, "notes": [], "settings": {},
           "cover_source": None}
    _STATE["run"] = run
    STAGE_TIMES.clear()
    return run


def _save_run_json(name, payload):
    from yue2.storage import write_json
    write_json(_ensure_run()["directory"] / name, payload)


def _release_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if not hasattr(engine, "_comfy_original_cancelled"):
    engine._comfy_original_cancelled = engine._cancelled


def _interrupted():
    comfy.model_management.throw_exception_if_processing_interrupted()
    return engine._comfy_original_cancelled()


engine._cancelled = _interrupted


def _cancelled_run(directory):
    _release_gpu_cache()
    from yue2.storage import write_json
    write_json(directory / "status.json", {"status": "cancelled", "message": "Saved stages retained"})


def _require_gpu_room():
    """Preflight against other GPU users, counting our own resident engine."""
    proc.check_gpu_room(include_pid=os.getpid())


def _acquire_exclusive_engine():
    """One engine at a time: stop verified Gradio engines, then own the lock.

    The ComfyUI server process is launched with ``python -u main.py ...`` and
    YUE2_OUTPUT_ROOT in its environment, so proc.identity() can verify it the
    same way it verifies the managed engine script.
    """
    proc.stop_owned(RUNTIME_ROOT, ENGINE_SCRIPT, OUTPUT_ROOT)
    if _STATE["lock"] is None:
        main_script = _comfy_main_script()
        _STATE["lock"] = proc.acquire_engine_lock(RUNTIME_ROOT, main_script, OUTPUT_ROOT)


def _comfy_main_script():
    for argument in sys.argv:
        if argument.endswith("main.py"):
            return Path(argument).resolve()
    raise RuntimeError("Cannot verify the ComfyUI server process; launch it with python -u main.py")


def _semantic_reporter(run, budget):
    directory = run["directory"]
    start = time.perf_counter()
    live = {"tokens": [], "first": None, "last": start}

    def report(phase, token):
        if phase != "semantic":
            return
        if _interrupted():
            raise InterruptProcessingException("YuE2 song generation was interrupted")
        now = time.perf_counter()
        if live["first"] is None:
            live["first"] = now
        offset = getattr(engine.yproto, "CODEC_OFFSET")
        size = getattr(engine.yproto, "CODEC_SIZE")
        if offset <= token < offset + size:
            live["tokens"].append(int(token) - offset)
        if len(live["tokens"]) != 1 and now - live["last"] < 5 and token != engine.yproto.MUSIC_END:
            return
        decode_time = now - live["first"]
        rate = (len(live["tokens"]) - 1) / decode_time if decode_time > 0 and len(live["tokens"]) > 1 else None
        snapshot = {
            "status": "partial_tokens_only", "content_tokens": len(live["tokens"]),
            "token_budget": budget, "represented_audio_seconds": len(live["tokens"]) / 25,
            "elapsed_seconds": now - start, "decode_tokens_per_second": rate,
            "time_to_first_token_seconds": live["first"] - start,
            "attention_dispatch": dict(getattr(engine, "ATTENTION_STATUS", {})),
        }
        _save_run_json("semantic-progress.json", snapshot)
        partial = directory / "semantic.partial.writing"
        with partial.open("wb") as stream:
            np.save(stream, np.asarray(live["tokens"], dtype=np.int32), allow_pickle=False)
        partial.replace(directory / "semantic.partial.npy")
        print(f"[YuE2 live] tokens {len(live['tokens'])}/{budget} | "
              f"{len(live['tokens'])/25:.1f}s audio | "
              f"{('%.1f tok/s' % rate) if rate else 'measuring'} | {_status_text()}", flush=True)
        live["last"] = now

    return report


# =============================================================================
# Nodes
# =============================================================================
class YuE2RunContext:
    """One saved run folder and one seed for the whole song, like the engine UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1})}}

    RETURN_TYPES = ("YUE2_RUN",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Creates this run's folder under /kaggle/working/yue2_native/runs and "
                   "fixes one seed for planning, tokens and acoustics. Seed -1 draws a new one.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Each queue attempt owns a fresh folder, including fixed seeds.

    def execute(self, seed):
        comfy.model_management.throw_exception_if_processing_interrupted()
        current = abs(int(seed)) % (2**31 - 1) if seed is not None and int(seed) != -1 else int(np.random.randint(0, 2**31 - 1))
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        run = _new_run(current)
        print(f"[YuE2] Run {run['directory'].name}; seed {current}", flush=True)
        return (run,)


class YuE2LoadEngine:
    """Loads the pinned FP16 YuE2 engine from your saved assets, exclusively."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run": ("YUE2_RUN",)},
                "optional": {"keep_loaded": ("BOOLEAN", {"default": True, "label_on": "keep for next song",
                                                         "label_off": "release after each song"}),
                             "after_analysis": ("YUE2_ABC",)}}

    RETURN_TYPES = ("YUE2_ENGINE",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Stops verified Gradio engines, owns the engine lock, checks GPU memory, "
                   "then loads YuE2-3B FP16 on GPU 0 with the CUDA-graph decoder. Reuse keeps "
                   "the transformer resident between songs.")

    def execute(self, run, keep_loaded=True, after_analysis=None):
        directory = run["directory"]
        from yue2.storage import write_json
        try:
            _STATE["release_after"] = not bool(keep_loaded)
            if engine.pipe is not None:
                print("[YuE2] Engine already loaded in this server; reusing it.", flush=True)
                _require_gpu_room()
                return {"ui": {"text": ["YuE2 engine already loaded (reused)."]}, "result": (engine,)}
            _acquire_exclusive_engine()
            _require_gpu_room()
            started = time.perf_counter()
            engine.initialize_pipeline()
            STAGE_TIMES["engine_load"] = time.perf_counter() - started
            message = (f"YuE2 engine loaded in {STAGE_TIMES['engine_load']:.1f}s | {_status_text()}")
            print("[YuE2] " + message, flush=True)
            if not keep_loaded:
                _STATE["release_after"] = True
            return {"ui": {"text": [message]}, "result": (engine,)}
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise
        except InterruptedError:
            _cancelled_run(directory)
            raise InterruptProcessingException("YuE2 engine load was interrupted")
        except Exception:
            write_json(directory / "status.json", {"status": "failed", "error": "engine_load", "stage_times": dict(STAGE_TIMES)})
            raise


class YuE2StyleLyrics:
    """Style prompt and section-tagged lyrics, exactly as the engine UI expects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "style": ("STRING", {"default": "", "multiline": True,
                                 "placeholder": "Genre, instruments, mood, vocal direction"}),
            "lyrics": ("STRING", {"default": "", "multiline": True,
                                  "placeholder": "[Verse]\\n...\\n\\n[Chorus]\\n..."})}}

    RETURN_TYPES = ("YUE2_PROMPT",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Write the style prompt and complete lyrics with section tags such as "
                   "[Verse], [Chorus], [Bridge] and [Outro]. Complete structures give the "
                   "model a real ending instead of a stop at the ceiling.")

    def execute(self, style, lyrics):
        cleaned_style = (style or "").strip()
        cleaned_lyrics = (lyrics or "").strip()
        if not cleaned_style:
            raise ValueError("Please enter a style or genre prompt.")
        if not cleaned_lyrics:
            raise ValueError("Please provide song lyrics with section tags (e.g. [Verse], [Chorus]).")
        return ({"style": cleaned_style, "lyrics": cleaned_lyrics},)


class YuE2PlanScore:
    """Optional symbolic plan: direct, melody score, or full melody+chords."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run": ("YUE2_RUN",),
            "engine": ("YUE2_ENGINE",),
            "prompt": ("YUE2_PROMPT",),
            "planning_mode": (COT_MODES, {"default": "Direct to audio"}),
            "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1})},
            "optional": {"abc_token_limit": ("INT", {"default": 4096, "min": 128, "max": 8192})}}

    RETURN_TYPES = ("YUE2_PLAN",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Builds the prompt prefix and, unless Direct to audio is selected, plans an "
                   "ABC score with an adjustable planning-token budget separate from the song ceiling. The "
                   "engine UI. The plan is saved into the run folder and checked for truncation.")

    def execute(self, run, engine, prompt, planning_mode, cfg_scale, abc_token_limit=4096):
        directory = run["directory"]
        from yue2.storage import write_json
        try:
            if not 128 <= int(abc_token_limit) <= 8192:
                raise ValueError("ABC token budget must be 128..8192")
            cot_mode = COT_MAP[planning_mode]
            pipe = _pipe()
            _release_gpu_cache()
            started = time.perf_counter()
            plan = pipe.plan(
                style=prompt["style"], lyrics=prompt["lyrics"], cot=cot_mode, abc=None,
                seed=run["seed"], abc_sampling={"max_tokens": int(abc_token_limit)},
                cfg_scale=float(cfg_scale), cancelled=engine._cancelled)
            STAGE_TIMES["planning"] = time.perf_counter() - started
            plan.save(directory)
            if plan.truncated:
                raise ValueError("ABC score hit its token limit before an ending; the score was "
                                 "saved to the run folder. Choose Direct to audio, simplify the "
                                 "composition, or supply a complete Custom ABC score.")
            write_json(directory / "plan-summary.json", {"mode": cot_mode, "abc_tokens": len(plan.abc_ids),
                                                 "prefix_tokens": len(plan.prefix), "truncated": plan.truncated,
                                                 "seconds": STAGE_TIMES["planning"]})
            print(f"[YuE2] Plan: {len(plan.abc_ids)} score tokens, prefix {len(plan.prefix)} "
                  f"in {STAGE_TIMES['planning']:.1f}s", flush=True)
            return (plan,)
        except InterruptedError:
            _cancelled_run(directory)
            raise InterruptProcessingException("Score planning was interrupted")
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise


class YuE2CustomABC:
    """Supply or paste a complete ABC score instead of planning one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("YUE2_ABC",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Paste an ABC score (from the planner, the ABC editor or a cover analysis) "
                   "and wire it into Generate Song Tokens. Planning is then promoted to melody "
                   "mode, exactly like the engine UI does.")

    def execute(self, abc):
        cleaned = (abc or "").strip()
        if not cleaned:
            raise ValueError("The Custom ABC score is empty.")
        return (cleaned,)


class YuE2SongTokens:
    """Stage 2: autoregressive music tokens under the safety ceiling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run": ("YUE2_RUN",),
            "engine": ("YUE2_ENGINE",),
            "prompt": ("YUE2_PROMPT",),
            "maximum_song_length": (DURATION_LABELS, {"default": "Up to 8 min"})},
            "optional": {
                "plan": ("YUE2_PLAN",),
                "custom_abc": ("YUE2_ABC",)}}

    RETURN_TYPES = ("YUE2_TOKENS",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Generates music tokens with the FP16 CUDA-graph decoder. The length menu is "
                   "a safety ceiling, not a target: the model may end naturally earlier, and a "
                   "cutoff at the ceiling is reported as such. Progress and partial tokens are "
                   "saved into the run folder.")

    def execute(self, run, engine, prompt, maximum_song_length, plan=None, custom_abc=None):
        directory = run["directory"]
        from yue2.storage import write_json
        try:
            pipe = _pipe()
            cleaned_abc = (custom_abc or "").strip() or None
            if plan is not None and cleaned_abc:
                raise ValueError("Wire either Plan Score or Custom ABC into Generate Song Tokens, not both.")
            target_max_tokens = engine.DURATION_MAP[maximum_song_length]
            if cleaned_abc:
                run["notes"].append("Custom ABC score supplied, so planning was promoted to melody mode.")
                started = time.perf_counter()
                plan = pipe.plan(style=prompt["style"], lyrics=prompt["lyrics"], cot="melody",
                                 abc=cleaned_abc, seed=run["seed"],
                                 abc_sampling={"max_tokens": 800},
                                 cfg_scale=1.0, cancelled=engine._cancelled)
                STAGE_TIMES["planning"] = time.perf_counter() - started
                plan.save(directory)
                if plan.truncated:
                    raise ValueError("The supplied ABC score is incomplete for the context window; "
                                     "it was saved to the run folder.")
            elif plan is None:
                plan = pipe.plan(style=prompt["style"], lyrics=prompt["lyrics"], cot="off", abc=None,
                                 seed=run["seed"], abc_sampling={"max_tokens": 800},
                                 cfg_scale=1.0, cancelled=engine._cancelled)
                plan.save(directory)
            if len(plan.prefix) + target_max_tokens > engine.yproto.CONTEXT:
                raise ValueError("Lyrics/score plus song budget exceed the 24,576-token context; shorten the prompt.")

            if cleaned_abc and engine.COVER_STATE.get("abc") == cleaned_abc:
                run["cover_source"] = engine.COVER_STATE.get("metadata")
                write_json(directory / "cover_source.json", run["cover_source"])

            guidance = float(getattr(plan.request, "guidance", 1.0) or 1.0)
            write_json(directory / "settings.json", {
                "style": prompt["style"], "lyrics": prompt["lyrics"], "abc": cleaned_abc,
                "mode": "off" if plan is None or (not cleaned_abc and getattr(plan.request, "cot", "off") == "off") else "planned",
                "seed": run["seed"], "max_tokens": target_max_tokens,
                "max_seconds": target_max_tokens / 25, "ode_steps": 16, "ode_method": "euler",
                "cfg_scale": guidance, "engine_sha256": pipe.runtime_sha256, "backend": pipe.backend,
                "dtype": "float16", "vae_dtype": "float32", "fused_projections": False,
                "model_weights": pipe.weights, "host": "comfyui",
                "prefix_tokens": len(plan.prefix)})
            run["settings"] = {"max_tokens": target_max_tokens}

            _release_gpu_cache()
            _require_gpu_room()
            print(f"[YuE2] AR preflight: prefix={len(plan.prefix)}, budget={target_max_tokens}, "
                  f"context={len(plan.prefix) + target_max_tokens}", flush=True)
            started = time.perf_counter()
            semantic = pipe.generate_semantic(
                plan, sampling={"max_tokens": target_max_tokens},
                cancelled=engine._cancelled, on_token=_semantic_reporter(run, target_max_tokens))
            STAGE_TIMES["semantic"] = time.perf_counter() - started
            np.save(directory / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32), allow_pickle=False)
            write_json(directory / "semantic.json", {"truncated": semantic.truncated,
                                                     "timing": semantic.timing,
                                                     "token_count": len(semantic.tokens)})
            tps = len(semantic.tokens) / STAGE_TIMES["semantic"] if STAGE_TIMES["semantic"] > 0 else 0.0
            timing = semantic.timing or {}
            print(f"[YuE2] Semantic: {len(semantic.tokens)} tokens in {STAGE_TIMES['semantic']:.1f}s "
                  f"({tps:.1f} tok/s) execution={timing.get('execution')} attention={timing.get('attention')}",
                  flush=True)
            run["semantic_truncated"] = semantic.truncated
            run["plan_truncated"] = plan.truncated
            return ({"semantic": semantic, "plan": plan, "max_tokens": target_max_tokens},)
        except InterruptedError:
            _cancelled_run(directory)
            raise InterruptProcessingException("Music-token generation was interrupted")
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise


class YuE2SynthesizeLatents:
    """Stage 3: flow-matching synthesis from tokens to 64-channel latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run": ("YUE2_RUN",),
            "tokens": ("YUE2_TOKENS",),
            "steps": ("INT", {"default": 16, "min": 2, "max": 64}),
            "method": (METHODS, {"default": "euler"})}}

    RETURN_TYPES = ("YUE2_LATENTS",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs the non-autoregressive acoustic solver (default 16 Euler steps, the "
                   "engine default; midpoint costs twice the evaluations). Latents and their "
                   "checksum manifest are saved before decoding so a late failure is recoverable.")

    def execute(self, run, tokens, steps, method):
        directory = run["directory"]
        from yue2.storage import write_json, sha256_file
        try:
            pipe = _pipe()
            import dataclasses
            pipe.generation_config = dataclasses.replace(
                pipe.generation_config, ode_steps=int(steps), ode_method=str(method))
            _release_gpu_cache()
            started = time.perf_counter()
            latents = pipe.synthesize(tokens["semantic"], cancelled=engine._cancelled)
            STAGE_TIMES["acoustic"] = time.perf_counter() - started
            np.save(directory / "latent.npy", latents, allow_pickle=False)
            write_json(directory / "latent_manifest.json", {
                "sha256": sha256_file(directory / "latent.npy"),
                "shape": list(latents.shape), "model_weights": pipe.weights})
            settings_path = directory / "settings.json"
            if settings_path.is_file():
                settings = json.loads(settings_path.read_text())
                settings.update({"ode_steps": int(steps), "ode_method": str(method)})
                write_json(settings_path, settings)
            print(f"[YuE2] Acoustic: {latents.shape[0]} latent frames in {STAGE_TIMES['acoustic']:.1f}s", flush=True)
            return ({"latents": latents},)
        except InterruptedError:
            _cancelled_run(directory)
            raise InterruptProcessingException("Acoustic synthesis was interrupted")
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise


class YuE2DecodeAudio:
    """Stage 4: tiled FP32 VAE decode to 48 kHz stereo on GPU 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run": ("YUE2_RUN",), "latents": ("YUE2_LATENTS",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decodes latents with the tiled FP32 YuE2 VAE (512-frame tiles, 16-frame "
                   "halo) on the second GPU and hands standard ComfyUI AUDIO to preview or save.")

    def execute(self, run, latents):
        directory = run["directory"]
        try:
            pipe = _pipe()
            _release_gpu_cache()
            started = time.perf_counter()
            audio = pipe.decode(latents["latents"])
            STAGE_TIMES["vae"] = time.perf_counter() - started
            if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
                raise ValueError("Expected finite stereo audio")
            if not np.any(audio):
                raise ValueError("Decoded audio is entirely silent")
            print(f"[YuE2] VAE decode: {len(audio)/48000:.1f}s audio in {STAGE_TIMES['vae']:.1f}s", flush=True)
            waveform = torch.from_numpy(audio).T.contiguous().unsqueeze(0).float()
            return ({"waveform": waveform, "sample_rate": 48000},)
        except InterruptedError:
            _cancelled_run(directory)
            raise InterruptProcessingException("VAE decoding was interrupted")
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise


class YuE2SaveSong:
    """Checked PCM-24 masters, full status report and the run ZIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run": ("YUE2_RUN",), "tokens": ("YUE2_TOKENS",), "audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Writes audio.wav / audio.flac (48 kHz stereo PCM-24) and playback.wav into "
                   "the run folder, validates headers and token duration, distinguishes a natural "
                   "ending from a token-limit cutoff, and zips the run for download.")

    def execute(self, run, tokens, audio):
        directory = run["directory"]
        from yue2.storage import write_json
        semantic = tokens["semantic"]
        try:
            waveform = audio["waveform"].detach().cpu()
            rate = int(audio["sample_rate"])
            if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] != 2:
                raise ValueError("Expected one stereo waveform [1,2,T]")
            data = waveform[0].detach().cpu().T.contiguous().numpy()
            if rate != 48000 or not np.isfinite(data).all() or not np.any(data):
                raise ValueError("Expected finite, non-silent stereo audio at 48000 Hz")
            audio_duration = len(data) / rate
            if not np.isclose(audio_duration, len(semantic.tokens) / 25, atol=0.1):
                raise ValueError("Audio length does not match generated tokens")
            playback = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            sf.write(str(directory / "audio.wav"), data, rate, subtype="PCM_24")
            sf.write(str(directory / "audio.flac"), data, rate, subtype="PCM_24")
            sf.write(str(directory / "playback.wav"), playback, rate, subtype="PCM_16")
            for saved in ("audio.wav", "audio.flac"):
                info = sf.info(str(directory / saved))
                if info.channels != 2 or info.samplerate != rate or not np.isclose(info.duration, audio_duration, atol=0.001):
                    raise ValueError(f"Export validation failed: {saved}")

            total_time = sum(STAGE_TIMES.get(k, 0.0) for k in ("planning", "semantic", "acoustic", "vae"))
            report = {
                "status": "audio_saved_with_cutoff" if semantic.truncated else "audio_saved_natural_end",
                "max_seconds": tokens["max_tokens"] / 25, "actual_seconds": audio_duration,
                "semantic_truncated": semantic.truncated, "abc_truncated": run.get("plan_truncated", False),
                "ending_reason": "token_limit" if semantic.truncated else "model_end_token",
                "timing": {k: v for k, v in STAGE_TIMES.items()},
                "ar_timing": semantic.timing, "attention_dispatch": dict(getattr(engine, "ATTENTION_STATUS", {})),
                "acoustic_ranges": _acoustic_ranges(),
                "boundary_seconds": _boundaries(),
                "notes": run["notes"], "files": ["audio.wav", "audio.flac", "playback.wav"],
                "host": "comfyui",
            }
            write_json(directory / "status.json", report)
            status_lines = [
                f"Done: {audio_duration:.1f}s of 48 kHz stereo audio." if semantic.truncated is False
                else f"Exported with cutoff: {audio_duration:.1f}s (reached the safety ceiling).",
                f"Ending reason: {report['ending_reason']}.",
                f"Stage times: " + ", ".join(f"{k}={v:.1f}s" for k, v in STAGE_TIMES.items()),
                "Saved: " + str(directory),
            ]
            if report["boundary_seconds"]:
                status_lines.append("Listen for seams at these seconds: " + str(report["boundary_seconds"]))
            for note in run["notes"]:
                status_lines.insert(0, "Note: " + note)
            import shutil
            try:
                archive_bytes = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
                if shutil.disk_usage(directory).free < archive_bytes + 64 * 2**20:
                    raise OSError("Not enough space for the additional ZIP; original files are saved")
                shutil.make_archive(str(directory), "zip", root_dir=directory)
            except OSError as archive_error:
                status_lines.append(f"ZIP unavailable: {archive_error}")
            _release_gpu_cache()
            _STATE["run"] = None  # The next queued song gets a fresh run folder.
            if _STATE.pop("release_after", False):
                release_engine()
            subfolder = directory.relative_to(folder_paths.get_output_directory()).as_posix()
            return {"ui": {"audio": [{"filename": "audio.flac", "subfolder": subfolder,
                                      "type": "output", "format": "audio/flac"}],
                           "text": ["\n".join(status_lines)]},
                    "result": ({"waveform": waveform, "sample_rate": rate},)}
        except InterruptProcessingException:
            _cancelled_run(directory)
            raise


def _acoustic_ranges():
    pipe = engine.pipe
    return list(getattr(pipe, "last_acoustic_ranges", []) or [])


def _boundaries():
    return [b / 25 for _, b in _acoustic_ranges()[:-1]]


class YuE2AnalyzeMelody:
    """Extracts a melody-only ABC score from an uploaded song (SheetSage2, FP32)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "transcribe_lyrics": ("BOOLEAN", {"default": False}),
            "language": ("STRING", {"default": "Auto-detect"})}}

    RETURN_TYPES = ("YUE2_ABC",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs the T4-compatible FP32 melody transcription on the uploaded song and "
                   "produces an ABC score plus a cover-analysis ZIP. Wire the score into "
                   "Generate Song Tokens (Custom ABC) and write the new style. This is symbolic "
                   "reconstruction, not voice or waveform copying.")

    def execute(self, audio, transcribe_lyrics=False, language="Auto-detect"):
        waveform = audio["waveform"]
        rate = int(audio["sample_rate"])
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError("Expected one audio channel batch")
        if engine.pipe is None:
            _require_gpu_room()
        else:
            from native_yue2.process import check_gpu_room
            check_gpu_room(include_pid=os.getpid())
        descriptor, name = tempfile.mkstemp(suffix=".wav", prefix="yue2_cover_")
        os.close(descriptor)
        temp = Path(name)
        try:
            sf.write(str(temp), waveform[0].detach().cpu().T.contiguous().numpy(), rate, subtype="FLOAT")
            abc, lyrics_out, _mode, status, archive = engine.analyze_cover_source(
                str(temp), language if language else "Auto-detect", "", bool(transcribe_lyrics))
        except Exception as exc:
            message = getattr(exc, "message", None) or str(exc)
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise RuntimeError(f"Cover melody analysis failed: {message}") from exc
        finally:
            temp.unlink(missing_ok=True)
        print("[YuE2 cover] " + status, flush=True)
        return {"ui": {"text": [status + f"\nAnalysis ZIP: {archive}", abc]},
                "result": (abc,)}


class YuE2ReleaseEngine:
    """Releases the transformer and VAE so the Gradio engine can run instead."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Unloads YuE2 weights from this ComfyUI server and releases the engine lock. "
                   "The notebook's Gradio cell calls the /yue2/release route automatically; this "
                   "node does the same from the canvas.")

    def execute(self):
        release_engine()
        return {"ui": {"text": ["YuE2 engine released from the ComfyUI server."]}, "result": ()}


def release_engine():
    if engine.pipe is not None:
        try:
            engine.pipe.close()
        except Exception as error:  # Release must never leave the server broken.
            print(f"[YuE2] Engine close reported: {error}", flush=True)
        engine.pipe = None
    _release_gpu_cache()
    if _STATE["lock"] is not None:
        try:
            _STATE["lock"].close()
        except OSError:
            pass
        _STATE["lock"] = None
    gc.collect()


def _checked_stage(function):
    import functools
    @functools.wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except InterruptProcessingException:
            raise
        except Exception as exc:
            run = _STATE.get("run")
            if run is not None:
                from yue2.storage import write_json
                write_json(run["directory"] / "status.json", {
                    "status": "failed", "stage": function.__qualname__, "error": str(exc)})
                if "out of memory" in str(exc).lower():
                    try:
                        report = proc.gpu_report()
                    except Exception as diagnostic_error:
                        report = {"diagnostic_error": str(diagnostic_error)}
                    write_json(run["directory"] / "gpu-memory-at-error.json", report)
            if "out of memory" in str(exc).lower():
                release_engine()
            raise
    return checked


for _stage in (YuE2LoadEngine, YuE2StyleLyrics, YuE2PlanScore, YuE2CustomABC,
               YuE2SongTokens, YuE2SynthesizeLatents, YuE2DecodeAudio,
               YuE2SaveSong, YuE2AnalyzeMelody):
    _stage.execute = _checked_stage(_stage.execute)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    YuE2RunContext, YuE2LoadEngine, YuE2StyleLyrics, YuE2PlanScore, YuE2CustomABC,
    YuE2SongTokens, YuE2SynthesizeLatents, YuE2DecodeAudio, YuE2SaveSong,
    YuE2AnalyzeMelody, YuE2ReleaseEngine)}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2RunContext": "YuE2 Run Context (folder + seed)",
    "YuE2LoadEngine": "Load YuE2 Engine (FP16, exclusive)",
    "YuE2StyleLyrics": "YuE2 Style and Lyrics",
    "YuE2PlanScore": "YuE2 Plan Score (optional ABC)",
    "YuE2CustomABC": "YuE2 Custom ABC Score",
    "YuE2SongTokens": "YuE2 Generate Song Tokens",
    "YuE2SynthesizeLatents": "YuE2 Acoustic Synthesis",
    "YuE2DecodeAudio": "YuE2 VAE Decode (48 kHz stereo)",
    "YuE2SaveSong": "YuE2 Save Song (masters + status + ZIP)",
    "YuE2AnalyzeMelody": "YuE2 Cover: Analyze Melody",
    "YuE2ReleaseEngine": "YuE2 Release Engine Memory",
}

WEB_DIRECTORY = "./web"

# --- server routes: read-only status + explicit release -----------------------
from aiohttp import web
from server import PromptServer

if getattr(PromptServer, "instance", None) is not None:
    from native_yue2.cover_upload import register_routes as register_cover_upload_routes
    register_cover_upload_routes(PromptServer.instance, folder_paths.get_input_directory(),
                                 folder_paths.get_temp_directory())

    @PromptServer.instance.routes.get("/yue2/status")
    async def yue2_status(request):
        payload = {
            "engine_loaded": engine.pipe is not None,
            "run": _STATE["run"]["directory"].name if _STATE["run"] else None,
            "stage_times": dict(STAGE_TIMES),
            "attention": dict(getattr(engine, "ATTENTION_STATUS", {})),
            "ar": dict(getattr(engine, "AR_DIAGNOSTICS", {})),
            "output_root": str(OUTPUT_ROOT),
        }
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    @PromptServer.instance.routes.post("/yue2/release")
    async def yue2_release(request):
        queue = getattr(PromptServer.instance, "prompt_queue", None)
        if queue is not None and queue.get_current_queue()[0]:
            return web.json_response({"released": False, "reason": "A workflow is running; stop it first."}, status=409)
        release_engine()
        return web.json_response({"released": True}, headers={"Cache-Control": "no-store"})
