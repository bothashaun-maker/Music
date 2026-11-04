"""Managed custom nodes: provenance, recoverable stages, and checked exports.

Installed as custom_nodes/kaggle_yue2_nodes/__init__.py. Model algorithms,
precision selection, attention, sampling and chunking remain native ComfyUI.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

import comfy.model_management
from comfy.text_encoders.yue2 import YuE2TEModel
from kaggle_yue2.recovery import (
    acoustic_ranges, atomic_json, checked_child, digest, load_phase, read_json,
    save_phase, sha256_file,
)

STATE = {}


def gpu_stats():
    return [
        {"device": i, "allocated_gib": torch.cuda.memory_allocated(i) / 2**30,
         "reserved_gib": torch.cuda.memory_reserved(i) / 2**30,
         "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30}
        for i in range(torch.cuda.device_count())
    ]


def write_status():
    atomic_json(STATE["directory"] / "stages.json", STATE["stages"])


def begin_run(run_spec):
    spec = json.loads(run_spec)
    root = Path(os.environ["YUE2_RUN_ROOT"]).resolve()
    directory = checked_child(root, spec["run_id"])
    if not (directory / "request.json").is_file():
        raise ValueError("Run must first be created by the Kaggle client")
    expected = read_json(directory / "request.json")
    if spec != expected:
        raise ValueError("Run specification disagrees with the saved request")
    STATE.clear()
    STATE.update(directory=directory, spec=spec, stages={"status": "running", "stages": {}})
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    write_status()


def run_directory():
    if not STATE:
        raise RuntimeError("Execute the run-context node before generation")
    return STATE["directory"]


def observe_generation(original):
    """Capture native stopping status; never reinterpret a cutoff as success."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        if not STATE:
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        phase = values["phase"]
        # Everything here is scalar metadata or token IDs, never model tensors.
        values["dtype"] = str(values["dtype"])
        effective = {
            "arguments": values,
            "runtime": STATE["spec"]["runtime"],
        }
        key = digest(effective)
        destination = run_directory() / f"{phase}_tokens.json"
        resume = STATE["spec"].get("resume_from")
        started = time.perf_counter()
        try:
            if resume and (Path(resume) / destination.name).is_file():
                tokens, truncated = load_phase(Path(resume) / destination.name, key, phase)
                restored = True
            else:
                tokens, truncated = original(self, *args, **kwargs)
                restored = False
            save_phase(destination, key, tokens, truncated)
            atomic_json(run_directory() / f"{phase}_effective.json", effective)
            STATE["stages"]["stages"][phase] = {
                "tokens": len(tokens), "truncated": truncated,
                "ending": "token_limit" if truncated else "end_token",
                "recovered": restored, "seconds": time.perf_counter() - started,
                "dtype": values["dtype"], "gpu": gpu_stats(),
            }
            write_status()
            if phase == "abc" and truncated:
                raise RuntimeError(
                    "ABC reached its token limit before an ending. Tokens were saved; "
                    "increase the ABC budget or simplify the composition before rendering."
                )
            if not tokens:
                raise RuntimeError(f"YuE2 produced no {phase} tokens")
            return tokens, truncated
        except BaseException as exc:
            STATE["stages"].update(status="failed", error=str(exc), failed_stage=phase)
            write_status()
            raise
    return observed


def observe_conditioning(original):
    @functools.wraps(original)
    def observed(self, prefix, tokens, dtype):
        if STATE:
            ranges = acoustic_ranges(len(tokens), len(prefix))
            STATE["stages"]["acoustic_layout"] = {
                "prefix_tokens": len(prefix), "frames": len(tokens),
                "ranges": ranges,
                "boundary_seconds": [a / 25 for a, _ in ranges[1:]],
                "continuity_requires_listening": len(ranges) > 1,
            }
            write_status()
            print(f"[Kaggle YuE2] Acoustic chunks: {ranges}; dtype={dtype}", flush=True)
        return original(self, prefix, tokens, dtype)
    return observed


# Keep wrappers idempotent when this managed module is imported again.
if not getattr(YuE2TEModel._generate, "_kaggle_observed", False):
    YuE2TEModel._generate = observe_generation(YuE2TEModel._generate)
    YuE2TEModel._generate._kaggle_observed = True
if not getattr(YuE2TEModel._acoustic_conditioning, "_kaggle_observed", False):
    YuE2TEModel._acoustic_conditioning = observe_conditioning(YuE2TEModel._acoustic_conditioning)
    YuE2TEModel._acoustic_conditioning._kaggle_observed = True


class YuE2RunContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "run_spec": ("STRING",)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "execute"
    CATEGORY = "audio/yue2/kaggle"

    def execute(self, clip, run_spec):
        begin_run(run_spec)
        return (clip,)


class YuE2PersistABC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ("STRING", {"forceInput": True}), "clip": ("CLIP",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "audio/yue2/kaggle"
    OUTPUT_NODE = True

    def execute(self, abc, clip):
        if not abc.strip():
            raise ValueError("No ABC score was produced")
        (run_directory() / "score.abc").write_text(abc)
        return (abc,)


class YuE2SaveLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "audio/yue2/kaggle"
    OUTPUT_NODE = True

    def execute(self, samples):
        tensor = samples["samples"].detach().float().cpu().contiguous()
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (1, 64):
            raise ValueError("Expected one YuE2 latent [1,64,T]")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Acoustic sampler produced non-finite latents; export stopped")
        path = run_directory() / "latents.safetensors"
        temporary = path.with_suffix(".writing")
        metadata = {"downscale_ratio_temporal": "1920", "type": "audio"}
        save_file({"samples": tensor}, str(temporary), metadata=metadata)
        os.replace(temporary, path)
        atomic_json(run_directory() / "latents_manifest.json", {
            "sha256": sha256_file(path), "shape": list(tensor.shape),
            "runtime": STATE["spec"]["runtime"], "metadata": metadata,
        })
        STATE["stages"].update(status="latents_saved", latent_frames=tensor.shape[-1], gpu=gpu_stats())
        write_status()
        comfy.model_management.throw_exception_if_processing_interrupted()
        return (samples,)


class YuE2LoadLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_spec": ("STRING",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "audio/yue2/kaggle"

    def execute(self, run_spec):
        begin_run(run_spec)
        source = Path(STATE["spec"]["resume_from"]).resolve()
        manifest = read_json(source / "latents_manifest.json")
        if manifest["runtime"]["model_sha256"] != STATE["spec"]["runtime"]["model_sha256"]:
            raise ValueError("Saved latents were made with a different checkpoint")
        path = source / "latents.safetensors"
        if sha256_file(path) != manifest["sha256"]:
            raise ValueError("Saved latents failed their checksum")
        tensors = load_file(str(path), device="cpu")
        tensor = tensors["samples"]
        if list(tensor.shape) != manifest["shape"] or not torch.isfinite(tensor).all():
            raise ValueError("Invalid saved latent shape or values")
        previous = source / "stages.json"
        if previous.is_file():
            old = read_json(previous)
            STATE["stages"].update({k: old[k] for k in ("stages", "acoustic_layout") if k in old})
        STATE["stages"]["decode_recovered_from"] = str(source)
        write_status()
        return ({"samples": tensor, "type": "audio", "downscale_ratio_temporal": 1920},)


class YuE2SaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "audio/yue2/kaggle"
    OUTPUT_NODE = True

    def execute(self, audio):
        waveform = audio["waveform"].detach().float().cpu()
        if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 2):
            raise ValueError("Expected one stereo waveform [1,2,T]")
        rate = int(audio["sample_rate"])
        if rate != 48000 or waveform.shape[-1] < rate:
            raise ValueError("Expected at least one second of 48 kHz audio")
        data = waveform[0].T.contiguous().numpy()
        if not np.isfinite(data).all():
            raise FloatingPointError("VAE produced non-finite audio; export stopped")
        peak = float(np.abs(data).max())
        squares = sum(float(np.square(block, dtype=np.float64).sum())
                      for block in np.array_split(data, max(1, len(data) // rate)))
        rms = (squares / data.size) ** 0.5
        if rms < 1e-6:
            raise ValueError("Decoded audio is effectively silent; latents remain available")
        # Prevent PCM conversion from clipping; record the applied gain explicitly.
        gain = min(1.0, 0.999 / peak) if peak else 1.0
        paths = []
        for name, fmt in (("audio.flac", "FLAC"), ("audio.wav", "WAV")):
            comfy.model_management.throw_exception_if_processing_interrupted()
            path = run_directory() / name
            temporary = path.with_suffix(".writing")
            with sf.SoundFile(temporary, "w", samplerate=rate, channels=2,
                              format=fmt, subtype="PCM_24") as stream:
                for start in range(0, len(data), rate):
                    stream.write(data[start:start + rate] * gain)
            os.replace(temporary, path)
            info = sf.info(path)
            if info.frames != len(data) or info.samplerate != rate or info.channels != 2:
                raise RuntimeError("Export header does not match decoded audio")
            paths.append(str(path))
        semantic = STATE["stages"]["stages"].get("semantic")
        truncated = semantic.get("truncated") if semantic is not None else None
        duration = len(data) / rate
        STATE["stages"].update(
            status=("audio_saved_with_cutoff" if truncated else
                    "audio_saved_natural_end" if truncated is False else "audio_saved_ending_unknown"),
            audio={"duration_seconds": duration, "sample_rate": rate, "channels": 2,
                   "subtype": "PCM_24", "peak_before_export": peak, "rms": rms,
                   "export_gain": gain, "paths": paths,
                   "maximum_requested_seconds": STATE["spec"]["song"]["max_duration"],
                   "near_requested_limit": duration >= 0.95 * STATE["spec"]["song"]["max_duration"]},
            gpu=gpu_stats(),
        )
        write_status()
        return (audio,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    YuE2RunContext, YuE2PersistABC, YuE2SaveLatents, YuE2LoadLatents, YuE2SaveAudio,
)}
