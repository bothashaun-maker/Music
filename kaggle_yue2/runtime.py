"""Notebook-facing orchestration. Imports do not install packages or start services."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from .recovery import atomic_json, digest, read_json, sha256_file

COMFY_COMMIT = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
MODEL_REVISION = "2f76ca75e6ee094169de899cc7fc99d6887e2196"
CHECKPOINT = "yue2_3b_int8_convrot.safetensors"
MODEL_SHA256 = "96fe199377309001ed8cd26a944baeee8cc31a20ba7c36d1d3c0a7e1f4149db6"
MODEL_SIZE = 3960938800
OBSERVER_VERSION = 1
PIP_VERSION = "25.3"
PIP_WHEEL = f"pip-{PIP_VERSION}-py3-none-any.whl"
PIP_URL = ("https://files.pythonhosted.org/packages/44/3c/"
           "d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/" + PIP_WHEEL)
PIP_SHA256 = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"


def command(args, *, cwd=None, env=None, capture=False):
    try:
        return subprocess.run([str(x) for x in args], cwd=cwd, env=env, check=True,
                              text=True, capture_output=capture)
    except subprocess.CalledProcessError as exc:
        if capture:
            print((exc.stderr or exc.stdout or "")[-6000:], file=sys.stderr)
        raise


def read_tail(path, lines=25):
    path = Path(path)
    if not path.exists():
        return "(no log yet)"
    # Seek backwards rather than loading hours of progress output into RAM.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 16000))
        return "\n".join(stream.read().decode(errors="replace").splitlines()[-lines:])


@dataclass(frozen=True)
class SongConfig:
    style: str
    lyrics: str
    max_duration: float = 30.0
    mode: str = "full"
    abc: str = ""
    abc_seed: int = 10
    semantic_seed: int = 17
    acoustic_seed: int = 7
    max_abc_tokens: int = 4096
    steps: int = 32
    semantic_cfg: float = 1.0
    tile_size: int = 512
    tile_overlap: int = 64
    plan_only: bool = False

    def __post_init__(self):
        if not isinstance(self.style, str) or not self.style.strip():
            raise ValueError("Provide a style prompt")
        if not isinstance(self.lyrics, str) or not self.lyrics.strip():
            raise ValueError("Provide complete, section-tagged lyrics")
        if not math.isfinite(self.max_duration) or not 8 <= self.max_duration <= 480:
            raise ValueError("Maximum duration must be 8..480 seconds (up to eight minutes)")
        if self.mode not in {"full", "melody", "off"}:
            raise ValueError("Planning mode must be full, melody or off")
        if not isinstance(self.abc, str) or type(self.plan_only) is not bool:
            raise ValueError("ABC must be text and plan_only must be a boolean")
        if self.mode == "off" and (self.abc.strip() or self.plan_only):
            raise ValueError("ABC and plan-only require full or melody planning")
        for seed in (self.abc_seed, self.semantic_seed, self.acoustic_seed):
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("Each seed must be an integer in 0..2**63-1")
        if type(self.max_abc_tokens) is not int or not 32 <= self.max_abc_tokens <= 16000:
            raise ValueError("ABC budget must be an integer in 32..16000")
        if type(self.steps) is not int or not 8 <= self.steps <= 64:
            raise ValueError("Acoustic steps must be an integer in 8..64")
        if not math.isfinite(self.semantic_cfg) or not 1 <= self.semantic_cfg <= 2:
            raise ValueError("Semantic CFG must be 1..2; start with 1 to save memory")
        if (type(self.tile_size) is not int or self.tile_size % 8 or not 64 <= self.tile_size <= 2048
                or type(self.tile_overlap) is not int or self.tile_overlap % 8
                or not 8 <= self.tile_overlap < self.tile_size // 2):
            raise ValueError("Use tile/overlap multiples of 8; overlap must be below half the tile")


def build_prompt(song, spec, *, decode_only=False):
    """Build an API graph, not a frontend workflow with stale widget values."""
    run_spec = json.dumps(spec, sort_keys=True)
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}}}
    if decode_only:
        graph["6"] = {"class_type": "YuE2LoadLatents", "inputs": {"run_spec": run_spec}}
    else:
        graph["2"] = {"class_type": "YuE2RunContext", "inputs": {"clip": ["1", 1], "run_spec": run_spec}}
        abc = song.abc
        if song.mode != "off" and not abc.strip():
            graph["3"] = {"class_type": "YuE2GenerateABC", "inputs": {
                "clip": ["2", 0], "style": song.style, "lyrics": song.lyrics,
                "seed": song.abc_seed, "mode": song.mode, "max_abc_tokens": song.max_abc_tokens,
                "temperature": 0.7, "top_p": 0.9, "top_k": 30,
                "repetition_penalty": 1.005, "penalty_window": 100,
            }}
            abc = ["3", 0]
        if song.mode != "off":
            # For supplied scores the music node still depends on run context.
            # A context input on the persistence node ensures it also runs first.
            graph["4"] = {"class_type": "YuE2PersistABC", "inputs": {"abc": abc, "clip": ["2", 0]}}
            abc = ["4", 0]
        if song.plan_only:
            return graph
        graph["5"] = {"class_type": "YuE2GenerateMusic", "inputs": {
            "clip": ["2", 0], "style": song.style, "lyrics": song.lyrics,
            "abc": abc, "seed": song.semantic_seed,
            "mode": song.mode if song.mode != "off" else "melody",
            "max_duration": song.max_duration, "temperature": 1.0, "top_p": 0.95,
            "top_k": 100, "repetition_penalty": 1.2, "cfg_scale": song.semantic_cfg,
        }}
        graph["6"] = {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["5", 1], "batch_size": 1}}
        graph["7"] = {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": song.acoustic_seed, "steps": song.steps,
            "cfg": 1.0, "sampler_name": "dpm_2", "scheduler": "sgm_uniform", "denoise": 1.0,
        }}
    graph["8"] = {"class_type": "YuE2SaveLatents", "inputs": {"samples": ["6" if decode_only else "7", 0]}}
    graph["9"] = {"class_type": "VAEDecodeAudioTiled", "inputs": {
        "samples": ["8", 0], "vae": ["1", 2], "tile_size": song.tile_size, "overlap": song.tile_overlap,
    }}
    graph["10"] = {"class_type": "YuE2SaveAudio", "inputs": {"audio": ["9", 0]}}
    return graph


class KaggleYuE2:
    def __init__(self, root="/kaggle/working/yue2", *, gpu=0):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkout = self.root / "ComfyUI"
        self.venv = self.root / "venv"
        self.python = self.venv / "bin/python"
        self.runs = self.root / "runs"
        self.runs.mkdir(exist_ok=True)
        self.log = self.root / "server.log"
        self.gpu = int(gpu)
        self.process = None
        self.base_url = None

    def environment(self):
        env = os.environ.copy()
        support = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = support + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["YUE2_RUN_ROOT"] = str(self.runs)
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _pip_wheel(self):
        """Fetch a verified wheel, without depending on pip or ensurepip."""
        wheel = self.root / "bootstrap" / PIP_WHEEL
        wheel.parent.mkdir(parents=True, exist_ok=True)
        if not wheel.exists():
            temporary = wheel.with_suffix(".part")
            try:
                with urllib.request.urlopen(PIP_URL, timeout=60) as response, temporary.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if sha256_file(temporary) != PIP_SHA256:
                    raise RuntimeError("Bootstrap pip wheel SHA256 mismatch; refusing to execute it")
                os.replace(temporary, wheel)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256_file(wheel) != PIP_SHA256:
            raise RuntimeError(f"Bootstrap pip wheel SHA256 mismatch: remove {wheel} and retry setup")
        return wheel

    def prepare_environment(self):
        """Create or repair the overlay even after an interrupted ensurepip step."""
        # Kaggle's distro Python may have a broken/missing ensurepip. Do not call
        # it, and do not use the mere existence of bin/python as a success marker.
        command([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", self.venv])
        check = r'''
import json, pathlib, sys, pip
root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(sys.prefix).resolve() == root and sys.prefix != sys.base_prefix
assert pathlib.Path(pip.__file__).resolve().is_relative_to(root), "pip is inherited, not local"
assert pip.__version__ == sys.argv[2]
print(json.dumps({"python": sys.version.split()[0], "pip": pip.__version__, "prefix": str(root)}))
'''
        checked = subprocess.run([str(self.python), "-I", "-c", check, str(self.venv), PIP_VERSION],
                                 capture_output=True, text=True)
        if checked.returncode:
            wheel = self._pip_wheel()
            bootstrap = r'''
import pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
wheel = sys.argv[2]
assert pathlib.Path(sys.prefix).resolve() == root and sys.prefix != sys.base_prefix
sys.path.insert(0, wheel)
sys.argv = ["pip", "--isolated", "--disable-pip-version-check", "install",
            "--no-index", "--no-deps", "--ignore-installed", "--no-compile", wheel]
runpy.run_module("pip", run_name="__main__")
'''
            command([self.python, "-I", "-c", bootstrap, self.venv, wheel])
        # Check pip belongs to the overlay: an inherited global pip is not enough.
        result = command([self.python, "-I", "-c", check, self.venv, PIP_VERSION], capture=True)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        atomic_json(self.root / "bootstrap.json", report)
        print(f"Environment ready: Python {report['python']}, local pip {report['pip']} (ensurepip not used).")
        return report

    def install(self, local_checkpoint=None):
        """Use an isolated overlay; never upgrade Kaggle's system PyTorch/NumPy."""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Stop the managed server before refreshing its installation")
        if not sys.platform.startswith("linux") or sys.version_info < (3, 10):
            raise RuntimeError("Use a Linux Kaggle Python GPU notebook (Python 3.10+)")
        if shutil.which("git") is None:
            raise RuntimeError("Git is required in the Kaggle image")
        free_gib = shutil.disk_usage(self.root).free / 2**30
        if free_gib < 12:
            raise RuntimeError(f"Only {free_gib:.1f} GiB free; allow at least 12 GiB for setup and outputs")
        if not self.checkout.exists():
            self.checkout.mkdir()
            command(["git", "init", "-q"], cwd=self.checkout)
        current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.checkout, capture_output=True, text=True)
        if current.returncode and not (self.checkout / "main.py").exists():
            command(["git", "fetch", "--depth=1", "https://github.com/Comfy-Org/ComfyUI.git", COMFY_COMMIT], cwd=self.checkout)
            command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=self.checkout)
        elif current.stdout.strip() != COMFY_COMMIT:
            raise RuntimeError("Existing ComfyUI checkout differs from the pinned revision; use a new root")
        dirty = command(["git", "diff", "--name-only", "HEAD"], cwd=self.checkout, capture=True).stdout
        if dirty.strip():
            raise RuntimeError("ComfyUI source has edits; use a fresh root rather than overwriting them")
        self.prepare_environment()
        pins = []
        for name in ("torch", "torchvision", "torchaudio", "numpy", "scipy", "numba", "llvmlite"):
            try:
                pins.append(f"{name}=={importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                pass
        if not any(x.startswith("torch==") for x in pins):
            raise RuntimeError("Kaggle's GPU PyTorch is missing; select a current GPU notebook image")
        constraint = self.root / "kaggle-stack-constraints.txt"
        constraint.write_text("\n".join(pins) + "\n")
        # No --upgrade: retain the image's binary scientific/CUDA stack.
        command([self.python, "-m", "pip", "install", "-r", self.checkout / "requirements.txt",
                 "-c", constraint, "transformers==4.57.6", "huggingface-hub==0.36.2", "soundfile==0.13.1"],
                env=self.environment())
        probe = r'''
import json, platform, torch
import numpy, scipy, soundfile, transformers, comfy_kitchen
import torch.nn.functional as F
assert torch.cuda.is_available(), "Enable Kaggle GPU acceleration"
assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 8), "Use a Kaggle image with PyTorch >=2.8"
devices=[]
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    devices.append({"index":i,"name":p.name,"vram_gib":p.total_memory/2**30,"capability":list(torch.cuda.get_device_capability(i))})
q=torch.randn((1,4,8,64),device="cuda",dtype=torch.float32)
k=torch.randn((1,2,8,64),device="cuda",dtype=torch.float32)
v=torch.randn_like(k)
assert torch.isfinite(F.scaled_dot_product_attention(q,k,v,enable_gqa=True)).all()
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"devices":devices}))
'''
        hardware = json.loads(command([self.python, "-c", probe], env=self.environment(), capture=True).stdout.strip().splitlines()[-1])
        if self.gpu >= len(hardware["devices"]) or self.gpu < 0:
            raise RuntimeError(f"GPU {self.gpu} is unavailable")
        atomic_json(self.root / "hardware.json", hardware)
        packages = json.loads(command([self.python, "-m", "pip", "list", "--format=json"], capture=True).stdout)
        freeze = "\n".join(sorted(f"{p['name']}=={p['version']}" for p in packages)) + "\n"
        (self.root / "installed-packages.txt").write_text(freeze)
        nodes = self.checkout / "custom_nodes/kaggle_yue2_nodes"
        nodes.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).with_name("comfy_nodes.py"), nodes / "__init__.py")
        target = self.checkout / "models/checkpoints" / CHECKPOINT
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_checkpoint:
            source = Path(local_checkpoint).resolve()
        elif target.is_file():
            source = target.resolve()
        else:
            downloader = r'''
from huggingface_hub import hf_hub_download
import sys
print(hf_hub_download("Comfy-Org/YuE2", "checkpoints/"+sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]))
'''
            source = Path(command([self.python, "-c", downloader, CHECKPOINT, MODEL_REVISION,
                                   self.root / "hub-cache"], env=self.environment(), capture=True).stdout.strip().splitlines()[-1])
        if source.stat().st_size != MODEL_SIZE or sha256_file(source) != MODEL_SHA256:
            raise RuntimeError("Checkpoint size/SHA256 mismatch; do not use this file")
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError("Checkpoint path already exists; use a new root to preserve it")
        else:
            target.symlink_to(source)
        runtime = {"comfy_commit": COMFY_COMMIT, "model_revision": MODEL_REVISION,
                   "model_sha256": MODEL_SHA256, "observer_version": OBSERVER_VERSION,
                   "observer_sha256": sha256_file(Path(__file__).with_name("comfy_nodes.py")),
                   "packages_sha256": digest(sorted(freeze.splitlines())), "hardware": hardware}
        atomic_json(self.root / "runtime.json", runtime)
        print(json.dumps(hardware, indent=2))
        print("Setup files prepared. GPU generation has not been validated until a song runs.")
        print("Native YuE2 uses FP32 text generation on T4. GPU selection is explicit; no automatic pooling of two T4s.")
        return runtime

    def request(self, method, endpoint, payload=None, *, timeout=30):
        if self.base_url is None:
            raise RuntimeError("Start this notebook's managed server first")
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + endpoint, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI {endpoint}: {exc.code} {exc.read().decode()[:4000]}") from exc

    def start(self, timeout=180):
        if self.process is not None and self.process.poll() is None:
            return self.request("GET", "/system_stats")
        if not (self.root / "runtime.json").is_file():
            raise RuntimeError("Run installation/preflight first")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        args = [self.python, "main.py", "--listen", "127.0.0.1", "--port", port,
                "--default-device", self.gpu, "--disable-auto-launch", "--disable-partner-nodes",
                "--disable-all-custom-nodes", "--whitelist-custom-nodes", "kaggle_yue2_nodes",
                "--cache-none", "--fp32-vae", "--use-split-cross-attention", "--disable-xformers",
                "--reserve-vram", "1.5", "--output-directory", self.runs]
        with self.log.open("a") as stream:
            self.process = subprocess.Popen([str(x) for x in args], cwd=self.checkout,
                                            env=self.environment(), stdout=stream, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("ComfyUI stopped during startup:\n" + read_tail(self.log))
            try:
                self.request("GET", "/system_stats", timeout=2)
                info = self.request("GET", "/object_info")
                required = {"CheckpointLoaderSimple", "YuE2GenerateABC", "YuE2GenerateMusic",
                            "EmptyYuE2LatentAudio", "KSampler", "VAEDecodeAudioTiled",
                            "YuE2RunContext", "YuE2PersistABC", "YuE2SaveLatents", "YuE2LoadLatents", "YuE2SaveAudio"}
                missing = required - info.keys()
                if missing:
                    raise ValueError(f"Required nodes failed to load: {sorted(missing)}\n{read_tail(self.log)}")
                atomic_json(self.root / "node-schemas.json", {name: info[name] for name in sorted(required)})
                print(f"Managed ComfyUI process started on GPU {self.gpu}. No public tunnel is required.")
                return self.request("GET", "/system_stats")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(1)
        self.stop()
        raise TimeoutError("ComfyUI startup timed out:\n" + read_tail(self.log))

    def _ensure_idle(self):
        queue = self.request("GET", "/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("The managed server is busy. Wait, cancel, or restart it before submitting another run")

    def cancel(self):
        # This server belongs exclusively to this notebook. Interrupt active work;
        # do not claim synchronous completion or delete anything in the queue.
        return self.request("POST", "/interrupt", {})

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        self.base_url = None

    def run(self, song, *, resume_from=None, decode_only=False, timeout=7200, poll_interval=5):
        if not isinstance(song, SongConfig):
            raise TypeError("Pass a SongConfig")
        self._ensure_idle()
        runtime = read_json(self.root / "runtime.json")
        source = str(Path(resume_from).resolve()) if resume_from else None
        if decode_only and not source:
            raise ValueError("Decode recovery requires a saved run directory")
        if source and not (Path(source) / "request.json").is_file():
            raise ValueError("Recovery directory has no request.json")
        if decode_only:
            old_song = read_json(Path(source) / "request.json")["song"]
            current_song = asdict(song)
            for key in ("tile_size", "tile_overlap"):
                old_song.pop(key, None)
                current_song.pop(key, None)
            if old_song != current_song:
                raise ValueError("Decode recovery must use the original song settings; only VAE tiles may change")
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
        directory = self.runs / run_id
        directory.mkdir(exist_ok=False)
        spec = {"run_id": run_id, "song": asdict(song), "runtime": runtime,
                "resume_from": source, "decode_only": decode_only}
        atomic_json(directory / "request.json", spec)
        graph = build_prompt(song, spec, decode_only=decode_only)
        atomic_json(directory / "api-prompt.json", graph)
        if source and decode_only:
            for name in ("score.abc", "abc_tokens.json", "semantic_tokens.json", "abc_effective.json", "semantic_effective.json"):
                if (Path(source) / name).is_file():
                    shutil.copyfile(Path(source) / name, directory / name)
        # Sampling every two seconds measures both cards without initializing
        # CUDA in the notebook kernel. Torch stage peaks are saved by the nodes.
        telemetry = None
        if shutil.which("nvidia-smi"):
            with (directory / "gpu.csv").open("w") as stream:
                telemetry = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,utilization.gpu",
                    "--format=csv", "--loop=2"], stdout=stream, stderr=subprocess.STDOUT)
        print(f"Run: {directory}")
        print(f"Maximum {song.max_duration / 60:.2f} minutes; early endings and cutoffs are reported separately.")
        if song.max_duration >= 420:
            print("7–8 minutes is experimental on Kaggle: check acoustic boundaries and the final outro.")
        submitted = False
        started = time.monotonic()
        heartbeat = started
        log_offset = self.log.stat().st_size if self.log.exists() else 0
        last_stage = None
        try:
            response = self.request("POST", "/prompt", {"prompt": graph, "client_id": run_id})
            if response.get("node_errors") or not response.get("prompt_id"):
                raise RuntimeError("Prompt validation failed: " + json.dumps(response))
            submitted = True
            prompt_id = response["prompt_id"]
            atomic_json(directory / "submission.json", response)
            while time.monotonic() - started < timeout:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("Managed ComfyUI process exited:\n" + read_tail(self.log))
                history = self.request("GET", f"/history/{prompt_id}")
                if prompt_id in history:
                    record = history[prompt_id]
                    atomic_json(directory / "history.json", record)
                    status = record.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError("ComfyUI generation failed: " + json.dumps(status.get("messages", []))[-6000:])
                    if status.get("completed"):
                        if song.plan_only and not decode_only:
                            if not (directory / "score.abc").is_file():
                                raise RuntimeError("Plan-only run completed without a saved score")
                            result = read_json(directory / "stages.json")
                            result["status"] = "score_saved"
                            atomic_json(directory / "stages.json", result)
                        else:
                            result = read_json(directory / "stages.json")
                            if not result.get("audio") or not (directory / "audio.flac").is_file():
                                raise RuntimeError("Server completed without a verified audio export")
                        self._summarize(directory, result, time.monotonic() - started)
                        return directory
                stage_path = directory / "stages.json"
                if stage_path.is_file():
                    stage = read_json(stage_path)
                    label = (stage.get("status"), tuple(stage.get("stages", {})))
                    if label != last_stage:
                        print(f"[{(time.monotonic() - started)/60:.1f} min] {label}")
                        last_stage = label
                if time.monotonic() - heartbeat >= 60:
                    print(f"[{(time.monotonic() - started)/60:.1f} min] Still processing:\n{read_tail(self.log, 2)}")
                    heartbeat = time.monotonic()
                time.sleep(poll_interval)
            raise TimeoutError(f"Run exceeded {timeout/60:.0f} minutes; saved stages remain in {directory}")
        except BaseException as exc:
            if submitted:
                try:
                    self.cancel()
                except Exception:
                    pass
            atomic_json(directory / "client-error.json", {"type": type(exc).__name__, "error": str(exc)})
            print(f"Stopped/failed. Saved work: {directory}\n{read_tail(self.log, 12)}")
            raise
        finally:
            if telemetry is not None:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait(timeout=5)
            if self.log.is_file():
                # Include bounded, run-specific diagnostics in downloadable ZIPs.
                with self.log.open("rb") as stream, (directory / "server-log.txt").open("wb") as destination:
                    size = self.log.stat().st_size
                    stream.seek(max(log_offset, size - 20 * 1024 * 1024))
                    shutil.copyfileobj(stream, destination, length=1024 * 1024)

    def decode_saved(self, directory, *, tile_size=None, timeout=3600):
        saved = read_json(Path(directory) / "request.json")
        values = saved["song"].copy()
        if tile_size is not None:
            values["tile_size"] = tile_size
            values["tile_overlap"] = min(values["tile_overlap"], tile_size // 4)
        return self.run(SongConfig(**values), resume_from=directory, decode_only=True, timeout=timeout)

    def _summarize(self, directory, result, elapsed):
        result["client_elapsed_seconds"] = elapsed
        atomic_json(directory / "summary.json", result)
        print(f"Result: {result['status']}")
        if "audio" in result:
            audio = result["audio"]
            print(f"Actual audio: {audio['duration_seconds']:.2f}s; requested maximum: {audio['maximum_requested_seconds']:.2f}s")
            if result["status"] == "audio_saved_with_cutoff":
                print("WARNING: token limit reached before the ending. The exported recording may be incomplete.")
            if not audio["near_requested_limit"]:
                print("The model ended below the selected maximum; no silence or repeated audio was added.")
        layout = result.get("acoustic_layout", {})
        if layout.get("continuity_requires_listening"):
            print("Listen around acoustic boundaries (seconds):", layout["boundary_seconds"])
        print("Artifacts:", directory)

    def archive(self, directory):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.runs) or directory == self.runs:
            raise ValueError("Archive one of this notebook's run directories")
        return Path(shutil.make_archive(str(directory), "zip", root_dir=directory))
