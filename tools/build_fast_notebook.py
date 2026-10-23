"""Build the standalone notebook around the original native FP16 Kaggle engine."""
from pathlib import Path
import base64
import json
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    files = {}
    for directory in ('kaggle_yue2', 'native_yue2'):
        for p in sorted((ROOT / directory).glob('*.py')):
            files[str(p.relative_to(ROOT))] = p.read_text()
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    bootstrap = '''
    import base64, json, os, sys, zlib
    from pathlib import Path
    if "app" in globals():
        app.stop()  # Release a managed server from the earlier ComfyUI notebook.
    if "native_process" in globals() and native_process.poll() is None:
        native_process.terminate()
        try:
            native_process.wait(timeout=20)
        except Exception:
            native_process.kill()
            native_process.wait(timeout=10)
    # Close only the original YuE2 pipeline if a previous notebook left it
    # inside this kernel. Other model variables/processes are not cleared.
    legacy_pipe = globals().get("pipe")
    if (legacy_pipe is not None and legacy_pipe.__class__.__module__ == "yue2.pipeline"
            and legacy_pipe.__class__.__name__ == "YuE2Pipeline"):
        if isinstance(globals().get("CANCEL"), dict):
            CANCEL["stop"] = True
        legacy_demo = globals().get("demo")
        if legacy_demo is not None and legacy_demo.__class__.__module__.startswith("gradio."):
            legacy_demo.close()
        legacy_pipe.close()
        globals().pop("pipe", None)
        import gc
        gc.collect()
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            torch_module.cuda.empty_cache()
        print("Closed the previous in-kernel YuE2 pipeline; saved song files remain.")
    support = Path("/kaggle/temp/yue2_native_support")
    support.mkdir(parents=True, exist_ok=True)
    payload = "PAYLOAD"
    for relative, contents in json.loads(zlib.decompress(base64.b64decode(payload))).items():
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    for name in list(sys.modules):
        if name.split(".")[0] in {"native_yue2", "kaggle_yue2"}:
            del sys.modules[name]
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from native_yue2.setup import install
    ROOT = "/kaggle/temp/yue2_native"
    OUTPUTS = "/kaggle/working/yue2_native/runs"
    native_python, native_env = install(ROOT, OUTPUTS)
    print("Setup completed. Run the next cell to open the original native engine UI.")
    '''.replace('PAYLOAD', payload)
    cells = [cell('markdown', '''
        # YuE2 on Kaggle — restored native FP16 engine (revision 7)

        Select **GPU T4 ×2** and enable Internet. This restores the original
        `yue2_infer 0.1.5` engine: FP16 transformer on GPU 0, CUDA-graph AR decoding
        with SDPA, and tiled FP32 VAE on GPU 1. Default generation uses **Direct to
        audio**, **16 Euler steps**, CFG **1.0**, and an **8-minute safety ceiling**.
        The model chooses a natural ending from your prompt and lyrics. It does
        not have to fill the ceiling. A token-limit cutoff is reported separately.

        **No diagnostic song runs automatically.** Run setup and open the UI,
        enter your full song lyrics and style, leave or adjust the safety ceiling, then click
        Generate Song. Optional score planning remains off unless you select it.
        Cover transcription is disabled in this version.

        **T4 attention change:** equivalent grouped-head views make PyTorch's
        memory-efficient kernel eligible, instead of the native T4 GQA math path.
        The first supported CUDA dispatch is verified and logged. Unsupported
        shapes retain native attention and are reported. CPU equivalence checks
        passed; actual T4 speed improvement has not been measured.

        Durations are maximums. A natural early ending or token cutoff is reported.
        Long acoustic generations may span multiple chunks; listen at the reported
        boundaries. Seven/eight-minute performance and continuity remain unverified.

        Runtime and model files use session-only `/kaggle/temp/yue2_native`.
        Song audio, exact tokens and recovery latents use
        `/kaggle/working/yue2_native/runs`. The paths may share underlying storage;
        setup prints actual free space rather than assuming a separate 50 GB quota.
        Old notebook caches and files are preserved.

        **Process protection:** setup and launch stop only verified older engines
        using this script/output root, including survivors of a kernel restart.
        A process lock prevents two managed engines from loading together. GPU
        memory is checked before loading and before generation. An unrecognized
        GPU process is reported and left running; setup stops before generation.
        CUDA OOM saves GPU process details and releases the failed engine's model.

        **After generation, download the complete run ZIP in the UI, or Save
        Version with outputs in Kaggle before ending the session.** A saved draft
        or a notebook download containing logs does not preserve audio files.
        '''), cell('code', bootstrap), cell('markdown', '''
        **Open the generator.** This cell downloads the original full-precision
        model assets on first launch and prints a Gradio share link. Open that link.
        The cell stays running while the UI is available. Reuse it for additional
        songs so the transformer can stay loaded. A stopped cell ends this process.

        The UI defaults to **Up to 8 min**. This allows seven/eight-minute songs
        while preserving natural shorter endings. Supply complete lyrics with
        sections, instrumental passages and an outro. A lower selected ceiling
        can cut a song off; it does not tell the model to finish there.
        The original generation defaults are restored; no speed is promised for
        long songs. The status panel reports actual AR execution and stage times.
        '''), cell('code', '''
        import subprocess
        engine = support / "native_yue2/engine.py"
        log_path = Path(OUTPUTS).parent / "runtime/engine.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from native_yue2.process import start_engine
        native_process = start_engine(native_python, engine, native_env, ROOT, OUTPUTS)
        try:
            with log_path.open("a") as log:
                for line in native_process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            if native_process.wait() != 0:
                raise RuntimeError(f"Native engine stopped. Inspect {log_path}")
        finally:
            if native_process.poll() is None:
                native_process.terminate()
                try:
                    native_process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    native_process.kill()
                    native_process.wait(timeout=10)
        '''), cell('markdown', '''
        **Find and preserve your song.** Each attempt has its own folder. After a
        successful generation the UI offers WAV, FLAC, and a complete ZIP containing
        settings, tokens, timings and recovery latents. Download it before stopping.
        The folder also appears under Kaggle Output at `yue2_native/runs`.
        `status.json` records the measured audio length and any cutoff; it does not
        establish musical completeness. Listen to the ending and chunk boundaries.

        Run the next cell after stopping the UI to list all saved attempts. It
        starts no model or generation. Files still need to be downloaded or saved
        with a Kaggle output version before the session is discarded.
        '''), cell('code', '''
        from IPython.display import FileLink, display
        for directory in sorted(Path(OUTPUTS).glob("*")):
            if directory.is_dir():
                status = directory / "status.json"
                print(directory.name, status.read_text() if status.exists() else "Partial run")
                for filename in ("audio.wav", "audio.flac"):
                    if (directory / filename).exists():
                        display(FileLink(str(directory / filename)))
        for archive in sorted(Path(OUTPUTS).glob("*.zip")):
            display(FileLink(str(archive)))
        '''), cell('markdown', '''
        **Optional decode recovery.** If audio decoding/export failed after
        `latent.npy` and its manifest were saved, stop the UI first and point the
        next cell at that run folder. This checks the latent checksum and original
        model weights, and decodes to a new folder without regenerating music
        tokens or the acoustic latents. For a new session, attach the saved run ZIP
        as a Kaggle input and extract it first. Setup and the original model files
        are still required. ComfyUI artifacts from revision 2 use a different format.
        '''), cell('code', '''
        RECOVER_RUN = None  # Folder containing latent.npy and latent_manifest.json
        if RECOVER_RUN:
            if "native_process" in globals() and native_process.poll() is None:
                raise RuntimeError("Stop the UI cell before decoding a saved run")
            recovery_env = dict(native_env, YUE2_RECOVER_RUN=str(Path(RECOVER_RUN).resolve()))
            subprocess.run([native_python, "-u", str(support / "native_yue2/engine.py")],
                           env=recovery_env, check=True)
        '''), cell('markdown', '''
        Based on the original AIQUEST Academy notebook in Movies commit
        `fcf4d45dc5dbca57b39e54311807410cdf41f737`.
        Sources: [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
        [YuE2 VAE](https://huggingface.co/m-a-p/YuE2-Vae).
        The native inference wheel and model revisions are pinned; wheel and
        model checksums are verified. Upstream model/license terms apply.
        Source and checks are included in the Music repository for review.
        '''),
    ]
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': True},
    }, 'nbformat': 4, 'nbformat_minor': 4}


if __name__ == '__main__':
    output = ROOT / 'yue2-3b-frontier-full-song-music-generation.ipynb'
    output.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + '\n')
    print(output)
