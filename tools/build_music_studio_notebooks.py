"""Generate the two independent, self-contained, one-cell Music Studio notebooks.

The studio notebook is generated here. Music-Studio-Dataset.ipynb is generated
by tools/build_music_studio_dataset_notebook.py, so the Dataset builder keeps the
same embedded-source, manifest-and-driver shape as the other JJP automatic
Dataset notebooks instead of running the whole studio package as a subprocess.
"""
from pathlib import Path
import base64
import hashlib
import json
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_music_studio_dataset_notebook as dataset_notebook  # noqa: E402


def build():
    dataset_notebook.refresh_sources()
    files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / 'music_studio').iterdir())
             if p.is_file() and p.suffix in ('.py', '.json', '.md', '.rs', '.html')}
    packed = zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)
    payload = base64.b64encode(packed).decode()
    digest = hashlib.sha256(packed).hexdigest()

    intro = '''# Music Studio — full browser interface on Kaggle

First create your saved input with **Music-Studio-Dataset.ipynb**. Attach that
ONE **Music Studio Dataset**, select **GPU T4 ×2**, enable **Internet**, and run
the **single startup cell** below. Open **OPEN MUSIC STUDIO** when it appears.
Keep the cell running. Press **Create** in the studio when you are ready;
startup never generates a diagnostic song.

This is the separate YuE2 Studio Linux browser interface: create songs, inspect
and edit scores/MIDI, re-render, cover/restyle recordings, manage a library,
split stems, make MIDI, time karaoke, process audio, create lyric videos,
use LoRAs and drive the studio through MCP. It does not replace the existing
native FP16/ComfyUI notebooks. Generation uses the saved Q8 GGUF model on GPU 0;
the second T4's memory is not pooled. Stems and speech recognition default to CPU.
The local writer, MIDI and style-description helpers also use CPU builds.

This pinned Linux fork has a different feature set from newer Windows releases.
T4 LoRA training, Windows VST plugins, AB2/chord-variety and newer extension/
continuation controls are not provisioned. The guide covers every available
panel plus alternatives and limitations. No subscription/API key is required
for the included local tools; optional external providers have their own terms.

Saved songs, SQLite library, settings, LoRAs and logs live under
`/kaggle/working/music_studio`. Models stay in your read-only Dataset and
temporary runtime state. **Save Version with output files** before ending
your session; a saved draft does not preserve your library. Reattach saved
outputs and follow the guide's restore procedure in a later session.
Keep the Cloudflare link/key private. Stop the previous music/movie/image
startup cell before starting another studio in the same Kaggle session.

[Complete tool guide PDF](https://github.com/bothashaun-maker/Music/blob/main/Music-Studio-Guide.pdf)
· [Setup and troubleshooting](https://github.com/bothashaun-maker/Music/blob/main/docs/music-studio/README.md)
'''
    config = "DATASET_PATH = 'auto'  # Or the exact saved Music Studio Dataset directory under /kaggle/input.\nGPU_INDEX = 0\nRESTORE_LIBRARY_PATH = 'auto'  # One attached saved music_studio library; use 'none' for a fresh library.\n"
    extras = "task_env['MUSIC_STUDIO_DATASET_PATH'] = DATASET_PATH\ntask_env['MUSIC_STUDIO_GPU'] = str(GPU_INDEX)\ntask_env['MUSIC_STUDIO_RESTORE_LIBRARY'] = RESTORE_LIBRARY_PATH\n"
    code = config + f'''\n# Self-contained support; no repository clone or separate repair cell required.
import base64, hashlib, json, os, signal, subprocess, sys, zlib
from pathlib import Path
packed = base64.b64decode({payload!r})
assert hashlib.sha256(packed).hexdigest() == {digest!r}, 'Embedded Music Studio source checksum failed'
support = Path('/kaggle/working/.music_studio_support/{digest[:12]}')
for relative, contents in json.loads(zlib.decompress(packed)).items():
    path = support / relative
    if not path.resolve().is_relative_to(support.resolve()):
        raise RuntimeError('Invalid embedded source path')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
task_env = dict(os.environ)
task_env['PYTHONPATH'] = str(support) + os.pathsep + task_env.get('PYTHONPATH', '')
''' + extras + f'''
task = subprocess.Popen([sys.executable, '-u', '-m', 'music_studio.launch'], env=task_env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, start_new_session=True)
try:
    for line in task.stdout:
        print(line, end='', flush=True)
    code = task.wait()
    if code:
        raise RuntimeError('Music Studio stopped with exit code ' + str(code) + '; inspect the error above and saved logs.')
except KeyboardInterrupt:
    print('Stopping this Music Studio process. Saved songs are retained.', flush=True)
finally:
    if task.poll() is None:
        os.killpg(task.pid, signal.SIGTERM)
        try:
            task.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(task.pid, signal.SIGKILL)
            task.wait(timeout=5)
'''
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info={'name': 'python'},
                                  kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True)),
                    cells=[dict(cell_type='markdown', metadata={}, source=intro.splitlines(True)),
                           dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code.splitlines(True))])
    target = ROOT / 'Music-Studio.ipynb'
    target.write_bytes((json.dumps(notebook, indent=1) + '\n').encode())
    compile(code, target.name, 'exec')
    print(target.name, 'one code cell;', len(packed), 'embedded bytes')

    dataset_notebook.build()


if __name__ == '__main__':
    build()
