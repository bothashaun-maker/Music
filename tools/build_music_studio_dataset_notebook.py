"""Generate Music-Studio-Dataset.ipynb — one cell, one private Kaggle Dataset.

Follows the same shape as the other automatic Dataset builders in the JJP
repositories: an embedded, checksummed runtime archive, a single driver cell
that validates the saved catalog, authenticates first, measures scratch space
and then calls `create_dataset()` once. Nothing about the 25 GB model set is
downloaded before those checks pass.
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]

# Only what the Dataset builder needs. The studio notebook's launcher, patches
# for the browser UI and the upload route are not part of this payload.
PAYLOAD = ('__init__.py', 'common.py', 'dataset.py', 'build_runtime.py', 'preflight.py',
           'patches.py', 'launch.py', 'kaggle_upload.rs', 'upload.html',
           'catalog.json', 'sources.json', 'NOTICES.md')


def source_bytes(name):
    """Bytes exactly as Kaggle will see them.

    A Windows checkout can hold CRLF while the committed and embedded sources are
    LF. Hashing the working-tree bytes once produced a sources.json whose patch
    checksums never matched the embedded payload, so every Kaggle build stopped
    with "Runtime build source changed" before any model was downloaded.
    """
    return (ROOT / 'music_studio' / name).read_bytes().replace(bytes([13, 10]), bytes([10]))

def refresh_sources():
    """Pin the build-input checksums the embedded runtime build verifies."""
    sources_path = ROOT / 'music_studio/sources.json'
    sources = json.loads(sources_path.read_text())
    sources['patches'] = {name: hashlib.sha256(source_bytes(name)).hexdigest()
                          for name in ('build_runtime.py', 'patches.py', 'kaggle_upload.rs', 'upload.html')}
    sources_path.write_bytes((json.dumps(sources, indent=2) + '\n').encode())
    return sources


def payload_archive():
    """One zip of the builder's own sources, so the notebook needs no clone."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in PAYLOAD:
            archive.writestr('music_studio/' + name, source_bytes(name))
    packed = buffer.getvalue()
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest()


def summary():
    catalog = json.loads((ROOT / 'music_studio/catalog.json').read_text())
    assets = catalog['assets']
    largest = max(assets, key=lambda a: a['bytes'])
    roles = sorted({a['role'] for a in assets})
    return dict(count=len(assets), total=sum(a['bytes'] for a in assets),
                largest=largest, roles=roles)


def intro(info, digest):
    return f'''# Music Studio Dataset — automatic private save

**Accelerator None / CPU build / Internet ON.** One cell checks your Kaggle
credentials, compiles the pinned x86-64/T4 runtime, verifies every model and
submits **one complete private Dataset** through the same native Kaggle
streaming uploader the other JJP notebooks use. No model passes through your PC.

1. Enable your private Kaggle Secret `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME`
   and `KAGGLE_KEY`). Never paste a token into code or chat.
2. Leave `DATASET_OWNER` blank when authentication reveals your username;
   otherwise set only your Kaggle username.
3. Run the single code cell. It prints **PRIVATE MUSIC STUDIO DATASET SUBMITTED**
   with a link. The first compilation is the long step; the cell stays busy.
4. Wait for Kaggle to process the Dataset, then attach that ONE Dataset to
   **Music-Studio.ipynb** through **Add Input → Your Datasets**.

The saved set is **{info["count"]} pinned files / {info["total"] / 1e9:.2f} GB** plus the compiled
runtime: {", ".join(info["roles"])}. Each file has an immutable source revision,
exact byte count and SHA-256. The compiled runtime is built once here on CPU and
reused by every later GPU session, which never compiles it.

Space is measured, not assumed. The build picks the first scratch location with
enough free room, and upload needs only the largest missing file
(**{info["largest"]["name"]}**, {info["largest"]["bytes"] / 1e9:.2f} GB) plus a reserve, so the
Dataset is never held twice on local disk. An attached, checksum-verified Music
Studio runtime is reused instead of recompiling. Only this builder's own
temporary downloads are released; attached inputs are never touched.

One private create request per run, never an update or overwrite of an existing
Dataset. An interrupted final request has an uncertain outcome: check the
Dataset page before retrying. Receipts and the build log stay under
`/kaggle/working/music_studio_dataset` and the chosen scratch location.

Embedded builder sources: `{digest}`
· [Setup and validation](https://github.com/bothashaun-maker/Music/blob/main/docs/music-studio/README.md)
· [Complete tool guide](https://github.com/bothashaun-maker/Music/blob/main/Music-Studio-Guide.pdf)
'''


def driver(payload, digest):
    return f'''DATASET_OWNER = ''  # Optional: your Kaggle username, never an API key.
BUILD_JOBS = 2      # CPU compilation; lower to 1 if the build reports insufficient memory.

from pathlib import Path
import base64, hashlib, importlib, io, json, sys, zipfile

SUPPORT = Path('/kaggle/working/music-studio-dataset-support') / {digest[:12]!r}
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}, 'Embedded Music Studio source checksum failed'
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'music_studio':
            raise RuntimeError('Invalid embedded source path: ' + name)
        target = SUPPORT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
sys.path.insert(0, str(SUPPORT))
importlib.invalidate_caches()
for name in [n for n in sys.modules if n == 'music_studio' or n.startswith('music_studio.')]:
    sys.modules.pop(name, None)  # Rerunning this cell must not reuse stale sources.

from music_studio.common import GIB, choose_scratch
from music_studio.dataset import SCRATCH_CANDIDATES, authenticated_api, create_dataset, username

catalog = json.loads((SUPPORT / 'music_studio/catalog.json').read_text())
api = authenticated_api()  # Check credentials before any compile, download or request.
owner = DATASET_OWNER or username(api)
if not owner:
    raise RuntimeError('Set DATASET_OWNER to your Kaggle username, then rerun. Nothing was built or downloaded.')
scratch = choose_scratch(8 * GIB, SCRATCH_CANDIDATES, 'music-studio-build')
cache = choose_scratch(max(a['bytes'] for a in catalog['assets']) + GIB, SCRATCH_CANDIDATES, 'music-studio-downloads')
result = create_dataset(catalog, '/kaggle/working/music_studio_dataset', Path('/kaggle/input'),
                        cache, owner, jobs=BUILD_JOBS, api=api, build_scratch=scratch)
print('Dataset handle:', result['handle'])
print('Local receipt:', result['receipt'])
'''


def verify_embedded(notebook_path):
    """Prove the written notebook payload matches its own pinned checksums."""
    notebook = json.loads(Path(notebook_path).read_text())
    code = ''.join(notebook['cells'][1]['source'])
    encoded = code.split('base64.b64decode(', 2)[1].split(')')[0].strip().strip(chr(39))
    payload = base64.b64decode(encoded)
    if hashlib.sha256(payload).hexdigest() not in code:
        raise RuntimeError('Embedded payload checksum is missing from the driver cell.')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        embedded = {name.rsplit('/', 1)[-1]: archive.read(name) for name in archive.namelist()}
    sources = json.loads(embedded['sources.json'])
    for name, expected in sources['patches'].items():
        if hashlib.sha256(embedded[name]).hexdigest() != expected:
            raise RuntimeError('Embedded ' + name + ' does not match its pinned sources.json checksum.')
    return True


def build():
    refresh_sources()
    payload, digest = payload_archive()
    code = driver(payload, digest)
    info = summary()
    notebook = dict(
        nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      language_info={'name': 'python'},
                      kaggle=dict(accelerator='none', isGpuEnabled=False, isInternetEnabled=True)),
        cells=[dict(cell_type='markdown', metadata={}, source=intro(info, digest).splitlines(True)),
               dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code.splitlines(True))])
    target = ROOT / 'Music-Studio-Dataset.ipynb'
    target.write_bytes((json.dumps(notebook, indent=1) + '\n').encode())
    compile(code, target.name, 'exec')
    print(target.name, '- one code cell;', len(payload), 'embedded base64 bytes;', digest[:12])
    verify_embedded(target)


if __name__ == '__main__':
    build()
