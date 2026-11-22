"""Generate Music-Studio-Dataset.ipynb — one cell, one private Kaggle Dataset.

Uses the compact automatic-builder pattern from Stem_Splitter_Model_Packs:
a clear setup note and one self-contained, checksummed driver cell. This Studio
Dataset includes the pinned runtime as well as all 18 assets the full Studio
launcher verifies. The CPU builder still compiles the CUDA 13/SM75 runtime; it
does not require a Kaggle GPU or provide CPU song generation.
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
            member = zipfile.ZipInfo('music_studio/' + name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, source_bytes(name), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    packed = buffer.getvalue()
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest()


def summary():
    catalog = json.loads((ROOT / 'music_studio/catalog.json').read_text())
    assets = catalog['assets']
    largest = max(assets, key=lambda a: a['bytes'])
    groups = {
        'Generation': ['generation', 'companion'],
        'Covers': ['cover'],
        'Stem separation': ['stems'],
        'Karaoke and transcription': ['karaoke', 'whisper'],
        'Local lyric writer': ['writer'],
        'Audio-to-MIDI': ['midi'],
        'MOSS audio description': ['caption'],
        'Tempo analysis': ['tempo'],
    }
    included = {asset['role'] for asset in assets}
    groups = {label: [asset['name'] for asset in assets if asset['role'] in roles]
              for label, roles in groups.items() if included.intersection(roles)}
    return dict(count=len(assets), total=sum(a['bytes'] for a in assets),
                largest=largest, groups=groups)


def intro(info, digest):
    groups = '\n'.join(
        f"- **{label}:** " + ', '.join(f'`{name}`' for name in names)
        for label, names in info['groups'].items())
    return f'''# Music Studio — saved runtime and model Dataset

**Kaggle settings: Accelerator None · Internet ON.** This is an automatic,
one-cell Dataset builder. It follows the compact model-pack pattern used by
Stem Splitter, with the extra pinned Linux runtime the Music Studio launcher
actually requires. Authentication and disk checks run before large downloads;
the verified files are submitted directly to one **private Kaggle Dataset**.
Nothing passes through your PC.

1. Enable Kaggle Internet. Add private notebook Secret `KAGGLE_API_TOKEN`, or
   `KAGGLE_USERNAME` and `KAGGLE_KEY`. Never paste credentials into the cell.
2. Leave `DATASET_OWNER = ''` to use the authenticated username, or enter only
   your Kaggle username if automatic detection is unavailable.
3. Run the single code cell and wait. The runtime compilation is CPU-based and
   may take a long time. Success prints **PRIVATE MUSIC STUDIO DATASET SUBMITTED**.
4. Wait for Kaggle to process it. In **Music-Studio.ipynb**, choose **GPU T4 ×2**,
   turn on Internet, then **Add Input → Your Datasets** and attach this Dataset.

## What the Studio actually needs

The catalog has **{info['count']} exact, checksum-pinned model files** totalling
**{info['total'] / 1e9:.2f} GB**. These are model assets (not training-base weights):

{groups}

These cover YuE2 generation and its companion/VAE, SheetSage cover generation,
HT-Demucs stems, Parakeet plus Whisper transcription, the Gemma lyric writer,
MuScriptor audio-to-MIDI, MOSS audio description, and tempo analysis. The
launcher verifies and links these catalog paths. The selected components cannot
be omitted from this single-Dataset Studio build. Artwork/video models are not
included; they are separate workflows.

## Important: CPU builder does not mean CPU runtime build

`Accelerator None` means Kaggle does not allocate a GPU to this builder. It
**still compiles** the pinned YuE2/CUDA runtime on CPU, including CUDA 13 device
code for T4/SM75. It is not a models-only download cell, and it does not make
song generation run on CPU. The normal Studio session uses the saved runtime on
GPU 0 and does not compile it again. The second T4's VRAM is not pooled.

The builder measures available disk, validates remote asset sizes before build,
checks exact bytes and SHA-256, and streams one file at a time through Kaggle's
native private Dataset-creation uploader. The largest model is
`{info['largest']['name']}` (**{info['largest']['bytes'] / 1e9:.2f} GB**); upload
space is needed for one missing model plus reserve, not another full model set.
Only this run's own successful temporary downloads are deleted; attached
inputs and existing user files are preserved. A verified runtime from an
attached Music Studio Dataset is reused when available.

This makes one private create request; it never updates or versions an existing
Dataset. If the cell is interrupted during final submission, check Kaggle
before retrying because the outcome may be uncertain. A success message/receipt
confirms submission, not completed Kaggle processing. Build and upload logs are
under `/kaggle/working/music_studio_dataset` and the selected scratch folder.

Embedded sources SHA-256: `{digest}`
· [Setup and validation](https://github.com/bothashaun-maker/Music/blob/main/docs/music-studio/README.md)
· [Complete Studio guide](https://github.com/bothashaun-maker/Music/blob/main/Music-Studio-Guide.pdf)
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
from music_studio.dataset import authenticated_api, create_dataset, username

catalog = json.loads((SUPPORT / 'music_studio/catalog.json').read_text())
api = authenticated_api()  # Check credentials before any compile, download or request.
owner = DATASET_OWNER or username(api)
if not owner:
    raise RuntimeError('Set DATASET_OWNER to your Kaggle username, then rerun. Nothing was built or downloaded.')
cache = choose_scratch(max(a['bytes'] for a in catalog['assets']) + GIB, ('/kaggle/temp', '/tmp', '/kaggle/working'), 'music-studio-downloads')
result = create_dataset(catalog, '/kaggle/working/music_studio_dataset', Path('/kaggle/input'),
                        cache, owner, jobs=BUILD_JOBS, api=api)
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
