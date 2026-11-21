"""CPU-only tests for the automatic Music Studio Dataset builder.

The Kaggle SDK, the runtime compilation and the 25 GB of model downloads are
replaced here; everything else is the real builder path, including the manifest,
the one-file-at-a-time upload loop, the receipts and the release of only owned
temporary downloads.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

from music_studio import common, dataset
from music_studio import build_runtime

ROOT = Path(__file__).resolve().parents[2]


def digest_of(name):
    return hashlib.sha256(name.encode()).hexdigest()


class DatasetFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_root = self.root / 'input'
        self.input_root.mkdir()
        self.receipt_dir = self.root / 'receipts'
        self.cache = self.root / 'cache'

    def record(self, name, size):
        return dict(name=name, path='models/' + name, bytes=size, sha256=digest_of(name),
                    source='https://example.invalid/' + name, role='test', license='test-only')

    def catalog(self, count=3):
        return dict(schema=1, profile=common.PROFILE,
                    assets=[self.record(f'asset-{n}.bin', 1024 * (n + 1)) for n in range(count)])

    def runtime(self):
        sources = json.loads((common.HERE / 'sources.json').read_text())
        return dict(name='music-studio-runtime.bundle', bytes=4096,
                    sha256=hashlib.sha256(b'runtime').hexdigest(), architecture='x86_64',
                    cuda_architectures=[75], build_mode='cuda',
                    source_fingerprint=common.fingerprint(sources),
                    compatibility={'libc': ['glibc', '2.35'], 'cuda': '13.0.2'})

    def fake_api(self):
        """The SDK call convention the builder patches: instance-level upload_files."""
        api = types.SimpleNamespace(config_values={'username': 'sanitized-owner'}, folders=[])

        def upload_files(folder, request, quiet=True, dir_mode='skip'):
            names = sorted(p.name for p in Path(folder).iterdir())
            request.files.extend(names)
            api.folders.append(names)

        def dataset_create_new(folder, public=False, quiet=False, convert_to_csv=False, dir_mode='skip'):
            api.metadata = json.loads((Path(folder) / 'dataset-metadata.json').read_text())
            api.public = public
            # The real SDK makes one upload_files call for the request; the builder
            # redirects that single call once per record it must send.
            request = types.SimpleNamespace(files=[])
            api.upload_files(folder=folder, request=request)
            api.request_files = request.files
            return {'status': 'ok'}

        api.upload_files = upload_files
        api.dataset_create_new = dataset_create_new
        return api

    def run_builder(self, catalog, api, owner=''):
        def build_runtime_record(scratch, jobs=2):
            scratch = Path(scratch)
            scratch.mkdir(parents=True, exist_ok=True)
            archive = scratch / 'music-studio-runtime.bundle'
            archive.write_bytes(b'runtime')
            (scratch / 'runtime-build.json').write_text(json.dumps(self.runtime()))
            for name in ('bundle', 'cargo', 'cuda'):
                (scratch / name).mkdir(exist_ok=True)
            return archive, self.runtime()

        def transfer(record, destination, reserve=0):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'x' * record['bytes'])
            return destination

        def copy_link(self_path, target, target_is_directory=False):
            # `folder/name` is only a handle for the SDK; copying keeps the same
            # one-file-at-a-time behaviour where symlinks would need privileges.
            Path(self_path).write_bytes(Path(target).read_bytes())

        printed = []
        with patch.object(dataset, 'authenticated_api', return_value=api), \
                patch.object(dataset, 'build_runtime_record', side_effect=build_runtime_record), \
                patch.object(dataset, 'transfer', side_effect=transfer), \
                patch.object(dataset, 'verify', side_effect=lambda path, record: path), \
                patch.object(dataset, 'validate_manifest', side_effect=lambda manifest: manifest), \
                patch.object(dataset, 'choose_scratch',
                             side_effect=lambda required, candidates, name, reserve=0: self.root / name), \
                patch.object(Path, 'symlink_to', copy_link), \
                patch('builtins.print', side_effect=lambda *a, **k: printed.append(' '.join(map(str, a)))):
            result = dataset.create_dataset(catalog, self.receipt_dir, self.input_root, self.cache,
                                            owner=owner, jobs=1, api=api)
        return result, '\n'.join(printed)

    def test_saved_catalog_is_validated(self):
        real = json.loads((common.HERE / 'catalog.json').read_text())
        dataset.validate_catalog(real)
        for mutate in (lambda c: c['assets'][0].__setitem__('sha256', 'not-a-digest'),
                       lambda c: c.__setitem__('assets', []),
                       lambda c: c.__setitem__('profile', 'other'),
                       lambda c: c['assets'][1].__setitem__('bytes', 'big'),
                       lambda c: c['assets'][2].__setitem__('name', 'nested/model.bin')):
            broken = json.loads(json.dumps(real))
            mutate(broken)
            with self.assertRaises(RuntimeError):
                dataset.validate_catalog(broken)

    def test_builder_streams_runtime_then_assets_then_manifest(self):
        catalog = self.catalog()
        api = self.fake_api()
        result, printed = self.run_builder(catalog, api)
        uploads = [name for folder in api.folders for name in folder]
        self.assertTrue(result['handle'].startswith('sanitized-owner/music-studio-dataset-v1-'))
        self.assertTrue(result['url'].startswith('https://www.kaggle.com/datasets/sanitized-owner/'))
        self.assertEqual(uploads[0], 'music-studio-runtime.bundle')
        self.assertEqual(uploads[-2:], ['MUSIC-STUDIO-NOTICES.md', 'music-studio-manifest.json'])
        self.assertEqual(len(uploads), len(catalog['assets']) + 3)
        receipt = json.loads((self.receipt_dir / 'dataset-upload.json').read_text())
        self.assertTrue(receipt['submitted'])
        self.assertTrue(receipt['requested_private'])
        self.assertFalse(api.public, 'the create request must be private')
        manifest = json.loads((self.receipt_dir / 'music-studio-manifest.json').read_text())
        self.assertEqual(len(manifest['files']), 3)
        self.assertEqual(manifest['model_bytes'], sum(r['bytes'] for r in catalog['assets']))
        self.assertIn('PRIVATE MUSIC STUDIO DATASET SUBMITTED', printed)
        self.assertTrue((self.receipt_dir / 'upload-progress.json').is_file())

    def test_owned_downloads_are_released_and_inputs_preserved(self):
        catalog = self.catalog(2)
        attached = self.input_root / 'dataset' / catalog['assets'][0]['name']
        attached.parent.mkdir(parents=True)
        attached.write_bytes(b'x' * catalog['assets'][0]['bytes'])
        api = self.fake_api()
        self.run_builder(catalog, api)
        self.assertTrue(attached.is_file(), 'an attached input must never be deleted')
        self.assertFalse((self.cache / catalog['assets'][1]['name']).exists(),
                         'this builder owns and releases its own downloaded cache file')

    def test_missing_owner_stops_before_any_work(self):
        api = types.SimpleNamespace(config_values={})
        with patch.object(dataset, 'build_runtime_record') as build, \
                patch.object(dataset, 'authenticated_api', return_value=api):
            with self.assertRaises(RuntimeError):
                dataset.create_dataset(self.catalog(), self.receipt_dir, self.input_root,
                                       self.cache, owner='', api=api)
        build.assert_not_called()


class CpuToolchainTests(unittest.TestCase):
    def test_cpu_helpers_cannot_inherit_cuda_toolkit_detection(self):
        common_options = [
            '-G', 'Ninja', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=ON',
            '-DCMAKE_CUDA_ARCHITECTURES=75-real',
            '-DCMAKE_CUDA_COMPILER=/tmp/cuda/bin/nvcc',
            '-DCUDAToolkit_ROOT=/tmp/cuda',
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/tmp/cuda/lib',
        ]
        cpu = build_runtime.cpu_cmake_options(common_options)
        self.assertIn('-DGGML_CUDA=OFF', cpu)
        self.assertIn('-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE', cpu)
        self.assertFalse(any(option.startswith(('-DCMAKE_CUDA', '-DCUDAToolkit_ROOT=',
                                               '-DCMAKE_EXE_LINKER_FLAGS=')) for option in cpu))
        # The separate CUDA trainer keeps the normal options untouched.
        self.assertIn('-DGGML_CUDA=ON', common_options)
        self.assertIn('-DCMAKE_CUDA_COMPILER=/tmp/cuda/bin/nvcc', common_options)


class NotebookPayloadTests(unittest.TestCase):
    def test_driver_embeds_a_verified_builder_archive(self):
        notebook = json.loads((ROOT / 'Music-Studio-Dataset.ipynb').read_text())
        self.assertEqual(len(notebook['cells']), 2)
        code_cell = notebook['cells'][1]
        self.assertEqual(code_cell['cell_type'], 'code')
        self.assertEqual(code_cell['outputs'], [])
        self.assertEqual(notebook['metadata']['kaggle']['accelerator'], 'none')
        self.assertFalse(notebook['metadata']['kaggle']['isGpuEnabled'])
        code = ''.join(code_cell['source'])
        compile(code, 'Music-Studio-Dataset.ipynb', 'exec')
        encoded = code.split("base64.b64decode(", 2)[1].split(')')[0].strip().strip("'")
        payload = base64.b64decode(encoded)
        self.assertIn(hashlib.sha256(payload).hexdigest(), code)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
        expected = {'music_studio/' + name for name in
                    ('__init__.py', 'common.py', 'dataset.py', 'build_runtime.py', 'patches.py',
                     'launch.py', 'kaggle_upload.rs', 'upload.html', 'catalog.json',
                     'sources.json', 'NOTICES.md')}
        self.assertEqual(names, expected)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            embedded = {name.rsplit('/', 1)[-1]: archive.read(name) for name in archive.namelist()}
        sources = json.loads(embedded['sources.json'])
        for name, expected_digest in sources['patches'].items():
            self.assertEqual(hashlib.sha256(embedded[name]).hexdigest(), expected_digest,
                             name + ' must match the checksum pinned in the embedded sources.json')


if __name__ == '__main__':
    unittest.main()
