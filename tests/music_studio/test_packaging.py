"""Exercise private upload, hostile archives, resumed transfers and library preservation."""
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import types
import typing
import unittest
from unittest.mock import patch

from music_studio import common, dataset
from music_studio.launch import restore_library

ROOT = Path(__file__).resolve().parents[2]


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, name, data):
        return dict(name=name, path='models/'+name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), source='https://example.invalid/'+name, role='test', license='test-only')

    def manifest(self, assets, runtime):
        catalog = dict(schema=1, profile=common.PROFILE, assets=assets)
        sources = json.loads((common.HERE / 'sources.json').read_text())
        runtime = dict(runtime, architecture='x86_64', cuda_architectures=[75], build_mode='cuda', source_fingerprint=common.fingerprint(sources))
        return catalog, dict(schema=1, profile=common.PROFILE, complete=True, catalog_fingerprint=common.fingerprint(catalog), files=assets,
                             runtime=runtime, model_bytes=sum(r['bytes'] for r in assets))

    def api(self):
        # Actual native SDK create/upload methods, with only network transport
        # replaced. This catches drift in the normal upload_files callback flow.
        spec = importlib.util.spec_from_file_location('sdk_fixture', Path(__file__).with_name('kaggle_upload_reference.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class HTTPError(Exception):
            pass
        module.__dict__.update(os=os, json=json, cast=typing.cast, List=typing.List, Dict=typing.Dict,
                               HTTPError=HTTPError, ApiCreateDatasetRequest=types.SimpleNamespace,
                               ApiCreateDatasetResponse=types.SimpleNamespace,
                               ApiBlobType=types.SimpleNamespace(DATASET=1), ResumableUploadContext=nullcontext)
        class API:
            DATASET_METADATA_FILE = 'dataset-metadata.json'
            OLD_DATASET_METADATA_FILE = 'datapackage.json'
            DATASET_COVER_IMAGE_FILES = []
            KERNEL_METADATA_FILE = 'kernel-metadata.json'
            MODEL_METADATA_FILE = 'model-metadata.json'
            MODEL_INSTANCE_METADATA_FILE = 'instance-metadata.json'
            CONFIG_NAME_USER = 'username'
            config_values = {'username': 'test-owner'}
            dataset_create_new = module.dataset_create_new
            upload_files = module.upload_files
            def __init__(self):
                self.uploaded = {}
                self.lookups = 0
                self.created = []
            def get_dataset_metadata_file(self, folder):
                return str(Path(folder) / self.DATASET_METADATA_FILE)
            def get_or_fail(self, data, key):
                return data[key]
            def get_or_default(self, data, key, default):
                return data.get(key, default)
            def dataset_status(self, ref):
                self.lookups += 1
                raise HTTPError('not created')
            def _upload_file_or_folder(self, folder, name, *args):
                self.uploaded[name] = (Path(folder) / name).read_bytes()
                return name
            def _new_file(self, uploaded):
                return uploaded
            def with_retry(self, fn):
                return fn
            def build_kaggle_client(self):
                def create(request):
                    self.created.append(request)
                    return types.SimpleNamespace(status='ok', error='')
                return nullcontext(types.SimpleNamespace(datasets=types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(create_dataset=create))))
        return API()

    def test_streaming_native_sdk_uses_one_private_request_and_preserves_existing_inputs(self):
        first, second = b'original saved input', b'new model bytes'
        assets = [self.record('first.bin', first), self.record('second.bin', second)]
        runtime = self.record('runtime.tar.gz', b'compiled runtime')
        runtime_path = self.root / runtime['name']
        runtime_path.write_bytes(b'compiled runtime')
        saved = self.root / 'input'
        saved.mkdir()
        (saved / 'first.bin').write_bytes(first)
        catalog, manifest = self.manifest(assets, runtime)
        api = self.api()
        def fetch(record, target):
            self.assertEqual(record['name'], 'second.bin')
            Path(target).write_bytes(second)
            return Path(target)
        output = self.root / 'output'
        with patch.object(common, 'catalog', return_value=catalog), patch.object(dataset, 'transfer', side_effect=fetch):
            url = dataset.publish(manifest, runtime_path, output_root=output, input_root=saved, api=api)
            again = dataset.publish(manifest, runtime_path, output_root=output, input_root=saved, api=api)
        self.assertEqual(url, again)
        self.assertEqual(len(api.created), 1)
        self.assertTrue(api.created[0].is_private)
        self.assertEqual(api.lookups, 1)  # the SDK's own lookup; no extra gate
        self.assertEqual(set(api.uploaded), {'first.bin', 'second.bin', 'runtime.tar.gz', 'music-studio-manifest.json', 'MUSIC-STUDIO-NOTICES.md'})
        self.assertEqual((saved / 'first.bin').read_bytes(), first)
        self.assertFalse((output / '.download-cache/second.bin').exists())
        self.assertTrue(json.loads((output / 'dataset-upload.json').read_text())['submitted'])

    def test_rejected_creation_has_no_success_receipt(self):
        assets = [self.record('first.bin', b'original')]
        runtime = self.record('runtime.tar.gz', b'runtime')
        for record, value in [(assets[0], b'original'), (runtime, b'runtime')]:
            (self.root / record['name']).write_bytes(value)
        catalog, manifest = self.manifest(assets, runtime)
        api = self.api()
        api.dataset_create_new = lambda *a, **k: types.SimpleNamespace(status='error', error='permission denied')
        output = self.root / 'out'
        with patch.object(common, 'catalog', return_value=catalog), self.assertRaisesRegex(RuntimeError, 'not confirmed'):
            dataset.publish(manifest, self.root / runtime['name'], output_root=output, input_root=self.root, api=api)
        self.assertFalse((output / 'dataset-upload.json').exists())
        self.assertEqual((self.root / 'first.bin').read_bytes(), b'original')

    def test_changed_model_or_catalog_cannot_be_reused(self):
        record = self.record('a.bin', b'aaaa')
        path = self.root / 'a.bin'
        path.write_bytes(b'aaab')
        with self.assertRaisesRegex(RuntimeError, 'Verification failed'):
            common.verify(path, record)
        catalog, manifest = self.manifest([record], self.record('runtime.tar.gz', b'rt'))
        with patch.object(common, 'catalog', return_value=catalog):
            common.validate_manifest(manifest)
            manifest['files'] = []
            with self.assertRaisesRegex(RuntimeError, 'allowlist'):
                common.validate_manifest(manifest)

    def test_arm_release_is_rejected_before_execution(self):
        header = bytearray(20)
        header[:6] = b'\x7fELF\x02\x01'
        header[18:20] = (183).to_bytes(2, 'little')
        path = self.root / 'server'
        path.write_bytes(header)
        with self.assertRaisesRegex(RuntimeError, 'x86-64'):
            common.elf_arch(path)
        header[18:20] = (62).to_bytes(2, 'little')
        path.write_bytes(header)
        self.assertEqual(common.elf_arch(path), 'x86_64')

    def test_archives_cannot_write_outside_runtime(self):
        for name, link in [('../escaped.txt', None), ('safe/link', '../../outside')]:
            archive = self.root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as output:
                item = tarfile.TarInfo(name)
                if link:
                    item.type = tarfile.SYMTYPE
                    item.linkname = link
                    output.addfile(item)
                else:
                    item.size = 1
                    output.addfile(item, io.BytesIO(b'x'))
            with self.assertRaises(RuntimeError):
                common.extract(archive, self.root / 'unpack')
            self.assertFalse((self.root / 'escaped.txt').exists())

    def test_internal_library_aliases_can_be_extracted(self):
        archive = self.root / 'runtime.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            item = tarfile.TarInfo('runtime/lib.so.1')
            item.size = 3
            output.addfile(item, io.BytesIO(b'lib'))
            alias = tarfile.TarInfo('runtime/lib.so')
            alias.type = tarfile.SYMTYPE
            alias.linkname = 'lib.so.1'
            output.addfile(alias)
        common.extract(archive, self.root / 'unpacked')
        self.assertEqual((self.root / 'unpacked/runtime/lib.so').read_bytes(), b'lib')

    def test_restore_preserves_audio_and_does_not_replace_current_work(self):
        saved = self.root / 'input/music_studio'
        (saved / 'media').mkdir(parents=True)
        (saved / 'media/song.wav').write_bytes(b'original source')
        with sqlite3.connect(saved / 'library.sqlite') as db:
            db.execute('create table songs(title text)')
            db.execute('insert into songs values (?)', ('Keeper',))
        output = self.root / 'working/music_studio'
        output.mkdir(parents=True)
        restore_library(output, self.root / 'input')
        with sqlite3.connect(output / 'library.sqlite') as db:
            self.assertEqual(db.execute('select title from songs').fetchone(), ('Keeper',))
            db.execute('insert into songs values (?)', ('New work',))
        self.assertEqual((output / 'media/song.wav').read_bytes(), b'original source')
        restore_library(output, self.root / 'input')
        with sqlite3.connect(output / 'library.sqlite') as db:
            self.assertEqual(db.execute('select count(*) from songs').fetchone()[0], 2)

    def test_credentials_are_removed_from_saved_error_messages(self):
        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'private-test-value'}):
            text = common.scrub('KAGGLE failed private-test-value Bearer secret-other https://host/?key=studio-key')
        self.assertNotIn('private-test-value', text)
        self.assertNotIn('secret-other', text)
        self.assertNotIn('studio-key', text)

    def test_notebooks_are_one_cell_self_contained_and_have_no_outputs(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('Music-Studio.ipynb', 'Music-Studio-Dataset.ipynb'):
            nb = json.loads((root / name).read_text())
            code = [c for c in nb['cells'] if c['cell_type'] == 'code']
            self.assertEqual(len(code), 1)
            self.assertEqual(code[0]['outputs'], [])
            compile(''.join(code[0]['source']), name, 'exec')


if __name__ == '__main__':
    unittest.main()
