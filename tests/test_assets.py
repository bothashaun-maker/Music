"""CPU-only tests of saved-asset building, offline setup, and notebook payloads."""
import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from native_yue2 import assets
from native_yue2 import setup
from kaggle_yue2 import runtime
from kaggle_yue2.recovery import sha256_file

ROOT = Path(__file__).resolve().parents[1]


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'inputs/user/datasets/music/versions/1/yue2-assets'
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.bundle.mkdir(parents=True)
        for repo, revision in assets.MODELS.values():
            p = self.bundle / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
            p.mkdir(parents=True)
            (p / 'config.json').write_text('{}')
            (p / 'model.safetensors').write_bytes(b'synthetic weights; not inference')
        (self.bundle / 'overlay-requirements.txt').write_text('')
        (self.bundle / 'wheelhouse').mkdir()
        self.pip = self.bundle / 'wheelhouse' / runtime.PIP_WHEEL
        self.pip.write_bytes(b'fake bootstrap')
        self.wheel = self.bundle / 'wheelhouse/yue2_infer-0.1.5-py3-none-any.whl'
        self.wheel.write_bytes(b'fake native wheel')
        (self.bundle / 'gradio').mkdir()
        (self.bundle / 'gradio/frpc_linux_amd64_v0.3').write_bytes(b'fake tunnel')
        self.manifest()

    def manifest(self):
        info = {'kind': 'yue2-kaggle-assets', 'schema': assets.SCHEMA,
                'models': {k: list(v) for k,v in assets.MODELS.items()},
                'image': assets.image_identity(), 'files': {}}
        for p in self.bundle.rglob('*'):
            if p.is_file() and p.name != 'assets.json':
                info['files'][str(p.relative_to(self.bundle))] = {
                    'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        (self.bundle / 'assets.json').write_text(json.dumps(info))
        return info

    def activate(self):
        with patch.object(runtime, 'PIP_SHA256', sha256_file(self.pip)):
            return assets.activate_assets(self.bundle, self.runtime, {'ORIGINAL': 'yes'})

    def test_auto_detects_nested_dataset_and_saved_output(self):
        self.assertEqual(assets.find_assets(input_root=self.root / 'inputs'), self.bundle)
        self.assertEqual(assets.find_assets(self.bundle), self.bundle)
        self.assertIsNone(assets.find_assets(None))

    def test_missing_and_ambiguous_datasets_do_not_download(self):
        with self.assertRaisesRegex(RuntimeError, 'will not download'):
            assets.find_assets(input_root=self.root / 'empty')
        other = self.root / 'inputs/other'
        other.mkdir()
        (other / 'assets.json').write_text((self.bundle / 'assets.json').read_text())
        with self.assertRaisesRegex(RuntimeError, 'multiple versions'):
            assets.find_assets(input_root=self.root / 'inputs')

    def test_offline_environment_uses_input_directly_and_only_copies_small_tools(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            env = self.activate()
        self.assertEqual(env['HF_HUB_CACHE'], str(self.bundle / 'hub'))
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(env['PIP_NO_INDEX'], '1')
        self.assertEqual(env['ORIGINAL'], 'yes')
        self.assertTrue((self.runtime / 'bootstrap' / runtime.PIP_WHEEL).exists())
        self.assertFalse((self.runtime / 'hub').exists())
        self.assertTrue(os.access(self.runtime / 'huggingface/gradio/frpc/frpc_linux_amd64_v0.3', os.X_OK))

    def test_corrupt_or_missing_files_rejected(self):
        self.pip.write_bytes(b'wrong contents')
        with self.assertRaisesRegex(RuntimeError, 'Missing/incomplete|checksum'):
            assets.validate_assets(self.bundle)
        self.pip.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing/incomplete'):
            assets.validate_assets(self.bundle)

    def test_full_validation_detects_weight_tampering(self):
        weight = next((self.bundle / 'hub').rglob('*.safetensors'))
        original = weight.read_bytes()
        weight.write_bytes(b'x' * len(original))
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            assets.validate_assets(self.bundle, full=True)

    def test_python_image_mismatch_rejected_before_network(self):
        identity = assets.image_identity()
        identity['python'] = [0, 1]
        with patch.object(assets, 'image_identity', return_value=identity):
            with self.assertRaisesRegex(RuntimeError, 'image version|image,|Kaggle Python'):
                self.activate()

    def test_manifest_cannot_escape_bundle(self):
        data = json.loads((self.bundle / 'assets.json').read_text())
        data['files']['../../outside'] = {'bytes': 1, 'sha256': 'x'}
        (self.bundle / 'assets.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'Invalid dataset'):
            assets.validate_assets(self.bundle)

    def test_offline_setup_no_network_and_no_gpu_for_builder_option(self):
        calls = []
        def command(args, **kwargs):
            calls.append(([str(x) for x in args], kwargs))
            return SimpleNamespace(stdout='{}\n')
        def version(name):
            return '2.11.0' if name in {'torch', 'torchaudio'} else '1.0.0'
        with patch.object(runtime, 'PIP_SHA256', sha256_file(self.pip)), \
             patch.object(setup, 'WHEEL_SHA256', sha256_file(self.wheel)), \
             patch.object(setup.KaggleYuE2, 'prepare_environment'), \
             patch.object(setup, 'command', side_effect=command), \
             patch.object(setup.shutil, 'which', return_value='/usr/bin/ffmpeg'), \
             patch.object(setup.importlib.metadata, 'version', side_effect=version), \
             patch.object(assets, 'validate_assets'), \
             patch('urllib.request.urlopen', side_effect=AssertionError('unexpected download')), \
             patch('native_yue2.process.check_gpu_room', side_effect=AssertionError('GPU touched')):
            _, env = setup.install(self.runtime, self.root / 'out/runs', assets=self.bundle, require_gpu=False)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        pip = [a for a, _ in calls if 'install' in a and '--no-deps' not in a][0]
        self.assertIn('--no-index', pip)
        self.assertIn(str(self.bundle / 'wheelhouse'), pip)
        probe = next(a[-1] for a, _ in calls if len(a) > 1 and a[1] == '-c')
        self.assertIn('if False:', probe)

    def test_builder_synthetic_hub_shared_blobs_and_complete_marker(self):
        # Exercise the complete builder without model downloads, GPU or real pip.
        destination = self.root / 'built'
        (self.runtime / 'bootstrap').mkdir()
        (self.runtime / 'bootstrap' / runtime.PIP_WHEEL).write_bytes(self.pip.read_bytes())
        (self.runtime / self.wheel.name).write_bytes(self.wheel.read_bytes())
        hf = ModuleType('huggingface_hub')
        def snapshot(repo, *, revision, cache_dir, **kwargs):
            cache = Path(cache_dir) / ('models--' + repo.replace('/', '--'))
            blobs = cache / 'blobs'
            blobs.mkdir(parents=True)
            snapshot = cache / 'snapshots' / revision
            snapshot.mkdir(parents=True)
            for filename, content in [('config.json', b'{}'), ('model.safetensors', b'weights')]:
                blob = blobs / filename
                blob.write_bytes(content)
                (snapshot / filename).symlink_to(blob)
            # Multiple snapshot paths may point at one content-addressed blob.
            (snapshot / 'duplicate.json').symlink_to(blobs / 'config.json')
        hf.snapshot_download = snapshot
        tunnel = ModuleType('gradio.tunneling')
        tunnel.BINARY_PATH = str(self.bundle / 'gradio/frpc_linux_amd64_v0.3')
        tunnel.BINARY_URL = 'synthetic'
        tunnel.CHECKSUMS = {'synthetic': sha256_file(tunnel.BINARY_PATH)}
        tunnel.Tunnel = SimpleNamespace(download_binary=lambda: None)
        with patch.dict(sys.modules, {'huggingface_hub': hf, 'gradio.tunneling': tunnel}), \
             patch.dict(os.environ, {'HF_HUB_OFFLINE': '0'}), \
             patch.object(setup, 'WHEEL_SHA256', sha256_file(self.wheel)), \
             patch.object(assets.subprocess, 'check_output', return_value='[{"name":"demo", "version":"1.0"}]'), \
             patch.object(assets.subprocess, 'run', return_value=None) as pip:
            assets.build_assets(destination, self.runtime)
        self.assertTrue((destination / 'assets.json').exists())
        self.assertFalse(any(p.is_symlink() for p in destination.rglob('*')))
        self.assertFalse(list((destination / 'hub').glob('models--*/blobs')))
        commands = [call.args[0] for call in pip.call_args_list]
        self.assertTrue(any('--no-deps' in c for c in commands))
        self.assertTrue(any('--no-index' in c and '--dry-run' in c for c in commands))
        self.assertTrue(assets.validate_assets(destination, full=True))

    def test_real_hub_cache_lookup_offline_in_fresh_process(self):
        import importlib.util
        if importlib.util.find_spec('huggingface_hub') is None:
            self.skipTest('Optional integration check needs huggingface-hub==0.36.2')
        script = r"""
import json, pathlib, socket, sys
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
root = pathlib.Path(sys.argv[1])
models = json.loads(sys.argv[2])
def denied(*args, **kwargs):
    raise AssertionError('network attempted during cached model lookup')
socket.socket.connect = denied
before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
for repo, revision in models.values():
    path = snapshot_download(repo, revision=revision, cache_dir=root)
    assert pathlib.Path(path, 'model.safetensors').is_file()
    config = hf_hub_download(repo, 'config.json', revision=revision, cache_dir=root)
    assert pathlib.Path(config).read_text() == '{}'
    try:
        hf_hub_download(repo, 'missing.json', revision=revision, cache_dir=root)
    except LocalEntryNotFoundError:
        pass
    else:
        raise AssertionError('missing file did not fail locally')
assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))
print('Real Hub offline snapshot/file lookup passed, with sockets blocked and no cache mutations')
"""
        result = subprocess.run([sys.executable, '-c', script, str(self.bundle / 'hub'),
                                 json.dumps(assets.MODELS)], check=True, capture_output=True, text=True,
                                env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        self.assertIn('passed', result.stdout)

    def test_partial_build_is_not_accepted(self):
        (self.bundle / 'assets.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'will not download'):
            assets.find_assets(input_root=self.root / 'inputs')

    def test_both_notebooks_parse_and_embed_exact_current_sources(self):
        for filename in ('yue2-build-kaggle-assets.ipynb', 'yue2-3b-frontier-full-song-music-generation.ipynb'):
            notebook = json.loads((ROOT / filename).read_text())
            found = False
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                tree = ast.parse(''.join(cell['source']))
                for statement in tree.body:
                    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'payload' for t in statement.targets):
                        files = json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(statement.value))))
                        for name, contents in files.items():
                            self.assertEqual((ROOT / name).read_text(), contents)
                        self.assertIn('native_yue2/assets.py', files)
                        found = True
            self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
