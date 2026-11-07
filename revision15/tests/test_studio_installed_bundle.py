"""Execute the shipped installer and nodes under ComfyUI's path-based module name."""
import ast
import base64
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import soundfile
import torch
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('STS_AUTOLAUNCH', '0')


class InstalledBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('installed_bundle_builder', ROOT/'tools/build_studio_notebook.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.files = json.loads(zlib.decompress(base64.b64decode(builder.build_payload())))
        self.support = self.root/'support'
        for name, text in self.files.items():
            path = self.support/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        function = next(n for n in ast.parse(builder.CODE_CELL).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'install_studio_pack')
        namespace = {'Path': Path, 'shutil': shutil}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'notebook_installer.py', 'exec'), namespace)
        self.install = namespace['install_studio_pack']
        self.pack_path = self.install(self.support, self.root/'ComfyUI')
        self.modules_before = set(sys.modules)
        self.addCleanup(self.clean_modules)

    def clean_modules(self):
        prefix = str(self.pack_path)
        for name in set(sys.modules)-self.modules_before:
            if name.startswith(prefix) or name.startswith('_timestamp_studio_acoustic_'):
                sys.modules.pop(name, None)

    def load_pack(self):
        comfy = ModuleType('comfy')
        mm = ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = lambda: None
        comfy.model_management = mm
        folders = ModuleType('folder_paths')
        folders.get_output_directory = lambda: str(self.root/'output')
        folders.get_input_directory = lambda: str(self.root/'input')
        folders.get_temp_directory = lambda: str(self.root/'temp')
        server = ModuleType('server')
        server.PromptServer = SimpleNamespace(instance=SimpleNamespace(routes=web.RouteTableDef()))
        self.mock_modules = patch.dict(sys.modules, {'comfy':comfy, 'comfy.model_management':mm,
                                                     'folder_paths':folders, 'server':server})
        self.mock_modules.start()
        self.addCleanup(self.mock_modules.stop)
        # ComfyUI can register a custom-node package using its absolute path.
        # This reproduces the namespace in the supplied Kaggle traceback.
        name = str(self.pack_path)
        spec = importlib.util.spec_from_file_location(name, self.pack_path/'__init__.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_installer_copies_every_payload_file_including_acoustic_helper(self):
        for name, text in self.files.items():
            relative = name.split('/', 1)[1]
            self.assertEqual((self.pack_path/relative).read_text(), text)
        self.assertTrue((self.root/'ComfyUI/user/default/workflows/Song Timestamp Studio.json').is_file())

    def test_path_loaded_installed_node_executes_acoustic_alignment_and_export(self):
        pack = self.load_pack()
        helper = pack.engine.validate_acoustic_bundle()
        labels = np.zeros(200, dtype=int)
        labels[25:29] = 1
        labels[30:34] = 2
        scores = np.full((200, 4), -12., dtype=np.float32)
        scores[np.arange(200), labels] = -.001
        frame_data = (scores, np.arange(200)*.02, {'<pad>':0, 'H':1, 'I':2, '|':3},
                      {'device':'cpu', 'cache_reused':True})
        with patch.object(helper, 'acoustic_frames', return_value=frame_data), \
             patch.object(pack.engine, 'detect_beats', return_value=(None, [])):
            opts = pack.STSLyricsOptions().execute('installed', '[Intro]\nHI', 'base', 'en', False)[0]
            output = pack.STSAlign().execute({'waveform':torch.full((1,1,64000),.1),
                                               'sample_rate':16000}, opts)
            result = output['result'][0]
            export = pack.STSExportAll().execute(result)
        line = next(e for e in result['entries'] if e['kind']=='sing')
        self.assertEqual(line['text'], 'HI')
        self.assertAlmostEqual(line['start'], .5)
        self.assertEqual(result['alignment_method'], 'ctc_acoustic')
        self.assertEqual(len(export['ui']['sts_download']), 1)
        self.assertEqual(len(list((self.root/'output').rglob('*.zip'))), 1)

    def test_missing_helper_fails_registration_with_its_actual_path(self):
        (self.pack_path/'acoustic.py').unlink()
        with self.assertRaisesRegex(RuntimeError, 'installation is incomplete: missing .*acoustic.py'):
            self.load_pack()

    def test_installed_node_routes_selected_isolation_to_both_acoustic_views(self):
        pack = self.load_pack()
        helper = pack.engine.validate_acoustic_bundle()
        separator = pack.engine._adjacent_helper('vocal.py')
        vocal_path = self.root/'vocals.wav'
        soundfile.write(vocal_path, np.full(64000,.2,np.float32),16000,subtype='FLOAT')
        labels = np.zeros(200,dtype=int)
        labels[25:29]=1; labels[30:34]=2
        scores = np.full((200,4),-12.,np.float32)
        scores[np.arange(200),labels]=-.001
        frames = (scores,np.arange(200)*.02,{'<pad>':0,'H':1,'I':2,'|':3},
                  {'device':'cached','cache_reused':True})
        with patch.object(separator,'isolate',return_value=(str(vocal_path),{'cache_reused':True})), \
             patch.object(helper,'acoustic_frames',return_value=frames) as inference, \
             patch.object(pack.engine,'detect_beats',return_value=(None,[])):
            opts = pack.STSLyricsOptions().execute('views','HI','auto','en',True)[0]
            result = pack.STSAlign().execute({'waveform':torch.full((1,1,64000),.1),
                                             'sample_rate':16000}, opts)['result'][0]
        self.assertEqual(inference.call_count,2)
        self.assertAlmostEqual(float(inference.call_args_list[0].args[0][0]),.1)
        self.assertAlmostEqual(float(inference.call_args_list[1].args[0][0]),.2)
        self.assertEqual(result['alignment']['variant'],'vocal_context_ctc')
        self.assertTrue(result['alignment']['isolation']['cache_reused'])
        self.assertAlmostEqual(result['entries'][0]['start'],.5)

    def test_dependency_error_is_not_masked_as_missing_source_package(self):
        (self.pack_path/'acoustic.py').write_text("raise ImportError('upstream dependency failure')\n")
        with self.assertRaisesRegex(ImportError, '^upstream dependency failure$'):
            self.load_pack()

    def test_incomplete_payload_fails_before_installing_a_partial_pack(self):
        (self.support/'TimestampStudio/acoustic.py').unlink()
        with self.assertRaisesRegex(RuntimeError, 'payload is incomplete: acoustic.py'):
            self.install(self.support, self.root/'fresh-ComfyUI')
        self.assertFalse((self.root/'fresh-ComfyUI/custom_nodes/TimestampStudio').exists())


if __name__ == '__main__':
    unittest.main()
