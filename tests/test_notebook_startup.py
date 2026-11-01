"""Reproduce missing kernel state without installing packages or loading a GPU."""
import ast
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / 'yue2-3b-frontier-full-song-music-generation.ipynb').read_text())
        cls.setup = ''.join(cls.notebook['cells'][1]['source'])
        launcher = ''.join(cls.notebook['cells'][3]['source'])
        cls.guard = launcher.split('# Setup verified; start the existing generator and optional tunnel below.')[0]

    def run_guard(self, *, ready=False, missing_python=False, fail=False, missing_values=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            support = root / 'support'
            (support / 'native_yue2').mkdir(parents=True)
            (support / 'native_yue2/tunnel.py').write_text('# test support module')
            python = root / 'python'
            if not missing_python:
                python.touch()
            calls = []
            def setup_executor(source, scope):
                # Recovery must execute the EXACT current setup, with its existing
                # dataset checks/GPU process guards, rather than bypassing them.
                self.assertEqual(source, self.setup)
                calls.append(source)
                if fail:
                    raise RuntimeError('synthetic setup failure')
                scope['_YUE2_SETUP_READY'] = True
            scope = {'_YUE2_SETUP_READY': ready, 'exec': setup_executor}
            if not missing_values:
                scope.update(native_python=str(python), native_env={}, ROOT=str(root),
                             OUTPUTS=str(root / 'runs'), support=support)
            guard = self.guard.replace('/kaggle/temp/yue2_native_support', str(support))
            with patch.object(sys, 'path', list(sys.path)):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic setup failure'):
                        exec(compile(guard, '<startup-check>', 'exec'), scope)
                else:
                    exec(compile(guard, '<startup-check>', 'exec'), scope)
                    self.assertIn(str(support), sys.path)
            return calls, scope

    def test_fresh_kernel_runs_setup_before_any_generator_import(self):
        calls, scope = self.run_guard(missing_values=True)
        self.assertEqual(len(calls), 1)
        self.assertTrue(scope['_YUE2_SETUP_READY'])

    def test_interrupted_setup_is_completed(self):
        calls, _ = self.run_guard(ready=False)
        self.assertEqual(len(calls), 1)

    def test_completed_setup_is_reused_and_import_path_repaired(self):
        calls, _ = self.run_guard(ready=True)
        self.assertEqual(calls, [])

    def test_removed_environment_triggers_checked_setup(self):
        calls, _ = self.run_guard(ready=True, missing_python=True)
        self.assertEqual(len(calls), 1)

    def test_failed_setup_stops_before_generator_import(self):
        calls, scope = self.run_guard(missing_values=True, fail=True)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('CloudflareAccess', scope)

    def test_completion_marker_only_follows_successful_install(self):
        tree = ast.parse(self.setup)
        markers = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_YUE2_SETUP_READY' for t in node.targets)]
        install = next(i for i,node in enumerate(tree.body) if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                       and node.value.func.id == 'install')
        self.assertLess(markers[0], install)
        self.assertGreater(markers[-1], install)
        self.assertIs(tree.body[markers[0]].value.value, False)
        self.assertIs(tree.body[markers[-1]].value.value, True)
        self.assertIn('assets=ASSET_DATASET', self.setup)


if __name__ == '__main__':
    unittest.main()
