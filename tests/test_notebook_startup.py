"""Fresh-kernel startup and auxiliary cells; never loads a GPU model."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

from native_yue2 import launch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'yue2-3b-frontier-full-song-music-generation.ipynb').read_text())


class StartupTests(unittest.TestCase):
    def test_start_cell_runs_from_empty_globals(self):
        code = ''.join(NOTEBOOK['cells'][2]['source'])
        with tempfile.TemporaryDirectory() as tmp:
            code = code.replace('/kaggle/temp/yue2_native_support', tmp+'/support')
            code = code.replace('/kaggle/temp/yue2_native', tmp+'/runtime')
            code = code.replace('/kaggle/working/yue2_native/runs', tmp+'/outputs/runs')
            worker = Mock(stdout=io.StringIO('STEP 1/2 COMPLETE\n'), pid=9999)
            worker.wait.return_value = 0
            worker.poll.return_value = 0
            with patch('subprocess.Popen', return_value=worker) as popen:
                exec(compile(code, '<fresh-kernel-start>', 'exec'), {'__name__':'test'})
            args = popen.call_args.args[0]
            self.assertEqual(args[2:4], ['-m', 'native_yue2.launch'])
            self.assertIn('--assets', args)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertTrue((Path(tmp)/'support/native_yue2/launch.py').is_file())
            self.assertIn('STEP 1/2 COMPLETE', (Path(tmp)/'outputs/runtime/startup.log').read_text())

    def test_listing_runs_from_empty_globals_without_gpu_or_setup(self):
        with patch('IPython.display.display'), patch('IPython.display.FileLink'):
            exec(compile(''.join(NOTEBOOK['cells'][4]['source']), '<listing>', 'exec'), {})

    def test_optional_recovery_runs_from_empty_globals(self):
        exec(compile(''.join(NOTEBOOK['cells'][6]['source']), '<recovery>', 'exec'), {})

    def run_worker(self, *, failed=False, cloudflare_fails=False, engine_code=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = Mock(stdout=iter(['* Running on local URL:  http://127.0.0.1:7868\n',
                                       '* Running on public URL: https://example.gradio.live\n']))
            process.wait.return_value = engine_code
            process.returncode = engine_code
            process.poll.return_value = engine_code
            tunnel = Mock()
            tunnel.start.return_value = 'https://test.trycloudflare.com'
            if cloudflare_fails:
                tunnel.start.side_effect = RuntimeError('synthetic tunnel failure')
            setup = Mock(return_value=('/fake/python', {'YUE2_ASSET_DATASET':'/saved/input'}))
            if failed:
                setup.side_effect = RuntimeError('synthetic setup interruption')
            with patch.object(launch, 'install', setup), patch.object(launch, 'start_engine', return_value=process) as start, \
                 patch.object(launch, 'CloudflareAccess', return_value=tunnel):
                if failed or engine_code:
                    with self.assertRaises(RuntimeError):
                        launch.launch(root, root/'outputs/runs')
                else:
                    launch.launch(root, root/'outputs/runs')
            status = json.loads((root/'outputs/runtime/startup-status.json').read_text())
            return setup, start, tunnel, status

    def test_setup_failure_is_saved_and_cannot_start_model(self):
        _, start, _, status = self.run_worker(failed=True)
        start.assert_not_called()
        self.assertEqual(status['phase'], 'failed')
        self.assertIn('synthetic setup interruption', status['error'])

    def test_setup_then_engine_share_same_saved_dataset(self):
        setup, start, tunnel, status = self.run_worker()
        setup.assert_called_once()
        self.assertEqual(start.call_args.args[2]['YUE2_ASSET_DATASET'], '/saved/input')
        tunnel.start.assert_called_once_with(7868, env={'YUE2_ASSET_DATASET':'/saved/input'})
        tunnel.close.assert_called_once()
        self.assertEqual(status['phase'], 'stopped')

    def test_tunnel_failure_keeps_existing_gradio_engine(self):
        _, start, tunnel, status = self.run_worker(cloudflare_fails=True)
        start.assert_called_once()
        tunnel.close.assert_called_once()
        self.assertEqual(status['phase'], 'stopped')

    def test_engine_error_is_saved_and_tunnel_closed(self):
        _, _, tunnel, status = self.run_worker(engine_code=1)
        self.assertEqual(status['phase'], 'failed')
        self.assertIn('Engine exited', status['error'])
        tunnel.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
