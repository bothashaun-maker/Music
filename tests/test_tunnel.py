"""CPU checks for the additive Cloudflare sidecar, no model generation."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from native_yue2 import tunnel


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
    def log_message(self, *args):
        pass


class TunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.access = tunnel.CloudflareAccess(self.root)
        self.addCleanup(self.access.close)

    def executable(self, text):
        p = self.root / 'fake-cloudflared'
        p.write_text('#!/bin/sh\n' + text)
        p.chmod(0o700)
        return p

    def test_verified_binary_reused_without_network(self):
        p = self.root / 'cloudflared'
        p.write_bytes(b'verified synthetic binary')
        with patch.object(tunnel, 'SHA256', tunnel.digest(p)), \
             patch.object(tunnel.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            self.assertEqual(tunnel.ensure_binary(self.root), p)

    def test_dataset_optional_binary_copied_without_download(self):
        assets = self.root / 'input'
        p = assets / 'cloudflare/cloudflared-linux-amd64'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'verified saved binary')
        with patch.object(tunnel, 'SHA256', tunnel.digest(p)), \
             patch.object(tunnel.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            result = tunnel.ensure_binary(self.root, assets)
        self.assertEqual(result.read_bytes(), p.read_bytes())
        self.assertTrue(os.access(result, os.X_OK))

    def test_bad_download_not_executed_or_published(self):
        with patch.object(tunnel.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong')):
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                tunnel.ensure_binary(self.root)
        self.assertFalse((self.root / 'cloudflared').exists())
        self.assertFalse((self.root / 'cloudflared.download').exists())

    def test_registered_tunnel_and_cleanup_use_actual_port(self):
        p = self.executable("echo \"$*\"\necho https://local-test.trycloudflare.com\n"
                            "echo Registered tunnel connection\nexec sleep 60\n")
        with patch.object(tunnel, 'ensure_binary', return_value=p):
            self.assertEqual(self.access.start(self.port, timeout=4), 'https://local-test.trycloudflare.com')
        child = self.access.process
        self.assertIsNone(child.poll())
        contents = self.access.log.read_text()
        self.assertIn(f'http://127.0.0.1:{self.port}', contents)
        self.assertIn('--protocol http2', contents)
        self.access.close()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(self.access.url)

    def test_early_failure_is_visible(self):
        p = self.executable('echo startup-failed\nexit 3\n')
        with patch.object(tunnel, 'ensure_binary', return_value=p):
            with self.assertRaisesRegex(RuntimeError, 'startup-failed'):
                self.access.start(self.port, timeout=4)
        self.assertIsNone(self.access.process)

    def test_timeout_closes_process_even_if_url_was_announced(self):
        p = self.executable('echo https://not-connected.trycloudflare.com\nexec sleep 60\n')
        with patch.object(tunnel, 'ensure_binary', return_value=p):
            with self.assertRaises(TimeoutError):
                self.access.start(self.port, timeout=.4)
        self.assertIsNone(self.access.process)

    def test_parent_exit_stops_tunnel_process(self):
        ready = self.root / 'ready'
        p = self.executable(f'touch "{ready}"\nexec sleep 60\n')
        script = '''
import os,subprocess,sys,time
from pathlib import Path
child = subprocess.Popen([sys.executable, sys.argv[1], '--watch', str(os.getpid()), sys.argv[2], '7860'])
for _ in range(100):
    if Path(sys.argv[3]).exists(): break
    time.sleep(.02)
else: raise RuntimeError('test child not ready')
print(child.pid, flush=True)
os._exit(0)
'''
        parent = subprocess.run([sys.executable, '-c', script, str(Path(tunnel.__file__)), str(p), str(ready)],
                                check=True, capture_output=True, text=True, timeout=6)
        pid = int(parent.stdout.strip())
        for _ in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split(') ',1)[1][0] == 'Z':
                break
            time.sleep(.02)
        else:
            os.kill(pid, signal.SIGKILL)
            self.fail('Cloudflare helper survived its notebook parent')

    def test_notebook_preserves_gradio_and_cleans_up_additional_link(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'yue2-3b-frontier-full-song-music-generation.ipynb').read_text())
        source = ''.join(notebook['cells'][3]['source'])
        self.assertIn('cloudflare_access.close()', source)
        self.assertIn('The Gradio link is still available.', source)
        self.assertIn('cloudflare_attempted = True', source)
        engine = (root / 'native_yue2/engine.py').read_text()
        self.assertIn('"share": True', engine)
        self.assertIn('cover_audio = gr.Audio(', engine)


if __name__ == '__main__':
    unittest.main()
