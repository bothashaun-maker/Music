"""Regression checks for interrupted restore, tunnel readiness and kernel shutdown."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from music_studio import common
from music_studio.launch import restore_library, tunnel_address


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'input/music_studio'
        (self.source / 'media').mkdir(parents=True)
        (self.source / 'media/song.wav').write_bytes(b'original audio')
        with sqlite3.connect(self.source / 'library.sqlite') as db:
            db.execute('create table songs(title text)')
            db.execute('insert into songs values (?)', ('Original',))
        self.output = self.root / 'output'
        self.output.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_copy_interruption_can_retry_without_partial_working_library(self):
        import shutil
        original = shutil.copytree
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError('simulated interrupted copy')
        with patch('music_studio.launch.shutil.copytree', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                restore_library(self.output, self.root / 'input')
        self.assertFalse((self.output / 'media').exists())
        self.assertFalse((self.output / 'library.sqlite').exists())
        restore_library(self.output, self.root / 'input')
        self.assertEqual((self.output / 'media/song.wav').read_bytes(), b'original audio')
        self.assertFalse((self.output / '.restore-stage').exists())

    def interrupt_commit(self):
        replace = Path.replace
        def interrupted(path, target):
            result = replace(path, target)
            if path.name == 'media':
                raise OSError('simulated lost move acknowledgement')
            return result
        with patch.object(Path, 'replace', interrupted):
            with self.assertRaisesRegex(OSError, 'acknowledgement'):
                restore_library(self.output, self.root / 'input')

    def test_commit_interruption_recovers_verified_moved_media(self):
        self.interrupt_commit()
        self.assertFalse((self.output / 'library.sqlite').exists())
        restore_library(self.output, self.root / 'input')
        with sqlite3.connect(self.output / 'library.sqlite') as db:
            self.assertEqual(db.execute('select title from songs').fetchone()[0], 'Original')

    def test_recovery_does_not_replace_work_created_after_interruption(self):
        self.interrupt_commit()
        current = self.output / 'media/new-work.wav'
        current.write_bytes(b'keep this')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            restore_library(self.output, self.root / 'input')
        self.assertEqual(current.read_bytes(), b'keep this')
        self.assertFalse((self.output / 'library.sqlite').exists())

    def test_restore_rejects_symlink_directories(self):
        (self.source / 'media/escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlinks'):
            restore_library(self.output, self.root / 'input')
        self.assertFalse((self.output / 'library.sqlite').exists())

    def test_quick_tunnel_needs_registered_connection(self):
        log = 'Your quick Tunnel has been created! https://example.trycloudflare.com\n'
        self.assertIsNone(tunnel_address(log))
        self.assertIsNone(tunnel_address(log + 'failed to connect to edge\n'))
        self.assertEqual(tunnel_address(log + 'INF Registered tunnel connection connIndex=0'), 'https://example.trycloudflare.com')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux notebook lifecycle')
    def test_kernel_death_stops_launcher_and_its_owned_process(self):
        worker = self.root / 'worker.py'
        worker.write_text('''import os,sys,time
from pathlib import Path
from music_studio.common import notebook_process
from music_studio.launch import owned_child,stop
notebook_process()
root=Path(sys.argv[1])
child=None
try:
 child=owned_child([sys.executable,'-c','import time;time.sleep(120)'],env=dict(os.environ),cwd=root,log=root/'child.log')
 (root/'ready').write_text(str(child.pid))
 while True: time.sleep(.1)
finally:
 stop(child)
 (root/'cleaned').write_text('stopped')
''')
        kernel = self.root / 'kernel.py'
        kernel.write_text('''import os,subprocess,sys,time
p=subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],env=dict(os.environ,MUSIC_STUDIO_PARENT_PID=str(os.getpid())))
print(p.pid,flush=True)
time.sleep(120)
''')
        env = dict(os.environ, PYTHONPATH=str(common.HERE.parent))
        process = subprocess.Popen([sys.executable, str(kernel), str(worker), str(self.root)], env=env, stdout=subprocess.PIPE, text=True)
        worker_pid = int(process.stdout.readline())
        try:
            deadline = time.monotonic() + 20
            while not (self.root / 'ready').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue((self.root / 'ready').exists(), 'worker did not start')
            child_pid = int((self.root / 'ready').read_text())
            process.kill()  # A kernel crash/restart does not run the notebook's finally.
            process.wait(timeout=5)
            while not (self.root / 'cleaned').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue((self.root / 'cleaned').exists(), 'launcher did not clean up after parent death')
            self.assertFalse(Path(f'/proc/{child_pid}').exists(), 'owned child survived kernel exit')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    unittest.main()
