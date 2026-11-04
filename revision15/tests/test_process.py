"""Real CPU child-process ownership/locks; GPU reports are synthetic fixtures."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from native_yue2.process import (identity, stop_owned, start_engine, check_gpu_room, _send_verified)

ROOT = Path(__file__).resolve().parents[1]


def hardware(free0=14500, own=0):
    return {'devices': [
        {'index':0,'uuid':'gpu0','total_mib':14900,'free_mib':free0},
        {'index':1,'uuid':'gpu1','total_mib':14900,'free_mib':14500}],
        'applications': ([{'pid':os.getpid(),'gpu_uuid':'gpu0','used_mib':str(own)}] if own else [])}


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='yue2-process-test-')
        self.root = Path(self.temp.name)
        self.outputs = self.root / 'saved/runs'
        self.outputs.mkdir(parents=True)
        self.saved = self.outputs / 'existing-song.txt'
        self.saved.write_text('preserve song data')
        self.script = self.root / 'engine.py'
        self.script.write_text('''
import os, time
from pathlib import Path
from native_yue2.process import acquire_engine_lock
if os.environ.get('TEST_LEGACY') != '1':
    lock = acquire_engine_lock(os.environ['YUE2_RUNTIME_ROOT'], __file__, os.environ['YUE2_OUTPUT_ROOT'])
Path(os.environ['YUE2_RUNTIME_ROOT'], 'ready-' + str(os.getpid())).write_text('ready')
time.sleep(60)
''')
        self.env = dict(os.environ, PYTHONPATH=str(ROOT), YUE2_RUNTIME_ROOT=str(self.root),
                        YUE2_OUTPUT_ROOT=str(self.outputs))
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
            if child.stdout is not None:
                child.stdout.close()
            if child.stderr is not None:
                child.stderr.close()
        self.temp.cleanup()

    def wait_ready(self, child):
        deadline = time.monotonic() + 5
        while not (self.root / f'ready-{child.pid}').exists():
            if child.poll() is not None:
                self.fail(f'Child stopped: {child.stderr.read() if child.stderr else child.returncode}')
            if time.monotonic() >= deadline:
                self.fail('Child startup timed out')
            time.sleep(.01)

    def spawn(self, **changes):
        env = dict(self.env, **changes)
        child = subprocess.Popen([sys.executable,'-u',str(self.script)],env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.children.append(child)
        self.wait_ready(child)
        return child

    def test_verified_engine_can_be_stopped_without_popen_reference(self):
        child = self.spawn()
        self.assertEqual(stop_owned(self.root,self.script,self.outputs),[child.pid])
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.saved.read_text(),'preserve song data')

    def test_older_unrecorded_engine_is_verified_and_stopped(self):
        child = self.spawn(TEST_LEGACY='1')
        self.assertFalse((self.root/'engine-owner.json').exists())
        self.assertEqual(stop_owned(self.root,self.script,self.outputs),[child.pid])
        self.assertIsNotNone(child.poll())

    def test_same_script_with_different_output_root_is_left_running(self):
        child = self.spawn(TEST_LEGACY='1',YUE2_OUTPUT_ROOT=str(self.root/'other-output'))
        self.assertEqual(stop_owned(self.root,self.script,self.outputs),[])
        self.assertIsNone(child.poll())

    def test_changed_start_identity_cannot_signal_process(self):
        child = self.spawn()
        record = identity(child.pid,self.script,self.outputs)
        record['start_ticks'] = str(int(record['start_ticks'])+1)
        self.assertFalse(_send_verified(record,signal.SIGTERM))
        self.assertIsNone(child.poll())

    def test_process_lock_rejects_second_direct_engine(self):
        child = self.spawn()
        result = subprocess.run([sys.executable,'-u',str(self.script)],env=self.env,
                                text=True,capture_output=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('already running',result.stderr)
        self.assertIsNone(child.poll())

    def test_start_replaces_only_verified_old_engine(self):
        first = self.spawn()
        with patch('native_yue2.process.gpu_report',return_value=hardware()):
            second = start_engine(sys.executable,self.script,self.env,self.root,self.outputs)
        self.children.append(second)
        self.wait_ready(second)
        self.assertIsNotNone(first.poll())
        self.assertIsNone(second.poll())
        self.assertEqual(self.saved.read_text(),'preserve song data')

    def test_low_gpu_memory_stops_before_popen(self):
        report = hardware(free0=11)
        report['applications'] = [{'pid':48,'gpu_uuid':'gpu0','used_mib':'7200'}]
        with patch('native_yue2.process.gpu_report',return_value=report), \
             patch('native_yue2.process.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError,'GPU users'):
                start_engine(sys.executable,self.script,self.env,self.root,self.outputs)
            spawn.assert_not_called()

    def test_warm_own_model_is_reusable_but_competing_model_is_not(self):
        check_gpu_room(hardware(free0=7000,own=7400),include_pid=os.getpid())
        with self.assertRaisesRegex(RuntimeError,'only 11 MiB'):
            check_gpu_room(hardware(free0=11,own=7400),include_pid=os.getpid())
