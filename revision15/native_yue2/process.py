"""Own exactly one native engine across notebook-cell and kernel restarts."""
from pathlib import Path
import fcntl
import json
import os
import select
import shutil
import signal
import subprocess
import time


def identity(pid, script, outputs):
    """Verify same UID, actual script argument, output root and PID start time."""
    directory = Path('/proc') / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        # Our launch command is exactly python -u /absolute/path/engine.py.
        if len(argv) < 3 or argv[1] != b'-u' or os.fsdecode(argv[2]) != str(Path(script).resolve()):
            return None
        environment = (directory / 'environ').read_bytes().split(b'\0')
        expected = os.fsencode('YUE2_OUTPUT_ROOT=' + str(Path(outputs).resolve()))
        if expected not in environment:
            return None
        return {'pid': int(pid), 'start_ticks': fields[19], 'uid': os.getuid(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'script': str(Path(script).resolve()), 'outputs': str(Path(outputs).resolve())}
    except (OSError, ValueError, IndexError):
        return None


def _write_owner(root, record):
    target = Path(root) / 'engine-owner.json'
    temporary = target.with_suffix(f'.{os.getpid()}.writing')
    temporary.write_text(json.dumps(record))
    os.replace(temporary, target)


def acquire_engine_lock(root, script, outputs):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'engine-running.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('A native engine is already running. Stop/reuse it before loading a second model.')
    record = identity(os.getpid(), script, outputs)
    if record is None:
        lock.close()
        raise RuntimeError('Cannot verify native engine process ownership')
    _write_owner(root, record)
    return lock  # Keep alive for the entire engine process, including UI/recovery.


def _send_verified(record, signum):
    pid = record['pid']
    descriptor = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            descriptor = os.pidfd_open(pid)
        if identity(pid, record['script'], record['outputs']) != record:
            return False
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signum)
        else:
            os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _wait_exit(record, timeout):
    """Wait for real process exit; cmdline/environ can vanish before it exits."""
    descriptor = None
    try:
        if hasattr(os, 'pidfd_open'):
            descriptor = os.pidfd_open(record['pid'])
            fields = (Path('/proc') / str(record['pid']) / 'stat').read_text().rsplit(') ', 1)[1].split()
            if fields[19] != record['start_ticks']:
                return True
            return bool(select.select([descriptor], [], [], timeout)[0])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fields = (Path('/proc') / str(record['pid']) / 'stat').read_text().rsplit(') ', 1)[1].split()
            except FileNotFoundError:
                return True
            if fields[19] != record['start_ticks'] or fields[0] == 'Z':
                return True
            time.sleep(.05)
        return False
    except (ProcessLookupError, FileNotFoundError):
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stop_owned(root, script, outputs, timeout=5):
    """Stop only verified engines, including older revisions with no owner file."""
    root = Path(root)
    candidates = {}
    owner = root / 'engine-owner.json'
    if owner.is_file():
        try:
            saved = json.loads(owner.read_text())
            if saved.get('script') == str(Path(script).resolve()) and saved.get('outputs') == str(Path(outputs).resolve()):
                if identity(saved['pid'], script, outputs) == saved:
                    candidates[saved['pid']] = saved
        except (OSError, ValueError, KeyError, TypeError):
            pass
    # Verify the exact script and output environment, not just a Python name.
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal() and int(entry.name) != os.getpid():
            record = identity(int(entry.name), script, outputs)
            if record is not None:
                candidates[record['pid']] = record
    for record in candidates.values():
        print(f"Stopping previous verified native engine PID {record['pid']}; saved runs are retained.", flush=True)
        _send_verified(record, signal.SIGTERM)
        if not _wait_exit(record, timeout):
            _send_verified(record, signal.SIGKILL)
            if not _wait_exit(record, timeout):
                raise RuntimeError(f"Verified native engine PID {record['pid']} has not stopped")
    return sorted(candidates)


def gpu_report():
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('nvidia-smi is unavailable; select the Kaggle GPU image')
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'], text=True).strip()
    rows = query(['--query-gpu=index,uuid,memory.total,memory.free'])
    devices = []
    for row in rows.splitlines():
        index, uuid, total, free = [v.strip() for v in row.split(',')]
        devices.append({'index': int(index), 'uuid': uuid, 'total_mib': int(total), 'free_mib': int(free)})
    applications = []
    for row in query(['--query-compute-apps=pid,gpu_uuid,used_gpu_memory']).splitlines():
        if not row.strip():
            continue
        pid, uuid, used = [v.strip() for v in row.split(',')]
        try:
            name = (Path('/proc') / pid / 'comm').read_text().strip()
        except OSError:
            name = 'not visible'
        applications.append({'pid': int(pid), 'gpu_uuid': uuid, 'used_mib': used,
                             'process_name': name, 'is_checking_process': int(pid) == os.getpid()})
    return {'caller_pid': os.getpid(), 'devices': devices, 'applications': applications}


def check_gpu_room(report=None, include_pid=None):
    report = gpu_report() if report is None else report
    print('GPU memory/process preflight:', json.dumps(report, indent=2), flush=True)
    requirements = {0: 12 * 1024, 1: 4 * 1024}
    for index, minimum in requirements.items():
        device = next((d for d in report['devices'] if d['index'] == index), None)
        if device is None:
            raise RuntimeError('Select GPU T4 x2; two GPUs are required')
        processes = [p for p in report['applications'] if p['gpu_uuid'] == device['uuid']]
        reusable = 0
        for process in processes:
            if process['pid'] == include_pid:
                try:
                    reusable += int(process['used_mib'])
                except (ValueError, TypeError):
                    pass
        if device['free_mib'] + reusable < minimum:
            raise RuntimeError(f"GPU {index} has only {device['free_mib']} MiB free; "
                               f"{reusable} MiB belongs to this reusable engine; require {minimum} MiB combined. GPU users: {processes}. "
                               'Unverified processes were left running. Close their model/session explicitly.')
    return report


def start_engine(python, script, env, root, outputs):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'engine-start.lock').open('a') as startup:
        fcntl.flock(startup, fcntl.LOCK_EX)
        stop_owned(root, script, outputs)
        check_gpu_room()  # Does not load a model or allocate CUDA tensors.
        process = subprocess.Popen([str(python), '-u', str(Path(script).resolve())], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        return process
