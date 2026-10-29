"""Optional Cloudflare access alongside the existing Gradio share link."""
import atexit
import hashlib
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.request

VERSION = '2026.10.3'
SHA256 = '053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0'
URL = f'https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/cloudflared-linux-amd64'


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def ensure_binary(root, assets=None):
    if platform.system() != 'Linux' or platform.machine().lower() not in {'x86_64', 'amd64'}:
        raise RuntimeError('The optional Cloudflare executable requires Linux x86_64')
    target = Path(root) / 'cloudflared'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or digest(target) != SHA256:
        saved = Path(assets) / 'cloudflare/cloudflared-linux-amd64' if assets else None
        temporary = target.with_suffix('.download')
        try:
            if saved is not None and saved.is_file():
                import shutil
                shutil.copyfile(saved, temporary)
            else:
                print('Fetching the optional Cloudflare executable on Kaggle; no model download.', flush=True)
                with urllib.request.urlopen(URL, timeout=30) as response, temporary.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(response, out)
            if digest(temporary) != SHA256:
                raise RuntimeError('Cloudflare executable checksum mismatch')
            temporary.chmod(0o700)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    target.chmod(0o700)
    return target


class CloudflareAccess:
    def __init__(self, root, *, assets=None, log=None):
        self.root = Path(root)
        self.assets = assets
        self.log = Path(log) if log else self.root / 'cloudflared.log'
        self.process = None
        self.url = None
        atexit.register(self.close)

    def start(self, port, *, timeout=60, env=None):
        self.close()
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError('Invalid local UI port')
        # The engine prints its local URL after binding the real Gradio port.
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/config', timeout=10) as response:
            if response.status != 200:
                raise RuntimeError('Local Gradio app is not ready')
        binary = ensure_binary(self.root, self.assets)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('w') as output:
            # A tiny stdlib wrapper sets Linux parent-death signalling BEFORE
            # exec. The tunnel exits if this notebook kernel dies/restarts.
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--watch', str(os.getpid()),
                 str(binary), str(port)],
                env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Cloudflare exited: ' + self.log.read_text(errors='replace')[-2400:])
                contents = self.log.read_text(errors='replace')
                match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com\b', contents)
                if match and 'Registered tunnel connection' in contents:
                    self.url = match.group(0)
                    return self.url
                time.sleep(0.25)
            raise TimeoutError(f'Cloudflare did not connect in {timeout}s; inspect {self.log}')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        self.url = None


def watch_and_exec(parent, binary, port):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'Cannot bind Cloudflare lifetime to notebook kernel')
    if os.getppid() != parent:
        raise RuntimeError('Notebook parent exited during tunnel startup')
    os.execv(binary, [binary, 'tunnel', '--no-autoupdate', '--protocol', 'http2',
                     '--url', f'http://127.0.0.1:{int(port)}'])


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--watch':
        raise SystemExit('Internal Cloudflare supervisor; launch using CloudflareAccess')
    watch_and_exec(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
