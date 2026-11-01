"""Run setup and the unchanged engine outside the notebook kernel."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback

from .process import start_engine
from .setup import install
from .tunnel import CloudflareAccess


def launch(root, outputs, *, assets='auto', cloudflare=True, recover=None):
    root, outputs = Path(root).resolve(), Path(outputs).resolve()
    records = outputs.parent / 'runtime'
    records.mkdir(parents=True, exist_ok=True)
    def status(phase, **details):
        path = records / 'startup-status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'phase': phase, **details}, indent=2) + '\n')
        os.replace(temp, path)
    process = access = None
    try:
        status('setup')
        print('STEP 1/2: checking the saved dataset and preparing the isolated runtime.', flush=True)
        python, env = install(str(root), str(outputs), assets=assets)
        print('STEP 1/2 COMPLETE. Starting the generator.', flush=True)
        status('starting_engine')
        engine = Path(__file__).with_name('engine.py')
        if recover:
            subprocess.run([python, '-u', str(engine)], env=dict(env, YUE2_RECOVER_RUN=str(Path(recover).resolve())), check=True)
            status('recovery_complete')
            return
        process = start_engine(python, engine, env, str(root), str(outputs))
        access = CloudflareAccess(root, assets=env.get('YUE2_ASSET_DATASET'), log=records / 'cloudflared.log')
        attempted = False
        for line in process.stdout:
            print(line, end='', flush=True)
            if 'Running on public URL:' in line:
                status('ready')
            local = re.search(r'Running on local URL:\s+http://127\.0\.0\.1:(\d+)', line)
            if cloudflare and local and not attempted:
                attempted = True
                try:
                    link = access.start(int(local.group(1)), env=env)
                    print(f'\nAdditional Cloudflare link: {link}', flush=True)
                    print('The existing Gradio link remains available. Neither starts a song automatically.', flush=True)
                except Exception as exc:
                    print(f'Cloudflare unavailable: {exc}\nContinuing with the existing Gradio link.', flush=True)
        if process.wait() != 0:
            raise RuntimeError(f'Engine exited with code {process.returncode}')
        status('stopped')
    except (KeyboardInterrupt, SystemExit):
        status('stopped')
        raise
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if access is not None:
            access.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--assets', default='auto')
    parser.add_argument('--no-cloudflare', action='store_true')
    parser.add_argument('--recover')
    args = parser.parse_args()
    # SIGTERM from the notebook's Stop button must run model/tunnel cleanup.
    def stop(signum, frame):
        raise KeyboardInterrupt('Notebook stopped the launcher')
    signal.signal(signal.SIGTERM, stop)
    if sys.platform.startswith('linux'):
        import ctypes
        parent = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot bind launcher to notebook lifetime')
        if os.getppid() != parent:
            raise KeyboardInterrupt('Notebook exited during launcher startup')
    launch(args.root, args.outputs, assets=args.assets, cloudflare=not args.no_cloudflare, recover=args.recover)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Generator stopped; saved files remain available.', flush=True)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
