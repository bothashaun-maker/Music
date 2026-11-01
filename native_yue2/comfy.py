"""Install a separate ComfyUI frontend around the unchanged native YuE2 stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

from kaggle_yue2.runtime import KaggleYuE2, command, COMFY_COMMIT, PIP_WHEEL
from .setup import install, REQUIREMENTS
from .tunnel import CloudflareAccess


def available_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        return probe.getsockname()[1]


def run(root, outputs, assets):
    root, outputs=Path(root).resolve(),Path(outputs).resolve()
    native_python, env=install(str(root),str(outputs),assets=assets)
    print('Preparing the separate ComfyUI interface. Existing model weights are reused.',flush=True)
    runtime=KaggleYuE2(root/'comfy_runtime')
    checkout=runtime.checkout
    if not checkout.exists():
        checkout.mkdir()
        command(['git','init','-q'],cwd=checkout)
        command(['git','fetch','--depth=1','https://github.com/Comfy-Org/ComfyUI.git',COMFY_COMMIT],cwd=checkout)
        command(['git','checkout','--detach','FETCH_HEAD'],cwd=checkout)
    current=command(['git','rev-parse','HEAD'],cwd=checkout,capture=True).stdout.strip()
    if current!=COMFY_COMMIT:
        raise RuntimeError('Existing ComfyUI checkout differs from the pinned interface version')
    (runtime.root/'bootstrap').mkdir(exist_ok=True)
    shutil.copyfile(root/'bootstrap'/PIP_WHEEL,runtime.root/'bootstrap'/PIP_WHEEL)
    runtime.prepare_environment()
    site=command([native_python,'-c','import site;print(site.getsitepackages()[0])'],env=env,capture=True).stdout.strip()
    comfy_site=command([runtime.python,'-c','import site;print(site.getsitepackages()[0])'],capture=True).stdout.strip()
    (Path(comfy_site)/'yue2-native-runtime.pth').write_text(site+'\n')
    constraints=runtime.root/'constraints.txt'
    constraints.write_text((root/'kaggle-constraints.txt').read_text()+'\n'+'\n'.join(REQUIREMENTS)+'\n')
    installer=dict(env)
    installer.pop('PIP_NO_INDEX',None)
    # Only frontend/server dependencies may be fetched; model access stays offline.
    command([runtime.python,'-m','pip','install','--no-cache-dir','-r',checkout/'requirements.txt',
             '-c',constraints],env=installer)
    source=Path(__file__).resolve().parents[1]
    destination=checkout/'custom_nodes/yue2_native'
    shutil.copytree(source/'comfy_yue2',destination,dirs_exist_ok=True)
    for name in ['song','cover']:
        workflow=source/'workflows'/f'yue2_{name}.json'
        shutil.copyfile(workflow,destination/'web'/f'{name}.json')
        menu=checkout/'user/default/workflows'
        menu.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(workflow,menu/f'YuE2 {name.title()}.json')
    settings_file=checkout/'user/default/comfy.settings.json'
    if not settings_file.exists():
        settings_file.write_text(json.dumps({'Comfy.TutorialCompleted':True})+'\n')
    outputs.mkdir(parents=True,exist_ok=True)
    media=outputs.parent/'input';media.mkdir(exist_ok=True)
    port=available_port()
    process=None
    access=CloudflareAccess(root,assets=env.get('YUE2_ASSET_DATASET'),log=outputs.parent/'runtime/comfy-cloudflared.log')
    child_env=dict(env,YUE2_HEADLESS='1')
    try:
        # Comfy's controller stays on CPU; the native nodes explicitly place
        # MoT on cuda:0 and VAE on cuda:1, exactly as in the working generator.
        log=outputs.parent/'runtime/comfyui.log'
        with log.open('a') as stream:
            process=subprocess.Popen([str(runtime.python),'-u','main.py','--listen','127.0.0.1',
                '--port',str(port),'--cpu','--disable-auto-launch','--output-directory',str(outputs.parent),
                '--input-directory',str(media)],cwd=checkout,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
        ready=False
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('ComfyUI stopped: '+log.read_text(errors='replace')[-6000:])
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/object_info',timeout=2) as response:
                    info=json.load(response)
                required={'YuE2NativeEngine','YuE2NativePlan','YuE2NativeSemantic','YuE2NativeAcoustic',
                          'YuE2NativeDecode','YuE2SaveSong','YuE2CoverABC','LoadAudio','PreviewAudio'}
                if not required<=info.keys():
                    raise RuntimeError('Required canvas nodes did not load: '+str(required-info.keys()))
                ready=True;break
            except (OSError,ValueError):
                time.sleep(1)
        if not ready:raise RuntimeError('ComfyUI startup timed out; inspect '+str(log))
        url=access.start(port,env=child_env,health_path='/system_stats')
        print('\nOPEN SONG CANVAS: '+url+'/?yue2_workflow=song',flush=True)
        print('OPEN WAV → ABC CANVAS: '+url+'/?yue2_workflow=cover',flush=True)
        print('The normal ComfyUI canvas is ready. Nothing generates until you press Run.',flush=True)
        while process.poll() is None:
            time.sleep(1)
        raise RuntimeError('ComfyUI exited; inspect '+str(log))
    finally:
        access.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--outputs',required=True)
    parser.add_argument('--assets',default='auto')
    args=parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt('Notebook stopped ComfyUI')
    signal.signal(signal.SIGTERM,stop)
    if os.name=='posix':
        import ctypes
        parent=os.getppid()
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'Cannot bind ComfyUI launcher to notebook lifetime')
        if os.getppid()!=parent:
            raise KeyboardInterrupt('Notebook exited during startup')
    run(args.root,args.outputs,args.assets)
