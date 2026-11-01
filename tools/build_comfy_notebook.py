"""Build the standalone standard ComfyUI notebook and visible workflows."""
import base64
import json
from pathlib import Path
import zlib
from build_fast_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def workflows():
    nodes=[];links=[]
    def node(i,kind,pos,inputs,outputs,widgets=(),size=(290,170)):
        item={'id':i,'type':kind,'pos':list(pos),'size':list(size),'flags':{},'order':len(nodes),'mode':0,
              'inputs':[{'name':n,'type':t,'link':None} for n,t in inputs],
              'outputs':[{'name':n,'type':t,'links':[]} for n,t in outputs],
              'properties':{'Node name for S&R':kind},'widgets_values':list(widgets)}
        nodes.append(item);return item
    def link(a,slot,b,input_slot):
        source=next(n for n in nodes if n['id']==a);target=next(n for n in nodes if n['id']==b)
        i=len(links)+1;t=source['outputs'][slot]['type']
        source['outputs'][slot]['links'].append(i);target['inputs'][input_slot]['link']=i
        links.append([i,a,slot,b,input_slot,t])
    def document():return {'last_node_id':max(n['id'] for n in nodes),'last_link_id':len(links),
        'nodes':nodes,'links':links,'groups':[],'config':{},'extra':{'ds':{'scale':0.65,'offset':[50,100]}},'version':0.4}
    node(1,'YuE2NativeEngine',(0,0),[],[('engine','YUE2_NATIVE')],size=(260,90))
    node(2,'YuE2Prompt',(0,140),[],[('prompt','YUE2_PROMPT')],
         ['Melodic rock, expressive vocals, guitar, piano, live drums','[Verse]\nPaste your complete lyrics here\n\n[Outro]\nWrite your ending here',''],size=(410,500))
    node(3,'YuE2Settings',(0,700),[],[('settings','YUE2_SETTINGS')],[17,'randomize',8,'off',16,'euler',1.0],size=(310,240))
    node(4,'YuE2NativePlan',(490,180),[('engine','YUE2_NATIVE'),('prompt','YUE2_PROMPT'),('settings','YUE2_SETTINGS')],
         [('plan','YUE2_PLAN'),('abc','STRING')])
    node(5,'YuE2NativeSemantic',(840,180),[('plan','YUE2_PLAN')],[('semantic','YUE2_SEMANTIC')])
    node(6,'YuE2NativeAcoustic',(1190,180),[('semantic','YUE2_SEMANTIC')],[('latents','YUE2_LATENTS')])
    node(7,'YuE2NativeDecode',(1540,180),[('latents','YUE2_LATENTS')],[('audio','AUDIO'),('result','YUE2_RESULT')])
    node(8,'YuE2SaveSong',(1900,160),[('audio','AUDIO'),('result','YUE2_RESULT')],[('folder','STRING')],size=(360,300))
    node(9,'PreviewAudio',(1900,540),[('audio','AUDIO')],[('audio','AUDIO')],size=(360,180))
    node(10,'YuE2ShowABC',(490,500),[('text','STRING')],[('text','STRING')],size=(600,350))
    for args in [(1,0,4,0),(2,0,4,1),(3,0,4,2),(4,0,5,0),(5,0,6,0),(6,0,7,0),(7,0,8,0),(7,1,8,1),(7,0,9,0),(4,1,10,0)]:link(*args)
    song=document()
    nodes=[];links=[]
    node(1,'YuE2NativeEngine',(0,0),[],[('engine','YUE2_NATIVE')],size=(270,100))
    node(2,'LoadAudio',(0,200),[],[('audio','AUDIO')],[''],size=(400,250))
    node(3,'YuE2CoverABC',(500,100),[('engine','YUE2_NATIVE'),('audio','AUDIO')],[('abc','STRING'),('analysis','STRING')],size=(650,500))
    link(1,0,3,0);link(2,0,3,1)
    return {'song':song,'cover':document()}


def build():
    for name,graph in workflows().items():
        (ROOT/'workflows'/f'yue2_{name}.json').write_text(json.dumps(graph,indent=2)+'\n')
    files={}
    for directory in ['native_yue2','kaggle_yue2','comfy_yue2']:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in ['.py','.js','.json']:
                files[str(path.relative_to(ROOT))]=path.read_text()
    for path in (ROOT/'workflows').glob('yue2_*.json'):
        files[str(path.relative_to(ROOT))]=path.read_text()
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
    code=r'''
import base64, json, os, signal, subprocess, sys, zlib
from pathlib import Path
SUPPORT=Path('/kaggle/temp/yue2_comfy_support')
ROOT='/kaggle/temp/yue2_native'
OUTPUTS='/kaggle/working/yue2_native/runs'
ASSET_DATASET='auto'
payload="PAYLOAD"
for relative,contents in json.loads(zlib.decompress(base64.b64decode(payload))).items():
    path=SUPPORT/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(contents)
env=dict(os.environ)
env['PYTHONPATH']=str(SUPPORT)+os.pathsep+env.get('PYTHONPATH','')
env['PYTHONUNBUFFERED']='1'
log_path=Path(OUTPUTS).parent/'runtime/comfy-startup.log'
log_path.parent.mkdir(parents=True,exist_ok=True)
worker=subprocess.Popen([sys.executable,'-u','-m','native_yue2.comfy',
    '--root',ROOT,'--outputs',OUTPUTS,'--assets',ASSET_DATASET],env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
try:
    with log_path.open('a') as log:
        for line in worker.stdout:
            log.write(line);log.flush()
            print(line,end='',flush=True)
    if worker.wait():raise RuntimeError(f'ComfyUI startup stopped. Full log: {log_path}')
finally:
    if worker.poll() is None:
        os.killpg(worker.pid,signal.SIGTERM)
        try:worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=5)
'''.replace('PAYLOAD',payload)
    cells=[cell('markdown','''
# YuE2 — standard ComfyUI canvas

Attach your existing **yue2-music-assets**, choose **GPU T4 x2**, and enable
**Internet**. Run the one code cell below. It installs the ComfyUI interface and
prints two Cloudflare links: **Song Canvas** and **WAV → ABC Canvas**. Nothing
runs until you press **Run** inside ComfyUI. There is no AIQUEST/Gradio wrapper.

The song canvas has separate, connected prompt, settings, planning, semantic,
acoustic, decoding, saving and standard audio-preview nodes. Edit the lyrics/style
and settings directly on those nodes. Native FP16/CUDA-graph generation, T4
attention patches, 16 Euler steps, CFG 1 and the 8-minute safety ceiling remain.
The model can end earlier naturally. Saved status distinguishes a natural ending
from a safety cutoff. Long-song quality, runtime and GPU peaks remain unverified.

For a cover/ABC transcription, open the **WAV → ABC** canvas, use ComfyUI's normal
**Load Audio** upload button, then Run. It displays the extracted approximate
melody ABC on the node. Analysis artifacts and score are saved in cover_sources.
You can paste that ABC into the song prompt node's ABC field, select melody
planning, and generate with your new style/lyrics. No lossy conversion is needed.

Your existing model dataset is reused without rebuilding. The new ComfyUI source,
frontend and server dependencies require a first-session Internet installation;
this is additional interface software, not another model download. It uses a
separate environment so the working native runtime is preserved. Comfy's controller
runs on CPU; native nodes explicitly use T4 0 for MoT and T4 1 for VAE as before.

Songs, tokens, latents and metadata stay in `/kaggle/working/yue2_native/runs`.
Use ComfyUI's audio preview/download and Save Version with outputs to preserve
songs before ending the session. Keep the code cell running while using the canvas.
The old working notebook remains available as a fallback. This interface still
requires live Kaggle validation; CPU/schema checks do not prove GPU generation.
'''),cell('code',code)]
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':True}},
        'nbformat':4,'nbformat_minor':4}


if __name__=='__main__':
    output=ROOT/'YuE2_ComfyUI_Kaggle.ipynb'
    output.write_text(json.dumps(build(),indent=1)+'\n')
    print(output)
