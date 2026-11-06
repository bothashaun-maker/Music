"""Explicit real-model smoke check; uses synthetic speech, not a song-quality benchmark.
Run with the Studio dependencies and FFmpeg's flite filter installed. Downloads
only the pinned acoustic model on first use. Not run during unit-test discovery.
"""
import json,os,subprocess,sys
from pathlib import Path
import tempfile
root=Path(os.environ.get('STUDIO_ACOUSTIC_TEST_DIR',str(Path(tempfile.gettempdir())/'studio-acoustic-smoke')));root.mkdir(parents=True,exist_ok=True)
os.environ['STS_OUTPUT_ROOT']=str(root/'output')
os.environ['HF_HOME']=str(root/'hf')
os.environ['HF_HUB_DISABLE_XET']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from timestamp_studio.acoustic import align_lines
passage=['The music begins.','The audience listens.','The singer speaks.','The chorus returns.',
         "A gentle melody fills the room.","The final words are clear."]
lines=passage+['Listen. Listen.']+passage+['The next verse begins.']
wave=np.zeros(16000*100,np.float32);expected=[];cursor=10.
for i,line in enumerate(lines):
 if i==6:cursor=38.
 if i==7:cursor=60.
 if i==13:cursor=83.
 txt=root/f'line{i}.txt';txt.write_text(line)
 wav=root/f'line{i}.wav'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',f'flite=textfile={txt}:voice=slt','-ar','16000','-ac','1',str(wav)],check=True)
 data,_=sf.read(wav,dtype='float32');begin=round(cursor*16000)
 wave[begin:begin+len(data)]=data;expected.append(cursor);cursor+=len(data)/16000+.5
sf.write(root/'synthetic-repeat.wav',wave,16000,subtype='FLOAT')
print('Real acoustic model test on synthetic speech; not the uploaded song',flush=True)
groups,meta=align_lines(wave,lines,lambda x:print(x,flush=True))
rows=[]
for target,words,text in zip(expected,groups,lines):
 actual=words[0]['start'];rows.append({'expected_clip_start':target,'aligned_first_word':actual,'text':text})
 print(f'{target:.3f} -> {actual:.3f} {text}',flush=True)
 assert abs(actual-target)<1.,(text,target,actual)
again,cached=align_lines(wave,lines)
assert cached['cache_reused'] is True
assert groups==again
(root/'results.json').write_text(json.dumps({'rows':rows,'metadata':meta,'cache_reuse_passed':True},indent=2))
print('PASS: all 14 lines in the correct occurrence; warm rerun reused acoustic frames.',flush=True)
