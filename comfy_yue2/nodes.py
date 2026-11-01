"""Visible ComfyUI stage nodes for the existing native FP16 YuE2 engine."""
import dataclasses
import json
import os
from pathlib import Path
import time
import uuid

_ENGINE = None


def interrupted():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    return False


def engine():
    global _ENGINE
    if _ENGINE is None:
        from native_yue2.process import check_gpu_room
        check_gpu_room(include_pid=os.getpid())
        os.environ['YUE2_HEADLESS'] = '1'
        import native_yue2.engine as loaded
        loaded._cancelled = interrupted
        _ENGINE = loaded
    return _ENGINE


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')


def guard(call):
    try:
        import torch
        with torch.inference_mode():
            return call()
    except BaseException as exc:
        # Comfy cancellation must propagate; errors never become silent audio.
        if 'out of memory' in str(exc).lower() and _ENGINE is not None:
            _ENGINE.pipe.close()
        raise


class NativeEngine:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'load'
    RETURN_TYPES = ('YUE2_NATIVE',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    def load(self):
        return (engine(),)


class Prompt:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'prompt'
    RETURN_TYPES = ('YUE2_PROMPT',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'style': ('STRING', {'multiline': True, 'default':'Melodic rock, expressive vocals, guitar, piano, live drums'}),
                             'lyrics': ('STRING', {'multiline':True, 'default':'[Verse]\nPaste your complete lyrics here\n\n[Outro]\nWrite the ending here'}),
                             'abc': ('STRING', {'multiline':True, 'default':''})}}
    def prompt(self, style, lyrics, abc):
        if not style.strip() or not lyrics.strip():
            raise ValueError('Enter your style and complete song lyrics')
        return ({'style':style.strip(),'lyrics':lyrics.strip(),'abc':abc.strip() or None},)


class Settings:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'settings'
    RETURN_TYPES = ('YUE2_SETTINGS',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'seed': ('INT', {'default':17,'min':0,'max':2147483647,'control_after_generate':True}),
            'maximum_minutes': ([1,2,3,6,7,8], {'default':8}),
            'planning': (['off','melody','full'],),
            'steps': ('INT', {'default':16,'min':1,'max':100}),
            'method': (['euler','midpoint'],),
            'cfg': ('FLOAT', {'default':1.0,'min':0.0,'max':10.0,'step':0.1})},
            'optional':{'abc_token_limit':('INT',{'default':4096,'min':128,'max':8192,'step':128,
                'tooltip':'Optional symbolic-score planning budget; independent of the song-duration ceiling. Unused in direct/off mode.'})}}
    def settings(self, seed, maximum_minutes, planning, steps, method, cfg, abc_token_limit=4096):
        if int(maximum_minutes) not in [1,2,3,6,7,8] or planning not in ['off','melody','full'] or method not in ['euler','midpoint']:
            raise ValueError('Unsupported generation settings')
        if not 128<=int(abc_token_limit)<=8192:
            raise ValueError('ABC planning token limit must be between 128 and 8192')
        return ({'abc_token_limit':int(abc_token_limit),'seed':int(seed),'max_tokens':int(maximum_minutes)*60*25,'planning':planning,
                 'steps':int(steps),'method':method,'cfg':float(cfg)},)


class Plan:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'plan'
    RETURN_TYPES = ('YUE2_PLAN','STRING')
    RETURN_NAMES = ('plan','abc')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'engine':('YUE2_NATIVE',),'prompt':('YUE2_PROMPT',),'settings':('YUE2_SETTINGS',)}}
    def plan(self, engine, prompt, settings):
        def run():
            interrupted()
            engine.release_cover_models()
            engine.pipe.generation_config = dataclasses.replace(engine.pipe.generation_config,
                ode_steps=settings['steps'], ode_method=settings['method'])
            engine.AR_OPTIONS['fuse_projections'] = False
            mode = 'melody' if prompt['abc'] and settings['planning']=='off' else settings['planning']
            directory = Path(os.environ['YUE2_OUTPUT_ROOT']) / (time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:10])
            directory.mkdir(parents=True)
            write(directory/'settings.json',dict(prompt=prompt,settings=settings,model_weights=engine.pipe.weights,
                                                 engine_sha256=engine.pipe.runtime_sha256))
            began=time.perf_counter()
            plan=engine.pipe.plan(style=prompt['style'],lyrics=prompt['lyrics'],cot=mode,abc=prompt['abc'],
                seed=settings['seed'],abc_sampling={'max_tokens':settings.get('abc_token_limit',4096)},
                cfg_scale=settings['cfg'],cancelled=interrupted)
            plan.save(directory)
            if plan.truncated:
                limit=settings.get('abc_token_limit',4096)
                write(directory/'status.json',{'status':'abc_planning_cutoff','abc_token_limit':limit,
                    'abc_tokens':len(plan.abc_ids),'score_saved':True})
                raise ValueError(f'ABC score reached its {limit}-token planning limit. Increase abc_token_limit on Song Settings, '
                    f'or use planning=off with an empty ABC field for direct generation. '
                    f'Partial score saved at {directory}; audio generation was not started.')
            if len(plan.prefix)+settings['max_tokens']>engine.yproto.CONTEXT:
                raise ValueError('Lyrics/score and song ceiling exceed the model context')
            state={'engine':engine,'plan':plan,'settings':dict(settings),'directory':directory,
                   'timing':{'planning':time.perf_counter()-began}}
            return state, plan.abc or ''
        return guard(run)


class Semantic:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'sample'
    RETURN_TYPES = ('YUE2_SEMANTIC',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'plan':('YUE2_PLAN',)}}
    def sample(self, plan):
        def run():
            import numpy as np
            import comfy.utils
            state=dict(plan);state['timing']=dict(plan['timing'])
            e=state['engine'];budget=state['settings']['max_tokens']
            progress=comfy.utils.ProgressBar(budget)
            tokens=[];last=[0.0]
            def report(phase, token):
                interrupted()
                if phase!='semantic':return
                if e.yproto.CODEC_OFFSET<=token<e.yproto.CODEC_OFFSET+e.yproto.CODEC_SIZE:
                    tokens.append(int(token)-e.yproto.CODEC_OFFSET)
                    progress.update_absolute(len(tokens),budget)
                if time.monotonic()-last[0]>=5:
                    np.save(state['directory']/'semantic.partial.npy',np.asarray(tokens,dtype=np.int32),allow_pickle=False)
                    write(state['directory']/'semantic-progress.json',{'tokens':len(tokens),'budget':budget,'audio_seconds':len(tokens)/25})
                    last[0]=time.monotonic()
            began=time.perf_counter()
            state['semantic']=e.pipe.generate_semantic(state['plan'],{'max_tokens':budget},cancelled=interrupted,on_token=report)
            np.save(state['directory']/'semantic.npy',np.asarray(state['semantic'].tokens,dtype=np.int32),allow_pickle=False)
            write(state['directory']/'semantic.json',{'truncated':state['semantic'].truncated,'timing':state['semantic'].timing,'token_count':len(state['semantic'].tokens)})
            state['timing']['semantic']=time.perf_counter()-began
            return (state,)
        return guard(run)


class Acoustic:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'synthesize'
    RETURN_TYPES = ('YUE2_LATENTS',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'semantic':('YUE2_SEMANTIC',)}}
    def synthesize(self, semantic):
        def run():
            import numpy as np
            from kaggle_yue2.recovery import sha256_file
            state=dict(semantic);state['timing']=dict(semantic['timing']);e=state['engine']
            # Each cached branch restores its own solver settings before synthesis.
            e.pipe.generation_config=dataclasses.replace(e.pipe.generation_config,
                ode_steps=state['settings']['steps'],ode_method=state['settings']['method'])
            began=time.perf_counter()
            state['latents']=e.pipe.synthesize(state['semantic'],cancelled=interrupted)
            np.save(state['directory']/'latent.npy',state['latents'],allow_pickle=False)
            write(state['directory']/'latent_manifest.json',{'sha256':sha256_file(state['directory']/'latent.npy'),
                'shape':list(state['latents'].shape),'model_weights':e.pipe.weights})
            state['timing']['acoustic']=time.perf_counter()-began
            state['acoustic_ranges']=list(e.pipe.last_acoustic_ranges)
            return (state,)
        return guard(run)


class Decode:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'decode'
    RETURN_TYPES = ('AUDIO','YUE2_RESULT')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'latents':('YUE2_LATENTS',)}}
    def decode(self, latents):
        def run():
            import torch,numpy as np
            interrupted()
            state=dict(latents);state['timing']=dict(latents['timing'])
            began=time.perf_counter()
            audio=state['engine'].pipe.decode(state['latents'])
            if audio.ndim!=2 or audio.shape[1]!=2 or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError('Decoder returned invalid/nonfinite/silent stereo audio')
            if not np.isclose(len(audio)/48000,len(state['semantic'].tokens)/25,atol=.1):
                raise ValueError('Decoded audio length differs from generated tokens')
            state['timing']['vae']=time.perf_counter()-began
            return {'waveform':torch.from_numpy(audio.T.copy()).unsqueeze(0),'sample_rate':48000},state
        return guard(run)


class SaveSong:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'save'
    RETURN_TYPES = ('STRING',)
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'audio':('AUDIO',),'result':('YUE2_RESULT',)}}
    def save(self, audio, result):
        import soundfile as sf,numpy as np,folder_paths,shutil
        waveform=audio['waveform'].squeeze(0).cpu().numpy().T
        directory=result['directory']
        for extension in ['wav','flac']:
            path=directory/f'audio.{extension}'
            sf.write(path,waveform,48000,subtype='PCM_24')
            info=sf.info(path)
            if info.channels!=2 or info.samplerate!=48000 or not np.isclose(info.duration,len(waveform)/48000,atol=.001):
                raise ValueError('Saved audio header validation failed')
        truncated=result['semantic'].truncated
        status={'status':'audio_saved_with_cutoff' if truncated else 'audio_saved_natural_end',
            'ending_reason':'token_limit' if truncated else 'model_end_token',
            'actual_seconds':len(waveform)/48000,'max_seconds':result['settings']['max_tokens']/25,
            'semantic_truncated':truncated,'timing':result['timing'],
            'ar_timing':result['semantic'].timing,'acoustic_ranges':result['acoustic_ranges'],
            'boundary_seconds':[b/25 for _,b in result['acoustic_ranges'][:-1]]}
        write(directory/'status.json',status)
        if shutil.disk_usage(directory).free>sum(p.stat().st_size for p in directory.iterdir() if p.is_file())+64*2**20:
            shutil.make_archive(str(directory),'zip',root_dir=directory)
        subfolder=str(directory.relative_to(Path(folder_paths.get_output_directory()).resolve()))
        return {'ui':{'audio':[{'filename':'audio.wav','subfolder':subfolder,'type':'output'}],
                      'text':[json.dumps(status,indent=2)]},'result':(str(directory),)}


class CoverABC:
    CATEGORY = 'audio/YuE2'
    FUNCTION = 'analyze'
    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('abc','analysis')
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'engine':('YUE2_NATIVE',),'audio':('AUDIO',)}}
    def analyze(self, engine, audio):
        import soundfile as sf,tempfile
        interrupted()
        # ComfyUI's native LoadAudio supplies uncompressed samples, not a lossy transcode.
        root=Path(os.environ['YUE2_COVER_ROOT']);root.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root,suffix='.wav',delete=False) as stream:
            path=Path(stream.name)
        try:
            sf.write(path,audio['waveform'].squeeze(0).cpu().numpy().T,audio['sample_rate'],subtype='FLOAT')
            abc,_,_,status,archive=engine.analyze_cover_source(str(path),'Auto-detect','',False,progress=lambda *a,**k:interrupted())
            return {'ui':{'text':[abc,status]},'result':(abc,archive or status)}
        finally:
            path.unlink(missing_ok=True)


NODE_CLASS_MAPPINGS={'YuE2NativeEngine':NativeEngine,'YuE2Prompt':Prompt,'YuE2Settings':Settings,
    'YuE2NativePlan':Plan,'YuE2NativeSemantic':Semantic,'YuE2NativeAcoustic':Acoustic,
    'YuE2NativeDecode':Decode,'YuE2SaveSong':SaveSong,'YuE2CoverABC':CoverABC}
NODE_DISPLAY_NAME_MAPPINGS={key:name for key,name in zip(NODE_CLASS_MAPPINGS,
    ['YuE2 Native FP16 Engine','Song Style / Lyrics / ABC','Song Settings','Plan / Prepare Prompt',
     'Generate Music Tokens','Acoustic Synthesis','Decode Stereo Audio','Save Song / Listen','Extract Melody ABC'])}

class ShowABC:
    CATEGORY='audio/YuE2'
    FUNCTION='show'
    RETURN_TYPES=('STRING',)
    OUTPUT_NODE=True
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'text':('STRING',{'forceInput':True})}}
    def show(self,text):return {'ui':{'text':[text]},'result':(text,)}

NODE_CLASS_MAPPINGS['YuE2ShowABC']=ShowABC
NODE_DISPLAY_NAME_MAPPINGS['YuE2ShowABC']='View ABC / Text'
