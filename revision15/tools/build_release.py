"""Rebuild the repaired uploaded branch and its standalone release ZIP."""
from pathlib import Path
import importlib.util
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def build():
    fast=module('r15_fast_builder','build_fast_notebook.py')
    studio=module('r15_studio_builder','build_studio_notebook.py')
    outputs={'yue2-3b-frontier-full-song-music-generation.ipynb':fast.build(),
             'yue2-build-kaggle-assets.ipynb':fast.build_asset_notebook(),
             'Song-Timestamp-Studio-Canvas.ipynb':studio.build()}
    for name,data in outputs.items():
        (ROOT/name).write_text(json.dumps(data,indent=1,ensure_ascii=False)+'\n')
    (DEST/'YuE2_Music_Revision15.ipynb').write_bytes((ROOT/'yue2-3b-frontier-full-song-music-generation.ipynb').read_bytes())
    (DEST/'Song-Timestamp-Studio-Canvas.ipynb').write_bytes((ROOT/'Song-Timestamp-Studio-Canvas.ipynb').read_bytes())
    archive=DEST/'yue2-music-revision-15.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ROOT.rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc'
                    and path.name != 'yue2-3b-frontier-full-song-music-generation.ipynb'):
                bundle.write(path,path.relative_to(ROOT))
    print(archive)


if __name__=='__main__':build()
