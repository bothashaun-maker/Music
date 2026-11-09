# JJP Music Launch Desk

[Full production-system PDF](JJP_Music_Production_System.pdf) · [Editable full guide](MUSIC_PRODUCTION_SYSTEM.md)

Choose the job you need. Already have a finished song? Skip generation.

| Your job | Open this | Kaggle setting |
| --- | --- | --- |
| Create a song | [Music Creation](https://github.com/bothashaun-maker/Music/blob/main/YuE2_Music_Revision15.ipynb) → Full Song | T4 ×2 · Internet ON · existing YuE2 assets |
| Cover / restyle | [Same notebook](https://github.com/bothashaun-maker/Music/blob/main/YuE2_Music_Revision15.ipynb) → Cover / Restyle | T4 ×2 · Internet ON · existing YuE2 assets |
| Separate stems | [Stem Studio](https://github.com/bothashaun-maker/kaggle_roformer_stem_splitter/blob/main/Stem_Splitter_ComfyUI_Kaggle.ipynb) | T4 ×2 · Internet ON · saved stem Dataset |
| Time lyrics / subtitles | [Timestamp Studio](https://github.com/bothashaun-maker/Music/blob/main/Song-Timestamp-Studio-Canvas.ipynb) | T4 ×2 recommended · Internet ON |
| Repair markers locally | [REAPER shift action](https://github.com/bothashaun-maker/Music/blob/main/reaper/README.md) | REAPER · no Kaggle run |
| Artwork | [Image Studio](https://github.com/bothashaun-maker/Movies/blob/main/Image_Studio_Kaggle_v3.ipynb) | Its own model packs/settings |
| Optional movie visuals | [Movie workflow](https://github.com/bothashaun-maker/Movies/blob/main/Current_Working_Workflow.ipynb) | Its own model packs/settings |

## Every session

1. Attach the saved inputs. Select the appropriate GPU setting.
2. Run the notebook's single startup code cell. Wait for its ready message and links.
3. Open the new Cloudflare canvas link. Keep the startup cell running.
4. Select the source and settings. Press Run once; listen and review.
5. Download the result ZIP or save a notebook version with output files before closing.

`FutureWarning` is not an error. An old Cloudflare link is not a permanent studio address.
Stop the previous startup cell before switching music tools in the same runtime.

## Model storage — first time only

- [Music asset builder](https://github.com/bothashaun-maker/Music/blob/main/yue2-build-kaggle-assets.ipynb): reuse your existing `yue2-music-assets`; no rebuild for a new song.
- [Stem model Dataset builder](https://github.com/bothashaun-maker/kaggle_roformer_stem_splitter/blob/main/Stem_Splitter_Model_Packs.ipynb): **Accelerator None**, Internet ON. Named Dataset: `janpierrrewiet/stem-studio-assets-v1-f6815b582a` (about 2.1 GiB).
- Timestamp models/frame caches can be reused from saved Studio outputs. There is no equivalent automatic named-Dataset builder in the current Timestamp notebook.
- [Image model packs](https://github.com/bothashaun-maker/Movies/blob/main/Image_Studio_Model_Packs.ipynb) and [upscaler model packs](https://github.com/bothashaun-maker/Movies/blob/main/Movie_Upscaler_Model_Packs.ipynb) are separate from music assets.

## Keep the order clear

New song or existing recording → optional stems/remix → **approved final audio** → timestamps → reviewed subtitles/markers → visuals/release pack.

Keep original WAVs and manual REAPER corrections. The model Dataset does not preserve generated songs.

[Song brief template](docs/music-production-system/templates/song_brief.md) · [Cue review sheet](docs/music-production-system/templates/cue_review.csv) · [Production log](docs/music-production-system/templates/production_log.csv)
