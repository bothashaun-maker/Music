# Music Studio Field Guide

*Edition II · 11 November 2026*

[Open the designed PDF](../../Music-Studio-Guide.pdf)

## 01 · Your studio desk

START HERE · ONE DATASET · ONE STARTUP CELL

Create a song, work on its score, or bring in a recording you already have. This guide follows the actual Linux browser studio supplied with Music-Studio.ipynb.

| What you need | Where to go |
| --- | --- |
| Prepare the saved tools/models | Music-Studio-Dataset.ipynb · CPU · first time only |
| Open your studio | Music-Studio.ipynb · T4 ×2 · every session |
| Create or re-perform a song | Create; Library → track menu → Re-render |
| Work on a recording | Upload large WAV; then Library / Studio tools |
| Save your work | Library downloads; Kaggle output files |

**Separate from your working notebooks**

This browser studio uses the official YuE2 Studio 3.8.5 with Q8 GGUF weights. Your existing native FP16 creator and separate Timestamp notebook remain available. Timestamp recognition is excluded from Music Studio.

- [Open Music-Studio.ipynb](https://github.com/bothashaun-maker/Music/blob/main/Music-Studio.ipynb)
- [Open automatic Dataset builder](https://github.com/bothashaun-maker/Music/blob/main/Music-Studio-Dataset.ipynb)
- [Read setup and current validation](https://github.com/bothashaun-maker/Music/blob/main/docs/music-studio/README.md)

## 02 · Prepare once. Save privately.

ACCELERATOR NONE · INTERNET ON

1. Import Music-Studio-Dataset.ipynb into Kaggle. Select Accelerator None and enable Internet.
2. In Kaggle Secrets, enable KAGGLE_API_TOKEN for this notebook. Keep the value in Secrets. Leave DATASET_OWNER blank unless username detection needs your Kaggle username.
3. Run its one code cell. Authentication is checked before compilation. It builds the pinned x86-64/T4 runtime and automatically creates one private Music Studio Dataset.
4. Wait for PRIVATE MUSIC STUDIO DATASET SUBMITTED, open the printed Dataset link, and allow Kaggle processing to finish.
5. In the GPU notebook choose Add Input → Your Datasets → the new Music Studio Dataset. Attach this one Dataset.

**Disk and time**

The model set is about 26.24 GB plus the runtime. Uploads stream one verified file at a time; the largest is about 5.19 GB. Runtime compilation needs at least 7 GiB free initially. Actual space is checked. First-time compilation/downloads take time; no timing guarantee is made.

The builder never overwrites an existing Dataset. An interrupted final request may have reached Kaggle: inspect its page before retrying. A submission receipt is not proof that Kaggle has finished processing.

## 03 · Open a session

GPU T4 ×2 · INTERNET ON · SAVED DATASET ATTACHED

1. Import a fresh Music-Studio.ipynb. Attach the saved Music Studio Dataset, select GPU T4 ×2, and enable Internet.
2. Stop any other music/movie/image startup cell in this Kaggle runtime. Run Music Studio’s single startup cell.
3. Wait for asset verification and the CUDA engine ready check. Open the printed OPEN MUSIC STUDIO link, including its private access key.
4. Keep the cell running while working in your browser. Choose your job and press Create or its tool button yourself.
5. Before finishing, preserve your library and media using Kaggle output files. Stop the startup cell to close its owned studio and tunnel processes.

**The link is temporary**

Each startup creates a fresh Cloudflare URL/key. Keep it private and use the latest link. An old bookmark is not a permanent studio address. A tunnel problem does not erase songs already saved on Kaggle.

GPU 0 runs generation and LoRA training, one heavy job at a time. The second T4 has separate memory. Stems, MIDI, the writing assistant and music-style description use CPU. No diagnostic song starts during setup.

Use the current Kaggle GPU image. This CUDA 13 runtime needs NVIDIA driver 580 or newer; startup checks it before loading the model.

## 04 · Find the right panel

CREATE · LIBRARY · STUDIO TOOLS · LORA · SETTINGS

| Panel | Use it for |
| --- | --- |
| Create | Title, style, lyrics, score, quality, seeds and generation |
| Library | Listen, rename, like, organize, inspect settings and reuse a track |
| Upload large WAV | Verified resumable WAV/MP3 imports without conversion |
| Studio tools | HT-Demucs / RoFormer stems, audio-to-MIDI and audio processing |
| LoRA | Browse/import adapters, prepare training material and train on the T4 |
| Settings | Music model, engine device, writer, artwork and network access |

The player follows you between panels. Select a track before opening a track-specific tool. Track menus contain actions that do not fit on the main card. A generated result, a stem and a processed version can be separate library items.

The official 3.8.5 release includes AB2, chord variety, re-render extension and score continuation. The notebook supplies native Linux tools and a CUDA trainer with the upstream Turing fix. Windows plugins and desktop editor windows require a local desktop. Recognition is handled in your separate notebook.

## 05 · Make your first song

A CLEAR STYLE · LYRICS WITH ROOM TO SING

1. Open Create. Give the song a useful title; it becomes a library label.
2. Write the style in English: language, genre/era, vocal character, instruments, mood, production and tempo. Describe the sound rather than depending on an artist’s name.
3. Write the sung words with section tags on separate lines: [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro]. Leave a blank line between sections.
4. Choose a score mode. Full plans melody and chords; Melody uses a melodic plan; Off generates without a symbolic score. Leave ABC empty to let the selected mode compose it.
5. For the first manual trial choose a modest duration ceiling and one variation. Press Create once. Watch its progress in Library and listen through the ending.

**An example style**

English, melodic synth-pop, expressive female lead, analog bass, clean drums, warm pads, hopeful mood, clear balanced mix, 112 BPM.

Duration is a ceiling, not a promise. Enough lyric material and clear sections help; a longer limit alone does not produce a longer complete song. Keep the original request and the version you prefer.

## 06 · Shape words and vocals

SECTION TAGS · SYLLABLES · INSTRUMENTALS

Keep nearby lyric lines similar in syllable count. Sparse, singable wording gives the model more room than a paragraph of prose. Avoid putting instructions such as “male singer enters” in the lyrics: the model can sing those words.

| Goal | Practical starting point |
| --- | --- |
| Clear delivery | Short lines; avoid packing many words into a fast section |
| A recurring hook | Write the chorus in full at each intended return |
| A duet | Describe both singers in the style; structure the exchange by sections |
| An instrumental | Choose the instrumental option where offered; use empty section tags if necessary |
| Another language | Name the sung language first in the style; keep section tags in English |

**Listen before committing**

An instrumental request can still produce vocal sounds; duets and voice/register prompts are not exact singer assignments. Check the whole result. Do not treat the preset name as a quality guarantee.

Use the assistant to draft or revise words, then review them yourself. For a cover, the original syllable pattern is particularly important. Changes in syllables can shift the melody’s phrasing.

## 07 · Choose settings with purpose

ONE CHANGE AT A TIME · SAVE THE REQUEST

| Control | What it changes |
| --- | --- |
| Duration ceiling | Maximum generation length; the model can end earlier |
| Composition / LM seed | Score and semantic choices; pin when comparing composition |
| Sound / noise seed | Acoustic realization; change for another performance |
| Steps | Acoustic solver work; more steps take longer and may change the sound |
| Guidance / CFG | Prompt-conditioning strength; begin with the offered default |
| Temperature / top-p / top-k | Sampling variety; larger changes can reduce consistency |
| Batch / variations | More candidates consume more memory and compute |
| Solver: Midpoint / AB2 | Choose the acoustic sampling method; compare at the same steps and seed |
| Chord variety / sections | Encourage varied harmonic choices during score planning |

Start with one song and one variation. Keep model, prompt and seeds fixed when comparing a setting. Export the request/settings alongside the keeper; a title alone does not identify how a version was made.

**T4 baseline**

The launcher chooses Q8, CUDA, one-song batching, staged model loading and a bounded context. It fails instead of silently substituting CPU generation. Do not choose a larger model simply because two T4s appear in the session.

Advanced settings include Midpoint or AB2 acoustic solvers and chord-variety controls. Start with the offered defaults. AB2 changes acoustic sampling; chord variety changes composition. Compare one setting at a time with both seeds pinned.

## 08 · Read and revise the score

ABC IS THE COMPOSITION · AUDIO IS THE PERFORMANCE

1. Use Full or Melody planning when you want an editable score. Inspect the returned score in the track’s details.
2. Use the score action to put it back into Create. Save a copy of the original ABC before editing.
3. Open the score/MIDI editor, or make a small ABC edit: one note, one chord, tempo or key. Keep the two voices and the measure structure valid.
4. Apply the editor’s changes back to the score. Check the score mode and lyrics, then create another performance.
5. Compare the result with the original. Preserve both versions until the revision is accepted.

ABC notation includes the meter, note-length unit, tempo, key and voice declarations. YuE2 uses its own score conventions; generic notation imported from elsewhere may need adjustment. A malformed score can fail or produce a poor result.

**Register changes**

Moving the vocal line by an octave changes the notes, not just a voice label. Review the range and adjust the style description too. Neither a score nor a seed promises an exact singer identity.

## 09 · Compose and edit MIDI

DRAW · PLAY · IMPORT · APPLY

1. Open the MIDI editor from the relevant score or MIDI track. Choose the track/instrument before editing notes.
2. Draw notes on the piano roll or record from an available keyboard input. Set tempo and grid so the phrase is easy to inspect.
3. For an existing .mid file, use MIDI import. Select the intended vocal and instrumental tracks and review octave choices.
4. Play back the editor to check the composition. MIDI playback is a synthesizer preview, not YuE2’s final voice or production.
5. Save MIDI, or Apply the composition to an ABC score and send it through the song generator.

**Browser permissions**

Open the HTTPS Cloudflare link. AudioWorklet/MIDI functions can depend on a secure browser context and device permission. Click Play once to allow browser audio. A disconnected controller does not stop note drawing or file import.

Keep the original .mid and the exported ABC. MIDI tempo, instrument choices and lyrics do not necessarily transfer perfectly between tools; inspect the imported result before a long render.

## 10 · Re-render a keeper

KEEP THE COMPOSITION · TRY A DIFFERENT REALIZATION

1. In Library, select the source song and open its track menu → Re-render.
2. To replay the same setup, retain the stored codes, seed and settings. To explore a different performance, change the sound seed or offered acoustic settings.
3. For a musical change, reuse and edit the ABC score instead. Re-rendering stored semantic codes does not invent new lyrics or rewrite the composition.
4. Give the new result a distinguishing title and compare it at matched listening level.
5. Keep the source track, request and codes. Delete a version only after reviewing the exported keeper.

**What exact replay means here**

The stored request/codes are reused. Hardware, software, quantization and numerical kernels can affect byte-level reproducibility. Do not equate a changed audio file with a new composition, or promise identical bytes across different engines.

Re-render also offers +15, +30, +60, +90 or +120 seconds when the saved source has enough context room. A duration ceiling is not a promise that the model will use every second. Keep both the source and extended version.

## 11 · Bring in your recording

ORIGINAL WAV/MP3 · RESUMABLE · NO CONVERSION

1. Choose Upload large WAV in the sidebar. The upload page opens in the same authenticated studio.
2. Choose your original WAV or MP3, up to 2 GiB, then press Upload / Resume. The notebook cell must remain running.
3. Wait for Ready in Library. Each 4 MiB piece has a checksum. If the connection fails, choose the same file and resume; previously saved pieces are checked against it.
4. Return to Studio and open Library. Select the imported recording before using covers, stems, MIDI or processing.
5. Inspect the source filename and checksum when you need a provenance check. The imported audio has not been transcoded.

**If a transfer pauses**

Pause stops after the current network operation. A completed import is not repeated when you resume the same upload. An incomplete transfer is retained in output storage; it is not a ready library song.

The studio’s original import control remains available for small files. Use the resumable route for large originals over Cloudflare. The pinned library import supports WAV and MP3; do not rename another format to make it look like one.

## 12 · Cover or restyle a song

SOURCE MELODY → REVIEWED ABC → NEW PERFORMANCE

1. Upload your recording and select it in Library. Choose its cover/transcribe-score action.
2. Let SheetSage2 estimate the melody. Inspect the score and correct obvious errors before generating.
3. Use Melody mode for a melody-only score. Supply the original lyrics, or new lines with a matching syllable pattern and compatible stresses.
4. Describe the target arrangement in the style. Check the voice range, tempo and any transposition.
5. Create a new song from that score and listen against the source. Keep the original recording and transcription artifacts.

**A cover is a new performance**

The tool estimates a symbolic melody and re-synthesizes it. Timing, voice, harmony and arrangement can change; extraction is not perfect source separation or an exact audio-to-audio copy.

Use recordings you have the necessary rights to use. SheetSage2 and the decoder companion have separate non-commercial terms; YuE2’s individual-creator exception does not automatically extend to them.

## 13 · Separate six stems

DRUMS · BASS · OTHER · VOCALS · GUITAR · PIANO

1. Select the source song in Library, open Studio tools and choose HT-Demucs for six stems or Mel-Band RoFormer for vocals/instrumental.
2. Choose the stems you want to keep. The model predicts all six even when you export fewer.
3. Begin with the offered overlap. Run once and wait for completion. HT-Demucs uses the bundled audio.cpp CPU server; RoFormer uses its own CPU tool.
4. Audition isolated parts and the recombined mix. Check difficult vocal tails, guitar/piano overlap and transitions.
5. Save the useful stems and their source/settings. Use the isolated vocal for subsequent analysis only after listening.

**Quality limits**

Separation can leave bleed, remove transients or create artifacts. More overlap costs more time and is not a guarantee of better sound. Keep the original full mix. This integrated tool exports 44.1 kHz stereo PCM16 WAV stems. Use the separate Stem Studio when its floating-point export is required.

The integrated RoFormer option is additional to your dedicated Stem Studio. It uses voc_fv6 Q8_0. Keep using the separate notebook when you need its other models or floating-point export.

## 14 · Turn audio into MIDI

MUSCRIPTOR MEDIUM · NOTES ARE AN ESTIMATE

1. Select a song, isolated stem or processed version. Open Audio to MIDI.
2. Choose the bundled Medium model. Other model sizes are not included in the saved Dataset.
3. Start transcription and inspect the live notes/progress. The native CPU helper uses the saved model.
4. Open the result in the piano-roll editor. Audition it against the original, mute tracks and correct note starts, lengths and instruments.
5. Save .mid for your DAW, or use a reviewed part as the song’s score input.

**A transcription is not the master**

Polyphonic mixtures can produce extra/missing notes and wrong instrument assignments. A vocal stem can be easier to inspect than a dense full mix. Check tempo and alignment when importing into REAPER.

MuScriptor’s weights are CC BY-NC 4.0. The MIDI editor itself does not require that transcription model when you draw notes or import your own MIDI.

## 15 · Extend a song. Continue an idea.

RE-RENDER EXTENSION · OPENING SCORE · YOUR OWN MELODY

1. For a longer performance, open a generated track’s menu → Re-render. Choose an available extension from +15 to +120 seconds, set the acoustic seed and render a new version.
2. For composition from an idea, import a short recording or hum. Use SheetSage2 to transcribe its melody, or write/import the opening in the MIDI/ABC editor.
3. Send the opening score into Create. Enable Continue score, choose Full or Melody planning, and describe the desired style and song structure.
4. Compose the continuation, inspect the resulting score, then render the performance. Compare where the original idea returns as a hook.
5. For a chord bed, use the score editor’s accompaniment/chord tools and suitable instruments. Keep the source MIDI/ABC alongside the chosen render.

**Two different operations**

Re-render extension works from a stored generated track. Continue score develops an opening melody into composition. Audio-to-score and audio-to-MIDI are musical tools; speech/timestamp recognition is not installed here.

Use short, clear reference material first. Inspect notes, key, meter and section boundaries before spending time on a long render.

## 16 · Process and master audio

COMPARE BEFORE/AFTER · KEEP THE ORIGINAL

1. Select the track or stem and open the audio processing tool.
2. Choose an operation: naturalizer/spectral processing, reference mastering, or a native Linux VST3 chain. Adjust one stage at a time.
3. Audition before and after on several sections. Match listening level so louder does not automatically seem better.
4. Keep the processed take as a new version only when it improves the intended result.
5. Export the chosen version, then check the file in your DAW. Retain the untouched source and settings.

**Plugins on Kaggle**

Attach a Dataset containing native Linux .vst3 bundles with Contents/x86_64-linux/*.so, then restart Studio and scan plugins. The launcher links these bundles into the host’s plugin folder. Kaggle cannot run Windows VST DLLs or show desktop plugin editors. Use default/saved compatible state for offline processing; use local REAPER for desktop editing.

Noise reduction and vocal processing can remove desired texture or add artifacts. Reference mastering matches aspects of a sound; it does not establish rights to the reference or guarantee a release-ready result.

## 17 · Make a lyric video

FINISH AUDIO FIRST · THEN RENDER

1. Select the approved audio version. If you want lyric overlays, export its LRC from your separate Timestamp notebook.
2. Open the video/clip tool. In Text, import that LRC file. Choose landscape 16:9, square 1:1 or vertical 9:16 as offered.
3. Choose a background/cover you may use. Add title, lyric layers, visualizer and supported effects; keep text readable.
4. Preview a difficult section: a repeated chorus, a long line and the ending. Then render the clip.
5. Download/save the video and play the exported file from beginning to end. Check audio sync, text clipping, transitions and final duration.

**Where the work happens**

This release’s video tools use browser media/WebAssembly APIs. Keep the browser tab open and allow audio. Older browsers or restrictive privacy settings may block audio/video capabilities.

Use CapCut or REAPER/video editing tools when the browser render cannot meet your delivery needs. A layout preset does not certify a platform’s current delivery specifications.

## 18 · Write with the assistant

LOCAL GEMMA · OR YOUR EXPLICIT PROVIDER

1. Open assistant settings. The saved bundle includes Gemma E4B Q4 and a CPU llama.cpp runtime.
2. Describe what you want the assistant to write or revise: the full song, lyrics, style or an existing score.
3. Review the returned words and musical instructions. Keep sung text separate from production guidance.
4. Apply only the parts you accept. Check that the score still fits the lyrics before rendering.
5. For an external provider, configure it yourself in Settings and review its own charges/privacy terms before use.

**Local does not mean on your laptop**

The local assistant runs inside Kaggle. CPU writing can take time and uses host RAM. Avoid running several heavy analysis tasks together. A provider error should be diagnosed from its log, not treated as proof that the song engine failed.

Gemma’s custom terms apply. The notebook contains no API key and does not automatically buy credits or submit a request to a paid provider.

## 19 · Use a style adapter

APPLICATION IS INCLUDED · T4 TRAINING IS NOT

1. Open LoRA/Adapters. Browse the catalog or import a compatible adapter you may use.
2. Check its model family and whether it targets composition or sound. A similarly named adapter for another engine is not compatible.
3. Select it in Create and begin with its documented strength/trigger word. Change one adapter at a time.
4. Compare with the same prompt and seeds without that adapter. Reject unwanted voice, pronunciation or mix changes.
5. Save accepted adapters and their source/license notes with your output library.

**Training on T4**

Studio 3.8.5 and the pinned HOT-Step trainer include the Turing F32 fallback for INT8 ConvRot training. Training weights and a CUDA SM75 trainer are included. No lyric aligner is installed. The source supports T4; actual training, VRAM use and speed still require a live Kaggle run.

Use the Training page for your own recordings and manually supplied lyrics/styles. The fast, balanced and thorough recipes train from mastered mixes without lyric timestamps. The next page covers preparation and checkpoints.

## 20 · Train a music adapter

YOUR RECORDINGS · MANUAL LYRICS · CHECKPOINTS

1. Open Training and create a training dataset. Add recordings from the library or upload them; use material you may train on.
2. Enter correct lyrics and section tags for each vocal song, or mark a song instrumental. Enter its style. Describe style can suggest musical metadata; it does not recognize lyrics.
3. Check that every item is ready. Choose Fast for a first trial, or Balanced / Thorough for more updates. Leave lyric timing off; the recognition and forced-alignment models are absent.
4. Start training with generation idle. Studio releases the generation engine while the CUDA trainer uses the selected T4. Keep the notebook running and inspect progress/loss.
5. Review a saved checkpoint, install/apply the adapter and compare a short test with fixed seeds. Save training datasets, runs and accepted adapters through Kaggle output files.

**Persistent training work**

Datasets and runs are linked into /kaggle/working/music_studio/training. Training models stay in the read-only model Dataset. Preserve output files before ending Kaggle; the model Dataset does not automatically save your new checkpoints.

This notebook does not pool both T4 GPUs. Use one heavy GPU job at a time. If training runs out of memory, retain logs/checkpoints and review the recipe; do not start a second engine.

## 21 · Describe audio by ear

MOSS-MUSIC · TEMPO ANALYSIS · EDIT THE RESULT

1. Select a recording and use its Describe style/audio action when available.
2. The saved pack includes the MOSS audio tower, language model and Beat This! tempo model; its native captioner is built with the other runtime tools.
3. Wait for the native CPU analysis. A dense track and a large model can take time; keep other heavy jobs idle.
4. Review the proposed genre, instruments, mood and BPM. Treat them as observations to correct, not authoritative metadata.
5. Use the useful description in your next style prompt or preparation record.

**Memory**

The CPU captioner uses the MOSS F16 audio tower and Q4_K_M language model, about 6.91 GB on disk together. Runtime memory is separate. The CPU profile analyzes up to two minutes of a track and avoids competing for generation VRAM.

Music-style description and training are separate operations. Correct the proposed genre, instruments and BPM yourself. Lyrics are entered manually; external writing providers are optional.

## 22 · Artwork, player and library

ORGANIZE THE WORK · AUDITION THE KEEPER

| Tool | Practical use |
| --- | --- |
| Library / playlists / likes | Name candidates, group a project, mark keepers and inspect source/settings |
| Cover art / pattern | Choose a deterministic pattern, upload your image, or use supported photo search |
| Player / equalizer | Audition the selected song; monitor changes at an appropriate volume |
| Visualizer / Winamp mode | Playback display and browser-player mode; this is not mastering |
| Metadata / request export | Keep style, lyrics, score, seeds and version history with the audio |

Automatic cover generation starts off. Photo searches and optional generation may use external services. Review source credits and terms; an attractive image does not establish rights. A pattern or your own image avoids relying on a stock result.

**Browser versus desktop**

The Linux fork uses browser playback/Webamp. Desktop window shaping, Explorer integration and Windows-only plugins are not part of the Kaggle experience. Browser sound may require an initial click.

## 23 · Drive it with an agent

OPTIONAL MCP AUTOMATION · SAME PRIVATE STUDIO

1. Open the studio’s MCP/agent settings and inspect the available connection information. The service exposes /mcp.
2. Connect your own MCP-capable agent to the current HTTPS studio address plus /mcp, using the studio access key as a Bearer authorization header.
3. Ask the agent to inspect studio status and available tools before changing anything. Tool counts can vary by release.
4. Give a bounded instruction, such as drafting lyrics or preparing a song for review. Review prompts and settings before generation.
5. Stop or disconnect the agent when finished. A new notebook startup rotates the URL/key; update the connection then.

**Keep the key private**

The public tunnel uses the same authentication guard for the UI and API. Do not put its key in GitHub, a shared prompt or a public notebook output. Your connected chat agent is not automatically connected to this new MCP endpoint.

MCP exposes real studio actions, not just a read-only help page. An agent can create, modify or remove work through supported tools. Use your own explicit instructions and preserve the original versions.

## 24 · Save and restore your work

MODEL DATASET ≠ SONG LIBRARY

1. Keep the approved audio, request/settings, score/codes, MIDI, stems, lyrics and any final video together.
2. Stop the startup cell before taking a final library snapshot so SQLite can close cleanly.
3. Save a Kaggle version with output files, or download the library/media you want to retain. A draft or notebook containing logs is not an audio backup.
4. For a later session, attach that saved output as another Kaggle input alongside the one model Dataset. Startup can restore a single saved Music Studio library into empty working output.
5. If there are multiple saved libraries, set RESTORE_LIBRARY_PATH to the desired library folder. Existing working libraries are preserved; restore does not overwrite them.

**What is durable**

/kaggle/working/music_studio contains the library, media, settings, adapters, uploads and logs. Model links and runtime state live in temporary storage. Saving the model Dataset alone does not save a newly generated song.

Save a new private project version after substantial changes. Preserve original recordings and manual corrections. Never publish private audio/projects merely because the code repository is public.

## 25 · When something stops

READ THE STAGE · KEEP THE EVIDENCE

| Symptom | Next action |
| --- | --- |
| No matching Dataset | Attach Music Studio Dataset; the older FP16 dataset is different |
| Builder authentication failure | Enable the private Kaggle Secret; check DATASET_OWNER if needed |
| Runtime build failed | Read runtime-build.log; keep the interrupted build for diagnosis |
| Port busy / low free VRAM | Stop the previous startup cell; unrelated processes are not killed |
| CUDA engine failed | Read studio.log and runtime-identity.json; do not switch blindly to CPU |
| Cloudflare link failed | Use the latest link/key; inspect cloudflare.log; keep saved media |
| Upload stopped | Re-select the same source file in Upload large WAV and resume |
| Training items incomplete | Enter lyrics with sections, or mark instrumental; enter/review styles |

Startup status, ready checks and runtime identity are saved with logs under music_studio. Dataset upload diagnostics live under music_studio_dataset. A FutureWarning is not the same as a failed model job.

**Validation scope**

Read VALIDATION.md for exactly what ran during development. Local compilation, browser/API tests and mocked private upload tests do not establish T4 speed, full-song quality, live Cloudflare reliability or a successful authenticated Kaggle upload.

## 26 · Know the tool boundaries

COMPONENT TERMS · SOURCE VERSIONS · PRACTICAL LIMITS

| Component | Terms / scope |
| --- | --- |
| Studio / C++ engines | MIT; retain upstream/dependency notices |
| YuE2 base weights | CC BY-NC plus individual-creator permission under its conditions |
| SheetSage2 / companion / MuScriptor | Separate CC BY-NC terms; no blanket creator exception |
| Gemma | Google Gemma terms and use policy |
| HT-Demucs / Beat This! | MIT; audio.cpp and RoFormer code retain their own notices |
| MOSS / Cloudflared | Apache-2.0; other dependencies retain their terms |

Realaudio/companion selection is Auto, On or Off. The companion has its own model terms; turning it off does not change the terms of other tools used. Keep the model Dataset private and review the licenses relevant to the selected workflow.

The PDF follows the visual format of your existing Music Production System guide. It describes the official 3.8.5 browser build and its selected tools, with unavailable newer Windows/hardware features identified.

- [Component notices and source links](https://github.com/bothashaun-maker/Music/blob/main/music_studio/NOTICES.md)
- [Pinned model catalog](https://github.com/bothashaun-maker/Music/blob/main/music_studio/catalog.json)
- [Validation record](https://github.com/bothashaun-maker/Music/blob/main/docs/music-studio/VALIDATION.md)

No Whisper, Parakeet, speech recognizer or MMS forced-alignment model is included. Musical score transcription remains available. RoFormer model usage follows its author’s terms; do not assume an unrestricted commercial license.
