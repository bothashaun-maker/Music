# Studio acoustic alignment — revision 16

These are historical revision-16 notes. The later exact analysis-audio evidence
and measured revision-17 repair are documented in STUDIO_MIX_VOCAL_REPAIR.md.

## Revision 16.2: stray punctuation and early lyric validation

The next supplied traceback reached the end of acoustic inference, then failed
on a standalone `[` immediately after a section heading. The parser now ignores
and logs punctuation-only lines instead of classifying them as sung lyrics.
Actual lyric lines and section headings retain their order and labels.
The complete supplied text is validated before optional vocal isolation, audio
loading or model inference. A nonempty sheet with no sung words fails explicitly
rather than falling through to automatic transcription. The acoustic helper also
validates its input before producing frames when called directly.

Regressions cover a stray bracket among repeated lyrics and section tags,
unsupported text failing before inference, and invalid sheets failing before
isolation or audio loading. Existing acoustic cache keys and frame computations
are unchanged: the interrupted run's saved frames remain reusable in its session
after removing the stray bracket. This is a parsing repair, not a measured
improvement to timing accuracy on the user's recording. The uploaded pip log
also reports conflicts with preinstalled Gradio/Diffusers requirements; these
did not trigger the supplied ValueError and are not changed by this repair.
All 138 local regression checks passed. The actual lyric text embedded in the
uploaded error report also passes text validation after the parser ignores the
stray bracket (488 words); this check does not run acoustic inference.

## Revision 16.1: installed helper import

The supplied Kaggle traceback showed that the notebook embedded `acoustic.py`
but its hardcoded installer copied only the older file list. The helper never
reached `custom_nodes/TimestampStudio/`. The engine's fallback then tried the
source package `timestamp_studio`, which is not present in that installed layout.

The installer now checks the required bundle and copies every payload file,
including future helpers. The engine loads the adjacent helper by file location
without assuming a Python package name or masking dependency failures. The node
pack validates that helper during registration, before setup prints a usable
canvas link. Five new regressions execute the actual installer, load the package
using the absolute-path module name seen in ComfyUI, run the installed alignment
and ZIP export, and exercise missing files and dependency errors. No model
weights or acoustic math are changed by this repair.
All 135 local regression checks passed, including those installed-layout tests.

The uploaded exports exposed a fundamental problem: freely transcribed words
were fuzzy-matched to the supplied lyrics, and missing words were interpolated.
Large recognition gaps and weak matches compressed repeated lines into a few
seconds. Text-match percentages did not validate their timing. Section-tag lookup
could also jump to a later lyric when previous estimated intervals overlapped.

The supplied-English-lyrics route now uses an acoustic CTC model and the entire
provided lyric sequence. A Viterbi path advances through that sequence while
blank states cover gaps; repeated words remain separate target occurrences.
It does not run the former fuzzy matcher or interpolate unheard lyric timings.

Implementation:

- Pinned `facebook/wav2vec2-base-960h` revision
  `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; inference uses the existing PyTorch
  CPU/CUDA stack rather than CTranslate2's separate CUDA-12 runtime.
- Twenty-second inference windows with two seconds of context bound model memory.
  CTC backtrace allocation is capped at 512 MiB. CPU inference uses at most four
  PyTorch threads. Model weights are about 360 MB and download on the Kaggle
  server on first use; they are stored under working outputs.
- Cached frame scores are keyed by audio samples, model revision and inference
  settings. Lyric edits reuse frames. Attached saved Studio outputs can supply
  model weights and cached frames without repeating those downloads/inference.
- Proper polyphase downsampling replaces interpolation without an antialias
  filter. The original recording is unchanged.
- Section tags follow the next lyric in text order, and opening Intro regions
  start at zero. Acoustic word onsets are not shifted to repair a section boundary.
- Supplied labels, repeated lines and their word times are retained. Acoustic
  scores are diagnostic values, not a word-match percentage or timing guarantee.
  Flagged lines appear in `_alignment_review.csv` for a listening check.
- Blank lyrics retain multilingual Whisper transcription. Its actual segment
  and word times are exported without dividing phrase durations by text length.
- Cut-plan CSV preserves milliseconds and exact durations. Section edges and
  the initial interval are covered without the earlier beat-snapping gaps.
- A **Download ZIP** button is added to the Export node and follows its latest
  result. Existing output routes, uploads, logs and diagnostics remain available.

Limits and verification:

Supplied-lyric acoustic alignment currently supports English/romanized lyrics.
The Whisper model selector affects automatic transcription with an empty lyric
box. Model scores are not calibrated timestamp accuracy; dense singing, growls,
backing voices and text that differs from the performance still require review.

Unit regressions cover repeated targets separated by long blank gaps, repeated
letters, impossible paths, direct routing without ASR matching, section order,
continuous cut coverage, CSV precision and antialias filtering. The explicit
`tools/acoustic_smoke.py` check exercises the real pinned model on synthetic
speech with repeated passages and long gaps, including cache reuse. This is a
technical integration check, not a singing-quality benchmark.
All 130 regression checks passed. The real-model smoke check placed all 14
synthetic lyric lines in the correct occurrences and verified cached-frame reuse.

The user's full WAV could not be accessed through the file-transfer tool because
it exceeds 32 MiB. No claim is made that this environment listened to or validated
that recording, or ran the updated notebook on Kaggle. The existing music
generation notebook, its model dataset, engine and full-song graph are unchanged.
