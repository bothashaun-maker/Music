# Studio acoustic alignment — revision 16

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
