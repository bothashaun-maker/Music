# Timestamp Studio: reported recognition failures

The supplied Kaggle console log showed `VAD filter removed 06:27.520 of audio`
for a 06:27.520 recording, and earlier attempts raised
`Library libcublas.so.12 is not found or cannot be loaded`. PyTorch used CUDA 13;
that does not establish availability of CTranslate2's CUDA 12 libraries. The
eventual export reported zero matched words. The old matcher spread unmatched
words across the recording, producing estimated times despite having no anchors.

Changes are confined to Timestamp Studio and its notebook builder:

- Music recognition uses `vad_filter=False` to retain the entire recording.
- GPU selection checks the CUDA libraries Whisper needs. Missing libraries use
  CPU INT8 with a visible explanation. A library error during model creation or
  lazy recognition iteration retries once on CPU; unrelated failures propagate.
  No CUDA packages are downloaded/installed over Kaggle's working PyTorch stack.
- Empty/silent/nonfinite audio, no recognition, and zero matched words produce
  explicit errors before export. The export node and ZIP builder also reject
  previous zero-match results. A single offset cannot repair zero-match output.
- Each alignment attempt retains options, audio levels, backend details and
  recognized segments/words in `timestamp_studio/diagnostics/<attempt>/alignment.json`,
  including failed attempts. Successful JSON exports carry transcription data
  and alignment status; README exports carry the status too.
- Warnings are shown on the canvas. Word matching percentage is not presented
  as an independently measured timestamp accuracy guarantee.
- Runtime messages appear below the running cell and are also retained in
  `timestamp_studio/logs/comfyui.log` under Kaggle working storage.

Local checks reproduce missing libraries, a lazy CUDA error, speech-filter
settings, empty recognition, zero-match rejection, known word anchors, automatic
transcription, and failed-attempt diagnostic retention. They use synthetic audio
and controlled recognizer outputs, not this user's recording or a real GPU/model.
No new Whisper recognition-quality or timing-accuracy claim is established.
The complete local regression suite passed all **118 tests**.

The music-creation notebook, its native engine and full-song graph remain
byte-identical to the working previous release. Existing asset datasets and
the resumable upload routes remain compatible. Accelerator **None** is supported
for Studio; CPU inference may take longer than a working GPU backend.

## Opening Intro region export

The REAPER CSV formatter used raw section-tag times, anchoring Intro to the first
vocal even though chapter/cut-plan sections already included the instrumental
opening from zero. It now uses the same section boundaries as the chapter
builder. An initial Intro always begins at 0:00. Lyric marker positions remain
unchanged. Five export-only regression checks cover the opening boundary,
chapter consistency, subsecond intros, lead-ins and marker-only exports. This
does not establish or repair the timing accuracy of the recognized lyric lines.
