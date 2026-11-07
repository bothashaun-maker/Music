# Studio revision 17: measured mix + vocal alignment repair

These are historical revision-17 results. The later, larger reference set and
revision-18 timing changes are documented in STUDIO_CUE_REPAIR.md.

The user supplied the exact 387.52-second analysis waveform, its original frame
cache, the resulting timestamps, and a REAPER project containing manually
corrected marker starts. The waveform SHA-256 was verified against both the
collector manifest and the original alignment metadata. All 87 lyric labels
matched the REAPER markers in order, including repeated passages.

Only markers 1–15 had been changed beyond timestamp-rounding differences. Those
15 were the explicitly confirmed reference set. The remaining 72 project marker
positions were unchanged Studio estimates, not independent accuracy references.
The repair project was initially kept private. The user later explicitly
authorized publishing that exact Studio17_REAPER_Repair.zip for download;
subsequent projects and audio remain private unless separately authorized.

## Failure and resulting change

The speech-trained acoustic model recognized little in several dense or growled
passages. Its complete forced lyric path nevertheless returned times. A weak
short first word could be placed several seconds before the rest of its line;
high scores on later words hid that error in a line average.

Vocal-only alignment improved much of the recording but misplaced part of a
repeated passage. Revision 17 therefore computes both full-mix and isolated-vocal
CTC frames and combines them with an equal geometric mean, normalized per frame.
One full-sequence Viterbi path then aligns the lyrics to the combined evidence.
It does not splice independently timed passages or apply a global offset.
The existing single-view route remains available with isolation OFF.

The supplied canvas now has **Vocal isolation ON**. Previously that optional
control invoked an uninstalled Demucs CLI and silently fell back to the mix.
The notebook now installs Demucs 4.1.0, Julius 0.2.8 and sphn 0.2.1. A separate
analysis vocal is computed with zero random shifts, bounded GPU chunks and CPU
full-track buffers. The original WAV is never rewritten. A selected isolation
failure is reported directly instead of silently changing the requested method.

Separator weights use `adefossez/HTDemucs` revision
`cbc8a9b1a87023b7fd74e7b3412e6321c0eab003`, file `955717e8.safetensors`, SHA-256
`d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`.
The approximately 80 MB weights, lossless float vocal analysis, and independent
CTC frame caches live in Studio outputs. Attached saved outputs can supply these
in later sessions. Original CTC cache keys and YuE2 asset datasets are unchanged.

Weak detached first words and stretched words now have explicit review reasons.
Scores remain model diagnostics; fused and single-view scores are not calibrated
accuracy measures. A bigger Whisper setting still affects empty-lyric
transcription only. This is stated in the node description.

## Actual recording measurements

All values below measure lyric-marker **starts**, before applying any manual
corrections to the new output:

| Confirmed markers 1–15 | Previous output | Revision 17 |
| --- | ---: | ---: |
| Mean absolute error | 1.061 s | 0.206 s |
| Median absolute error | 0.102 s | 0.085 s |
| Maximum absolute error | 7.526 s | 1.526 s |
| Within 0.25 s | 9/15 | 12/15 |
| Within 0.50 s | 12/15 | 14/15 |

The growled opening line remains the outlier and is flagged for review. It is
not claimed to be repaired automatically. Scores flagged 53 of the full 87
lines for review. These warnings do not establish that their timing is wrong
or right. The remaining 72 marker starts and within-line word times lack
independently confirmed references. Results from one recording do not establish
accuracy on other songs.
These 15 references were used during development, not as a held-out benchmark.

The full production engine processed all 488 supplied words on CPU, produced
the normal export ZIP, and left the source analysis WAV unchanged. With weights
and the original mix frames already available, that run took 176 seconds here.
A subsequent run reused the saved vocal and both frame caches and returned the
same marker starts without separation/model inference. Kaggle and T4 speed were
not measured. The local runtime was Python 3.12 / PyTorch 2.11 CPU; this is not a
claim of a live Kaggle Python 3.13 or GPU test.

Other tested methods, including unanchored Whisper alignment and alternative
gap handling, created errors or failed words and are not part of the release.
Research-only packages/models used for comparison are not added to the notebook.

The private REAPER repair copy keeps all first 15 manually corrected markers
exactly, updates later markers and section regions using the new output, and
preserves every non-marker project byte. Its unreferenced later timings remain
estimates to review. Only that specific, subsequently approved project download
was published in GitHub; no audio was published.

## Checks and reproducibility

Local regressions cover model checksums, preserved source samples, vocal cache
reuse/corruption, synchronized acoustic views, non-mutating fusion, word-level
review, selected separation failures, deployed ComfyUI path-based loading, and
the installed node's two-view route. The evidence collector now includes the
second frame cache when available and directs downloads through Kaggle's file
panel. It still performs no model inference.
All 152 local regression checks passed, including the installed-node two-view
route. Those tests supplement the actual recording run; they do not replace it
or establish a live Kaggle/Cloudflare test.

To measure an exported candidate against explicitly confirmed markers:

```bash
python tools/audit_studio_timing.py --project confirmed.rpp \
  --candidate candidate_timestamps.zip --reference-markers 1:15
```

The IDs must be confirmed by the person who corrected them; the audit does not
infer reference accuracy from changed positions. It matches by ID and exact
label/order, so repeated lines cannot collapse onto the same reference.

The native YuE2 generation engine, music notebook, generation graph, and asset
builder remain unchanged. Rebuild through `tools/build_release.py`.
