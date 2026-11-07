# Studio revision 18: word gaps, vocal attacks and repeated-phrase cues

The user's next project provided 71 additional manually adjusted marker starts.
The original 15 references remain unchanged, giving 86 confirmed references.
Marker 69 was unchanged and was conservatively excluded. The latest project is
used privately for evaluation; publishing the earlier approved revision-17 ZIP
does not authorize publishing this new project or its contents.

## Result on the supplied recording

| Confirmed references | Revision 17 | Revision 18 |
| --- | ---: | ---: |
| New 71: mean absolute start error | 0.362 s | 0.126 s |
| New 71: within 0.5 s | 60/71 | 71/71 |
| All 86: mean absolute start error | 0.335 s | 0.130 s |
| All 86: median absolute start error | 0.131 s | 0.074 s |
| All 86: within 0.25 s | 66/86 | 75/86 |
| All 86: within 0.5 s | 74/86 | 85/86 |

These references were used during development and model/method comparison, not
as a held-out benchmark. Only line-marker starts have manual references. The
numbers do not validate every within-line word boundary or performance on other
recordings.

The known growled opening line still starts approximately **1.69 s late**, about
0.16 s worse than revision 17's estimate. Its low acoustic score remains flagged
for review. Do not describe this release as eliminating all manual corrections.
The latest fully corrected REAPER project itself is untouched.

## Decoder changes

With isolation ON, the decoder now uses the isolated-vocal acoustic scores as
its primary view. Mix scores remain cached and recorded for diagnostics; the
previous equal-geometric mixture is retained as a helper, not the default route.
Isolation OFF retains the existing single-view route.

Soft state costs distinguish different kinds of gaps:

- Blanks between letters inside a word cost 0.1 per 20-ms frame.
- Gaps between words on one lyric line cost 0.02 per frame.
- Lyric-line boundaries and explicit ellipsis pauses are free.
- Sustained label self-loops are unpenalized; there is no word-duration cap.

This discourages a weak first letter from attaching to noise many seconds before
the rest of a word. In the actual project, one first-word cue that was 6.49 s
early is now within about 0.03 s of its corrected position. The decoder retains
full-song order and long instrumental gaps and does not apply a global offset.

## Small onset refinement

A bounded RMS attack check can move a sufficiently supported first word back
to a local vocal attack. It uses a 20-ms envelope, 5-ms spacing, at most 0.4 s
of lookback, a 2:1 energy contrast, and a threshold 20% above the local floor.
Very weak words (score below 0.05), continuous energy and indistinct attacks are
left alone. Refinement cannot cross the previous lyric line's end.

`ctc_start` retains the original acoustic boundary; `timing_source` distinguishes
an adjusted attack from a CTC-only boundary. Word scores describe CTC frames,
not measured precision of the refined onset.

## Repeated-phrase refinement

Some weak leading words remained detached from a repeated chorus. The precision
pass compares the actual vocal audio of a clearer occurrence of the exact same
lyric phrase, using normalized MFCCs/deltas and subsequence dynamic time warping.
It transfers the cue through that audio match, not through copied durations,
fixed chorus offsets or positions from the manually corrected project.

Conservative bounds:

- At least three words per identical phrase; single-word loops are skipped.
- The clearest template must have a first-word acoustic score of at least 0.25;
  the uncertain target must be below 0.2.
- Only cue changes between 0.35 and 3 s are considered, with mean cosine distance
  at most 0.55. This distance is a similarity diagnostic, not timing confidence.
- Templates are limited to 15 s and query windows to 25 s (at most 3.75 million
  DTW cells). Search stays near the target occurrence.
- At most three leading words can change. Prefix proposals must preserve word
  order, the stable suffix and the previous lyric line's end; otherwise they are
  rejected. No word is compressed to zero duration to force acceptance.

On the supplied recording this adjusted two later chorus cues to within 0.13 s
and 0.05 s of the new manual references. Other proposals were rejected or too
small to change. The timestamp JSON records template occurrence, similarity,
original CTC boundaries and the changed prefix. The supplied labels are retained.

## Verification and compatibility

The exact 387.52-second analysis waveform was replayed through the production
engine. All 488 words had positive, ordered intervals. The vocal and both frame
caches were reused with model loading and separation explicitly blocked in the
test; no new inference, downloads, or Kaggle GPU time were used. The cached full
run took about five seconds in this environment; that is not a cold-run or
Kaggle performance guarantee.

Additional regressions exercise detached letters, line/ellipsis gaps, sustained
notes, invalid state costs, local attacks, weak/continuous energy, repeated-audio
cue recovery, prefix expansion, unchanged suffixes, rejection of unrelated audio,
and ambiguous single-word loops. All 164 local tests passed, including existing
deployed-node layout checks. Live Kaggle GPU/tunnel behavior was not retested.

Alternative singing/phoneme models and pronunciation/decoder experiments did
not improve the tested cases reliably and are excluded from the release. No
additional weights, libraries, asset rebuild, or new canvas controls are needed.
The YuE2 generator, original media, model datasets and approved project download
are unchanged.

Reproduce the reference measurement with the original private files:

```bash
python tools/audit_studio_timing.py --project confirmed.rpp \
  --candidate candidate_timestamps.zip --reference-markers 1:68,70:87
```

Rebuild the standalone Studio notebook with `tools/build_release.py`.
