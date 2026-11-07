# Collect the existing run for a timing repair

The bracket crash repair did not establish that lyric timing is accurate. The
current speech-trained CTC model can produce a complete path through the lyrics
without locating the performance correctly. We need the actual acoustic evidence
and the manually corrected REAPER timings to diagnose the reported drift.

The original uploaded WAV exceeds this chat's 32 MiB file-transfer limit. The
collector packages the exact float32 mono/16-kHz analysis samples plus matching
cached frame scores in ZIP files below 30 MiB each. It does not alter or replace
the original song, install packages, download models, or run alignment. A hash
check prevents quietly collecting a different recording's frame scores.

## In the existing Kaggle session

1. Stop the Studio startup cell with **Cancel Run**. Keep this session and its
   files; do not restart it or import a new notebook. This closes the canvas.
2. Add one code cell below it and paste:

   ```python
   import urllib.request
   source = urllib.request.urlopen(
       "https://raw.githubusercontent.com/bothashaun-maker/Music/main/revision15/tools/collect_studio_evidence.py",
       timeout=30,
   ).read()
   exec(compile(source, "collect_studio_evidence.py", "exec"), {"__name__": "__main__"})
   ```

3. Run **only that new cell**. It prints download links and paths for
   `studio_evidence_...zip` under `/kaggle/working/timestamp_studio/`.
4. Attach those evidence ZIPs to the engineering chat, together with the saved
   REAPER `.rpp` project containing the corrected intro and first verse. The
   project stores marker/region positions; its referenced media is not embedded.

If more than one song is present, the collector stops and lists exact paths.
Set the `SONG` variable in the collector to the intended path rather than
guessing. If the matching cache is missing, it stops; it never starts model
inference as a fallback. If no export with the same audio hash is found, attach
the latest timestamp ZIP separately. No Kaggle alignment rerun is requested.

## Verification and scope

Five local regressions check exact sample recovery after mono conversion and
resampling, unchanged source bytes, matching-export selection, missing-cache
failure, ambiguous-song failure, archive splitting without truncation, and
reconstruction of a 387.52-second sample array including its final partial chunk.
These verify collection mechanics, not accuracy on the user's recording.
The Studio alignment algorithm and working music-generation notebook are
unchanged by this evidence-collection addition. No timing repair is claimed yet.
