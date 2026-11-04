# Revision 14 audit → revision 15

This repair is based on the uploaded `yue2-music-revision-14.zip`, not a rollback
to an older GitHub implementation. The uploaded source is preserved separately
in the engineer's workspace. Existing root-level GitHub workflows are untouched;
this directory contains the repaired branch and its reproducible builders.

## Repaired execution failures

- **Wrong engine handle:** Load Engine returned `True`, while Plan Score and
  Song Tokens treated their `engine` input as a module. Both cold/warm paths now
  return the actual engine module. A stage-chain test executes load, plan,
  tokens, acoustic, decode and PCM-24 export together.
- **Process-lock mismatch:** the verifier required an absolute `main.py` path;
  the notebook launched the relative string `main.py`. Launch paths now match
  the verifier. Existing unrelated processes are not killed or adopted.
- **Run caching/overwrites:** a fixed seed could reuse a cached Run Context and
  overwrite artifacts. Run Context explicitly invalidates per queue attempt and
  allocates a unique folder. Transformer reuse is preserved.
- **Cover order:** the engine loader had no actual dependency on analysis,
  despite its title claiming otherwise. An `after_analysis` link enforces it.
  The loader is no longer an output node that independently triggers work.
- **Stop behavior:** Comfy interrupts now reach native cancellation hooks in
  score/token/acoustic stages. Cancellation retains artifacts. Busy workflow
  release requests are rejected instead of unloading models during inference.
- **Keep-loaded switch:** the option previously did nothing after saving and
  was ignored on the warm path. Its per-attempt release policy now applies.
- **ABC limits:** hidden 800/1200 caps are replaced by a visible optional
  4096 default, adjustable 128–8192. This is a separate planning ceiling;
  it does not force completion or alter the song-length budget.
- **Artifact integrity:** planner summaries no longer overwrite the native
  `plan.json`; nonfinite/silent/wrong-rate exports fail before file writes;
  allocation failures retain diagnostics and release the failed engine.
- **Canvas reports:** ABC notation, stage status and timestamp reports now have
  read-only text displays in their nodes; previously the backend returned text
  without a frontend handler to show it.
- **Playback:** standard Comfy `PreviewAudio` nodes are connected after saving.
- **Startup lifecycle:** stopping either notebook stops its server and tunnel.
  An existing server is reported instead of incorrectly reusing an undefined
  `comfy_process`. Native dependency pins also constrain Comfy installation.

## Timestamp Studio repairs

- Each upload has a separate ID. Pieces are checked for expected size and
  published atomically; a failed retry cannot truncate a previously good piece.
- Resume status skips complete pieces. Finish is idempotent, so retrying a lost
  success response does not duplicate the song. The assembler runs off the HTTP
  event loop. Two same-named uploads are published to separate files.
- Bad lengths/counts/extensions/IDs are rejected before assembly. Failed size
  checks retain pieces for retry. Imported paths are resolved before checking
  allowed roots. Displayed filenames/paths are HTML-escaped.
- Link downloads now run as background jobs with small polled status requests,
  avoiding one long HTTP response through Cloudflare. Local HTTP success/failure
  cases pass; this is not a live public-tunnel test.
- The uploader link is actually printed; it was absent from the original builder
  even though the documentation instructed the user to open it. Normal native
  upload also receives the documented server-size setting, subject to tunnel
  limits; the chunked page remains the route for large files.
- Alignment working WAVs have unique names, use float samples and are removed
  afterward. Choosing a missing song no longer silently substitutes another.
- The V1 `IS_CHANGED` hook invalidates cached audio when file content changes.
- Audition is wired after Nudge; exports and the ear check now use the same offset.
- Export folders and ZIP paths are unique, preserving earlier results.
- The Studio builder writes inside its source tree by default instead of an
  unavailable `/home/z/my-project/download` directory. Its notebook is included
  in the release; it was missing from the uploaded ZIP.

## What stays the same

The native model/weight revisions, FP16 MoT and FP32 VAE patches, T4 attention
path, model dataset layout, full-song ceiling and natural endings are retained.
The timing matcher/Whisper alignment algorithm is not rewritten. Timestamp
estimates still require listening/auditing; they are not guaranteed exact.
Models are not regenerated or downloaded by the repair tests. The interface and
alignment dependencies may still require first-session installation/downloads;
the YuE2 weight dataset does not need rebuilding.

## Validation and practical limits

The original uploaded branch's checks plus new stage-chain and real local HTTP
checks passed on Python 3.13.5 / PyTorch 2.11.0 CPU. A 75 MiB synthetic WAV arrived
byte-identical, with a readable WAV header. Tests cover interrupted/missing pieces,
resume, duplicate finish, simultaneous same-named files, changed audio cache,
planning budget forwarding, engine reuse/release, cancellation, valid masters,
native plan preservation, and rejection of NaN exports. Generated notebooks parse
and their payloads match the shipped sources. **101 tests passed** in the final suite. Python notebook cells and embedded
frontend JavaScript also passed syntax checks.

These are CPU/local-server checks. They are not a completed Kaggle GPU song,
live Cloudflare transfer, a quality benchmark or a timestamp-accuracy guarantee.
The archive contained no new Kaggle runtime traceback to attribute further
failures to hardware or the network. Keep saved outputs before ending a session.
