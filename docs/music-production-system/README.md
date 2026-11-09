# JJP Music Production System — document source

- [Designed PDF](../../JJP_Music_Production_System.pdf)
- [GitHub launch desk](../../MUSIC_LAUNCH_DESK.md)
- [Editable full guide](../../MUSIC_PRODUCTION_SYSTEM.md)
- [Song brief](templates/song_brief.md), [cue review](templates/cue_review.csv), [production log](templates/production_log.csv)

`guide.json` is the single source for the designed guide and full Markdown.
Rebuild from repository root:

```bash
python tools/build_music_production_system.py
```

The document builder needs ReportLab and DejaVu Sans TTF fonts. It is a local
documentation build tool, not a Kaggle cell or a new music runtime. It imports
no model/generation code. The generated PDF uses selectable text, clickable
notebook links, PDF section bookmarks and a link back to the launch desk.
`layout-check.json` records the fitted body height for every page; the build
fails rather than truncating an overflowing section.

The user's supplied JJP Launch System PDF was used as a design/organizational
reference. It is not copied into this public repository. Personal history,
private audio/projects, unverified song measurements, platform forecasts and
blanket commercial-clearance claims from that reference are not reproduced.

The guide documents the existing workflows as of 9 November 2026. No music,
cover, timestamp or stem runtime, notebook, model or saved Dataset is changed.

## Document checks

The final PDF was rendered and visually reviewed on 9 November 2026. It has 16
pages, 16 outline entries, selectable text, 20 external link annotations and a
working launch-desk return link on each body page. Extracted text bounds stay
inside the page margins; no body section is truncated. Linked GitHub source
files returned HTTP 200. CSV templates parse correctly. Git diff checks confirm
that no notebooks, music runtimes, model assets, saved project archives or tests
were changed by this documentation addition.
