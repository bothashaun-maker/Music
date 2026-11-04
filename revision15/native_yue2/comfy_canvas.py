"""Builders for the two pre-installed YuE2 ComfyUI canvases (song + cover).

The graphs reference the KaggleYuE2 node pack only. Every stage is a real,
individually executed node - nothing is collapsed into one wrapper node. The
notebook's setup writes these JSON files into custom_nodes/KaggleYuE2/web/ so
the browser extension can open them via the ?yue2_workflow= link parameter.
"""

HEADER_SONG = "YuE2 Full Song - native FP16 engine (T4 x2)"
HEADER_COVER = "YuE2 Cover / Restyle - melody score, then audio"


def _node(nodes, nid, kind, pos, inputs, outputs, widgets, size=(340, 120), title=None):
    node = {"id": nid, "type": kind, "pos": list(pos), "size": list(size), "flags": {},
            "order": len(nodes), "mode": 0,
            "inputs": [{"name": name, "type": typ, "link": None} for name, typ in inputs],
            "outputs": [{"name": name, "type": typ, "links": []} for name, typ in outputs],
            "properties": {"Node name for S&R": kind}, "widgets_values": widgets}
    if title:
        node["title"] = title
    nodes.append(node)
    return node


def _wire(nodes, links, pairs):
    lookup = {n["id"]: n for n in nodes}
    for src, out_slot, dst, input_name in pairs:
        source = lookup[src]
        target = lookup[dst]
        slot = next(i for i, x in enumerate(target["inputs"]) if x["name"] == input_name)
        ident = len(links) + 1
        typ = source["outputs"][out_slot]["type"]
        links.append([ident, src, out_slot, dst, slot, typ])
        source["outputs"][out_slot]["links"].append(ident)
        target["inputs"][slot]["link"] = ident


def _finish(nodes, links, title):
    return {"last_node_id": max(n["id"] for n in nodes),
            "last_link_id": len(links), "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {"ds": {"scale": 0.55, "offset": [40, 60]},
                                                  "yue2_title": title},
            "version": 0.4}


def build_song_canvas(style="", lyrics=""):
    nodes, links = [], []
    _node(nodes, 1, "YuE2RunContext", (40, 80), [], [("YUE2_RUN", "YUE2_RUN")], [-1],
          title="One run folder + one seed per song")
    _node(nodes, 2, "YuE2LoadEngine", (40, 260), [("run", "YUE2_RUN")], [("YUE2_ENGINE", "YUE2_ENGINE")],
          [True], title="FP16 engine, exclusive on GPU 0")
    _node(nodes, 3, "YuE2StyleLyrics", (40, 520), [], [("YUE2_PROMPT", "YUE2_PROMPT")],
          [style, lyrics], (420, 320), title="Style + complete section-tagged lyrics")
    _node(nodes, 4, "YuE2PlanScore", (500, 80),
          [("run", "YUE2_RUN"), ("engine", "YUE2_ENGINE"), ("prompt", "YUE2_PROMPT")],
          [("YUE2_PLAN", "YUE2_PLAN")],
          ["Direct to audio", 1.0, 4096], title="Optional ABC plan - Direct to audio by default")
    _node(nodes, 5, "YuE2SongTokens", (950, 260),
          [("run", "YUE2_RUN"), ("engine", "YUE2_ENGINE"), ("prompt", "YUE2_PROMPT"),
           ("plan", "YUE2_PLAN"), ("custom_abc", "YUE2_ABC")], [("YUE2_TOKENS", "YUE2_TOKENS")],
          ["Up to 8 min"], (360, 200), title="Music tokens - ceiling, not a target")
    _node(nodes, 6, "YuE2SynthesizeLatents", (1360, 260),
          [("run", "YUE2_RUN"), ("tokens", "YUE2_TOKENS")], [("YUE2_LATENTS", "YUE2_LATENTS")],
          [16, "euler"], title="Flow matching - engine default 16 Euler")
    _node(nodes, 7, "YuE2DecodeAudio", (1730, 260),
          [("run", "YUE2_RUN"), ("latents", "YUE2_LATENTS")], [("AUDIO", "AUDIO")], [],
          title="Tiled FP32 VAE on GPU 1")
    _node(nodes, 8, "YuE2SaveSong", (2080, 260),
          [("run", "YUE2_RUN"), ("tokens", "YUE2_TOKENS"), ("audio", "AUDIO")],
          [("AUDIO", "AUDIO")], [], (380, 200), title="PCM-24 masters + status + ZIP")
    _wire(nodes, links, [(1, 0, 2, "run"), (1, 0, 4, "run"), (1, 0, 5, "run"),
                         (1, 0, 6, "run"), (1, 0, 7, "run"), (1, 0, 8, "run"),
                         (2, 0, 4, "engine"), (2, 0, 5, "engine"),
                         (3, 0, 4, "prompt"), (3, 0, 5, "prompt"), (4, 0, 5, "plan"),
                         (5, 0, 6, "tokens"), (5, 0, 8, "tokens"),
                         (6, 0, 7, "latents"), (7, 0, 8, "audio")])
    _node(nodes, 9, "PreviewAudio", (2510, 260), [("audio", "AUDIO")], [("AUDIO", "AUDIO")], [],
          (380, 180), title="Listen to the saved song")
    # Continue link IDs when appending the final standard player.
    _wire(nodes, links, [(8, 0, 9, "audio")])
    return _finish(nodes, links, HEADER_SONG)


def build_cover_canvas(style="", lyrics=""):
    nodes, links = [], []
    _node(nodes, 1, "YuE2RunContext", (40, 80), [], [("YUE2_RUN", "YUE2_RUN")], [-1],
          title="One run folder + one seed per song")
    _node(nodes, 2, "LoadAudio", (40, 260), [], [("AUDIO", "AUDIO")], [""],
          (420, 180), title="Upload the song you want to cover (WAV / FLAC / MP3)")
    _node(nodes, 3, "YuE2AnalyzeMelody", (500, 260),
          [("audio", "AUDIO")], [("YUE2_ABC", "YUE2_ABC")], [False, "Auto-detect"], (420, 200),
          title="SheetSage2 melody score (FP32 on T4) - runs before the engine loads")
    _node(nodes, 4, "YuE2StyleLyrics", (40, 560), [], [("YUE2_PROMPT", "YUE2_PROMPT")],
          [style, lyrics], (420, 320), title="New style + your original lyrics")
    _node(nodes, 5, "YuE2LoadEngine", (500, 560), [("run", "YUE2_RUN"), ("after_analysis", "YUE2_ABC")],
          [("YUE2_ENGINE", "YUE2_ENGINE")], [True], title="FP16 engine, loads after analysis")
    _node(nodes, 6, "YuE2SongTokens", (970, 260),
          [("run", "YUE2_RUN"), ("engine", "YUE2_ENGINE"), ("prompt", "YUE2_PROMPT"),
           ("plan", "YUE2_PLAN"), ("custom_abc", "YUE2_ABC")], [("YUE2_TOKENS", "YUE2_TOKENS")],
          ["Up to 8 min"], (360, 200), title="Melody score, then audio")
    _node(nodes, 7, "YuE2SynthesizeLatents", (1380, 260),
          [("run", "YUE2_RUN"), ("tokens", "YUE2_TOKENS")], [("YUE2_LATENTS", "YUE2_LATENTS")],
          [16, "euler"], title="Flow matching - engine default 16 Euler")
    _node(nodes, 8, "YuE2DecodeAudio", (1750, 260),
          [("run", "YUE2_RUN"), ("latents", "YUE2_LATENTS")], [("AUDIO", "AUDIO")], [],
          title="Tiled FP32 VAE on GPU 1")
    _node(nodes, 9, "YuE2SaveSong", (2100, 260),
          [("run", "YUE2_RUN"), ("tokens", "YUE2_TOKENS"), ("audio", "AUDIO")],
          [("AUDIO", "AUDIO")], [], (380, 200), title="PCM-24 masters + status + ZIP")
    _wire(nodes, links, [(1, 0, 5, "run"), (1, 0, 6, "run"), (1, 0, 7, "run"), (1, 0, 8, "run"),
                         (1, 0, 9, "run"), (2, 0, 3, "audio"), (3, 0, 6, "custom_abc"), (3, 0, 5, "after_analysis"),
                         (4, 0, 6, "prompt"), (5, 0, 6, "engine"),
                         (6, 0, 7, "tokens"), (6, 0, 9, "tokens"),
                         (7, 0, 8, "latents"), (8, 0, 9, "audio")])
    _node(nodes, 10, "PreviewAudio", (2520, 260), [("audio", "AUDIO")], [("AUDIO", "AUDIO")], [],
          (380, 180), title="Listen to the saved cover")
    _wire(nodes, links, [(9, 0, 10, "audio")])
    return _finish(nodes, links, HEADER_COVER)


CANVAS_BUILDERS = {"song": build_song_canvas, "cover": build_cover_canvas}


def build_all(style="", lyrics=""):
    return {name: builder(style, lyrics) for name, builder in CANVAS_BUILDERS.items()}
