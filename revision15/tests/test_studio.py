"""Structure + formatter tests for the Timestamp Studio canvas notebook.

Fast CPU checks only: notebook shape, payload/source identity, node-pack
loading and wiring semantics, and the engine's formatter edge cases.
The full known-truth alignment run is a delivery-time validation (see
tools/build_studio_notebook.py outputs and the delivery notes).
"""
import ast
import base64
import importlib
import json
import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("STS_AUTOLAUNCH", "0")


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        from build_studio_notebook import build

        cls.nb = build()

    def test_notebook_is_single_executable_cell(self):
        cells = self.nb["cells"]
        self.assertEqual([c["cell_type"] for c in cells], ["markdown", "code", "markdown"])
        code = "".join(cells[1]["source"])
        ast.parse(code)
        for probe in ("STSAlign", "sts_workflow=studio", "faster-whisper==1.2.1",
                      "COMFYUI_COMMIT", "Registered tunnel connection"):
            self.assertIn(probe, code)

    def test_payload_embeds_exact_repo_sources(self):
        code = "".join(self.nb["cells"][1]["source"])
        tree = ast.parse(code)
        encoded = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "payload" for t in n.targets))
        files = json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(files["TimestampStudio/__init__.py"],
                         (ROOT / "timestamp_studio/comfy_pack.py").read_text())
        self.assertEqual(files["TimestampStudio/sts_engine.py"],
                         (ROOT / "timestamp_studio/engine.py").read_text())
        self.assertEqual(files["TimestampStudio/acoustic.py"],
                         (ROOT / "timestamp_studio/acoustic.py").read_text())
        self.assertEqual(files["TimestampStudio/vocal.py"],
                         (ROOT / "timestamp_studio/vocal.py").read_text())
        graph = json.loads(files["TimestampStudio/web/sts-studio.json"])
        types = {n["type"] for n in graph["nodes"]}
        self.assertIn("STSExportAll", types)
        self.assertIn("STSAlign", types)
        # Node 1 is the big-file-safe uploader (7fcf70d); native LoadAudio stays
        # in the node menu for small files but is not on the default canvas.
        self.assertIn("STSLoadSong", types)
        self.assertIn("PreviewAudio", types)
        options = next(n for n in graph['nodes'] if n['type'] == 'STSLyricsOptions')
        self.assertIs(options['widgets_values'][-1], True)

    def test_canvas_wiring_is_a_single_connected_graph(self):
        code = "".join(self.nb["cells"][1]["source"])
        tree = ast.parse(code)
        encoded = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "payload" for t in n.targets))
        files = json.loads(zlib.decompress(base64.b64decode(encoded)))
        graph = json.loads(files["TimestampStudio/web/sts-studio.json"])
        self.assertEqual(len(graph["nodes"]), 7)
        self.assertEqual(len(graph["links"]), 6)
        # every non-native node output feeds exactly the documented consumers
        kinds = {n["id"]: n["type"] for n in graph["nodes"]}
        self.assertEqual(kinds[3], "STSAlign")
        self.assertEqual(kinds[5], "STSExportAll")
        # links: LoadAudio->Align, Options->Align, Align->Nudge, Nudge->Export, Align->Audition, Audition->Preview
        endpoints = {(l[1], l[3]) for l in graph["links"]}
        self.assertEqual(endpoints, {(1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (6, 7)})

    def test_pack_nodes_load_and_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "TimestampStudio"
            pkg.mkdir()
            (pkg / "__init__.py").write_text((ROOT / "timestamp_studio/comfy_pack.py").read_text())
            (pkg / "sts_engine.py").write_text((ROOT / "timestamp_studio/engine.py").read_text())
            (pkg / "acoustic.py").write_text((ROOT / "timestamp_studio/acoustic.py").read_text())
            (pkg / "vocal.py").write_text((ROOT / "timestamp_studio/vocal.py").read_text())
            sys.path.insert(0, tmp)
            try:
                for name in list(sys.modules):
                    if name.startswith("TimestampStudio"):
                        del sys.modules[name]
                pack = importlib.import_module("TimestampStudio")
            finally:
                sys.path.remove(tmp)
        for name in ("STSLyricsOptions", "STSAlign", "STSNudge", "STSAudition", "STSExportAll"):
            self.assertIn(name, pack.NODE_CLASS_MAPPINGS)
            self.assertTrue(hasattr(pack.NODE_CLASS_MAPPINGS[name], "INPUT_TYPES"))
        (opts,) = pack.STSLyricsOptions().execute("my_song", "  ", "auto", "auto", False)
        self.assertEqual(opts["lyrics"], "")  # empty lyrics = auto-transcribe mode
        self.assertEqual(opts["song_name"], "my_song")
        base = {"entries": [], "nudge": 0.0}
        (shifted,) = pack.STSNudge().execute(base, 1.5)
        self.assertAlmostEqual(shifted["nudge"], 1.5)
        (again,) = pack.STSNudge().execute(shifted, 0.5)
        self.assertAlmostEqual(again["nudge"], 2.0)  # chained nudges accumulate

    def test_engine_formatter_edge_cases(self):
        ns = {"__name__": "sts_engine_test"}
        code = (ROOT / "timestamp_studio/engine.py").read_text()
        exec(compile(code, "engine.py", "exec"), ns)
        self.assertEqual(ns["fmt_clock"](59.9996), "01:00.000")
        self.assertEqual(ns["fmt_clock"](6.0, comma=True, force_hours=True), "00:00:06,000")
        self.assertEqual(ns["fmt_lrc_stamp"](59.996), "[01:00.00]")
        self.assertEqual(ns["fmt_yt"](3599.999), "1:00:00")
        self.assertEqual(ns["_csv_field"]('say "hi"'), '"say ""hi"""')
        entries = [{"kind": "tag", "text": "Chorus", "start": 10.0},
                   {"kind": "sing", "text": "hello world", "start": 12.0}]
        lrc = ns["fmt_lrc"](entries)
        self.assertIn("[00:10.00][Chorus]", lrc)      # section tags keep brackets
        self.assertIn("[00:12.00]hello world", lrc)   # sung lines are plain text


if __name__ == "__main__":
    unittest.main()
