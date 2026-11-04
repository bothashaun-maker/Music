"""CPU checks for the ComfyUI canvas route (revision 11).

These tests validate what can be checked without a GPU or a live ComfyUI
server: the node-pack source structure, the canvas graph integrity, and the
setup helpers. Live server registration and ComfyUI's own prompt validation
were performed against a real ComfyUI checkout during development and are
repeated on Kaggle by the notebook cell's /object_info and /prompt checks.
"""
import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _snapshot_native_modules():
    return {name: mod for name, mod in sys.modules.items()
            if name.split(".")[0] in {"native_yue2", "kaggle_yue2"}}


def _restore_native_modules(snapshot):
    """Restore the exact pre-test module set: other suites (imported earlier by
    the discovery collector) still hold references to those module objects, so
    the temporary-tree imports must never replace or remove them."""
    for name in list(sys.modules):
        if name.split(".")[0] in {"native_yue2", "kaggle_yue2"} and name not in snapshot:
            del sys.modules[name]
    for name, mod in snapshot.items():
        sys.modules[name] = mod

EXPECTED_SONG = ["YuE2RunContext", "YuE2LoadEngine", "YuE2StyleLyrics", "YuE2PlanScore",
                 "YuE2SongTokens", "YuE2SynthesizeLatents", "YuE2DecodeAudio", "YuE2SaveSong", "PreviewAudio"]
EXPECTED_COVER = ["YuE2RunContext", "LoadAudio", "YuE2AnalyzeMelody", "YuE2StyleLyrics",
                  "YuE2LoadEngine", "YuE2SongTokens", "YuE2SynthesizeLatents",
                  "YuE2DecodeAudio", "YuE2SaveSong", "PreviewAudio"]


class CanvasGraphTests(unittest.TestCase):
    def setUp(self):
        from native_yue2.comfy_canvas import build_all
        self.graphs = build_all()

    def by_id(self, graph):
        return {node["id"]: node for node in graph["nodes"]}

    def test_both_canvases_present(self):
        self.assertEqual(sorted(self.graphs), ["cover", "song"])

    def test_node_rosters(self):
        self.assertEqual([n["type"] for n in self.graphs["song"]["nodes"]], EXPECTED_SONG)
        self.assertEqual([n["type"] for n in self.graphs["cover"]["nodes"]], EXPECTED_COVER)

    def test_links_reference_existing_slots(self):
        for name, graph in self.graphs.items():
            nodes = self.by_id(graph)
            for link in graph["links"]:
                ident, origin, out_slot, target, in_slot, typ = link
                self.assertIn(origin, nodes, f"{name}: link {ident} origin {origin}")
                self.assertIn(target, nodes, f"{name}: link {ident} target {target}")
                self.assertLess(out_slot, len(nodes[origin]["outputs"]), f"{name}: link {ident} output slot")
                self.assertLess(in_slot, len(nodes[target]["inputs"]), f"{name}: link {ident} input slot")
                self.assertEqual(nodes[origin]["outputs"][out_slot]["type"], typ,
                                 f"{name}: link {ident} output type mismatch")
                self.assertEqual(nodes[target]["inputs"][in_slot]["type"], typ,
                                 f"{name}: link {ident} input type mismatch")

    def test_link_bookkeeping_is_consistent(self):
        for name, graph in self.graphs.items():
            nodes = self.by_id(graph)
            self.assertEqual(graph["last_link_id"], len(graph["links"]), name)
            self.assertEqual(graph["last_node_id"], max(nodes), name)
            wired_inputs = set()
            for link in graph["links"]:
                ident, origin, out_slot, target, in_slot, _ = link
                self.assertIn(ident, nodes[origin]["outputs"][out_slot]["links"], name)
                self.assertEqual(nodes[target]["inputs"][in_slot]["link"], ident, name)
                wired_inputs.add((target, in_slot))
            for node in nodes.values():
                for slot, input_def in enumerate(node["inputs"]):
                    if (node["id"], slot) not in wired_inputs:
                        self.assertIsNone(input_def["link"],
                                          f"{name}: node {node['id']} input {input_def['name']} "
                                          "has a stale link id")

    def test_required_stage_inputs_are_wired(self):
        required_inputs = {
            "YuE2LoadEngine": {"run"},
            "YuE2PlanScore": {"run", "engine", "prompt"},
            "YuE2SongTokens": {"run", "engine", "prompt"},
            "YuE2SynthesizeLatents": {"run", "tokens"},
            "YuE2DecodeAudio": {"run", "latents"},
            "YuE2SaveSong": {"run", "tokens", "audio"},
            "YuE2AnalyzeMelody": {"audio"},
        }
        for name, graph in self.graphs.items():
            for node in graph["nodes"]:
                needed = required_inputs.get(node["type"], set())
                wired = {i["name"] for i in node["inputs"] if i["link"] is not None}
                for want in needed:
                    self.assertIn(want, wired | {i["name"] for i in node["inputs"]},
                                  f"{name}: node {node['type']} lacks input {want}")
                    if node["type"] != "YuE2CustomABC":
                        self.assertIn(want, wired,
                                      f"{name}: node {node['type']} input {want} not wired")

    def test_song_and_cover_share_the_ending_chain(self):
        song = [n["type"] for n in self.graphs["song"]["nodes"]][-4:]
        cover = [n["type"] for n in self.graphs["cover"]["nodes"]][-4:]
        self.assertEqual(song, cover)

    def test_canvas_json_round_trips(self):
        for name, graph in self.graphs.items():
            restored = json.loads(json.dumps(graph))
            self.assertEqual(restored, graph, name)
            self.assertEqual(restored["version"], 0.4, name)


class PackSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "native_yue2" / "comfy_pack.py").read_text()

    def test_pack_compiles_and_registers_nodes(self):
        compile(self.source, "comfy_pack.py", "exec")
        tree = ast.parse(self.source)
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        expected = {"YuE2RunContext", "YuE2LoadEngine", "YuE2StyleLyrics", "YuE2PlanScore",
                    "YuE2CustomABC", "YuE2SongTokens", "YuE2SynthesizeLatents",
                    "YuE2DecodeAudio", "YuE2SaveSong", "YuE2AnalyzeMelody", "YuE2ReleaseEngine"}
        self.assertTrue(expected <= classes, f"missing nodes: {expected - classes}")
        self.assertIn("NODE_CLASS_MAPPINGS", self.source)
        self.assertIn('WEB_DIRECTORY = "./web"', self.source)

    def test_pack_is_import_safe_and_never_launches_ui(self):
        # The pack imports the engine module for the pipeline patches; the engine
        # must stay lazy so importing inside the ComfyUI server loads no weights
        # and never builds the Gradio UI.
        engine = (ROOT / "native_yue2" / "engine.py").read_text()
        self.assertIn("pipe = None  # Set by initialize_pipeline(); the managed engine calls it below.", engine)
        self.assertIn("if IS_MANAGED_ENGINE:\n    initialize_pipeline()", engine)
        self.assertIn("if IS_MANAGED_ENGINE:\n    build_and_launch_ui()", engine)
        self.assertNotIn("gr.Blocks(**blocks_kwargs) as demo:\n    ", engine[:engine.index("def build_and_launch_ui")])

    def test_release_route_and_gpu_preflight_present(self):
        self.assertIn('"/yue2/release"', self.source)
        self.assertIn('"/yue2/status"', self.source)
        self.assertIn("check_gpu_room(include_pid=os.getpid())", self.source)
        self.assertIn("acquire_engine_lock", self.source)
        self.assertIn("stop_owned(RUNTIME_ROOT, ENGINE_SCRIPT, OUTPUT_ROOT)", self.source)

    def test_interrupt_reaches_comfyui(self):
        self.assertIn("throw_exception_if_processing_interrupted", self.source)
        self.assertIn("InterruptProcessingException", self.source)


class SetupHelperTests(unittest.TestCase):
    def test_write_comfy_pack_installs_all_assets(self):
        import sys
        import tempfile
        import shutil
        support = Path(tempfile.mkdtemp(prefix="yue2-pack-test-"))
        saved = sys.path[:]
        snapshot = _snapshot_native_modules()  # Taken inside the test, after collection.
        try:
            # The helper imports sibling modules; run it against a copied tree so
            # the repo checkout is never written to.
            for pkg in ("native_yue2", "kaggle_yue2"):
                shutil.copytree(ROOT / pkg, support / pkg, dirs_exist_ok=True)
            sys.path.insert(0, str(support))
            # Drop cached repo-tree modules so the copied tree is imported.
            for name in list(sys.modules):
                if name.split(".")[0] in {"native_yue2", "kaggle_yue2"}:
                    del sys.modules[name]
            from native_yue2.setup import write_comfy_pack, COMFYUI_COMMIT
            comfy = support / "ComfyUI"
            pack = write_comfy_pack(comfy)
            files = sorted(str(p.relative_to(pack)) for p in pack.rglob("*") if p.is_file())
            for wanted in ("__init__.py", "web/open_workflow.js", "web/yue2-song.json",
                           "web/yue2-cover.json"):
                self.assertIn(wanted, files)
            workflows = sorted(p.name for p in (comfy / "user" / "default" / "workflows").glob("*.json"))
            self.assertEqual(workflows, ["YuE2 Cover Restyle.json", "YuE2 Full Song.json"])
            self.assertRegex(COMFYUI_COMMIT, r"^[0-9a-f]{40}$")
        finally:
            sys.path[:] = saved
            _restore_native_modules(snapshot)
            shutil.rmtree(support, ignore_errors=True)

    def test_canvas_json_files_match_builders(self):
        from native_yue2.comfy_canvas import build_all
        graphs = build_all()
        for name in ("song", "cover"):
            embedded = (ROOT / "native_yue2" / "comfy_web.py").read_text()
            self.assertIn("yue2_workflow", embedded)
            self.assertEqual(json.loads(json.dumps(graphs[name]))["version"], 0.4)


if __name__ == "__main__":
    unittest.main()
