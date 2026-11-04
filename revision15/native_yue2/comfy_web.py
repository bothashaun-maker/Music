"""Frontend extension source for the KaggleYuE2 node pack.

OPEN_WORKFLOW_JS is written to custom_nodes/KaggleYuE2/web/open_workflow.js at
setup time. The Cloudflare link carries ?yue2_workflow=song or =cover; the
extension loads that canvas once ComfyUI finishes mounting, then strips the
parameter so refreshes preserve subsequent edits.
"""

OPEN_WORKFLOW_JS = """import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const REPORT_NODES = new Set(["YuE2LoadEngine", "YuE2SaveSong", "YuE2AnalyzeMelody", "YuE2ReleaseEngine"]);

const WORKFLOWS = {
    song: { file: "yue2-song.json",
            require: "YuE2SaveSong",
            title: "YuE2 Full Song - native FP16 engine" },
    cover: { file: "yue2-cover.json",
             require: "YuE2AnalyzeMelody",
             title: "YuE2 Cover / Restyle - melody score, then audio" },
};

app.registerExtension({
    name: "KaggleYuE2.OpenWorkflow",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!REPORT_NODES.has(nodeData.name)) return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            original?.apply(this, arguments);
            if (!Array.isArray(message?.text)) return;
            if (!this._resultReport) {
                this._resultReport = ComfyWidgets.STRING(
                    this, "result_report", ["STRING", { multiline: true }], app).widget;
                this._resultReport.inputEl.readOnly = true;
                this._resultReport.options = { ...this._resultReport.options, serialize: false };
            }
            this._resultReport.value = message.text.join("\\n\\n");
            const size = this.computeSize();
            this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
            this.graph?.setDirtyCanvas(true, true);
        };
    },
    setup() {
        const url = new URL(window.location.href);
        const name = url.searchParams.get("yue2_workflow");
        if (!name || !Object.prototype.hasOwnProperty.call(WORKFLOWS, name)) return;
        const spec = WORKFLOWS[name];
        const open = async () => {
            try {
                // Let ComfyUI finish mounting and restoring its previous tab first.
                const deadline = Date.now() + 60000;
                while (!app.vueAppReady || !app.positionConversion) {
                    if (Date.now() > deadline) throw new Error("ComfyUI did not finish starting.");
                    await new Promise((resolve) => setTimeout(resolve, 100));
                }
                await new Promise((resolve) => setTimeout(resolve, 500));
                const response = await fetch(new URL(spec.file, import.meta.url), { cache: "no-store" });
                if (!response.ok) throw new Error(`Canvas download failed: ${response.status}`);
                const graph = await response.json();
                const types = new Set((graph.nodes || []).map((node) => node.type));
                if (!types.has(spec.require) || !types.has("YuE2RunContext")) {
                    throw new Error("The installed YuE2 canvas is incomplete.");
                }
                await app.loadGraphData(graph, true, true, (graph.extra?.yue2_title || spec.title) + ".json");
                // Newer frontends pop the Templates dialog over the loaded canvas;
                // dismiss it so the graph is immediately visible.
                for (const delay of [200, 800]) {
                    setTimeout(() => {
                        const closer = document.querySelector(".p-dialog-close-button, [data-testid='close-button']");
                        if (closer instanceof HTMLElement) closer.click();
                        else window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true }));
                    }, delay);
                }
                // Refreshes must preserve subsequent edits rather than reload the template.
                url.searchParams.delete("yue2_workflow");
                window.history.replaceState(null, "", url);
            } catch (error) {
                console.error("Could not open the YuE2 canvas", error);
                window.alert(`Could not open the YuE2 canvas: ${error.message}`);
            }
        };
        // Do not block ComfyUI setup while waiting for its mounted state.
        void open();
    },
});
"""
