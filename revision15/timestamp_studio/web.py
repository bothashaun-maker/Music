"""Frontend extension source for the TimestampStudio node pack.

STS_OPEN_WORKFLOW_JS is written to custom_nodes/TimestampStudio/web/open_workflow.js
at setup time. It does two things:

1. OpenWorkflow — the Cloudflare link carries ?sts_workflow=studio; the extension
   loads that canvas once ComfyUI finishes mounting, then strips the parameter so
   refreshes preserve subsequent edits. Same proven pattern as the YuE2 / H3 links.

2. SongLoader — keeps the "1 · Your song" list current. When a song arrives
   (uploader page / link download / Kaggle import) the uploader tab broadcasts it
   over BroadcastChannel('sts-uploads') and every open canvas refreshes the
   STSLoadSong combo in place — no page reload needed. New nodes also fetch the
   list once when created, so a fresh canvas always shows the songs on disk.
"""

STS_OPEN_WORKFLOW_JS = """import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

const REPORT_NODES = new Set(["STSAlign", "STSExportAll"]);

const WORKFLOWS = {
    studio: { file: "sts-studio.json",
              require: "STSExportAll",
              title: "Song Timestamp Studio" },
};

const NO_SONG = "(no songs yet — open the 📤 upload link from Kaggle)";
const LOAD_NODE = "STSLoadSong";
let lastAdded = null;

async function stsSongs() {
    try {
        const r = await fetch("/sts/songs", { cache: "no-store" });
        if (!r.ok) return null;
        return await r.json();
    } catch (error) { return null; }
}

async function refreshSongList(node) {
    if (node.type !== LOAD_NODE) return;
    const widget = (node.widgets || []).find((w) => w.name === "audio");
    if (!widget) return;
    const data = await stsSongs();
    if (!data) return;
    const values = (data.input && data.input.length) ? data.input.slice() : [NO_SONG];
    widget.options.values = values;
    if (!values.includes(widget.value)) {
        if (lastAdded && values.includes(lastAdded)) widget.value = lastAdded;
        else widget.value = values[0];
    }
    if (node.graph && node.graph.change) node.graph.change();
}

async function refreshAllSongNodes() {
    const nodes = (app.graph && app.graph._nodes) || [];
    for (const node of nodes) await refreshSongList(node);
}

app.registerExtension({
    name: "TimestampStudio.OpenWorkflow",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!REPORT_NODES.has(nodeData.name)) return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            original?.apply(this, arguments);
            const download = message?.sts_download?.[0];
            if (nodeData.name === "STSExportAll" && download) {
                this._stsDownload = download;
                if (!this._stsDownloadButton) {
                    this._stsDownloadButton = this.addWidget("button", "Download ZIP", null, () => {
                        const query = new URLSearchParams(this._stsDownload).toString();
                        window.open(api.apiURL("/view?" + query), "_blank");
                    }, { serialize: false });
                }
            }
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
        const name = url.searchParams.get("sts_workflow");
        if (name && Object.prototype.hasOwnProperty.call(WORKFLOWS, name)) {
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
                    if (!types.has(spec.require) || !types.has("STSAlign")) {
                        throw new Error("The installed Timestamp Studio canvas is incomplete.");
                    }
                    await app.loadGraphData(graph, true, true, (graph.extra?.sts_title || spec.title) + ".json");
                    // The canvas JSON re-applies the saved widgets AFTER nodeCreated's
                    // async refresh — re-sync the song list once the graph has settled.
                    await refreshAllSongNodes();
                    setTimeout(() => void refreshAllSongNodes(), 1200);
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
                    url.searchParams.delete("sts_workflow");
                    window.history.replaceState(null, "", url);
                } catch (error) {
                    console.error("Could not open the Timestamp Studio canvas", error);
                    window.alert(`Could not open the Timestamp Studio canvas: ${error.message}`);
                }
            };
            // Do not block ComfyUI setup while waiting for its mounted state.
            void open();
        }
        // Live song list: the uploader tab shouts, every canvas tab listens.
        try {
            const channel = new BroadcastChannel("sts-uploads");
            channel.onmessage = (event) => {
                lastAdded = (event.data && event.data.name) || null;
                void refreshAllSongNodes();
            };
        } catch (error) { /* BroadcastChannel unavailable — page reload still works */ }
        // Also catch graphs the user loads themselves (Workflows menu / F5):
        // re-sync for the first 20 s so the placeholder never survives a settled graph.
        let guardTicks = 0;
        const guard = setInterval(() => {
            void refreshAllSongNodes();
            if (++guardTicks >= 10) clearInterval(guard);
        }, 2000);
    },
    nodeCreated(node) {
        if (node.type === LOAD_NODE) void refreshSongList(node);
    },
});
"""
