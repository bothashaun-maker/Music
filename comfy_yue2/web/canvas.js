import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
app.registerExtension({
  name: "YuE2.NativeCanvas",
  setup() {
    const choice = new URLSearchParams(location.search).get("yue2_workflow");
    if (choice === "song" || choice === "cover") {
      // Do not block Comfy bootstrap while waiting for its canvas store.
      requestAnimationFrame(() => requestAnimationFrame(async () => {
      const response = await fetch(new URL(`./${choice}.json`, import.meta.url));
      if (!response.ok) throw new Error("Unable to load the installed YuE2 canvas");
      const graph = await response.json();
      // Extension setup precedes Vue mounting the LiteGraph canvas.
      await new Promise(resolve => {
        const ready = () => document.querySelector("canvas")
          ? requestAnimationFrame(() => requestAnimationFrame(resolve))
          : requestAnimationFrame(ready);
        ready();
      });
      await app.loadGraphData(graph);
      const nodes = app.graph._nodes;
      if (nodes.length) {
        const left = Math.min(...nodes.map(n => n.pos[0]));
        const top = Math.min(...nodes.map(n => n.pos[1]));
        const right = Math.max(...nodes.map(n => n.pos[0] + n.size[0]));
        const bottom = Math.max(...nodes.map(n => n.pos[1] + n.size[1]));
        const canvas = app.canvas.canvas;
        const scale = Math.min((canvas.clientWidth - 150) / (right - left),
                               (canvas.clientHeight - 150) / (bottom - top), 1);
        app.canvas.ds.scale = scale;
        app.canvas.ds.offset = [-left + 90 / scale, -top + 90 / scale];
        app.graph.setDirtyCanvas(true, true);
      }
      }));
    }
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["YuE2CoverABC", "YuE2SaveSong", "YuE2ShowABC"].includes(nodeData.name)) return;
    const previous = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
      previous?.apply(this, arguments);
      if (!this.yue2Text) {
        this.yue2Text = ComfyWidgets.STRING(this, "result", ["STRING", {multiline: true}], app).widget;
        this.yue2Text.inputEl.readOnly = true;
        this.yue2Text.options = {...this.yue2Text.options, serialize: false};
      }
      this.yue2Text.value = (message.text || []).join("\n\n");
      this.setSize(this.computeSize());
      app.graph.setDirtyCanvas(true, true);
    };
  }
});
