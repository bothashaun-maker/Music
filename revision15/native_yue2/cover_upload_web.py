"""Additional upload button in the standard LoadAudio node; no canvas replacement."""

COVER_UPLOAD_JS = r'''import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHUNK_BYTES = 1024 * 1024;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function request(path, options = {}) {
    for (let attempt = 0; ; attempt++) {
        const control = new AbortController();
        const timeout = setTimeout(() => control.abort(), 180000);
        try {
            const response = await api.fetchApi(path, {...options, signal: control.signal});
            const body = await response.json().catch(() => ({}));
            if (!response.ok) {
                const error = new Error(body.error || `HTTP ${response.status}`);
                error.retryable = response.status >= 500 || response.status === 429;
                throw error;
            }
            return body;
        } catch (error) {
            if (attempt >= 4 || error.retryable === false) throw error;
            await sleep(1000 * (attempt + 1));
        } finally { clearTimeout(timeout); }
    }
}

app.registerExtension({
    name: "KaggleYuE2.CoverUpload",
    nodeCreated(node) {
        if (node.type !== "LoadAudio" && node.comfyClass !== "LoadAudio") return;
        const audio = (node.widgets || []).find(widget => widget.name === "audio");
        if (!audio) return;
        let busy = false;
        let resume = null;
        const button = node.addWidget("button", "Upload song (resumable)", null, async () => {
            if (busy) return;
            const picker = document.createElement("input");
            picker.type = "file";
            picker.accept = ".wav,.flac,.mp3,.ogg,.m4a,.aac,.opus,.aiff,.aif,.webm,.wma,audio/*";
            picker.onchange = async () => {
                const file = picker.files?.[0];
                if (!file || busy) return;
                busy = true;
                const label = text => {button.name = text; node.graph?.setDirtyCanvas(true, true);};
                const key = `yue2-cover:${file.name}:${file.size}:${file.lastModified}`;
                try {
                    if (!file.size || file.size > 2 * 1024 ** 3)
                        throw new Error("Choose a nonempty song no larger than 2 GiB");
                    let id = resume?.key === key ? resume.id : null;
                    try {id = sessionStorage.getItem(key) || id;} catch {}
                    id ||= crypto.randomUUID().replaceAll("-", "");
                    resume = {key, id};
                    try {sessionStorage.setItem(key, id);} catch {}
                    const metadata = {upload: id, name: file.name, size: file.size,
                                      n: Math.ceil(file.size / CHUNK_BYTES)};
                    const query = new URLSearchParams(metadata).toString();
                    label("Checking saved upload pieces…");
                    const status = await request("/yue2/cover-upload/status?" + query);
                    const completed = new Set(status.completed || []);
                    for (let index = 0; index < metadata.n; index++) {
                        label(`Uploading song ${Math.floor(100 * index / metadata.n)}% (${index+1}/${metadata.n})`);
                        if (!completed.has(index)) {
                            await request(`/yue2/cover-upload/chunk?${query}&i=${index}`, {
                                method: "POST", headers: {"Content-Type": "application/octet-stream"},
                                body: file.slice(index * CHUNK_BYTES, (index + 1) * CHUNK_BYTES)});
                        }
                    }
                    label("Finishing upload…");
                    const result = await request("/yue2/cover-upload/finish", {
                        method: "POST", headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(metadata)});
                    if (!result.ok || !result.filename) throw new Error("Upload did not finish");
                    if (audio.options && Array.isArray(audio.options.values)
                        && !audio.options.values.includes(result.filename))
                        audio.options.values = [...audio.options.values, result.filename];
                    audio.value = result.filename;
                    audio.callback?.(result.filename);
                    node.graph?.change();
                    resume = null;
                    try {sessionStorage.removeItem(key);} catch {}
                    label("Upload song (resumable) — ready");
                } catch (error) {
                    label("Upload song (resumable) — retry");
                    window.alert(`Song upload failed: ${error.message}\nChoose the same file with this button to resume. Completed pieces are kept.`);
                } finally {busy = false; picker.remove();}
            };
            picker.click();
        }, {serialize: false});
    },
});
'''
