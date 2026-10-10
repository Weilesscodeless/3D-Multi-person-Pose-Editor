import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EDITORS = {
    studio: { url: new URL("./editor/studio.html", import.meta.url), version: "20261002-v5", title: "机位与姿态编辑器编辑器",
              fields: ["scene_json", "output_mode", "width", "height", "extra_prompt"], data: "scene_json", image: "reference_image_1" },
    freePose: { url: new URL("./editor/freepose.html", import.meta.url), version: "gaoshang-view-v47", title: "3D多人姿势编辑器",
                fields: ["pose_json", "extra_prompt", "width", "height"], data: "pose_json", image: "reference_image" },
};
const editorFor = node => (node.comfyClass || node.type) === "GaoshangQwenFreePose" ? EDITORS.freePose : EDITORS.studio;
const widget = (node, name) => node.widgets.find(item => item.name === name);
let activeEditor = null;

// Prefer the actual upstream image preview, falling back to LoadImage's file name.
function upstreamPreview(node,inputName){
    const input=node.inputs?.find(i=>i.name===inputName);
    let link=input?.link,seen=new Set();
    for(let depth=0;link!=null&&depth<12;depth++){
        const edge=app.graph.links[link];if(!edge)return null;
        const source=app.graph.getNodeById(edge.origin_id);if(!source||seen.has(source.id))return null;seen.add(source.id);
        const preview=source.imgs?.[source.imageIndex||0]?.src;if(preview)return preview;
        if(source.comfyClass==='LoadImage'||source.type==='LoadImage'){
            const file=source.widgets?.find(w=>w.name==='image')?.value;
            if(typeof file==='string'&&file){const cleaned=file.replace(/ \[(input|output|temp)\]$/,'');const slash=cleaned.lastIndexOf('/');return new URL('/view?'+new URLSearchParams({filename:cleaned.slice(slash+1),subfolder:slash>=0?cleaned.slice(0,slash):'',type:'input'}),location.origin).href;}
        }
        link=source.inputs?.find(i=>i.type==='IMAGE'&&i.link!=null)?.link;
    }
    return null;
}

// Show the applied mannequin image right away (on the node and on PreviewImage nodes fed by
// its 人偶姿态图 output) without running the workflow. The frontend draws previews from
// app.nodeOutputs, which must point at a served file, so the PNG goes to ComfyUI's temp folder.
async function showPoseImage(node) {
    let reference;
    try { reference = JSON.parse(widget(node, "pose_json")?.value || "{}").poseReference; } catch { return; }
    if (typeof reference !== "string" || !reference.startsWith("data:image/png;base64,")) return;
    let hash = 2166136261;
    for (let i = 0; i < reference.length; i += 7) hash = Math.imul(hash ^ reference.charCodeAt(i), 16777619);
    const form = new FormData();
    form.append("image", await (await fetch(reference)).blob(), `gaoshang_freepose_${(hash >>> 0).toString(16)}.png`);
    form.append("type", "temp");
    form.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) return;
    const { name, subfolder } = await response.json();
    const images = [{ filename: name, subfolder: subfolder || "", type: "temp" }];
    const slot = node.outputs?.findIndex(output => output.type === "IMAGE") ?? -1;
    const previews = (node.outputs?.[slot]?.links || []).map(id => app.graph.getNodeById(app.graph.links[id]?.target_id))
        .filter(target => target?.type === "PreviewImage");
    for (const target of [node, ...previews]) app.nodeOutputs[target.id] = { ...(app.nodeOutputs[target.id] || {}), images };
    app.graph.setDirtyCanvas(true, true);
}

function openEditor(node) {
    activeEditor?.();
    const editor = editorFor(node);
    const overlay = document.createElement("dialog");
    overlay.style.cssText = "width:96vw;max-width:1700px;height:94vh;max-height:1100px;padding:0;border:1px solid #dce3ed;border-radius:12px;background:#f4f6f8;box-shadow:0 20px 90px #0007;overflow:hidden;";
    const frame = document.createElement("iframe");
    frame.title = editor.title;
    frame.style.cssText = "border:0;width:100%;height:100%;display:block";
    frame.src = editor.url.href + "?embedded=1&v=" + editor.version;
    const referencePicker = document.createElement("input");
    referencePicker.type = "file";
    referencePicker.accept = "image/*";
    referencePicker.hidden = true;
    referencePicker.addEventListener("click", event => event.stopPropagation());
    // Keep the picker inside the active modal, invoked synchronously by the user click.
    frame.gaoshangChooseReference = onFile => {
        referencePicker.value = "";
        referencePicker.onchange = () => {
            const file = referencePicker.files[0];
            if (file) onFile(file);
        };
        referencePicker.click();
    };
    // Multi-file / folder variant for the free-pose OpenPose gallery.
    frame.gaoshangChooseFiles = (onFiles, { directory = false } = {}) => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = "image/*";
        picker.multiple = true;
        picker.webkitdirectory = directory;
        picker.hidden = true;
        picker.addEventListener("click", event => event.stopPropagation());
        picker.onchange = () => { if (picker.files.length) onFiles([...picker.files]); picker.remove(); };
        overlay.append(picker);
        picker.click();
    };
    overlay.append(frame, referencePicker);
    const oldFocus = document.activeElement;
    function close() {
        window.removeEventListener("message", receive);
        overlay.close();
        overlay.remove();
        oldFocus?.focus();
        activeEditor = null;
    }
    function receive(event) {
        if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
        if (event.data?.type === "gaoshang-ready") {
            const payload = Object.fromEntries(
                editor.fields.map(name => [name, widget(node, name).value])
            );
            if (node.comfyClass === "GaoshangQwenPose" || node.type === "GaoshangQwenPose") {
                payload.qwenBinding = {
                    peEnabled: node.inputs?.find(input => input.name === "pe_clip")?.link != null,
                    count: [1, 2, 3].filter(i => node.inputs?.find(input => input.name === `reference_image_${i}`)?.link != null).length,
                    assignments: [1, 2, 3].map(i => widget(node, `person_${i}_reference`).value),
                };
            }
            if (editor === EDITORS.freePose) {
                payload.width = widget(node, "width")?.value;
                payload.height = widget(node, "height")?.value;
            }
            payload.referencePreview=upstreamPreview(node,editor.image);
            if (editor === EDITORS.freePose) payload.referencePreviews = Array.from({length:9}, (_,i) => upstreamPreview(node, i ? `reference_image_${i+1}` : 'reference_image'));
            // FreePose + SAM3D 工具台：用户尚未亲自应用过姿势时，拉取 GaoshangMHRToOpenPose
            // 落盘的多人骨架，编辑器打开即自动摆好（跑过一次工作流后生效）。
            if (editor === EDITORS.freePose) {
                fetch("/gaoshang_pose/sam3d_multi")
                    .then(response => response.ok ? response.json() : null)
                    .then(data => {
                        if (data?.people?.length) payload.multiPose = data;
                    })
                    .catch(() => {})
                    .finally(() => frame.contentWindow.postMessage({ type: "gaoshang-load", payload }, location.origin));
            } else {
                frame.contentWindow.postMessage({ type: "gaoshang-load", payload }, location.origin);
            }
        }
        if (event.data?.type === "gaoshang-close") close();
        if (event.data?.type === "gaoshang-apply") {
            const payload = event.data.payload;
            if (!payload || typeof payload[editor.data] !== "string") return;
            for (const name of editor.fields) {
                const item = widget(node, name);
                let value = payload[name];
                if (name === "scene_json" && typeof value === "string") {
                    try { const parsed = JSON.parse(value); if (parsed) { delete parsed.gaoshang_auto; delete parsed.gaoshang_sam3d; value = JSON.stringify(parsed); } } catch (error) { /* keep as-is */ }
                }
                item.value = value;
                item.callback?.(value);
            }
            node.setDirtyCanvas(true, true);
            node.properties = node.properties || {};
            node.properties.gaoshang_auto = false;
            node.properties.gaoshang_user_posed = true;  // 用户已亲自应用过姿势，之后打开不再自动灌入 SAM3D  // editor result: stop auto refresh
            app.graph.change();
            close();
            if (editor === EDITORS.freePose) void showPoseImage(node).catch(error => console.warn("高尚: pose preview failed", error));
        }
    }
    window.addEventListener("message", receive);
    overlay.addEventListener("cancel", event => { event.preventDefault(); close(); });
    document.body.append(overlay);
    overlay.showModal();
    activeEditor = close;
}

app.registerExtension({
    name: "Gaoshang.PoseStudio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["GaoshangPoseStudio", "GaoshangQwenPose", "GaoshangQwenFreePose"].includes(nodeData.name)) return;
        const freePose = nodeData.name === "GaoshangQwenFreePose";
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const scene = widget(this, freePose ? "pose_json" : "scene_json");
            scene.type = "gaoshang_hidden";
            scene.computeSize = () => [0, -4];
            if (scene.element) scene.element.style.display = "none";
            this.addWidget("button", freePose ? "打开3D多人姿势编辑器" : "打开机位与姿态编辑器", null, () => openEditor(this), { serialize: false });
            this.size = freePose ? [380, 300] : nodeData.name === "GaoshangQwenPose" ? [420, 470] : [350, 290];
            if (freePose) {
                this.title = '3D多人姿势编辑器';
                for (let i=1;i<=9;i++) {
                    const input=this.inputs?.find(input => input.name === (i===1?'reference_image':`reference_image_${i}`));
                    if(input) input.label=`${i}号人物图片`;
                }
            }
            return result;
        };
        if (!freePose) return;
        // Workflows saved with a pose show it again on load; outputs are reset while loading, so wait a tick.
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configure?.apply(this, arguments);
            this.title = '3D多人姿势编辑器';
            for (let i=2;i<=9;i++) if (!this.inputs?.some(input => input.name === `reference_image_${i}`)) this.addInput(`reference_image_${i}`, 'IMAGE');
            for (let i=1;i<=9;i++) { const input=this.inputs?.find(input => input.name === (i===1?'reference_image':`reference_image_${i}`)); if(input) input.label=`${i}号人物图片`; }
            setTimeout(() => void showPoseImage(this).catch(() => {}), 600);
            return result;
        };
    },
});

// Auto-load flow: when GaoshangQwenPose builds a scene from a connected
// POSE_KEYPOINT (scene_json was empty), echo it into the widget so the editor
// opens with the photo's pose. A scene applied from the editor is never
// overwritten; re-running refreshes an auto scene on a new photo.
api.addEventListener("executed", event => {
    const detail = event.detail || {};
    const scene = detail.output?.gaoshang_auto_scene?.[0];
    if (typeof scene === "string" && scene.trim()) {
        const node = (app.graph._nodes ?? []).find(n => n.comfyClass === "GaoshangQwenPose");
        if (node) {
            const target = widget(node, "scene_json");
            if (target) {
                node.properties = node.properties || {};
                const current = String(target.value || "").trim();
                if (!(current && current !== "{}" && !node.properties.gaoshang_auto)) {  // editor-applied wins
                    target.value = scene;
                    node.properties.gaoshang_auto = true;
                    target.callback?.(scene);
                    node.setDirtyCanvas(true, true);
                }
            }
        }
    }
    // FreePose 画幅自动跟 1 号输入图（16 对齐）：未应用过编辑器状态时回写 width/height
    // 控件——编辑器画布、输出 latent、白模 latent 三者始终同幅同比例。
    const autoSize = detail.output?.gaoshang_auto_size?.[0];
    if (Array.isArray(autoSize) && autoSize.length === 2) {
        for (const node of app.graph._nodes ?? []) {
            if (node.comfyClass !== "GaoshangQwenFreePose") continue;
            if (node.properties?.gaoshang_user_posed) continue;  // 用户应用过的画幅优先
            for (const [name, value] of [["width", autoSize[0]], ["height", autoSize[1]]]) {
                const item = widget(node, name);
                if (item && Number(item.value) !== Number(value)) {
                    item.value = value;
                    item.callback?.(value);
                }
            }
            node.setDirtyCanvas(true, true);
        }
    }
});

