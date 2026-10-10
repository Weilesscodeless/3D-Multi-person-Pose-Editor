// Free-pose editor: one VNCCS MakeHuman mannequin. Editing uses a free orbit view;
// the output is always the VNCCS front capture (yaw 0 / pitch 0, white background,
// flat white ambient light), which is what the VNCCS_QI2_PoseStudio LoRA was trained on.
import { PoseViewerCore } from '../vnccs/vnccs_pose_studio_core.mjs';
import { loadMorphPack, solveMorph, buildStaticModelData } from '../vnccs/vnccs_pose_morph_runtime.mjs';
import { HAND_PRESETS } from '../vnccs/vnccs_hand_presets.mjs';
import { readSkeletonImage } from './pose-import.mjs';
import { liftOpenPose, WORLD_KEYPOINT_NAMES } from './openpose-lift.mjs';

const $ = selector => document.querySelector(selector);
const embedded = new URLSearchParams(location.search).has('embedded');
const INSTRUCTION = 'Draw character from image2';
const FRONT = { yaw: 0, pitch: 0 };
const CAPTURE_BACKGROUND = [255, 255, 255];
const CAPTURE_LIGHTS = [{ type: 'ambient', color: '#ffffff', intensity: 1.0 }];
const EDITOR_VERSION = 'v52';
const versionChip = $('#editor-version');
if (versionChip) versionChip.textContent = EDITOR_VERSION;
const DEFAULT_MESH = { age: 25, gender: 0.5, weight: 0.5, muscle: 0.5, height: 0.5, breast_size: 0, firmness: 0.5, show_genitals: false };
const BODY_SLIDERS = [
    ['gender', '性别（女 ← → 男）', 0, 1, 0.01], ['age', '年龄', 1, 90, 1], ['height', '身高', 0, 1, 0.01],
    ['weight', '体重', 0, 1, 0.01], ['muscle', '肌肉', 0, 1, 0.01],
];
const NEUTRAL_TRANSFORM = { x: 0, y: 0, z: 0, zoom: 1 };
// Head / neck / torso / limb proportions as scale factors (1 = MakeHuman default).
const PROPORTIONS = [
    ['head', '头部大小', 0.75, 1.3], ['neck', '脖子长度', 0.5, 2],
    ['shoulder', '肩宽', 0.7, 1.4], ['spine', '躯干长度', 0.7, 1.4],
    ['upper_arm', '上臂长度', 0.6, 1.6], ['forearm', '小臂长度', 0.6, 1.6],
    ['thigh', '大腿长度', 0.6, 1.6], ['shin', '小腿长度', 0.6, 1.6],
];
const DEFAULT_PROPORTIONS = Object.fromEntries(PROPORTIONS.map(([key]) => [key, 1]));
// VNCCS bone-length groups (both sides together); the core maps a group value v to scale 0.5 + v.
const PROPORTION_GROUPS = {
    shoulder: ['shoulder_l', 'shoulder_r'], spine: ['spine'],
    upper_arm: ['upper_arm_l', 'upper_arm_r'], forearm: ['forearm_l', 'forearm_r'],
    thigh: ['thigh_l', 'thigh_r'], shin: ['shin_l', 'shin_r'],
};
// Our 2D joint names → MakeHuman bones whose head sits on that joint.
const JOINT_BONES = { ls: 'upperarm_l', le: 'lowerarm_l', lw: 'hand_l', rs: 'upperarm_r', re: 'lowerarm_r', rw: 'hand_r', lh: 'thigh_l', lk: 'calf_l', la: 'foot_l', rh: 'thigh_r', rk: 'calf_r', ra: 'foot_r' };
// The MakeHuman body pack ships uncompressed: cloud drives (网盘) delete archive files such as the
// upstream .bin.gz, which left the editor stuck on loading. The .gz is only a fallback for old copies;
// the VNCCS loader accepts both (it checks the gzip magic bytes).
const BODY_PACK_URLS = ['pose_studio_makehuman.v2.bin', 'pose_studio_makehuman.v2.bin.gz'].map(name => new URL(`../vnccs/assets/${name}`, import.meta.url));

async function loadBodyPack() {
    for (const url of BODY_PACK_URLS) {
        try { return await loadMorphPack(url); } catch { /* missing or truncated: try the next copy */ }
    }
    throw new Error('人体数据文件缺失或不完整（web/vnccs/assets/pose_studio_makehuman.v2.bin，约 86MB）。可能被网盘删除或没下载完整，请从 GitHub 重新下载插件：插件安装包');
}

let doc = { mesh: { ...DEFAULT_MESH }, proportions: { ...DEFAULT_PROPORTIONS }, pose: null, transform: { ...NEUTRAL_TRANSFORM }, width: 1024, height: 1024, openpose: null };
const PERSON_COLORS = ['#e53935','#1e88e5','#43a047','#fdd835','#8e24aa','#fb8c00','#00acc1','#ec407a','#795548'];
const COLOR_NAMES = ['red','blue','green','yellow','purple','orange','cyan','pink','brown'];
let people = [doc], activePerson = 0, referencePreviews = [];
let extraPrompt = '';
let pack = null;
let ready = false;
let selectedBoneName = null;
let previewTimer = null;
let morphTimer = null;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const round16 = value => clamp(Math.round(value / 16) * 16, 64, 4096);
const promptText = () => [people.length === 1 ? INSTRUCTION : `Create one coherent scene with exactly ${people.length} people. Use image1 as the pose and composition guide.\n` + people.map((_,i) => `Person ${i+1}: replace the ${COLOR_NAMES[i]} mannequin in image1 with the character from image${i+2}. Preserve that character's face, hairstyle and clothing; match that mannequin's pose, position, scale and facing direction.`).join('\n') + '\nKeep all identities separate. Render natural people, without mannequin colors, labels or extra people.', ...extraPrompt.split('\n').map(line => line.trim())].filter(Boolean).join('\n');

function toast(text) {
    const element = $('#toast');
    element.textContent = text;
    element.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => element.classList.remove('show'), 2400);
}

const canvas = $('#viewport-canvas');
const stage = $('#stage');
canvas.width = stage.clientWidth || 800;
canvas.height = stage.clientHeight || 600;
const viewer = new PoseViewerCore(canvas, {
    skinMode: 'naked',
    enableTextureSkinning: true,
    showSkeletonHelper: true,
    showCaptureFrame: true,
    syncMode: 'end',
    useHandControlPopover: false,
    captureHistoryContext: () => ({ transform: { ...doc.transform } }),
    onHistoryRestore: pose => {
        if (pose.editorState) {
            doc.transform = { ...pose.editorState.transform };
            viewer.setActiveCharacterAppearance({ transform: doc.transform });
            updateCamera(false);
        }
        refreshControls();
    },
    onBoneSelectionChange: ({ boneName }) => { selectedBoneName = boneName; refreshBoneSliders(); },
    onPoseChange: () => { refreshControls(); schedulePreview(); },
});
viewer.maxHistory = 60; // match the studio editor; VNCCS keeps only 10

function modelData(morph, staticData) {
    const bones = staticData.bones.map((bone, index) => {
        const headPos = Array.from(morph.bonePositions.subarray(index * 6, index * 6 + 3));
        const tailPos = Array.from(morph.bonePositions.subarray(index * 6 + 3, index * 6 + 6));
        return { name: bone.name, parent: bone.parent || null, headPos, tailPos, length: Math.hypot(...tailPos.map((v, i) => v - headPos[i])) };
    });
    return {
        vertices: morph.vertices, uvs: staticData.uvs, indices: staticData.indices, bones,
        skinIndices: staticData.skinIndices, skinWeights: staticData.skinWeights,
        landmarks: morph.landmarks || {}, landmark_indices: morph.landmarkIndices || {},
    };
}

// Absolute IK positions depend on bone lengths, so a body-shape change keeps only rotations.
function rotationsOnly(pose) {
    const { bonePositions, ikEffectorPositions, poleTargetPositions, hipBonePosition, camera, cameraParams, ...rest } = pose || {};
    return rest;
}

function savedPose() {
    const { camera, cameraParams, ...pose } = viewer.getPose();
    return pose;
}

// The core has no neck group, so the neck scales the head bone's offset from neck_01
// the same way its limb groups scale child offsets, and caches it as the new rest.
function applyNeck() {
    viewer._setBoneOffsetScale('head', doc.proportions.neck);
    for (const bone of viewer.boneList) bone.updateMatrixWorld(true);
    viewer.skeleton?.update();
    viewer._cacheShapedRestBonePositions(['head']);
    viewer.updateIKEffectorPositions?.();
    viewer.requestRender();
}

function setProportion(key, value) {
    doc.proportions = { ...doc.proportions, [key]: value };
    if (key === 'head') viewer.updateHeadScale(value);
    else if (key === 'neck') applyNeck();
    else for (const group of PROPORTION_GROUPS[key]) viewer.updateBoneLengthScale(group, value - 0.5);
}

// resetPose() restores the core's own groups but not our neck offset.
function resetPose() {
    viewer.resetPose();
    applyNeck();
}

function loadModel(pose) {
    const morph = solveMorph(pack, doc.mesh, doc.mesh.keepHeight ?? null);
    viewer.setSkinMode('naked');
    // loadData() applies these cached scales while it builds the new rig.
    viewer.headScale = doc.proportions.head;
    viewer.boneLengthParams = {
        ...viewer.boneLengthParams,
        ...Object.fromEntries(Object.entries(PROPORTION_GROUPS).flatMap(([key, groups]) => groups.map(group => [group, doc.proportions[key] - 0.5]))),
    };
    viewer.loadData(modelData(morph, buildStaticModelData(pack, morph.includeGenitals)), true);
    applyNeck();
    viewer.updateLights(CAPTURE_LIGHTS);
    viewer.setPose(pose || {}, true);
    viewer.setActiveCharacterAppearance({ color: PERSON_COLORS[activePerson], transform: doc.transform });
}

// The capture camera never moves; `snap` also brings the free editing view back to it.
function updateCamera(snap) {
    const args = [doc.width, doc.height, 1, 0, 0, FRONT.yaw, FRONT.pitch];
    if (snap) viewer.snapToCaptureCamera(...args);
    else viewer.updateCaptureCamera(...args);
    refreshControls();
    schedulePreview();
}

function fitFrame(snap = false) {
    viewer.setActiveCharacterAppearance({ transform: NEUTRAL_TRANSFORM });
    const framing = viewer.computeModelFitFraming(doc.width, doc.height, FRONT.yaw, FRONT.pitch, 0.08);
    const pivot = viewer.sceneCameraTarget;
    if (framing && pivot) {
        const zoom = framing.zoom;
        doc.transform = {
            x: clamp((1 - zoom) * pivot.x + zoom * framing.offsetX, -50, 50),
            y: clamp((1 - zoom) * pivot.y + zoom * framing.offsetY, -50, 50),
            z: clamp((1 - zoom) * pivot.z, -40, 40),
            zoom,
        };
    }
    viewer.setActiveCharacterAppearance({ transform: doc.transform });
    updateCamera(snap);
}

function setPersonColor(index, color) {
    if (index === activePerson) {
        viewer.setActiveCharacterAppearance({ color, transform: people[index].transform });
    } else {
        viewer.upsertPassiveCharacterFromActive(String(index), {
            pose: people[index].pose || {}, color, transform: people[index].transform,
        });
    }
}

async function capture(width, height, withOverlay = false) {
    viewer.updateLights(CAPTURE_LIGHTS);
    // 单人保持白模惯例；多人输出仍用红蓝区分人物，轮廓靠加粗描边保证可读。
    if (people.length === 1) viewer.setActiveCharacterAppearance({color:'#ffffff'});
    try { return viewer.capture(width, height, 1, CAPTURE_BACKGROUND, 0, 0, FRONT.yaw, FRONT.pitch); }
    finally { viewer.setActiveCharacterAppearance({color:PERSON_COLORS[activePerson]}); }
}

// 输出图描粗轮廓：Sobel 提边 + 膨胀加粗 + 描深（色块人偶的剪影/遮挡边界变清晰）。
async function drawBoldEdges(dataUrl, width, height) {
    const image = new Image();
    image.src = dataUrl;
    try { await image.decode(); } catch { return dataUrl; }
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, width, height);
    const frame = ctx.getImageData(0, 0, width, height);
    const data = frame.data;
    const luminance = new Float32Array(width * height);
    for (let index = 0; index < width * height; index++) {
        luminance[index] = 0.299 * data[index * 4] + 0.587 * data[index * 4 + 1] + 0.114 * data[index * 4 + 2];
    }
    const edge = new Uint8Array(width * height);
    const threshold = 36;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const at = (yy, xx) => luminance[yy * width + xx];
            const gx = -at(y - 1, x - 1) - 2 * at(y, x - 1) - at(y + 1, x - 1) + at(y - 1, x + 1) + 2 * at(y, x + 1) + at(y + 1, x + 1);
            const gy = -at(y - 1, x - 1) - 2 * at(y - 1, x) - at(y - 1, x + 1) + at(y + 1, x - 1) + 2 * at(y + 1, x) + at(y + 1, x + 1);
            if (Math.hypot(gx, gy) > threshold) edge[y * width + x] = 1;
        }
    }
    // 膨胀两圈加粗
    const bold = new Uint8Array(edge);
    for (let pass = 0; pass < 2; pass++) {
        const source = new Uint8Array(bold);
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                if (!bold[y * width + x]) {
                    if (source[y * width + x - 1] || source[y * width + x + 1] || source[(y - 1) * width + x] || source[(y + 1) * width + x]) bold[y * width + x] = 1;
                }
            }
        }
    }
    for (let index = 0; index < width * height; index++) {
        if (bold[index]) {
            data[index * 4] = Math.round(data[index * 4] * 0.25);
            data[index * 4 + 1] = Math.round(data[index * 4 + 1] * 0.25);
            data[index * 4 + 2] = Math.round(data[index * 4 + 2] * 0.25);
        }
    }
    ctx.putImageData(frame, 0, 0);
    return canvas.toDataURL('image/png');
}

// 蒙皮之上叠标准 OpenPose 配色骨架：保留体积感 + 下游可读的肢段结构。
async function drawColoredSkeletonOverlay(dataUrl, width, height) {
    const THREE = viewer.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = new Image();
    image.src = dataUrl;
    try { await image.decode(); } catch { return dataUrl; }
    ctx.drawImage(image, 0, 0, width, height);
    const camera = viewer.captureCamera;
    if (!camera || !THREE) return dataUrl;
    camera.updateMatrixWorld?.();
    const project = v3 => { const p = v3.clone().project(camera); return [(p.x + 1) / 2 * width, (1 - p.y) / 2 * height]; };
    const ORDER_NAMES = ['head','neck','rs','re','rw','ls','le','lw','rh','rk','ra','lh','lk','la'];
    const LIMB_PAIRS = [[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,0]];
    const PALETTE = ['#ff0000','#ff5500','#ffaa00','#ffff00','#aaff00','#55ff00','#00ff00','#00ff55','#00ffaa','#00ffff','#00aaff','#0055ff','#0000ff','#5500ff'];
    const lineWidth = Math.max(2, Math.round(Math.min(width, height) / 95));
    const dotRadius = Math.max(3, Math.round(Math.min(width, height) / 110));
    const keepActive = activePerson;
    let drawn = 0;
    for (let index = 0; index < people.length; index++) {
        try {
            if (index !== keepActive) selectPerson(index);
            viewer.skinnedMesh.updateMatrixWorld(true);
            const joints = {};
            for (const [key, boneName] of Object.entries(JOINT_BONES)) {
                const bone = viewer.bones[boneName];
                if (bone) joints[key] = bone.getWorldPosition(new THREE.Vector3());
            }
            const headBone = viewer.bones.head || viewer.bones['head'];
            if (headBone) joints.head = headBone.getWorldPosition(new THREE.Vector3());
            if (joints.ls && joints.rs) joints.neck = joints.ls.clone().lerp(joints.rs, 0.5);
            if (joints.lh && joints.rh) joints.hip = joints.lh.clone().lerp(joints.rh, 0.5);
            if (!ORDER_NAMES.every(name => joints[name])) continue;
            const points = ORDER_NAMES.map(name => project(joints[name]));
            LIMB_PAIRS.forEach(([a, b], colorIndex) => {
                ctx.strokeStyle = PALETTE[colorIndex];
                ctx.lineWidth = lineWidth;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(points[a][0], points[a][1]);
                ctx.lineTo(points[b][0], points[b][1]);
                ctx.stroke();
            });
            points.forEach((point, dotIndex) => {
                ctx.fillStyle = PALETTE[dotIndex];
                ctx.beginPath();
                ctx.arc(point[0], point[1], dotRadius, 0, Math.PI * 2);
                ctx.fill();
            });
            drawn++;
        } catch (error) { console.warn('[高尚] 骨架叠加失败（人 ' + (index + 1) + '）:', error); }
    }
    if (activePerson !== keepActive) selectPerson(keepActive);
    return drawn ? canvas.toDataURL('image/png') : dataUrl;
}

function schedulePreview() {
    if (!ready) return;
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
        await viewer.waitForCaptureReady();
        const scale = Math.min(1, 480 / Math.max(doc.width, doc.height));
        const url = await drawBoldEdges(await capture(Math.round(doc.width * scale), Math.round(doc.height * scale), true), Math.round(doc.width * scale), Math.round(doc.height * scale));
        if (url) { $('#capture-preview').src = url; $('#inset-preview').src = url; }
    }, 300);
}

// --- OpenPose one-click posing ---------------------------------------------

const worldOf = name => viewer.bones[name].getWorldPosition(new viewer.THREE.Vector3()).toArray();

function restJoints() {
    const rest = Object.fromEntries(Object.entries(JOINT_BONES).map(([key, bone]) => [key, worldOf(bone)]));
    rest.neck = rest.ls.map((v, i) => (v + rest.rs[i]) / 2);
    rest.hipMid = rest.lh.map((v, i) => (v + rest.rh[i]) / 2);
    return { rest, head: worldOf('head'), pelvis: worldOf('pelvis') };
}

function applyOpenPose() {
    const { points, flips } = doc.openpose;
    viewer.recordState();
    doc.transform = { ...NEUTRAL_TRANSFORM };
    viewer.setActiveCharacterAppearance({ transform: doc.transform });
    resetPose();
    viewer.skinnedMesh.updateMatrixWorld(true);
    const { rest, head, pelvis } = restJoints();
    const { kps, facingAway } = liftOpenPose(points, rest, flips);
    // The spine IK target is the head bone origin, not the nose: keep the nose direction at head-bone distance.
    const headDistance = Math.hypot(...head.map((v, i) => v - rest.neck[i]));
    const direction = kps.head.map((v, i) => v - kps.neck[i]);
    const norm = Math.hypot(...direction) || 1;
    kps.head = kps.neck.map((v, i) => v + direction[i] / norm * headDistance);
    const THREE = viewer.THREE;
    const worldKps = Object.fromEntries(Object.entries(WORLD_KEYPOINT_NAMES).map(([key, name]) => [name, new THREE.Vector3(...kps[key].map((v, i) => v + pelvis[i]))]));
    // Hip sockets, head, hands and feet have no reliable 2D source; keep the mannequin's own.
    viewer.applyWorldKeypointImport(worldKps, { drawFigure: false, placeHipRoots: false, alignHead: false, alignHands: false, alignFeet: false, dispatchPoseChange: false });
    fitFrame(true);
    refreshFlips(facingAway);
    return facingAway;
}

async function importEntry(entry) {
    const status = $('#import-status');
    for (const button of document.querySelectorAll('.fp-thumb')) button.classList.toggle('active', button.dataset.key === entry.key);
    const label = entry.key.startsWith('builtin:') ? `${SOURCE_NAMES.builtin} ${entryLabel(entry)}` : entry.name;
    try {
        const image = new Image();
        image.src = entry.url;
        await image.decode();
        const people = readSkeletonImage(image);
        // Single-person editor: take the tallest skeleton.
        const height = person => Math.max(...Object.values(person).map(p => p[1])) - Math.min(...Object.values(person).map(p => p[1]));
        const person = people.slice().sort((a, b) => height(b) - height(a))[0];
        doc.openpose = { key: entry.key, name: label, points: person, flips: {} };
        const facingAway = applyOpenPose();
        const notes = [facingAway ? '识别为背面' : '识别为正面'];
        if (people.length > 1) notes.push(`图中 ${people.length} 人，已取最大的一个`);
        if (people.warnings?.length) notes.push('遮挡关节已近似补全');
        status.textContent = `已按「${label}」摆好：${notes.join('，')}。前后不对可用下方深度修正。`;
    } catch (error) {
        entry.failed = true;
        document.querySelector(`.fp-thumb[data-key="${CSS.escape(entry.key)}"]`)?.classList.add('failed');
        status.textContent = `「${label}」无法识别：${error.message}`;
    }
}

// 多人 JSON 一键摆姿：{people:[{pose_keypoints_2d: 18×3 OpenPose}], canvas_width, canvas_height}
// 每个人物自动建一个人偶并套用姿势（DWPose / SAM3D 转出的 JSON 均可）。
// 站位按照片里的真实水平位置摆放（相邻的人保持相邻），不缩放、不网格铺开。
const OP18_TO_GAOSHANG = ['head','neck','rs','re','rw','ls','le','lw','rh','rk','ra','lh','lk','la'];
// SAM3D 真 3D：OpenPose18 世界坐标(Y上,米) 直接喂 viewer，绕过 2D 抬升。
function applyWorldPose(world3d) {
    viewer.recordState();
    doc.transform = { ...NEUTRAL_TRANSFORM };
    viewer.setActiveCharacterAppearance({ transform: doc.transform });
    resetPose();
    viewer.skinnedMesh.updateMatrixWorld(true);
    const { rest, pelvis } = restJoints();
    const THREE = viewer.THREE;
    const worldKps = Object.fromEntries(Object.entries(WORLD_KEYPOINT_NAMES).map(([key, name]) => {
        let source;
        const slot = OP18_TO_GAOSHANG.indexOf(key);
        if (slot >= 0) source = world3d[slot];
        else if (key === 'hipMid') source = [
            (world3d[8][0] + world3d[11][0]) / 2,
            (world3d[8][1] + world3d[11][1]) / 2,
            (world3d[8][2] + world3d[11][2]) / 2,
        ];
        else source = [0, 0, 0];
        return [name, new THREE.Vector3(source[0] + pelvis[0], source[1] + pelvis[1], source[2] + pelvis[2])];
    }));
    viewer.applyWorldKeypointImport(worldKps, { drawFigure: false, placeHipRoots: false, alignHead: false, alignHands: false, alignFeet: false, dispatchPoseChange: false });
}
async function importMultiPersonJSON(data) {
    const status = $('#import-status');
    try {
        const raw = (data.people || []).map(person => person.pose_keypoints_2d || []).filter(kp => Array.isArray(kp) && kp.length >= 17 * 3);
        if (!raw.length) throw new Error('JSON 中没有可用的 pose_keypoints_2d');
        const canvasWidth = Number(data.canvas_width) || 1024;
        const entries = (data.people || []).filter(person => Array.isArray(person.pose_keypoints_2d) && person.pose_keypoints_2d.length >= 17 * 3);
        const built = entries.map(entry0 => {
            const kp = entry0.pose_keypoints_2d;
            const joints = {};
            OP18_TO_GAOSHANG.forEach((name, slot) => {
                const x = Number(kp[slot * 3]), y = Number(kp[slot * 3 + 1]), c = Number(kp[slot * 3 + 2]);
                if (Number.isFinite(x) && Number.isFinite(y) && c > 0.1) joints[name] = [x, y];
            });
            if (!OP18_TO_GAOSHANG.every(name => joints[name])) return null;
            const xs = Object.values(joints).map(point => point[0]);
            const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
            // 片内身高：头到脚的像素高，用来做相对大小（远的人小 → 纵深）
            const feetY = Math.max(joints.la[1], joints.ra[1], joints.lk[1], joints.rk[1]);
            const camT = Array.isArray(entry0.cam_t) && entry0.cam_t.length === 3 ? entry0.cam_t : null;
            const w3 = Array.isArray(entry0.pose_world3d) && entry0.pose_world3d.length === 54 ? entry0.pose_world3d : null;
            const origin = Array.isArray(entry0.origin) && entry0.origin.length === 3 ? entry0.origin : null;
            return {
                joints, frameX: (meanX / canvasWidth) * 2 - 1,
                heightPx: Math.max(1, feetY - joints.head[1]),
                camT, w3, origin,
                worldPos: Array.isArray(entry0.world_pos) && entry0.world_pos.length === 3 ? entry0.world_pos : null,
                body: entry0.body && typeof entry0.body === 'object' ? entry0.body : null,
                // 可见度三态：true=全身可见（重建可信）/false=被遮挡（大小采用组内可见者基准）/null=无数据
                fullyVisible: entry0.fully_visible === true ? true : (entry0.fully_visible === false ? false : null),
                sam3d: entry0.sam3d && typeof entry0.sam3d === 'object' ? entry0.sam3d : null,
            };
        }).filter(item => OP18_TO_GAOSHANG.every(name => item.joints[name]));
        if (!built.length) throw new Error('没有 14 关节齐全的人物（低置信关节太多）');
        if (built.length > 9) built.length = 9;
        if (!ready) throw new Error('人偶模型尚未就绪，请稍候再试');
        // 核弹清场：清掉 viewer 里所有游离被动角色（历史残留的"多余骨架"），
        // people 裁回 1 人再按数据重建，杜绝任何陈旧实体。
        if (viewer.clearPassiveCharacters) viewer.clearPassiveCharacters();
        while (people.length > 1) removePerson();
        while (people.length < built.length) addPerson();
        for (let index = 0; index < built.length; index++) {
            selectPerson(index);
            if (built[index].body) {
                // 体型/性别自动套用（检测链估计）：改 doc.mesh 后重建人偶再摆姿势
                const bm = built[index].body;
                const mesh = { ...doc.mesh };
                for (const k of ['gender', 'height', 'weight', 'muscle'])
                    if (Number.isFinite(Number(bm[k]))) mesh[k] = Number(bm[k]);
                mesh.keepHeight = null;  // 检测估计的身高为权威，解除旧锚
                doc.mesh = mesh;
                loadModel(savedPose());
            }
            doc.openpose = { key: 'json:' + (index + 1), name: 'JSON ' + (index + 1) + ' 号', points: built[index].joints, flips: {} };
            let applied = false;
            if (built[index].sam3d) {
                try {
                    applied = viewer.applySAM3DImport(built[index].sam3d, 0, { dispatchPoseChange: false });
                    // applySAM3DImport 内部默认画调试火柴人（hmr2 线框，点不到也非角色），
                    // 导入完立刻清掉，只留人偶本体。
                    if (applied && viewer._hmr2FigureGroup) {
                        viewer._clearImportedFigureGroup?.('_hmr2FigureGroup');
                        viewer._hmr2FigureGroup = null;
                    }
                } catch (error) { console.warn('[高尚] SAM3D 刚性导入失败，退回 2D:', error); }
            }
            if (!applied) applyOpenPose();
        }
        // 站位与纵深：水平间距向中心压缩（相邻保持相邻），整体留边不贴框；
        // 每人大小 = 组基准 × 片内相对身高（片里小的人站得远，自带纵深）。
        const frameXs = built.map(item => item.frameX);
        const center = frameXs.reduce((sum, value) => sum + value, 0) / frameXs.length;
        const maxHeight = Math.max(...built.map(item => item.heightPx));
        const allOrigin = built.every(item => item.origin);
        const allCam = built.every(item => item.camT);
        if (allOrigin) {
            // 单一相机透视一致落位：大小/纵深/横向全部由原图几何决定，讲同一个镜头的故事。
            //  · 大小 = 片内像素高占比（近大远小，还原照片透视）
            //  · 纵深 = 大小反推（size ∝ 1/d：片内矮一半 ≈ 后退一倍距离，k 米标定）
            //  · 横向 = 原图像素位置 → 框坐标（近的按其真实框位、远的随其后退略收拢）
            const canvasHeight = Number(data.canvas_height) || maxHeight;
            const canvasWidth = Number(data.canvas_width) || 1024;
            const DEPTH_K = 3.2; // 米：片内身高每缩到 1/n，后退约 k*(n-1)
            // 单位换算：MakeHuman 网格原生 14.33 单位高（≈1 分米/单位），transform.x/z
            // 是网格原生单位不是米——直接给米会小 8 倍（人偶 12.9 单位高、间距 1.6 单位=贴脸）。
            const UNITS_PER_METER = 8.2;
            const maxPx = Math.max(...built.map(item => item.heightPx));
            // 同距判定：真实深度（cam_t z）齐全且组内差 < 0.6m = 紧抱/并排同距。
            // 此时片内高度差是真实身高差不是距离差 → 不做"矮的退后"、不透视收拢，
            // 大小差如实等于身高差；有远近的普通合影（z 差大）仍走透视反推。
            const camZs = built.map(item => (item.camT && Number.isFinite(item.camT[2])) ? item.camT[2] : null);
            const sameDepth = camZs.every(z => z !== null) && (Math.max(...camZs) - Math.min(...camZs)) < 0.6;
            // 真实站位（共享相机 cam_t 横向，米）→ 网格单位（随各自 zoom 缩放），
            // 直接采用计算结果，不做最小间距拉开——紧抱就呈现紧抱。
            const haveW = built.every(item => Array.isArray(item.worldPos));
            let placeX = null;
            if (haveW) {
                const zoomOf = i => Math.max(0.3, Math.min(1.1,
                    Math.max(0.2, Math.min(1.05, built[i].heightPx / canvasHeight)) / 0.8));
                placeX = built.map((item, i) => item.worldPos[0] * UNITS_PER_METER * zoomOf(i));
            }
            // 组内大小基准：全身可见者（掩膜高/脸高在人类区间）的重建可信，保留各自
            // 投影 zoom；被遮挡者的重建整体缩水（00096 骨长实测），zoom 采用组内可见
            // 者的最大基准——两个等大成年人呈现等大，可见者的几何一分不改。
            const ownZoomOf = i => Math.max(0.3, Math.min(1.1,
                Math.max(0.2, Math.min(1.05, built[i].heightPx / canvasHeight)) / 0.8));
            const visibleZooms = built
                .map((item, i) => item.fullyVisible === true ? ownZoomOf(i) : null)
                .filter(z => z !== null);
            const refZoom = visibleZooms.length ? Math.max(...visibleZooms) : null;
            for (let index = 0; index < built.length; index++) {
                selectPerson(index);
                const fraction = Math.max(0.2, Math.min(1.05, built[index].heightPx / canvasHeight));
                const depthScale = maxPx / Math.max(built[index].heightPx, 1);
                const zDepth = sameDepth ? 0 : (depthScale - 1) * DEPTH_K; // 米
                const rawX = haveW ? placeX[index]
                    : built[index].frameX * (sameDepth ? 1 : (1 - Math.min(0.35, zDepth * 0.12))); // 透视收拢
                const bound = haveW ? 20 : 2.5;
                const rawZ = haveW ? zDepth * UNITS_PER_METER : zDepth;
                const zoom = (built[index].fullyVisible === false && refZoom !== null) ? refZoom : ownZoomOf(index);
                doc.transform = {
                    ...doc.transform,
                    x: Math.max(-bound, Math.min(bound, rawX)),
                    z: Math.max(-bound, Math.min(bound, rawZ)),
                    zoom,
                };
                viewer.setActiveCharacterAppearance({ transform: doc.transform });
                saveActivePerson();
            }
        } else if (allCam) {
            // SAM3D 真实站位：cam_t(相机系: x右 y下 z前,米) -> 世界(x平移, z纵深)；
            // 姿势仍走已验证的 2D 抬升，间距与远近来自真实三维位置。
            const meanX = built.reduce((sum, item) => sum + item.camT[0], 0) / built.length;
            const minZ = Math.min(...built.map(item => item.camT[2]));
            const offsets = built.map(item => item.camT[0] - meanX);
            const widen = 1;
            for (let index = 0; index < built.length; index++) {
                selectPerson(index);
                doc.transform = {
                    ...doc.transform,
                    x: Math.max(-2.5, Math.min(2.5, offsets[index])),
                    z: Math.max(0, built[index].camT[2] - minZ),
                    zoom: (built.length === 1 ? 1 : Math.max(0.34, 1.28 / built.length)) * (built[index].heightPx / maxHeight),
                };
                viewer.setActiveCharacterAppearance({ transform: doc.transform });
                saveActivePerson();
            }
        } else {
            const zoomBase = built.length === 1 ? 1 : Math.max(0.34, 1.28 / built.length);
            for (let index = 0; index < built.length; index++) {
                selectPerson(index);
                const framePoint = Math.max(-0.42, Math.min(0.42, center + (built[index].frameX - center) * 0.55));
                const target = viewer.characterTranslationForFramePoint(framePoint, 0);
                doc.transform = { ...doc.transform, ...(target || {}), zoom: zoomBase * (built[index].heightPx / maxHeight) };
                viewer.setActiveCharacterAppearance({ transform: doc.transform });
                saveActivePerson();
            }
        }
        selectPerson(0);
        updateCamera(true);
        status.textContent = `已按 JSON 摆好 ${built.length} 个人偶（站位按照片），可逐个微调或修正前后朝向。`;
    } catch (error) {
        status.textContent = '多人 JSON 导入失败：' + error.message;
    }
}
const resam3dButton = $('#resam3d-button');
if (resam3dButton) resam3dButton.onclick = async () => {
    $('#import-status').textContent = '正在拉取 SAM3D 多人骨架…';
    try {
        const response = await fetch('/gaoshang_pose/sam3d_multi');
        const data = await response.json();
        if (!data.people?.length) throw new Error('暂无数据，请先跑一次工作流');
        await importMultiPersonJSON(data);
    } catch (error) {
        $('#import-status').textContent = 'SAM3D 重摆失败：' + error.message;
    }
};
// 人物形象预设：改 doc.mesh → 整个 rig 即时重建（姿势保留），肉眼立辨生效。
const BUILDS = { slim: { weight: 0.25, muscle: 0.3 }, standard: { weight: 0.5, muscle: 0.5 }, strong: { weight: 0.72, muscle: 0.85 } };
// 切性别只换身材比例、身高不变：锚定当前身高，solveMorph 层等比缩放顶回。
const solveHeight = params => {
    if (!pack) return null;
    const v = solveMorph(pack, params).vertices;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 1; i < v.length; i += 3) { if (v[i] < minY) minY = v[i]; if (v[i] > maxY) maxY = v[i]; }
    return maxY - minY;
};
// keepHeight 存在每人 doc.mesh 里（随存档持久化、切人不串）：首次改性别/年龄时
function applyBodyPreset(patch) {
    if (patch.gender != null && patch.gender !== doc.mesh.gender && doc.mesh.keepHeight == null)
        doc.mesh = { ...doc.mesh, keepHeight: solveHeight(doc.mesh) };
    doc.mesh = { ...doc.mesh, ...patch };
    loadModel(savedPose());
    saveActivePerson();
    schedulePreview();
    $('#import-status').textContent = '人物形象已更新（' + (activePerson + 1) + ' 号人物）：性别 ' + (doc.mesh.gender > 0.7 ? '男' : doc.mesh.gender < 0.3 ? '女' : '中性') + '，体型 ' + (doc.mesh.muscle > 0.7 ? '壮硕' : doc.mesh.muscle < 0.4 ? '偏瘦' : '标准');
}
document.querySelectorAll('#body-gender button').forEach(button => {
    button.onclick = () => applyBodyPreset({ gender: Number(button.dataset.gender) });
});
document.querySelectorAll('#body-build button').forEach(button => {
    button.onclick = () => applyBodyPreset(BUILDS[button.dataset.build] || {});
});

const multiJsonInput = $('#multi-json-input');
if (multiJsonInput) multiJsonInput.addEventListener('change', async event => {
    const file = event.target.files && event.target.files[0];
    event.target.value = '';
    if (!file) return;
    try { importMultiPersonJSON(JSON.parse(await file.text())); }
    catch (error) { $('#import-status').textContent = 'JSON 解析失败：' + error.message; }
});

function refreshFlips(facingAway) {
    $('#depth-section').hidden = !doc.openpose;
    const flips = doc.openpose?.flips || {};
    for (const button of document.querySelectorAll('[data-flip]')) button.classList.toggle('on', Boolean(flips[button.dataset.flip]));
    if (facingAway !== undefined) $('[data-flip=body]').textContent = facingAway ? '背面 → 改正面' : '正面 → 改背面';
}

for (const button of document.querySelectorAll('[data-flip]')) {
    button.onclick = () => {
        if (!doc.openpose) return;
        const key = button.dataset.flip;
        doc.openpose.flips = { ...doc.openpose.flips, [key]: !doc.openpose.flips[key] };
        applyOpenPose();
    };
}

// Two sources: the plugin's built-in 高尚姿势图库 skeletons (read-only) and the user's
// library in input/gaoshang_openpose (see openpose_library.py), which survives reopening
// the editor. Standalone (no ComfyUI server) the library only lasts for this page.
const LIBRARY_URL = '/gaoshang_pose/openpose_library';
const BUILTIN_URL = '/gaoshang_pose/builtin_poses';
const SOURCE_NAMES = { builtin: '高尚姿势图库', library: '我的图库' };
const galleries = { builtin: [], library: [] };
let gallerySource = 'builtin';
let libraryAvailable = false;
const byName = (a, b) => a.name.localeCompare(b.name, 'zh', { numeric: true });
const entryLabel = entry => entry.name.replace(/_bone_structure|\.(png|jpe?g|webp)$/gi, '');

function renderGallery() {
    const gallery = galleries[gallerySource];
    const filter = $('#gallery-filter').value.trim().toLowerCase();
    const shown = gallery.filter(entry => entry.name.toLowerCase().includes(filter));
    const container = $('#gallery');
    container.replaceChildren(...shown.map(entry => {
        const button = document.createElement('button');
        button.className = 'fp-thumb' + (entry.failed ? ' failed' : '') + (doc.openpose?.key === entry.key ? ' active' : '');
        button.dataset.key = entry.key;
        button.title = entry.name;
        button.innerHTML = `<img loading="lazy" alt=""><span></span>`;
        button.querySelector('img').src = entry.url;
        button.querySelector('span').textContent = entryLabel(entry);
        button.onclick = () => importEntry(entry);
        return button;
    }));
    for (const button of document.querySelectorAll('#gallery-source button')) {
        button.classList.toggle('active', button.dataset.source === gallerySource);
        button.textContent = `${SOURCE_NAMES[button.dataset.source]} ${galleries[button.dataset.source].length || ''}`.trim();
    }
    $('#gallery-count').textContent = gallery.length && filter ? `${shown.length} / ${gallery.length}` : '';
    $('#gallery-filter').hidden = gallery.length < 2;
    $('#clear-gallery').hidden = gallerySource !== 'library' || !libraryAvailable || !gallery.length;
    if (!gallery.length) container.innerHTML = gallerySource === 'builtin'
        ? '<p class="muted">内置骨架图需要在 ComfyUI 中打开编辑器才能读取。</p>'
        : '<p class="muted">选 OpenPose 骨架图（黑底彩色），点缩略图即摆好姿势。也可把图片拖到这里。选过的图会存入图库，下次打开自动载入。</p>';
}

for (const button of document.querySelectorAll('#gallery-source button')) {
    button.onclick = () => { gallerySource = button.dataset.source; renderGallery(); };
}

async function loadBuiltin() {
    try {
        const response = await fetch(BUILTIN_URL, { cache: 'no-store' });
        if (!response.ok) return;
        const { files } = await response.json();
        galleries.builtin = files.map(name => ({ key: 'builtin:' + name, name, url: `${BUILTIN_URL}/${encodeURIComponent(name)}` }));
        renderGallery();
    } catch { /* standalone preview without the ComfyUI server */ }
}

async function loadLibrary() {
    try {
        const response = await fetch(LIBRARY_URL, { cache: 'no-store' });
        if (!response.ok) return;
        const { subfolder, files } = await response.json();
        libraryAvailable = true;
        galleries.library = files.map(name => ({ key: 'library:' + name, name, url: '/view?' + new URLSearchParams({ filename: name, subfolder, type: 'input' }) })).sort(byName);
        renderGallery();
    } catch { /* standalone preview without the ComfyUI server */ }
}

async function uploadToLibrary(images) {
    const status = $('#import-status');
    let done = 0, failed = 0;
    const queue = images.slice();
    const worker = async () => {
        for (let file = queue.shift(); file; file = queue.shift()) {
            const form = new FormData();
            form.append('image', file, file.name);
            form.append('subfolder', 'gaoshang_openpose');
            form.append('type', 'input');
            form.append('overwrite', 'true');
            try { if (!(await fetch('/upload/image', { method: 'POST', body: form })).ok) failed++; }
            catch { failed++; }
            status.textContent = `正在存入图库 ${++done} / ${images.length}…`;
        }
    };
    await Promise.all(Array.from({ length: 4 }, worker));
    await loadLibrary();
    status.textContent = failed ? `${failed} 张未能存入图库` : '';
    toast(`已存入我的图库 ${images.length - failed} 张，下次打开无需重新选择`);
}

async function addFiles(files) {
    const images = [...files].filter(file => file.type.startsWith('image/') || /\.(png|jpe?g|webp)$/i.test(file.name));
    if (!images.length) { toast('没有找到图片'); return; }
    gallerySource = 'library';
    if (libraryAvailable) { await uploadToLibrary(images); return; }
    for (const entry of galleries.library) URL.revokeObjectURL(entry.url);
    galleries.library = images.map(file => ({ key: 'library:' + file.name, name: file.name, url: URL.createObjectURL(file) })).sort(byName);
    renderGallery();
    toast(`已载入 ${galleries.library.length} 张骨架图，点缩略图即可摆姿`);
}

$('#clear-gallery').onclick = async () => {
    if (!libraryAvailable || !confirm(`清空我的图库中的 ${galleries.library.length} 张骨架图？\n（删除 ComfyUI/input/gaoshang_openpose 里的副本，原文件夹和 高尚姿势图库 不受影响）`)) return;
    await fetch(LIBRARY_URL + '/clear', { method: 'POST' });
    await loadLibrary();
    toast('我的图库已清空');
};

for (const id of ['#pick-folder', '#pick-files']) {
    const input = $(id);
    input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
    // Inside the ComfyUI modal, let the host page open the picker (same as the studio editor).
    input.addEventListener('click', event => {
        const chooseInHost = window.frameElement?.gaoshangChooseFiles;
        if (!chooseInHost) return;
        event.preventDefault();
        try { chooseInHost(addFiles, { directory: id === '#pick-folder' }); }
        catch { toast('文件窗口未能打开，请把图片拖到左侧'); }
    });
}
$('#gallery-filter').addEventListener('input', renderGallery);
const galleryBox = $('#gallery');
galleryBox.addEventListener('dragover', event => { event.preventDefault(); galleryBox.classList.add('drag'); });
galleryBox.addEventListener('dragleave', () => galleryBox.classList.remove('drag'));
galleryBox.addEventListener('drop', event => { event.preventDefault(); galleryBox.classList.remove('drag'); addFiles(event.dataTransfer.files); });

// --- 我的姿势: whole editor states saved on the server (see saved_poses.py) ---

const SAVED_URL = '/gaoshang_pose/saved_poses';
let savedPoses = [];
let savedAvailable = false;
let activeSavedName = null;

function renderSaved() {
    const container = $('#saved-poses');
    $('#save-pose').disabled = !savedAvailable;
    $('#saved-count').textContent = savedPoses.length ? `${savedPoses.length} 个` : '';
    if (!savedAvailable) { container.innerHTML = '<p class="muted">在 ComfyUI 中打开编辑器才能保存姿势。</p>'; return; }
    if (!savedPoses.length) { container.innerHTML = '<p class="muted">摆好的姿势点上面的按钮存下来，下次打开点缩略图就能恢复，换工作流也能用。</p>'; return; }
    container.replaceChildren(...savedPoses.map(entry => {
        const button = document.createElement('button');
        button.className = 'fp-thumb' + (entry.name === activeSavedName ? ' active' : '');
        button.title = `${entry.name}\n点击载入`;
        button.innerHTML = '<img alt=""><span></span><i class="fp-delete" title="删除">×</i>';
        button.querySelector('img').src = entry.thumbnail;
        button.querySelector('span').textContent = entry.name;
        button.onclick = event => (event.target.closest('.fp-delete') ? deleteSavedPose(entry) : loadSavedPose(entry));
        return button;
    }));
}

async function loadSavedList() {
    try {
        const response = await fetch(SAVED_URL, { cache: 'no-store' });
        if (response.ok) {
            savedPoses = (await response.json()).poses;
            savedAvailable = true;
        }
    } catch { /* standalone preview without the ComfyUI server */ }
    renderSaved();
}

// Named after the OpenPose image it came from when there is one, else 姿势 N; never an existing name.
function defaultPoseName() {
    const taken = new Set(savedPoses.map(entry => entry.name));
    const base = doc.openpose?.name ? doc.openpose.name.replace(/[\\/:*?"<>|]/g, ' ').trim() : '姿势';
    if (doc.openpose?.name && !taken.has(base)) return base;
    for (let n = savedPoses.length + 1; ; n++) if (!taken.has(`${base} ${n}`)) return `${base} ${n}`;
}

const postPose = (name, record, overwrite) => fetch(SAVED_URL, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, record, overwrite }),
});

async function saveCurrentPose(presetName) {
    if (!savedAvailable || !ready) return;
    const name = (typeof presetName === 'string' ? presetName : prompt('给这个姿势起个名字', defaultPoseName()))?.trim();
    if (!name) return;
    await viewer.waitForCaptureReady();
    const scale = 256 / Math.max(doc.width, doc.height);
    const thumbnail = capture(Math.round(doc.width * scale), Math.round(doc.height * scale));
    updateCamera(false);
    const record = {
        version: 1, thumbnail,
        doc: { ...doc, pose: savedPose() },
        grips: { l: Number($('#grip-l').value), r: Number($('#grip-r').value) },
    };
    try {
        let response = await postPose(name, record, false);
        if (response.status === 409) {
            if (!confirm(`已有名为「${name}」的姿势，要覆盖吗？`)) return;
            response = await postPose(name, record, true);
        }
        const result = await response.json();
        if (!response.ok) { toast(result.error || '保存失败'); return; }
        activeSavedName = result.name;
        await loadSavedList();
        toast(`已保存「${result.name}」`);
    } catch { toast('保存失败：连不上 ComfyUI'); }
}

async function loadSavedPose(entry) {
    try {
        const response = await fetch(`${SAVED_URL}/${encodeURIComponent(entry.name)}`, { cache: 'no-store' });
        if (!response.ok) { toast('这个姿势已经不存在了'); await loadSavedList(); return; }
        const record = await response.json();
        viewer.recordState();
        doc = docFrom({...record.doc, width:doc.width, height:doc.height});
        people[activePerson] = doc;
        loadModel(doc.pose);
        for (const side of ['l', 'r']) $(`#grip-${side}`).value = record.grips?.[side] ?? 0;
        updateCamera(true);
        refreshFlips();
        activeSavedName = entry.name;
        for (const button of document.querySelectorAll('#gallery .fp-thumb')) button.classList.toggle('active', button.dataset.key === doc.openpose?.key);
        renderSaved();
        $('#import-status').textContent = `已载入「${entry.name}」。`;
    } catch (error) { toast(`载入失败：${error.message}`); }
}

async function deleteSavedPose(entry, skipConfirm = false) {
    if (!skipConfirm && !confirm(`删除姿势「${entry.name}」？`)) return;
    try {
        await fetch(`${SAVED_URL}/${encodeURIComponent(entry.name)}/delete`, { method: 'POST' });
        if (activeSavedName === entry.name) activeSavedName = null;
        await loadSavedList();
        toast(`已删除「${entry.name}」`);
    } catch { toast('删除失败：连不上 ComfyUI'); }
}

$('#save-pose').onclick = () => saveCurrentPose();

// --- Viewport interaction: our editor's feel on top of the VNCCS core -------

let interactionMode = 'move';
let interactionInstalled = false;
let finishPersonDrag = null;

function setInteractionMode(mode) {
    finishPersonDrag?.();
    interactionMode = ['move', 'pose', 'view'].includes(mode) ? mode : 'move';
    const moving = interactionMode === 'move';
    const viewing = interactionMode === 'view';
    for (const button of document.querySelectorAll('[data-edit-mode]')) {
        button.setAttribute('aria-pressed', String(button.dataset.editMode === interactionMode));
    }
    $('#edit-mode-hint').textContent = moving ? '← → 转身 · ↑ ↓ 前后 · Shift 微调 · 中键平移 · 右键旋转'
        : viewing ? '右键拖动旋转视角 · 中键拖动平移 · 滚轮缩放'
        : '拖关节摆姿 · Shift+拖移动人物 · 中键平移 · 右键旋转';
    $('#selection-hint').textContent = moving
        ? '拖动当前编号人物 · ← → 转身 · ↑ 远离镜头 · ↓ 靠近镜头'
        : viewing ? '右键旋转／中键平移视角 · 点“回到正面视角”复位'
        : '拖关节摆姿 · Shift+拖 移动人物 · 右键旋转视角';
    canvas.style.cursor = moving ? 'grab' : 'default';
    if (moving || viewing) {
        viewer.deselectBone?.();
        viewer.deselectIKEffector?.();
        viewer.deselectPoleTarget?.();
        viewer.transform?.detach();
        selectedBoneName = null;
    }
    if (viewer.transform) viewer.transform.enabled = interactionMode === 'pose';
    if (viewer.orbit) viewer.orbit.enabled = true;
    canvas.focus({preventScroll:true});
}

function nudgePerson(key, fine = false) {
    const step = fine ? 1 : 5;
    if (key === 'ArrowLeft' || key === 'ArrowRight') {
        const rotation = viewer.modelRotation;
        const yaw = ((rotation.y + (key === 'ArrowLeft' ? -step : step) + 180) % 360 + 360) % 360 - 180;
        viewer.setModelRotation(rotation.x, yaw, rotation.z);
    } else {
        const delta = (fine ? 0.05 : 0.25) * (key === 'ArrowUp' ? -1 : 1);
        doc.transform = {...doc.transform, z:clamp(doc.transform.z + delta, -20, 20)};
        viewer.setActiveCharacterAppearance({transform:doc.transform});
    }
    refreshControls();
    schedulePreview();
}

for (const button of document.querySelectorAll('[data-edit-mode]')) {
    button.onclick = () => setInteractionMode(button.dataset.editMode);
}
document.addEventListener('keydown', event => {
    if (!ready || interactionMode !== 'move' || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest?.('input, textarea, select, [contenteditable="true"], [role="slider"]')) return;
    if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    if (!event.repeat) viewer.recordState();
    (groupMode() ? applyGroupNudge : nudgePerson)(event.key, event.shiftKey);
});

function setupInteraction() {
    if (interactionInstalled) return;
    interactionInstalled = true;
    const THREE = viewer.THREE;
    // 视角控制三模式统一（与核心默认一致）：中键=平移，右键=旋转，滚轮=缩放；
    // 左键留给各模式本体（移动人物/拖关节），view 模式下左键不转视角。
    viewer.orbit.mouseButtons = { LEFT: THREE.MOUSE.NONE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
    // OrbitControls has its own listeners. In view mode, bypass rig picking
    // even when dragging directly over a person or a previously selected joint.
    for (const method of ['handlePointerDown', 'handlePointerMove', 'handlePointerUp']) {
        const original = viewer[method].bind(viewer);
        viewer[method] = event => { if (interactionMode !== 'view') return original(event); };
    }
    // Capture phase keeps whole-person dragging separate from IK and camera orbit.
    canvas.addEventListener('pointerdown', event => {
        if (interactionMode === 'view' || event.button !== 0 || (!event.shiftKey && interactionMode !== 'move') || !ready) return;
        event.stopImmediatePropagation();
        event.preventDefault();
        finishPersonDrag?.();
        canvas.focus({preventScroll:true});
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = 'grabbing';
        viewer.orbit.enabled = false;
        viewer.recordState();
        const camera = viewer.camera;
        const distance = camera.position.distanceTo(viewer.orbit.target);
        const perPixel = 2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) / (canvas.clientHeight * camera.zoom);
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
        const start = { x: event.clientX, y: event.clientY, transform: { ...doc.transform } };
        // 全组拖拽：勾选"全组"时，拖拽增量同时应用到所有人（相对关系不变）
        const groupStart = (groupMode() && people.length > 1) ? people.map(p => ({ ...p.transform })) : null;
        const move = moveEvent => {
            if (moveEvent.pointerId !== event.pointerId) return;
            const delta = right.clone().multiplyScalar((moveEvent.clientX - start.x) * perPixel)
                .add(up.clone().multiplyScalar(-(moveEvent.clientY - start.y) * perPixel));
            doc.transform = { ...start.transform, x: clamp(start.transform.x + delta.x, -20, 20), y: clamp(start.transform.y + delta.y, -20, 20), z: clamp(start.transform.z + delta.z, -20, 20) };
            viewer.setActiveCharacterAppearance({ transform: doc.transform });
            if (groupStart) {
                const active = activePerson;
                const dx = doc.transform.x - groupStart[active].x;
                const dy = doc.transform.y - groupStart[active].y;
                const dz = doc.transform.z - groupStart[active].z;
                for (let index = 0; index < people.length; index++) {
                    if (index === active) continue;
                    selectPerson(index);
                    doc.transform = { ...groupStart[index],
                        x: clamp(groupStart[index].x + dx, -20, 20),
                        y: clamp(groupStart[index].y + dy, -20, 20),
                        z: clamp(groupStart[index].z + dz, -20, 20) };
                    viewer.setActiveCharacterAppearance({ transform: doc.transform });
                    saveActivePerson();
                }
                selectPerson(active);
            }
            refreshControls();
        };
        const end = endEvent => {
            if (endEvent?.pointerId != null && endEvent.pointerId !== event.pointerId) return;
            window.removeEventListener('pointermove', move, true);
            window.removeEventListener('pointerup', end);
            window.removeEventListener('pointercancel', end);
            window.removeEventListener('blur', end);
            canvas.removeEventListener('lostpointercapture', end);
            finishPersonDrag = null;
            if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
            viewer.orbit.enabled = true;
            canvas.style.cursor = interactionMode === 'move' ? 'grab' : 'default';
            schedulePreview();
        };
        finishPersonDrag = end;
        window.addEventListener('pointermove', move, true);
        window.addEventListener('pointerup', end);
        window.addEventListener('pointercancel', end);
        window.addEventListener('blur', end);
        canvas.addEventListener('lostpointercapture', end);
    }, { capture: true });
    canvas.addEventListener('pointermove', event => {
        if (finishPersonDrag || (interactionMode === 'move' && event.buttons === 0)) event.stopImmediatePropagation();
    }, {capture:true});
    // Runs after the core's own handler: if it grabbed a joint, the drag must not also orbit.
    canvas.addEventListener('pointerdown', event => {
        if (interactionMode !== 'pose' || event.button !== 0) return;
        if (viewer.directDrag?.active || viewer.selectedBone || viewer.selectedIKEffector || viewer.selectedPoleTarget || viewer.transform?.dragging) {
            viewer.orbit.enabled = false;
            window.addEventListener('pointerup', () => { viewer.orbit.enabled = true; }, { once: true });
        }
    });
    setInteractionMode(interactionMode);
}

// --- Controls ---------------------------------------------------------------

const setOutput = (id, text) => { $(`#${id}-value`).textContent = text; };

function refreshBoneSliders() {
    const bone = selectedBoneName && viewer.bones?.[selectedBoneName];
    $('#bone-name').textContent = bone ? selectedBoneName : '未选择';
    $('#bone-sliders').classList.toggle('fp-disabled', !bone);
    for (const axis of ['x', 'y', 'z']) {
        const degrees = bone ? Math.round(bone.rotation[axis] * 180 / Math.PI) : 0;
        $(`#r${axis}`).value = degrees;
        setOutput(`r${axis}`, bone ? `${degrees}°` : '—');
    }
}

function refreshControls() {
    $('#zoom').value = doc.transform.zoom; setOutput('zoom', doc.transform.zoom.toFixed(2));
    $('#tx').value = doc.transform.x; setOutput('tx', doc.transform.x.toFixed(1));
    $('#ty').value = doc.transform.y; setOutput('ty', doc.transform.y.toFixed(1));
    $('#tz').value = doc.transform.z; setOutput('tz', doc.transform.z.toFixed(1));
    const turn = Math.round(viewer.modelRotation?.y || 0);
    $('#turn').value = turn; setOutput('turn', `${turn}°`);
    const rot3 = viewer.modelRotation || { x: 0, y: 0, z: 0 };
    $('#pitch').value = Math.round(rot3.x); setOutput('pitch', `${Math.round(rot3.x)}°`);
    $('#roll').value = Math.round(rot3.z); setOutput('roll', `${Math.round(rot3.z)}°`);
    $('#output-width').value = doc.width;
    $('#output-height').value = doc.height;
    $('#preview-size').textContent = `${doc.width}×${doc.height}`;
    for (const button of document.querySelectorAll('#ratios button')) {
        const [a, b] = button.dataset.ratio.split(':').map(Number);
        button.classList.toggle('active', Math.abs(doc.width / doc.height - a / b) < 0.02);
    }
    // 全组滑杆的增量基准（refreshControls 每次刷新后重记）
    for (const id of ['zoom', 'tx', 'ty', 'tz', 'turn', 'pitch', 'roll']) sliderPrev[id] = Number($(`#${id}`).value);
    for (const [key] of BODY_SLIDERS) {
        const input = $(`#body-${key}`);
        if (input) { input.value = doc.mesh[key]; setOutput(`body-${key}`, key === 'age' ? String(doc.mesh[key]) : Number(doc.mesh[key]).toFixed(2)); }
    }
    for (const side of ['l', 'r']) setOutput(`grip-${side}`, Number($(`#grip-${side}`).value).toFixed(2));
    for (const [key] of PROPORTIONS) {
        const input = $(`#prop-${key}`);
        if (input) { input.value = doc.proportions[key]; setOutput(`prop-${key}`, `${Math.round(doc.proportions[key] * 100)}%`); }
    }
    $('#description').textContent = promptText();
    refreshBoneSliders();
}

// One undo entry per slider gesture: record before the first input event.
function bindSlider(input, onValue) {
    let gesture = false;
    input.addEventListener('input', () => {
        if (!gesture && ready) { viewer.recordState(); gesture = true; }
        onValue(Number(input.value));
    });
    input.addEventListener('change', () => { gesture = false; });
}

// 输出栏滑杆：单人=直接赋值；全组=按增量作用于所有人（zoom 用比值，平移用差值）
const sliderPrev = {};
for (const [id, key] of [['zoom', 'zoom'], ['tx', 'x'], ['ty', 'y'], ['tz', 'z']]) {
    bindSlider($(`#${id}`), value => {
        if (groupMode()) {
            const prev = sliderPrev[id] ?? value;
            sliderPrev[id] = value;
            // 大小=整组等比缩放（位置绕组中心同步扩散，相对关系不变）；
            // 平移=差值同施所有人
            if (key === 'zoom') { applyGroupScale(value / Math.max(prev, 0.01)); return; }
            const delta = value - prev;
            eachPerson(() => {
                doc.transform = { ...doc.transform, [key]: clamp(doc.transform[key] + delta, -20, 20) };
            });
            return;
        }
        doc.transform = { ...doc.transform, [key]: value };
        viewer.setActiveCharacterAppearance({ transform: doc.transform });
        refreshControls(); schedulePreview();
    });
}
// 三自由度旋转：轴（yaw）、俯仰（pitch=x）、侧倾（roll=z）。旋转中心=人偶自身脚下；
// 单人=绝对角（其余轴保持），全组=增量同施所有人（相对关系不变）。
const setRot = (axis, value) => {
    const r = viewer.modelRotation || { x: 0, y: 0, z: 0 };
    const next = { x: r.x, y: r.y, z: r.z, [axis]: value };
    viewer.setModelRotation(next.x, next.y, next.z);
    refreshControls(); schedulePreview();
};
for (const [id, axis] of [['turn', 'y'], ['pitch', 'x'], ['roll', 'z']]) {
    bindSlider($(`#${id}`), value => {
        if (groupMode()) {
            const prev = sliderPrev[id] ?? value;
            sliderPrev[id] = value;
            const delta = value - prev;
            eachPerson(() => {
                const r = viewer.modelRotation;
                const v = ((r[axis] + delta + 540) % 360) - 180;
                viewer.setModelRotation(axis === 'y' ? v : r.x, axis === 'y' ? r.y : (axis === 'x' ? v : r.z), axis === 'z' ? v : r.z);
            });
            return;
        }
        sliderPrev[id] = value;
        setRot(axis, value);
    });
}
for (const axis of ['x', 'y', 'z']) {
    bindSlider($(`#r${axis}`), value => {
        const bone = viewer.bones?.[selectedBoneName];
        if (!bone) return;
        bone.rotation[axis] = value * Math.PI / 180;
        bone.updateMatrixWorld(true);
        viewer.updateIKEffectorPositions?.();
        viewer.updateMarkers?.();
        viewer.requestRender();
        setOutput(`r${axis}`, `${value}°`);
        schedulePreview();
    });
}
for (const button of document.querySelectorAll('[data-preset]')) {
    button.onclick = () => {
        viewer.applyHandPreset(button.dataset.hand, HAND_PRESETS[button.dataset.preset]);
        $(`#grip-${button.dataset.hand}`).value = button.dataset.preset === 'FIST' ? 1 : 0;
        refreshControls(); schedulePreview();
    };
}
for (const side of ['l', 'r']) {
    bindSlider($(`#grip-${side}`), value => {
        viewer.interpolateHandPose(HAND_PRESETS.OPEN, HAND_PRESETS.FIST, value, side);
        refreshControls(); schedulePreview();
    });
}
$('#snap-view').onclick = () => updateCamera(true);
$('#fit-frame').onclick = $('#fit-frame-2').onclick = () => { viewer.recordState(); fitFrame(); };
$('#reset-bone').onclick = () => { viewer.recordState(); viewer.resetSelectedBone(); refreshControls(); schedulePreview(); };
$('#reset-pose').onclick = () => { viewer.recordState(); resetPose(); doc.openpose = null; refreshFlips(); refreshControls(); schedulePreview(); };
$('#undo').onclick = () => viewer.undo();
$('#redo').onclick = () => viewer.redo();

function setSize(width, height) {
    doc.width = round16(width);
    doc.height = round16(height);
    updateCamera(false);
}
// 相框比例锁定：frameAspect 来自输入图（applyPayload 时记录），编辑器内只能等比缩放。
// 改任一边另一边按锁定比例跟上；± 按钮整体等比 10%；输出人偶图即此画幅。
let frameAspect = null;
const snapHeight = w => round16(w / (frameAspect || doc.width / doc.height));
const scaleFrame = factor => { viewer.recordState(); setSize(doc.width * factor, snapHeight(doc.width * factor)); refreshControls(); };
$('#output-width').onchange = event => {
    const value = Number(event.target.value) || doc.width;
    viewer.recordState();
    setSize(value, snapHeight(value));
    refreshControls();
};
$('#output-height').onchange = event => {
    const value = Number(event.target.value) || doc.height;
    viewer.recordState();
    const w = frameAspect ? value * frameAspect : value * doc.width / doc.height;
    setSize(w, value);
    refreshControls();
};
$('#frame-scale-down').onclick = () => scaleFrame(0.9);
$('#frame-scale-up').onclick = () => scaleFrame(1.1);
// 预制比例：一键切换画幅比例（长边保持），切换后宽高联动跟随新比例
for (const button of document.querySelectorAll('#ratios button')) {
    button.onclick = () => {
        const [a, b] = button.dataset.ratio.split(':').map(Number);
        const longSide = Math.max(doc.width, doc.height);
        viewer.recordState();
        setSize(a >= b ? longSide : longSide * a / b, a >= b ? longSide * b / a : longSide);
        frameAspect = doc.width / doc.height;
        refreshControls();
    };
}

// ── 全组变换：「全部」按钮（左上角人物按钮旁）选中所有人：右侧只剩输出栏可调，
// 滑杆/拖拽/方向键微调整组，单人之间相对关系不变；点具体人物号退出全组。 ──
let groupSelected = false;
const groupMode = () => groupSelected;
function applyGroupSelectionUI() {
    const button = $('#select-all');
    if (button) button.setAttribute('aria-pressed', String(groupSelected));
    for (const section of document.querySelectorAll('aside section')) {
        if (groupSelected && section.id !== 'output-section') section.classList.add('fp-disabled');
        else section.classList.remove('fp-disabled');
    }
    const status = $('#people-status');
    if (status && groupSelected) status.textContent = `已选中全部 ${people.length} 人 · 滑杆/拖拽微调整组 · 点人物号退出`;
}
$('#select-all').onclick = () => { viewer.recordState?.(); groupSelected = !groupSelected; applyGroupSelectionUI(); renderPeople(); };
const eachPerson = fn => {
    const active = activePerson;
    for (let index = 0; index < people.length; index++) {
        selectPerson(index);
        fn(index);
        viewer.setActiveCharacterAppearance({ transform: doc.transform });
        saveActivePerson();
    }
    selectPerson(active);
    refreshControls();
    schedulePreview();
};
const groupCenter = () => ({
    x: people.reduce((s, p) => s + p.transform.x, 0) / people.length,
    z: people.reduce((s, p) => s + p.transform.z, 0) / people.length,
});
const applyGroupScale = factor => {
    viewer.recordState();
    const { x: cx, z: cz } = groupCenter();
    eachPerson(() => {
        doc.transform = { ...doc.transform,
            x: clamp(cx + (doc.transform.x - cx) * factor, -20, 20),
            z: clamp(cz + (doc.transform.z - cz) * factor, -20, 20),
            zoom: clamp(doc.transform.zoom * factor, 0.2, 6) };
    });
};
const applyGroupRotate = degrees => {
    viewer.recordState();
    const rad = degrees * Math.PI / 180, cos = Math.cos(rad), sin = Math.sin(rad);
    const { x: cx, z: cz } = groupCenter();
    eachPerson(() => {
        const rotation = viewer.modelRotation;
        const yaw = ((rotation.y + degrees + 180) % 360 + 360) % 360 - 180;
        viewer.setModelRotation(rotation.x, yaw, rotation.z);
        const dx = doc.transform.x - cx, dz = doc.transform.z - cz;
        doc.transform = { ...doc.transform,
            x: clamp(cx + dx * cos - dz * sin, -20, 20),
            z: clamp(cz + dx * sin + dz * cos, -20, 20) };
    });
};
const applyGroupNudge = (key, fine) => {
    const step = fine ? 1 : 5;
    const delta = (fine ? 0.05 : 0.25) * (key === 'ArrowUp' ? -1 : 1);
    eachPerson(() => {
        if (key === 'ArrowLeft' || key === 'ArrowRight') {
            const rotation = viewer.modelRotation;
            const yaw = ((rotation.y + (key === 'ArrowLeft' ? -step : step) + 180) % 360 + 360) % 360 - 180;
            viewer.setModelRotation(rotation.x, yaw, rotation.z);
        } else {
            doc.transform = { ...doc.transform, z: clamp(doc.transform.z + delta, -20, 20) };
        }
    });
};
$('#group-scale-down').onclick = () => applyGroupScale(0.9);
$('#group-scale-up').onclick = () => applyGroupScale(1.1);
$('#group-rot-l').onclick = () => applyGroupRotate(-15);
$('#group-rot-r').onclick = () => applyGroupRotate(15);

const bodyContainer = $('#body-sliders');
for (const [key, label, min, max, step] of BODY_SLIDERS) {
    bodyContainer.insertAdjacentHTML('beforeend', `<label class="slider-label">${label}<output id="body-${key}-value"></output></label><input id="body-${key}" type="range" min="${min}" max="${max}" step="${step}">`);
    $(`#body-${key}`).addEventListener('input', event => {
        if ((key === 'gender' || key === 'age') && doc.mesh.keepHeight == null)
            doc.mesh = { ...doc.mesh, keepHeight: solveHeight(doc.mesh) };  // 首次锚定，拖动全程沿用
        if (key === 'height') doc.mesh = { ...doc.mesh, keepHeight: null };  // 手动调身高 = 解除约束
        doc.mesh = { ...doc.mesh, [key]: Number(event.target.value) };
        refreshControls();
        clearTimeout(morphTimer);
        morphTimer = setTimeout(() => { loadModel(rotationsOnly(viewer.getPose())); updateCamera(false); }, 120);
    });
}
$('#reset-body').onclick = () => { doc.mesh = { ...DEFAULT_MESH }; loadModel(rotationsOnly(viewer.getPose())); updateCamera(false); };

const proportionContainer = $('#proportion-sliders');
for (const [key, label, min, max] of PROPORTIONS) {
    proportionContainer.insertAdjacentHTML('beforeend', `<label class="slider-label">${label}<output id="prop-${key}-value"></output></label><input id="prop-${key}" type="range" min="${min}" max="${max}" step="0.01">`);
    $(`#prop-${key}`).addEventListener('input', event => {
        setProportion(key, Number(event.target.value));
        refreshControls();
        schedulePreview();
    });
}
$('#reset-proportions').onclick = () => {
    for (const [key] of PROPORTIONS) setProportion(key, 1);
    refreshControls();
    schedulePreview();
};

for (const button of document.querySelectorAll('.panel-tabs button')) {
    button.onclick = () => {
        for (const other of document.querySelectorAll('.panel-tabs button')) other.classList.toggle('active', other === button);
        for (const tab of ['output', 'pose', 'body', 'proportion']) $(`#${tab}-panel`).hidden = tab !== button.dataset.tab;
    };
}
$('#extra-prompt').addEventListener('input', event => { extraPrompt = event.target.value; refreshControls(); });
$('#copy-description').onclick = async () => { try { await navigator.clipboard.writeText(promptText()); toast('已复制'); } catch { toast('复制失败'); } };

document.addEventListener('keydown', event => {
    if (event.target.closest?.('textarea, input[type=number], input[type=search]')) return;
    if (!(event.ctrlKey || event.metaKey)) return;
    const key = event.key.toLowerCase();
    if (key === 'z' && !event.shiftKey) { event.preventDefault(); viewer.undo(); }
    else if ((key === 'z' && event.shiftKey) || key === 'y') { event.preventDefault(); viewer.redo(); }
});
new ResizeObserver(() => viewer.resize(stage.clientWidth, stage.clientHeight)).observe(stage);

// Each person owns their editable state; the core clones inactive rigs into the same scene.
function saveActivePerson() {
    doc.pose = savedPose();
    doc.grips = {l:Number($('#grip-l').value),r:Number($('#grip-r').value)};
    doc.history = [...viewer.history]; doc.future = [...viewer.future];
    people[activePerson] = doc;
    for (const person of people) { person.width=doc.width; person.height=doc.height; }
}
function renderPeople() {
    $('#person-buttons').replaceChildren(...people.map((person,i) => {
        const button=document.createElement('button');
        button.textContent=`${i+1}号`; button.style.background=PERSON_COLORS[i];
        button.setAttribute('aria-pressed',String(i===activePerson));
        button.onclick=()=>{ groupSelected=false; applyGroupSelectionUI(); selectPerson(i); }; return button;
    }));
    $('#add-person').disabled=people.length>=9;
    $('#remove-person').disabled=people.length<=1;
    $('#people-status').textContent=`${people.length}/9 人 · 当前编辑 ${activePerson+1}号 · 图片输入按人物编号对应`;
    const preview=referencePreviews[activePerson];
    $('#person-preview-wrap').hidden=!preview;
    if(preview) $('#person-preview').src=preview;
    $('#person-preview-title').textContent=`${activePerson+1}号人物图 · image${activePerson+2}`;
}
function activatePerson(index) {
    activePerson=index; doc=people[index];
    viewer.removePassiveCharacter(String(index));
    loadModel(doc.pose);
    viewer.history=doc.history || []; viewer.future=doc.future || [];
    selectedBoneName=null;
    for(const side of ['l','r']) $(`#grip-${side}`).value=doc.grips?.[side] || 0;
    refreshFlips(); renderPeople(); updateCamera(false);
}
function selectPerson(index) {
    if(!ready || index===activePerson || index<0 || index>=people.length) return;
    clearTimeout(morphTimer);
    // Flush body slider changes before cloning, including the shaped rig.
    loadModel(savedPose());
    saveActivePerson();
    viewer.upsertPassiveCharacterFromActive(String(activePerson),{pose:doc.pose,color:PERSON_COLORS[activePerson],transform:doc.transform});
    activatePerson(index);
}
function rebuildPeople() {
    const selected=activePerson;
    viewer.clearPassiveCharacters();
    for(let i=0;i<people.length;i++) {
        if(i===selected) continue;
        activePerson=i; doc=people[i]; loadModel(doc.pose);
        viewer.upsertPassiveCharacterFromActive(String(i),{pose:savedPose(),color:PERSON_COLORS[i],transform:doc.transform});
    }
    activatePerson(selected);
}
function addPerson() {
    if(!ready || people.length>=9) return;
    const index=people.length;
    const next=docFrom({width:doc.width,height:doc.height});
    // Spawn near the active person; all positions remain freely editable.
    next.transform={...doc.transform,x:clamp(doc.transform.x+2,-20,20)};
    people.push(next); selectPerson(index);
}
function removePerson() {
    if(!ready || people.length<=1) return;
    const last=people.length-1;
    if(activePerson===last) selectPerson(last-1);
    viewer.removePassiveCharacter(String(last)); people.pop();
    renderPeople(); refreshControls(); schedulePreview();
}
function arrangePeople() {
    if(!ready) return;
    const selected=activePerson;
    saveActivePerson();
    const cols=Math.ceil(Math.sqrt(people.length)), rows=Math.ceil(people.length/cols);
    for(let i=0;i<people.length;i++) {
        selectPerson(i);
        viewer.recordState();
        fitFrame(false);
        doc.transform.zoom /= Math.max(cols,rows);
        viewer.setActiveCharacterAppearance({transform:doc.transform});
        const target=viewer.characterTranslationForFramePoint(-1+(2*(i%cols)+1)/cols,1-(2*Math.floor(i/cols)+1)/rows);
        if(target) doc.transform={...doc.transform,...target};
        viewer.setActiveCharacterAppearance({transform:doc.transform});
    }
    saveActivePerson(); selectPerson(selected); updateCamera(true);
}
$('#arrange-people').onclick=arrangePeople;
$('#add-person').onclick=addPerson;
$('#remove-person').onclick=removePerson;

// --- Node bridge ------------------------------------------------------------

async function serialize() {
    clearTimeout(morphTimer);
    loadModel(savedPose());
    await viewer.waitForCaptureReady();
    const poseReference = await drawBoldEdges(await capture(doc.width, doc.height, true), doc.width, doc.height);
    updateCamera(false);
    doc.pose = savedPose();
    saveActivePerson();
    return JSON.stringify({ version: 3, kind: 'vnccs-free-pose', ...doc, people, activePerson, camera: FRONT, poseReference });
}

// Shared by the node payload and 我的姿势: any saved editor state → a complete doc.
function docFrom(saved) {
    return {
        mesh: { ...DEFAULT_MESH, ...saved.mesh, breast_size: 0 }, pose: saved.pose || null, // flat chest is fixed
        proportions: { ...DEFAULT_PROPORTIONS, ...saved.proportions },
        transform: { ...NEUTRAL_TRANSFORM, ...saved.transform },
        // The mannequin size lives only here; the node's width/height are the separate output size.
        width: round16(Number(saved.width) || doc.width), height: round16(Number(saved.height) || doc.height),
        openpose: saved.openpose || null, grips: saved.grips || {l:0,r:0},
    };
}

function applyPayload(payload) {
    const saved = JSON.parse(payload.pose_json || '{}');
    const restored = saved.kind === 'vnccs-free-pose';
    // 取景框与节点输出同尺寸（=原图比例），而非沿用存档里的旧画幅
    if (Number.isFinite(Number(payload.width)) && Number(payload.width) >= 64) {
        doc.width = round16(Number(payload.width));
        doc.height = round16(Number(payload.height) || doc.height);
    }
    // 比例锁：以本次传入的画幅（输入图/auto 或用户上次应用值）为准，编辑器内不可改比例
    frameAspect = doc.width / doc.height;
    if (restored) {
        const records = Array.isArray(saved.people) && saved.people.length ? saved.people.slice(0,9) : [saved];
        people = records.map(item => docFrom({...item,width:saved.width,height:saved.height}));
        activePerson = clamp(Number(saved.activePerson)||0,0,people.length-1);
        doc = people[activePerson];
    }
    referencePreviews = payload.referencePreviews || [payload.referencePreview];
    extraPrompt = payload.extra_prompt || '';
    $('#extra-prompt').value = extraPrompt;
    if (payload.referencePreview) {
        $('#person-preview').src = payload.referencePreview;
        $('#person-preview-wrap').hidden = false;
    }
    return restored;
}

let pendingPayload = null;
async function start(payload) {
    const restored = payload ? applyPayload(payload) : false;
    loadModel(doc.pose);
    rebuildPeople();
    setupInteraction();
    ready = true;
    renderPeople();
    if (restored) updateCamera(true);
    else fitFrame(true);
    refreshFlips();
    void loadBuiltin();
    void loadLibrary();
    void loadSavedList();
    await viewer.waitForCaptureReady();
    // 工具台 SAM3D 直入：仅在没有任何已保存姿势时自动摆一次；
    // 应用过/有状态 → 永不自动重置，要新数据点"按 SAM3D 重摆"。
    if (payload?.multiPose?.people?.length && !restored) {
        await importMultiPersonJSON(payload.multiPose);
    }
    $('#loading').hidden = true;
    $('#apply-editor').disabled = false;
    $('#save-status').textContent = restored && payload?.multiPose?.people?.length
        ? '已载入上次编辑的姿势；跑过新图后点「按 SAM3D 重摆」换新骨架'
        : payload?.multiPose?.people?.length
            ? '已按 SAM3D 自动摆好，可微调后应用到节点'
            : (restored ? '已载入节点中的姿势' : '新姿势：可用左侧 OpenPose 图一键摆姿');
    schedulePreview();
}

window.addEventListener('message', event => {
    if (event.source !== parent || event.origin !== location.origin || event.data?.type !== 'gaoshang-load') return;
    if (pack) start(event.data.payload).catch(showError);
    else pendingPayload = event.data.payload;
});
$('#cancel-editor').onclick = () => parent.postMessage({ type: 'gaoshang-close' }, location.origin);
$('#apply-editor').onclick = async () => {
    $('#apply-editor').disabled = true;
    try {
        const pose_json = await serialize();
        parent.postMessage({ type: 'gaoshang-apply', payload: {
            pose_json, extra_prompt: extraPrompt, width: doc.width, height: doc.height,
        } }, location.origin);
    } catch (error) { showError(error); }
    finally { $('#apply-editor').disabled = false; }
};

function showError(error) {
    console.error(error);
    $('#loading-text').textContent = '加载失败：' + (error?.message || error);
    $('#loading').hidden = false;
}

if (!embedded) { $('#apply-editor').hidden = true; $('#cancel-editor').hidden = true; }
window.freePose = { get people() { return people; }, get activePerson() { return activePerson; }, addPerson, selectPerson, removePerson, arrangePeople, viewer, get doc() { return doc; }, serialize, prompt: promptText, fitFrame, importEntry, addFiles, saveCurrentPose, loadSavedPose, deleteSavedPose };

(async () => {
    await viewer.init();
    if (!viewer.initialized) throw new Error('WebGL 初始化失败');
    // The official QI2.1 workflow runs with the skydome disabled; it would otherwise be captured.
    viewer.setDirectionalSkydomeVisible(false);
    refreshControls();
    if (embedded) parent.postMessage({ type: 'gaoshang-ready' }, location.origin);
    pack = await loadBodyPack();
    if (embedded && !pendingPayload) return; // start() runs when the node payload arrives
    await start(pendingPayload);
})().catch(showError);
