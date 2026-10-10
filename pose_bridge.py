"""Gaoshang scene_json IO bridge: project the editor scene to flat OpenPose JSON,
a skeleton image, and the editor-baked mannequin image, so other workflows can
consume the pose without the Gaoshang generation chain."""
import io
import json

import numpy as np
import torch

from .pose_core import read_scene, render_pose, project_points
from .pose_reference import reference_image


def _to_tensor(image):
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255).unsqueeze(0)


class GaoshangPoseBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene_json": ("STRING", {"forceInput": True, "multiline": True}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("openpose_json", "骨架图", "人偶图")
    FUNCTION = "convert"
    CATEGORY = "3D多人姿势编辑器"
    DESCRIPTION = "场景JSON桥：把机位与姿态编辑器的 scene_json 投影成平面 OpenPose JSON（DWPose 同构，可接 RenderPeopleKps/UC 链）、2D 骨架图，并解码编辑器烘焙的人偶图（编辑器未应用时人偶图为黑图）。"

    @classmethod
    def IS_CHANGED(cls, scene_json, width, height):
        return scene_json

    def convert(self, scene_json, width, height):
        state = read_scene(scene_json)
        skeleton = render_pose(state, width, height)
        people = []
        for person in state["people"]:
            points = project_points(state, width, height, person)
            flat = []
            for point in points:
                flat.extend([point[0], point[1], 1.0] if point else [0.0, 0.0, 0.0])
            flat.extend([0.0] * 12)  # eyes/ears have no 3D source
            people.append({"pose_keypoints_2d": flat})
        openpose = json.dumps({"canvas_width": width, "canvas_height": height, "people": people}, ensure_ascii=False)
        try:
            mannequin = reference_image(scene_json)
        except Exception as error:  # editor not applied yet; keep the graph alive
            print(f"[GaoshangPoseBridge] 人偶图跳过：{error}")
            from PIL import Image
            mannequin = Image.new("RGB", (width, height), (0, 0, 0))
        return {"ui": {"openpose_json": [openpose]}, "result": (openpose, _to_tensor(skeleton), _to_tensor(mannequin))}


class GaoshangBboxesFromKeypoints:
    """Per-person bounding boxes (pixels) from a DWPose POSE_KEYPOINT payload,
    for SAM3DBody_Predict multi-person detection on single images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_keypoint": ("POSE_KEYPOINT",),
            "padding": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                                  "tooltip": "框外扩比例（相对图宽）"}),
        }}

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "convert"
    CATEGORY = "3D多人姿势编辑器"
    DESCRIPTION = "DWPose 关键点 → 每人一个像素 bbox（BOUNDING_BOX 列表），接 SAM3DBody_Predict.bboxes 实现单图多人。"

    @classmethod
    def IS_CHANGED(cls, pose_keypoint, padding):
        return ""

    def convert(self, pose_keypoint, padding):
        frames = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]
        frame = frames[0]
        width, height = int(frame["canvas_width"]), int(frame["canvas_height"])
        pad = max(1, round(min(width, height) * padding))
        boxes = []
        for person in frame["people"]:
            flat = person["pose_keypoints_2d"]
            xs = [flat[i * 3] for i in range(18) if flat[i * 3 + 2] > 0.3]
            ys = [flat[i * 3 + 1] for i in range(18) if flat[i * 3 + 2] > 0.3]
            if not xs:
                continue
            x0 = max(0, min(xs) - pad)
            y0 = max(0, min(ys) - pad * 3)
            x1 = min(width, max(xs) + pad)
            y1 = min(height, max(ys) + pad)
            boxes.append({"x": float(x0), "y": float(y0),
                          "width": float(x1 - x0), "height": float(y1 - y0)})
        return (boxes,)


class GaoshangSceneFromMHR:
    """SAM3D-Body pose_data -> Gaoshang studio scene with real estimated depth.

    Camera-frame OpenPose joints (x right, y down, z forward, metres) are lifted
    to the Gaoshang y-up ground world; the scene echoes back through
    ui.gaoshang_auto_scene so the editor auto-loads it like the DWPose path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_data": ("MHR_POSE_DATA",),
            "width": ("INT", {"default": 1152, "min": 64, "max": 4096, "step": 16,
                              "tooltip": "原画布宽（仅用于编辑器比例/占位渲染）"}),
            "height": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_json",)
    FUNCTION = "convert"
    CATEGORY = "3D多人姿势编辑器"
    DESCRIPTION = "SAM3D-Body 3D 姿态 → 高尚场景（真深度）。运行后经 ui 回传自动填进 GaoshangQwenPose 的 scene_json，编辑器直接可调。"

    @classmethod
    def IS_CHANGED(cls, pose_data, width, height):
        return ""

    def convert(self, pose_data, width, height):
        import base64
        import io as _io
        import math as _math

        from .pose_core import render_pose as _render_pose
        from comfy_extras.sam3d_body.export.glb_shared import openpose_render_keypoints

        frames = pose_data.get("frames") if isinstance(pose_data, dict) else pose_data
        people = frames[0] if frames else []
        names = ["head", "neck", "rs", "re", "rw", "ls", "le", "lw", "rh", "rk", "ra", "lh", "lk", "la"]
        cam_people = []
        for person in people:
            kp3 = openpose_render_keypoints(person, pose_data if isinstance(pose_data, dict) else None, "body", dim=3)
            cam_t = person.get("pred_cam_t")
            if kp3 is None or cam_t is None:
                continue
            import numpy as _np
            pts = _np.asarray(kp3, dtype=_np.float32) + _np.asarray(cam_t, dtype=_np.float32).reshape(1, 3)
            cam_people.append({name: [float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i, name in enumerate(names)})
        if not cam_people:
            raise ValueError("SAM3D 未检出人物，无法建场。")

        ground = min(p[n][1] for p in cam_people for n in ("la", "ra"))
        far = max(p["neck"][2] for p in cam_people)
        world = []
        for p in cam_people:
            w = {n: [p[n][0], ground - p[n][1], far - p[n][2]] for n in names}
            world.append(w)
        mid_y = sum((max(p[n][1] for n in names) + min(p[n][1] for n in names)) / 2 for p in world) / len(world)
        cx = sum(p["neck"][0] for p in world) / len(world)
        cz = sum(p["neck"][2] for p in world) / len(world)
        span = max(max(p[n][1] for n in names) for p in world)
        distance = max(1.2, span / (2 * _math.tan(37 * _math.pi / 360)) + 0.9)

        people_out = []
        for i, w in enumerate(world, 1):
            origin = w["neck"]
            pts3 = {n: [w[n][k] - origin[k] for k in range(3)] for n in names}
            pts3["hip"] = [(pts3["lh"][k] + pts3["rh"][k]) / 2 for k in range(3)]
            pts3["head"][1] += .005
            pts3["head"][2] = -.11
            people_out.append({"id": f"p{i}", "name": f"人物 {i:02d}", "points": pts3, "bodyAngle": 0,
                               "position": list(origin), "preset": "imported", "identity": ""})

        state = {"gaoshang_sam3d": True, "azimuth": 0, "elevation": 0, "distance": distance,
                 "cameraTarget": [cx, mid_y, cz], "zoom": max(0, min(10, (6.8 - distance) / .5)),
                 "ratio": f"{width}:{height}", "people": people_out}
        buffer = _io.BytesIO()
        _render_pose(state, width, height).save(buffer, "PNG")
        state["poseReference"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        scene_json = json.dumps(state, ensure_ascii=False)

        # Reprojection self-check: gaoshang projection vs SAM3D native 2D joints.
        import numpy as _np2
        from .pose_core import project_points as _project_points
        report = []
        for person_out, person in zip(people_out, people):
            kp2d = openpose_render_keypoints(person, pose_data if isinstance(pose_data, dict) else None, "body", dim=2, H=height, W=width)
            if kp2d is None:
                continue
            projected = _project_points(state, width, height, person_out)
            dists = []
            for j in range(14):
                gx, gy = projected[j]
                mx, my = float(kp2d[j, 0]), float(kp2d[j, 1])
                if 0 <= mx < width and 0 <= my < height:
                    dists.append(_math.hypot(gx - mx, gy - my))
            if dists:
                report.append({"person": person_out["id"], "joints": len(dists), "mean_px": round(sum(dists) / len(dists), 1), "max_px": round(max(dists), 1)})
        return {"ui": {"gaoshang_auto_scene": [scene_json], "gaoshang_reproj": [json.dumps(report)]}, "result": (scene_json,)}


def _enrich_sam3d_person(person, model):
    """单人原始 MHR 参数 -> VNCCS applySAM3DImport 可吃的富化载荷。

    照抄 ComfyUI_VNCCS_Utils/vnccs_sam3d/pose_import.py 的富化流程：mhr_forward
    跑 posed + rest 两遍得到关节旋转/坐标/静止对照，供 MakeHuman 刚性求解器使用。
    """
    import torch
    from ComfyUI_VNCCS_Utils.vnccs_sam3d.processing.process import (
        _FACE_BS_CACHE, _get_mhr_rest_verts, _load_sam3d_model, _to_batched_tensor,
    )
    loaded = _load_sam3d_model(model)
    mhr_head = loaded["model"].head_pose
    device = torch.device(loaded["device"])

    def _plain(value):
        if isinstance(value, np.ndarray):
            return [float(v) for v in value.reshape(-1)]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return value
    pose_data = {k: _plain(person.get(k)) for k in ("global_rot", "body_pose_params", "hand_pose_params")}
    zeros_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
    zeros_rot = torch.zeros((1, 3), dtype=torch.float32, device=device)
    zeros_body = torch.zeros((1, 133), dtype=torch.float32, device=device)
    zeros_hand = torch.zeros((1, 108), dtype=torch.float32, device=device)
    zeros_scale = torch.zeros((1, mhr_head.num_scale_comps), dtype=torch.float32, device=device)
    zeros_shape = torch.zeros((1, mhr_head.num_shape_comps), dtype=torch.float32, device=device)
    zeros_expr = torch.zeros((1, mhr_head.num_face_comps), dtype=torch.float32, device=device)
    common = dict(global_trans=zeros_trans, scale_params=zeros_scale,
                  shape_params=zeros_shape, expr_params=zeros_expr,
                  return_joint_rotations=True, return_joint_coords=True)
    with torch.no_grad():
        posed_out = mhr_head.mhr_forward(
            global_rot=_to_batched_tensor(pose_data.get("global_rot"), device, width=3),
            body_pose_params=_to_batched_tensor(pose_data.get("body_pose_params"), device, width=133),
            hand_pose_params=_to_batched_tensor(pose_data.get("hand_pose_params"), device, width=108),
            return_keypoints=True, **common)
        rest_out = mhr_head.mhr_forward(
            global_rot=zeros_rot, body_pose_params=zeros_body, hand_pose_params=zeros_hand, **common)

    def _collect(outputs, want_keypoints):
        rots = coords = kps = None
        for tensor in outputs[1:]:
            if tensor.ndim == 4 and tensor.shape[-1] == 3 and tensor.shape[-2] == 3:
                rots = tensor.detach().cpu().numpy()
            elif tensor.ndim == 3 and tensor.shape[-1] == 3 and tensor.shape[-2] > 127:
                kps = tensor.detach().cpu().numpy()
            elif tensor.ndim == 3 and tensor.shape[-1] == 3 and tensor.shape[-2] != 3:
                coords = tensor.detach().cpu().numpy()
        if rots is not None and rots.ndim == 4: rots = rots[0]
        if coords is not None and coords.ndim == 3: coords = coords[0]
        if want_keypoints and kps is not None:
            if kps.ndim == 3: kps = kps[0]
            kps = kps[:70]
        return rots, coords, kps

    posed_rots, posed_coords, posed_kps = _collect(posed_out, True)
    rest_rots, rest_coords, _ = _collect(rest_out, False)
    if posed_rots is not None: pose_data["joint_rotations"] = posed_rots.tolist()
    if posed_coords is not None: pose_data["joint_coords"] = posed_coords.tolist()
    if posed_kps is not None: pose_data["canonical_keypoints_3d"] = posed_kps.tolist()
    if rest_rots is not None: pose_data["rest_joint_rotations"] = rest_rots.tolist()
    if rest_coords is not None: pose_data["rest_joint_coords"] = rest_coords.tolist()
    try:
        _get_mhr_rest_verts(mhr_head, device)
        parents = _FACE_BS_CACHE.get("joint_parents")
        if parents is not None:
            pose_data["joint_parents"] = np.asarray(parents, dtype=np.int32).tolist()
    except Exception:
        pass
    num_joints = 0
    for candidate in (pose_data.get("joint_rotations"), pose_data.get("rest_joint_rotations"), pose_data.get("joint_coords")):
        if isinstance(candidate, list):
            num_joints = max(num_joints, len(candidate))
    known = {1: "pelvis", 2: "thigh_l", 3: "calf_l", 4: "foot_l", 18: "thigh_r", 19: "calf_r", 20: "foot_r",
             35: "spine_01", 36: "spine_02", 37: "spine_03", 38: "clavicle_r", 39: "upperarm_r", 40: "lowerarm_r",
             42: "hand_r", 74: "clavicle_l", 75: "upperarm_l", 76: "lowerarm_l", 78: "hand_l", 110: "neck_01", 113: "head"}
    pose_data["joint_names"] = [known.get(i, f"joint_{i:03d}") for i in range(num_joints)]
    pose_data["sam3d_pose_space"] = "mhr_forward_canonical"
    return pose_data




class GaoshangMergeBboxes:
    """合并两路人物框（如 YOLO 分割 + DWPose 关键点），IoU 去重后输出。

    拥抱/侧视等场景单一检测器容易漏人或重复框住同一人：两路合并互相补漏，
    NMS 去掉扣在同一人身上的重复框（保留置信路顺序：先 A 后 B）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bboxes_a": ("BOUNDING_BOX",),
            "bboxes_b": ("BOUNDING_BOX",),
            "iou_threshold": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 0.9, "step": 0.05}),
        }}

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "merge"
    CATEGORY = "3D多人姿势编辑器"

    @classmethod
    def IS_CHANGED(cls, bboxes_a, bboxes_b, iou_threshold):
        return float("nan")

    def merge(self, bboxes_a, bboxes_b, iou_threshold):
        def normalize(source):
            items = source if isinstance(source, list) else [source]
            flat = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            out = []
            for box in flat:
                if isinstance(box, dict):
                    x = float(box.get("x", 0)); y = float(box.get("y", 0))
                    w = float(box.get("width", 0)); h = float(box.get("height", 0))
                else:
                    x, y, w, h = (float(v) for v in box[:4])
                if w > 4 and h > 4:
                    out.append((x, y, w, h))
            return out

        # 实例分割为主：勾肩/拥抱两人的掩膜框彼此重叠度可达 90%（左人框几乎
        # 整个装在右人框里），任何几何去重(IoU/包含/面积)都无法区分"重叠的真
        # 两人"和"同一人的重复框"——全部会误杀。YOLO 实例数即人数，直接采用；
        # DWPose 关键点框仅在 YOLO 一无所获时兜底（单人/极端遮挡）。
        primary = normalize(bboxes_a)
        fallback = normalize(bboxes_b) if not primary else []
        boxes = primary if primary else fallback
        return ([{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in boxes],)

class GaoshangYoloMaskedCrops:
    """YOLO 实例分割 → 每人一张"只剩自己"的掩膜图 + 每人一框。

    紧密拥抱/勾肩的多人掩膜框彼此近全重叠（每框都含两人），矩形裁剪无法
    隔离个体，SAM3D 会重建出融合人。本节点把其他人像素涂黑后整帧输出
    （batch），配 list[list[box]] 逐帧单框喂 SAM3DBody_Predict，实现真正
    的按实例隔离重建。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "confidence_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.9, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE", "BOUNDING_BOX")
    RETURN_NAMES = ("masked_images", "bboxes")
    FUNCTION = "split"
    CATEGORY = "3D多人姿势编辑器"

    @classmethod
    def IS_CHANGED(cls, image, confidence_threshold):
        return float("nan")

    def split(self, image, confidence_threshold):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        import folder_paths as _fp
        import os as _os

        model_path = None
        for root in _fp.get_folder_paths("yolo"):
            candidate = _os.path.join(root, "person_yolov8m-seg.pt")
            if _os.path.isfile(candidate):
                model_path = candidate
                break
        if model_path is None:
            raise ValueError("找不到 person_yolov8m-seg.pt（models/yolo/）")

        arr = (image.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        batch = []
        boxes_out = []
        for bi in range(arr.shape[0]):
            rgb = arr[bi]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            result = YOLO(model_path)(bgr, conf=confidence_threshold, iou=0.75, verbose=False)[0]
            if result.boxes is None or result.boxes.conf is None:
                continue
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            polys = result.masks.xy if result.masks is not None else []
            for idx in range(len(confs)):
                if int(clss[idx]) != 0 or float(confs[idx]) <= confidence_threshold:
                    continue
                if idx >= len(polys) or len(polys[idx]) < 8:
                    continue
                pts = np.asarray(polys[idx])
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                H, W = rgb.shape[:2]
                pad = 0.05 * max(x2 - x1, y2 - y1)
                bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
                bx2, by2 = min(W, x2 + pad), min(H, y2 + pad)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                masked = rgb.copy()
                masked[mask == 0] = 0
                batch.append(torch.from_numpy(masked.astype(np.float32) / 255.0))
                boxes_out.append({"x": float(bx1), "y": float(by1),
                                  "width": float(bx2 - bx1), "height": float(by2 - by1)})
        if not batch:
            return (image, [])
        images = torch.stack(batch, dim=0)
        return (images, [[b] for b in boxes_out])


class GaoshangFaceSeedCrops:
    """脸种子 SAM2 逐人分离：face_yolov8m 检脸 → 每脸一正点+其余脸负点 →
    SAM2 多掩码候选取"含自己脸、不含他人脸、面积最大"的全身掩膜 →
    逐人涂黑他人整帧输出 + 逐帧单框，喂 SAM3DBody_Predict 按实例重建。

    侧视紧抱等 YOLO 实例分割分不开的场景：两张掩膜横跨全幅彼此包含，
    而脸是唯一可靠的个体锚点——SAM2 点提示按"谁是谁"切开纠缠掩膜
    （00213 实测两掩膜 IoU 0.004，质心分居左右）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "sam2_model": ("SAM2MODEL",),
            "face_confidence": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                          "tooltip": "人脸检测置信度阈值"}),
            "max_people": ("INT", {"default": 9, "min": 1, "max": 9, "step": 1,
                                   "tooltip": "最多按脸数取置信度前 N 人"}),
        }}

    RETURN_TYPES = ("IMAGE", "BOUNDING_BOX")
    RETURN_NAMES = ("masked_images", "bboxes")
    FUNCTION = "split"
    CATEGORY = "3D多人姿势编辑器"
    DESCRIPTION = "人脸种子SAM2逐人分离：紧抱/勾肩等实例分割失效场景的掩膜裁剪（上游接 DownloadAndLoadSAM2Model，segmentor=single_image）。"

    @classmethod
    def IS_CHANGED(cls, image, sam2_model, face_confidence, max_people):
        return float("nan")

    _face_model = None
    _face_model_path = None
    _person_model = None
    _person_model_path = None

    @classmethod
    def _person_instances(cls, frame):
        """YOLO 人物实例框（conf 0.3，按置信度排序）——只用于数人和补种子，
        前后站位时后位的脸常被挡/背对，实例数是比脸数更可靠的人数下限。"""
        import os as _os
        import folder_paths as _fp
        path = None
        for root in _fp.get_folder_paths("yolo"):
            candidate = _os.path.join(root, "person_yolov8m-seg.pt")
            if _os.path.isfile(candidate):
                path = candidate
                break
        if path is None:
            return []
        if cls._person_model is None or cls._person_model_path != path:
            from ultralytics import YOLO
            cls._person_model = YOLO(path)
            cls._person_model_path = path
        result = cls._person_model(frame, conf=0.3, iou=0.75, verbose=False)[0]
        if result.boxes is None or result.boxes.conf is None:
            return []
        pairs = sorted(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()),
                       key=lambda kv: -float(kv[1]))
        return [(float(a[0]), float(a[1]), float(a[2]), float(a[3])) for a, _ in pairs]

    @classmethod
    def _load_face_model(cls):
        import os as _os
        import folder_paths as _fp
        candidates = []
        for root in _fp.get_folder_paths("ultralytics"):
            candidates += [_os.path.join(root, "bbox", "face_yolov8m.pt"),
                           _os.path.join(root, "face_yolov8m.pt")]
        candidates.append(_os.path.join(_fp.models_dir, "ultralytics", "bbox", "face_yolov8m.pt"))
        path = next((p for p in candidates if _os.path.isfile(p)), None)
        if path is None:
            raise ValueError("找不到 face_yolov8m.pt（models/ultralytics/bbox/）")
        if cls._face_model is None or cls._face_model_path != path:
            from ultralytics import YOLO
            cls._face_model = YOLO(path)
            cls._face_model_path = path
        return cls._face_model

    def split(self, image, sam2_model, face_confidence, max_people):
        import contextlib
        import cv2
        import torch

        predictor = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        face_model = self._load_face_model()

        arr = (image.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        batch, boxes_out, report = [], [], []
        for bi in range(arr.shape[0]):
            rgb = arr[bi]
            H, W = rgb.shape[:2]
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fr = face_model(frame, conf=face_confidence, iou=0.5, imgsz=640, verbose=False)[0]
            faces = []
            if fr.boxes is not None and fr.boxes.conf is not None:
                xyxy = fr.boxes.xyxy.cpu().numpy()
                confs = fr.boxes.conf.cpu().numpy()
                for (x0, y0, x1, y1), conf in zip(xyxy, confs):
                    faces.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                                  "cx": float((x0 + x1) / 2), "cy": float((y0 + y1) / 2), "conf": float(conf)})
            faces.sort(key=lambda f: -f["conf"])
            faces = faces[:max_people]
            # 前后站位/遮挡兜底：人物实例数多于脸数（后位脸被挡/背对/漏检）时，
            # 先低阈值重试检脸；仍缺的用"不含任何脸心的人物实例"躯干点当种子补位
            # （与脸种子统一规则：互为负点、掩膜须含自己种子不含他人种子）。
            instances = self._person_instances(frame)
            wanted = min(max_people, len(instances))
            if len(faces) < wanted:
                fr2 = face_model(frame, conf=min(face_confidence, 0.1), iou=0.5, imgsz=640, verbose=False)[0]
                if fr2.boxes is not None and fr2.boxes.conf is not None:
                    for (x0, y0, x1, y1), conf in zip(fr2.boxes.xyxy.cpu().numpy(), fr2.boxes.conf.cpu().numpy()):
                        cx, cy = float((x0 + x1) / 2), float((y0 + y1) / 2)
                        if any((f["cx"] - cx) ** 2 + (f["cy"] - cy) ** 2 < (0.06 * min(H, W)) ** 2 for f in faces):
                            continue
                        faces.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                                      "cx": cx, "cy": cy, "conf": float(conf)})
                faces.sort(key=lambda f: -f["conf"])
                faces = faces[:wanted]
            for box in instances:
                if len(faces) >= wanted:
                    break
                bx0, by0, bx1, by1 = box
                cx, cy = (bx0 + bx1) / 2, by0 + (by1 - by0) * 0.35  # 躯干点
                if any(bx0 - 0.05 * (bx1 - bx0) <= f["cx"] <= bx1 + 0.05 * (bx1 - bx0)
                       and by0 - 0.05 * (by1 - by0) <= f["cy"] <= by1 + 0.05 * (by1 - by0) for f in faces):
                    continue  # 该实例已有脸种子
                faces.append({"x0": float(bx0), "y0": float(by0), "x1": float(bx1), "y1": float(by1),
                              "cx": float(cx), "cy": float(cy), "conf": 0.0, "fallback": True})
            if not faces:
                continue

            autocast = torch.autocast(device.type, dtype=dtype) if device.type != "cpu" else contextlib.nullcontext()
            with autocast:
                predictor.set_image(rgb)
                masks_picked = []
                for i, face in enumerate(faces):
                    coords = [[face["cx"], face["cy"]]] + [[g["cx"], g["cy"]] for j, g in enumerate(faces) if j != i]
                    labels = [1] + [0] * (len(coords) - 1)
                    cand, scores, _ = predictor.predict(
                        point_coords=np.asarray(coords, dtype=np.float32),
                        point_labels=np.asarray(labels, dtype=np.int32),
                        multimask_output=True)
                    cands = [np.asarray(cand[k], dtype=bool) for k in range(cand.shape[0])]
                    def _ok(m):
                        inside = m[min(H - 1, int(face["cy"])), min(W - 1, int(face["cx"]))]
                        others = [m[min(H - 1, int(g["cy"])), min(W - 1, int(g["cx"]))]
                                  for j, g in enumerate(faces) if j != i]
                        return bool(inside) and not any(others)
                    pool = [k for k in range(len(cands)) if _ok(cands[k])]
                    if not pool:
                        pool = [k for k in range(len(cands))
                                if cands[k][min(H - 1, int(face["cy"])), min(W - 1, int(face["cx"]))]]
                    if not pool:
                        continue
                    masks_picked.append((face, cands[max(pool, key=lambda k: int(cands[k].sum()))]))

            frame_report = {"frame": bi, "faces": faces, "masks": []}
            for face, mask in masks_picked:
                ys, xs = np.nonzero(mask)
                if len(xs) == 0 or (ys.max() - ys.min()) < 30 or (xs.max() - xs.min()) < 30:
                    continue
                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                pad = 0.05 * max(x2 - x1, y2 - y1)
                bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
                bx2, by2 = min(W, x2 + pad), min(H, y2 + pad)
                masked = rgb.copy()
                masked[mask == 0] = 0
                batch.append(torch.from_numpy(masked.astype("float32") / 255.0))
                boxes_out.append({"x": float(bx1), "y": float(by1),
                                  "width": float(bx2 - bx1), "height": float(by2 - by1)})
                frame_report["masks"].append({"area": int(len(xs)), "x0": int(xs.min()), "x1": int(xs.max()),
                                              "y0": int(ys.min()), "y1": int(ys.max()),
                                              "cx": round(float(xs.mean()), 1),
                                              "face_cx": round(face["cx"], 1), "face_cy": round(face["cy"], 1),
                                              "face_h": round(float(face["y1"] - face["y0"]), 1)})
            if len(frame_report["masks"]) >= 2:
                picked = [m for _, m in masks_picked][:len(frame_report["masks"])]
                ious = []
                for a in range(len(picked)):
                    for b in range(a + 1, len(picked)):
                        inter = int((picked[a] & picked[b]).sum())
                        union = int((picked[a] | picked[b]).sum())
                        ious.append(round(inter / max(1, union), 4))
                frame_report["pair_IoU"] = ious
            report.append(frame_report)

        # 侧写文件：掩膜真实几何（照片真值）供 MHRToOpenPose 回校骨架尺度/编辑器定大小
        try:
            import os as _os
            import time as _time
            import folder_paths as _fp
            folder = _os.path.join(_fp.get_input_directory(), "gaoshang_openpose")
            _os.makedirs(folder, exist_ok=True)
            crops = {"version": int(_time.time() * 1000), "frames": report}
            with open(_os.path.join(folder, "face_crops.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(crops, ensure_ascii=False))
        except Exception as error:
            print("GaoshangFaceSeedCrops write face_crops failed:", error)

        if not batch:
            return (image, [])
        images = torch.stack(batch, dim=0)
        ui = {"gaoshang_faces": [json.dumps(report, ensure_ascii=False)]}
        return {"ui": ui, "result": (images, [[b] for b in boxes_out])}


class GaoshangTwoWaySwitch:
    """单一总开关两态分发（脸种子版专用）：generate=True → 生成路通/检测路断；
    False 反之。断的一路返回 ExecutionBlocker，下游整链不执行（大模型零加载）。
    取代 布尔源+取反+双闸 的积木组合——那套在缓存回放与前端 widget 序列化上
    反复出问题（旧缓存放行陈旧结果、残留死控件），本节点 IS_CHANGED 恒变禁缓存。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "generate": ("BOOLEAN", {"default": True, "label_on": "ON=生成白模", "label_off": "OFF=检测"}),
            "detection_input": ("*", {"forceInput": True}),
            "generation_input": ("*", {"forceInput": True}),
        }}

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("检测路输出", "生成路输出")
    FUNCTION = "switch"
    CATEGORY = "3D多人姿势编辑器"
    DESCRIPTION = "总开关：OFF 只跑 SAM3D 检测路，ON 只跑 人偶图→白模 生成路。"

    @classmethod
    def IS_CHANGED(cls, generate, detection_input, generation_input):
        return float("nan")

    def switch(self, generate, detection_input, generation_input):
        from comfy_execution.graph import ExecutionBlocker
        if generate:
            return (ExecutionBlocker(None), generation_input)
        return (detection_input, ExecutionBlocker(None))


class GaoshangMHRToOpenPose:
    """SAM3D-Body 多人姿态 -> OpenPose JSON（STRING）。

    同时把结果写入 input/gaoshang_openpose/sam3d_multi.json：高尚自由姿势编辑器
    打开时会读取该文件，自动把每个人偶摆成 SAM3D 估计的姿势（跑过一次本节点后生效）。
    """
    OUTPUT_NODE = True  # 无下游也要执行：侧写文件是本节点的本职产物

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mhr_pose": ("MHR_POSE_DATA",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openpose_json",)
    FUNCTION = "convert"
    CATEGORY = "3D多人姿势编辑器"

    @classmethod
    def IS_CHANGED(cls, mhr_pose):
        return float("nan")  # 侧写文件 + sam3d 载荷需每次落盘，禁用输出缓存

    _face_crops_cache = None
    _face_crops_version = None

    @classmethod
    def _load_face_crops(cls):
        """上游 GaoshangFaceSeedCrops 落盘的掩膜几何（按 version 缓存）。用途只有一个：
        可见度判定（掩膜高/脸高在 5-8 = 全身可见）——被遮挡者的重建不可信，
        编辑器让其 zoom 采用组内可见者基准，不做任何体型推断。"""
        import os as _os
        import folder_paths as _fp
        path = _os.path.join(_fp.get_input_directory(), "gaoshang_openpose", "face_crops.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        version = data.get("version")
        if cls._face_crops_version != version:
            masks = []
            for frame in data.get("frames", []):
                masks.extend(frame.get("masks", []))
            cls._face_crops_cache = masks
            cls._face_crops_version = version
        return cls._face_crops_cache or None

    def convert(self, mhr_pose):
        from comfy_extras.sam3d_body.export.glb_shared import openpose_render_keypoints
        frames = mhr_pose.get("frames") or []
        # 逐帧单框批次（掩膜分离链）每帧一人：拍平所有帧；单帧多框（合并框链）拍平后等价。
        total_persons = sum(len(f) for f in frames)
        print("GaoshangMHRToOpenPose.convert ENTER, frames:", len(frames), "persons:", total_persons)
        sam3d_model = None
        people = []
        width = height = 0
        try:
            frames = mhr_pose.get("frames") or []
            size = mhr_pose.get("image_size") or (0, 0)
            height, width = int(size[0]), int(size[1])
            kept_cam = []
            raw_persons = [person for frame in frames for person in frame]
            # 图像空间去重：同一人被双检测器/掩膜碎片各重建一次 → 2D 重心几乎重合。
            def _person_stats(person):
                body = openpose_render_keypoints(person, mhr_pose, "body", dim=2, H=height, W=width)
                if body is None or body.size < 18 * 2:
                    return None
                kp = []
                for row in body:
                    ok = bool(np.isfinite(row).all())
                    kp += [float(row[0]) if ok else 0.0, float(row[1]) if ok else 0.0, 1.0 if ok else 0.0]
                xs = [kp[i * 3] for i in range(18) if kp[i * 3 + 2] > 0.1]
                ys = [kp[i * 3 + 1] for i in range(18) if kp[i * 3 + 2] > 0.1]
                if not xs:
                    return None
                conf = sum(kp[i * 3 + 2] for i in range(18)) / 18.0
                return {"kp": kp, "mx": sum(xs) / len(xs), "my": sum(ys) / len(ys), "conf": conf}
            stats = [ _person_stats(p) for p in raw_persons ]
            def _cam_of(p):
                try:
                    return [float(v) for v in np.asarray(p.get("pred_cam_t")).reshape(-1)[:3]]
                except Exception:
                    return None
            cams = [_cam_of(p) for p in raw_persons]
            # 逐帧单框链（脸种子掩膜分离）：一人一帧，结构上不可能同人重复——
            # 直接全保留。图像空间去重只服务旧的单帧多框链（双检测器重复）。
            # 前后站位两人 2D 重心可近同、独立重建的 cam_t 也会收敛（实测 z 差
            # 0.09m），任何近邻判同都会误杀，故此链路禁用。
            per_frame_single = len(frames) > 1 and all(len(f) == 1 for f in frames)
            keep_idx = list(range(len(raw_persons))) if per_frame_single else []
            for i, st_i in enumerate(stats):
                if st_i is None:
                    keep_idx.append(i)  # 无法统计的（低质）也交给后续流程自然过滤
                    continue
                dup = None
                for j in keep_idx:
                    st_j = stats[j] if j < len(stats) else None
                    if st_j is None:
                        continue
                    near2d = abs(st_i["mx"] - st_j["mx"]) < 0.09 * width and abs(st_i["my"] - st_j["my"]) < 0.14 * height
                    if not near2d:
                        continue
                    # 前后站位两人 2D 重心可几乎同 x（同列站位是合法场景）——
                    # 真重复（同像素重建两次）的 cam_t 几乎重合（<5cm），
                    # 真人前后 z 差至少 0.3m：加深度条件区分（无 cam_t 才退回纯 2D）。
                    ci, cj = cams[i], cams[j] if j < len(cams) else None
                    near3d = ci is None or cj is None or (abs(ci[0] - cj[0]) < 0.25 and abs(ci[2] - cj[2]) < 0.25)
                    if near2d and near3d:
                        dup = j
                        break
                if dup is None:
                    keep_idx.append(i)
                elif st_i["conf"] > (stats[dup]["conf"] if stats[dup] else -1):
                    keep_idx.remove(dup)
                    keep_idx.append(i)
            for person in [raw_persons[i] for i in keep_idx]:
                # 同一人被双检测器各重建一次 → cam_t 几乎重合（同像素，<5cm），丢弃后者。
                # 半径取 0.1m：紧抱真人 cam_t 差常见 0.2-0.3m，0.3m 半径会误杀（00213 实证）。
                cam_vec = None
                try:
                    cam_vec = [float(v) for v in np.asarray(person.get("pred_cam_t")).reshape(-1)[:3]]
                except Exception:
                    pass
                cam_radius = 0.05 if per_frame_single else 0.1
                if cam_vec and any(abs(cam_vec[0]-k[0])<cam_radius and abs(cam_vec[2]-k[2])<cam_radius for k in kept_cam):
                    continue
                if cam_vec:
                    kept_cam.append(cam_vec)
                body = openpose_render_keypoints(person, mhr_pose, "body", dim=2, H=height, W=width)
                if body is None or body.size < 18 * 2:
                    continue
                kp = []
                for row in body:
                    ok = bool(np.isfinite(row).all())
                    kp += [float(row[0]) if ok else 0.0, float(row[1]) if ok else 0.0, 1.0 if ok else 0.0]
                entry = {"pose_keypoints_2d": kp}
                # 与 GaoshangSceneFromMHR 完全同源的真三维姿势（旧编辑器验证过的数学）：
                # dim=3(相机系Y下,已摆姿势) + cam_t → 地面 Y 上世界系，取颈部原点的相对关节。
                try:
                    kp3 = openpose_render_keypoints(person, mhr_pose, "body", dim=3)
                    cam_t = np.asarray(person.get("pred_cam_t"), dtype=np.float32).reshape(1, 3)
                    if kp3 is not None and getattr(kp3, "size", 0) == 18 * 3 and cam_t.size == 3:
                        pts = np.asarray(kp3, dtype=np.float32) + cam_t
                        ground = min(float(pts[10, 1]), float(pts[13, 1]))   # 脚踝最低点
                        far = float(pts[1, 2])                               # 颈部深度
                        world = pts.copy()
                        world[:, 1] = ground - pts[:, 1]
                        world[:, 2] = far - pts[:, 2]
                        origin = world[1].copy()                             # 颈部为原点
                        rel = world - origin.reshape(1, 3)
                        rel[0, 1] += .005
                        rel[0, 2] = -.11
                        entry["pose_world3d"] = [float(v) for v in rel.reshape(-1)]
                        entry["origin"] = [float(v) for v in origin]
                except Exception as error3:
                    print("GaoshangMHRToOpenPose world3d failed:", error3)
                # VNCCS 原生 SAM3D 载荷（作者的 MakeHuman 刚性求解器直接吃）：
                # 姿势/朝向/手部全部真值，2D 启发式彻底退役。模型只加载一次。
                # 姿势/朝向/手部全部真值，2D 启发式彻底退役。模型只加载一次。
                try:
                    if sam3d_model is None:
                        from ComfyUI_VNCCS_Utils.vnccs_sam3d.processing.load_model import LoadSAM3DBodyModel
                        sam3d_model = LoadSAM3DBodyModel().load_model("Auto")[0]
                    entry["sam3d"] = _enrich_sam3d_person(person, sam3d_model)
                    print("GaoshangMHRToOpenPose: sam3d payload OK, joints=", len(entry["sam3d"].get("joint_rotations", [])))
                except Exception as error4:
                    import traceback as _tb
                    with open(r"D:/qwen-2.1/gaoshang_debug.log", "a", encoding="utf-8") as _h:
                        _h.write(_tb.format_exc() + chr(10))
                try:
                    eul = np.asarray(person.get("global_rot"), dtype=np.float32).reshape(3)
                    cz, sz = np.cos(eul[0]), np.sin(eul[0])
                    cy, sy = np.cos(eul[1]), np.sin(eul[1])
                    cx, sx = np.cos(eul[2]), np.sin(eul[2])
                    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
                    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
                    forward = (Rz @ Ry @ Rx) @ np.array([0, 0, 1], dtype=np.float32)
                    entry["yaw_deg"] = float(np.degrees(np.arctan2(forward[0], forward[2])))
                except Exception:
                    pass
                # 真 3D 通道：dim=3 是相机系(Y下)米制 pre-cam_t；翻转 y/z 得世界 Y 上，
                # 保留每人真实站位/身高/朝向，编辑器直接吃世界关节而不再 2D 推深度。
                try:
                    body3 = openpose_render_keypoints(person, mhr_pose, "body", dim=3)
                    if body3 is not None and body3.size == 18 * 3:
                        world3 = np.asarray(body3, dtype=np.float32).copy()
                        world3[:, 1] *= -1.0
                        world3[:, 2] *= -1.0
                        entry["pose_keypoints_3d"] = [list(map(float, row)) for row in world3]
                except Exception as error3:
                    print("GaoshangMHRToOpenPose 3d failed:", error3)
                if cam_vec:
                    entry["cam_t"] = [round(float(v), 4) for v in cam_vec]
                # 可见度判定（非体型推断）：掩膜高/脸高在人类区间 5-8 = 全身可见，
                # 重建可信；被遮挡者（紧抱后位）的重建整体会缩水（00096 实测骨长比
                # 0.65=投影比 0.68），编辑器让其大小采用组内可见者基准。
                try:
                    crops = self._load_face_crops()
                    if crops:
                        best = None
                        for m in crops:
                            d = ((m["face_cx"] - kp[0]) ** 2 + (m["face_cy"] - kp[1]) ** 2) ** 0.5
                            if best is None or d < best[0]:
                                best = (d, m)
                        if best and best[0] < 0.15 * max(width, height) and best[1].get("face_h", 0) > 0:
                            ratio = (best[1]["y1"] - best[1]["y0"]) / best[1]["face_h"]
                            entry["fully_visible"] = bool(5.0 <= ratio <= 8.0)
                except Exception as error7:
                    print("GaoshangMHRToOpenPose visibility check failed:", error7)
                people.append(entry)
        except Exception as error:
            print("GaoshangMHRToOpenPose convert failed:", error)
        # 同平面定尺：单目深度各自多解（远处大人≡近处小人），两人基本并排/紧抱的
        # 假设是唯一几何规尺——每人 3D 骨架按 λ=z*/z_i 等比缩放（z*=组内基准深度），
        # 投影不变（所见即所得），两组骨架在同一深度下尺度可比。2D 投影一律不改写。
        try:
            zs = [p["cam_t"][2] for p in people if p.get("cam_t") and p.get("pose_world3d")]
            if len(people) >= 2 and len(zs) == len(people):
                ordered = sorted(zs)
                z_ref = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
                for p in people:
                    lam = z_ref / p["cam_t"][2]
                    if 0.5 < lam < 2.0:
                        p["pose_world3d"] = [v * lam for v in p["pose_world3d"]]
                        p["origin"] = [v * lam for v in p["origin"]]
                        p["scale_same_plane"] = round(float(lam), 3)
                    else:
                        lam = 1.0
                    # 共享相机系真实站位（同平面规尺下）：横向=cam_t.x×λ，纵深统一 z*。
                    # 编辑器按此左右次序摆放并强制最小间距——紧抱真实横距仅 0.2-0.3m，
                    # 直接采用人偶会叠模。
                    p["world_pos"] = [round(float(p["cam_t"][0]) * lam, 3), 0.0, round(float(z_ref), 3)]
        except Exception as error6:
            print("GaoshangMHRToOpenPose same-plane normalization failed:", error6)
        # 体型与性别估计（同平面规尺下的骨架度量，缩水不影响比值）：肩宽/髋宽比是
        # 经典性别二态特征（男~1.4 女~1.15-1.25）；相对身高/躯干宽给编辑器 morph 滑杆。
        # 被遮挡者绝对尺寸不可信（历史实测缩水），高度/体重取组默认，性别比值保留。
        try:
            def _v(p, i):
                w = p.get("pose_world3d") or []
                return np.array([w[i * 3], w[i * 3 + 1], w[i * 3 + 2]], dtype=np.float32) if len(w) >= 54 else None
            def _h3d(p):
                w = p.get("pose_world3d") or []
                if len(w) < 54: return None
                ys = [w[i * 3 + 1] for i in range(14)]
                return max(ys) - min(ys)
            def _torso(p):
                rs, ls, rh, lh = _v(p, 2), _v(p, 5), _v(p, 8), _v(p, 11)
                if any(x is None for x in (rs, ls, rh, lh)): return None
                return (float(np.linalg.norm(rs - ls)) + float(np.linalg.norm(rh - lh))) / 2
            heights = [h for h in (_h3d(p) for p in people) if h and h > 0.2]
            torsos = [t for t in (_torso(p) for p in people) if t and t > 0.05]
            h_med = sorted(heights)[len(heights) // 2] if heights else None
            t_med = sorted(torsos)[len(torsos) // 2] if torsos else None
            for p in people:
                rs, ls, rh, lh = _v(p, 2), _v(p, 5), _v(p, 8), _v(p, 11)
                body = {}
                if all(x is not None for x in (rs, ls, rh, lh)):
                    sh = float(np.linalg.norm(rs - ls)); hp = float(np.linalg.norm(rh - lh))
                    if hp > 0.02:
                        # 骨架肩髋比对性别无判别力：已知两男标定 1.96-2.30，
                        # 覆盖全部观测范围（重建噪声支配）。性别默认中性，
                        # 比值仅存诊断；可靠性别待 VLM 通道（见档案）。
                        body["gender"] = 0.5
                        body["shoulder_hip"] = round(sh / hp, 2)
                h = _h3d(p); t = _torso(p)
                if h_med and h and h > 0.2 and p.get("fully_visible") is not False:
                    body["height"] = round(0.5 + min(0.25, max(-0.25, (h / h_med - 1) / 0.30 * 0.25)), 2)
                else:
                    body["height"] = 0.5
                if t_med and t and h and h > 0.2 and p.get("fully_visible") is not False:
                    rel = (t / h) / (t_med / h_med)
                    body["weight"] = round(0.5 + min(0.22, max(-0.18, (rel - 1) / 0.12 * 0.2)), 2)
                else:
                    body["weight"] = 0.5
                body["muscle"] = round(min(0.85, max(0.15, 0.25 + 0.45 * ((body.get("gender", 0.5) - 0.15) / 0.7) + 0.15 * ((body["weight"] - 0.5) * 4))), 2)
                if body:
                    p["body"] = body
        except Exception as error8:
            print("GaoshangMHRToOpenPose body estimation failed:", error8)
        import time as _time
        payload = {"people": people, "canvas_width": width, "canvas_height": height,
                   "version": int(_time.time())}
        out = json.dumps(payload, ensure_ascii=False)
        try:
            import os as _os
            import folder_paths as _fp
            folder = _os.path.join(_fp.get_input_directory(), "gaoshang_openpose")
            _os.makedirs(folder, exist_ok=True)
            with open(_os.path.join(folder, "sam3d_multi.json"), "w", encoding="utf-8") as handle:
                handle.write(out)
        except Exception as error:
            print("GaoshangMHRToOpenPose write failed:", error)
        return {"ui": {"mhr_people": [str(len(people))]}, "result": (out,)}
