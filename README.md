# 3D多人姿势编辑器 · 3D Multi-person Pose Editor

ComfyUI 浏览器端 3D 多人姿势编辑器 + 照片→姿势全自动检测链 + 白模生成，为 Qwen-Image-2.1 姿势控制而生。

从照片里紧抱/勾肩/前后站位的多人，自动分离、逐人 3D 重建、在编辑器里摆成任意姿势，再一键生成纯白人偶模型（白模）用于下游生成。

## 功能一览

**检测链（照片 → 多人 3D 姿势）**
- **脸种子 SAM2 分离**：face_yolov8m 检脸 → 每脸一正点 + 其余脸负点 → SAM2 多掩码候选取全身掩膜。侧视紧抱、前后站位等实例分割失效场景均可分离（实测双人掩膜 IoU 0.004）
- **人物实例兜底**：脸被挡/背对/漏检时，YOLO 人物实例自动补种子，不再丢人
- **SAM3D-Body 逐人重建**：掩膜涂黑他人 → 逐人 3D 姿态估计（真深度）
- **同平面定尺**：单目尺度不确定用"同平面假设"固定（λ=z*/zᵢ 等比缩放，投影所见即所得）
- **可见度基准 zoom**：被遮挡者重建缩水时，大小自动采用组内可见者基准

**编辑器（1-9 人偶，浏览器内 3D）**
- MakeHuman 真人比例网格人偶，拖关节摆姿 / FK 旋转 / IK
- 统一视角控制：**中键平移、右键旋转**（全模式一致）
- **「全部」组选择**：整组等比缩放/平移/旋转，单人相对关系不变
- 画幅默认跟随输入图比例，宽高联动等比缩放
- 内置姿势库 / OpenPose JSON / SAM3D 骨架一键导入
- 「按 SAM3D 重摆」一键套用照片姿势

**生成链（人偶图 → 白模）**
- ViggleTurbo 6 步白模 LoRA：逼真纯白人偶、纯黑背景、姿势逐像素一致（实测关节偏差 0.009）
- 总开关两态：OFF 只跑检测路（秒回、零大模型加载），ON 只跑白模生成路

## 环境要求

- ComfyUI ≥ 0.38（需内置 TextEncodeQwenImage21 与 SAM3D-Body 节点）
- 依赖插件：comfyui-segment-anything-2（Kijai）
- 生成链模型：Qwen-Image-2.1（int8 convrot UNET、qwen3vl TE、VAE）+ VNCCS_QI2_PoseStudioV1.1 LoRA + Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256

| 检测模型 | 放置路径 |
|---|---|
| face_yolov8m.pt | models/ultralytics/bbox/ |
| person_yolov8m-seg.pt | models/yolo/ |
| sam2_hiera_base_plus.safetensors | models/sam2/ |
| sam_3d_body_dinov3_int8_convrot.safetensors | models/detection/ |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Weilesscodeless/3D-Multi-person-Pose-Editor.git
```

可选：把 OpenPose 姿势图（黑底彩骨标准格式）放入插件目录 `高尚姿势图库/`，即可在编辑器左侧使用内置姿势库（因原始资产授权未明，仓库不含此目录）。

## 使用

完整示例工作流见 `workflows/3D多人姿势编辑器-完整工作流.json`。

1. **总开关 OFF** 跑一次 → 检测链落盘多人骨架（几秒）
2. 打开编辑器 → **「按 SAM3D 重摆」** → 按照片姿势自动摆好所有人偶
3. 微调（拖关节 / 组变换 / 画幅）→ **应用**
4. **总开关 ON** 再跑 → 输出纯白人偶白模图（Gaoshang_host_whitemodel）

编辑器操作：移动模式拖人物/方向键转身，关节模式拖关节，Shift+拖 移动人物，中键平移、右键旋转、滚轮缩放。

## 节点列表

| 节点 | 功能 |
|---|---|
| 3D多人姿势编辑器（GaoshangQwenFreePose） | 1-9 人浏览器 3D 编辑器，输出提示词/latent/人偶图 |
| 机位与姿态编辑器（GaoshangQwenPose） | 1-3 人 + 相机机位编辑 |
| 脸种子SAM2分离（GaoshangFaceSeedCrops） | 紧抱/前后站位掩膜分离，SAM2 点提示 |
| 总开关（两态）（GaoshangTwoWaySwitch） | OFF=检测路 / ON=生成路 单开关分发 |
| SAM3D转场景（GaoshangSceneFromMHR） | SAM3D 3D 姿态 → 编辑器场景 |
| 场景JSON桥（GaoshangPoseBridge） | 场景 → OpenPose JSON / 骨架图 |
| 关键点转多人bbox（GaoshangBboxesFromKeypoints） | DWPose 关键点 → 逐人 bbox |
| YOLO 掩膜裁剪（GaoshangYoloMaskedCrops） | YOLO 实例分割逐人掩膜裁剪（脸种子的备选） |

> 节点类名保留 Gaoshang 前缀以兼容旧工作流，重命名仅影响显示名。

## 致谢与许可

- 基于高尚 Gaoshang-Pose（MIT, © 2026 Work-Fisher）与 VNCCS PoseStudio 深度改造
- MakeHuman 人偶网格 CC0（见 web/vnccs/assets/ 许可文件）
- three.js（MIT，见 web/editor/vendor/LICENSE）
- 本项目代码 MIT
