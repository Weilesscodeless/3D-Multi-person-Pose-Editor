from .nodes import GaoshangPoseStudio
from .qwen_bridge import GaoshangQwenPose
from .gguf_clip import GaoshangQwen21GGUFCLIP
from .free_pose import GaoshangQwenFreePose
from .pose_bridge import GaoshangPoseBridge, GaoshangBboxesFromKeypoints, GaoshangSceneFromMHR
from .openpose_library import register_routes
from .saved_poses import register_routes as register_saved_pose_routes

register_routes()
register_saved_pose_routes()

NODE_CLASS_MAPPINGS = {"GaoshangPoseStudio": GaoshangPoseStudio, "GaoshangQwenPose": GaoshangQwenPose}
NODE_DISPLAY_NAME_MAPPINGS = {"GaoshangPoseStudio": "机位与姿态编辑器", "GaoshangQwenPose": "Qwen2.1 人物与姿态编码"}
NODE_CLASS_MAPPINGS['GaoshangQwen21GGUFCLIP'] = GaoshangQwen21GGUFCLIP
NODE_CLASS_MAPPINGS['GaoshangQwenFreePose'] = GaoshangQwenFreePose
NODE_DISPLAY_NAME_MAPPINGS['GaoshangQwenFreePose'] = '3D多人姿势编辑器'
NODE_CLASS_MAPPINGS['GaoshangPoseBridge'] = GaoshangPoseBridge
NODE_DISPLAY_NAME_MAPPINGS['GaoshangPoseBridge'] = '场景JSON桥'
NODE_CLASS_MAPPINGS['GaoshangBboxesFromKeypoints'] = GaoshangBboxesFromKeypoints
NODE_DISPLAY_NAME_MAPPINGS['GaoshangBboxesFromKeypoints'] = '关键点转多人bbox'
NODE_CLASS_MAPPINGS['GaoshangSceneFromMHR'] = GaoshangSceneFromMHR
from .pose_bridge import GaoshangMHRToOpenPose
NODE_CLASS_MAPPINGS['GaoshangMHRToOpenPose'] = GaoshangMHRToOpenPose
from .pose_bridge import GaoshangMergeBboxes
NODE_CLASS_MAPPINGS['GaoshangMergeBboxes'] = GaoshangMergeBboxes
from .pose_bridge import GaoshangYoloMaskedCrops
NODE_CLASS_MAPPINGS['GaoshangYoloMaskedCrops'] = GaoshangYoloMaskedCrops
from .pose_bridge import GaoshangFaceSeedCrops
NODE_CLASS_MAPPINGS['GaoshangFaceSeedCrops'] = GaoshangFaceSeedCrops
NODE_DISPLAY_NAME_MAPPINGS['GaoshangFaceSeedCrops'] = '脸种子SAM2分离'
NODE_DISPLAY_NAME_MAPPINGS['GaoshangSceneFromMHR'] = 'SAM3D转场景（真3D）'
NODE_DISPLAY_NAME_MAPPINGS['GaoshangQwen21GGUFCLIP'] = 'Qwen2.1 GGUF CLIP 加载器'
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .pose_bridge import GaoshangTwoWaySwitch
NODE_CLASS_MAPPINGS["GaoshangTwoWaySwitch"] = GaoshangTwoWaySwitch
NODE_DISPLAY_NAME_MAPPINGS["GaoshangTwoWaySwitch"] = "总开关（两态）"
