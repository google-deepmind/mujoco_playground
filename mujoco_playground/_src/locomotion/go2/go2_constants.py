from etils import epath

from mujoco_playground._src import mjx_env

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2"

MJX_XML_PATH = (
    ROOT_PATH / "xmls" / "scene_mjx_collision_free.xml"
)

MUJOCO_XML_PATH = (
    ROOT_PATH / "xmls" / "scene.xml"
)

ONNX_DIR = mjx_env.ROOT_PATH / "experimental" / "sim2sim" / "onnx"
