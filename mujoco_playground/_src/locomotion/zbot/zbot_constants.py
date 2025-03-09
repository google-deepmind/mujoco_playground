"""Constants for ZBot."""

import pathlib

from etils import epath

ROOT_PATH = pathlib.Path(__file__).parent.resolve()

FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
    }[task_name]


FEET_SITES = ["l_foot", "r_foot"]
LEFT_FEET_GEOMS = ["r_foot1"]
RIGHT_FEET_GEOMS = ["r_foot1"]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "Z-BOT2_MASTER-BODY-SKELETON"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
