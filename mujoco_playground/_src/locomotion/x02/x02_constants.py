# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for Berkeley Humanoid."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "x02"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
   ROOT_PATH / "xmls" / "scene_rough_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "l_foot",
    "r_foot",
]

LEFT_FEET_GEOMS = [
    "l_foot00",
    #"l_foot01",
    #"l_foot02",
    #"l_foot03",
    #"l_foot04",
    #"l_foot05",
    #"l_foot06",
    #"l_foot07",
    #"l_foot08",
    #"l_foot09",
    #"l_foot10",
    #"l_foot11",
]

RIGHT_FEET_GEOMS = [
    "r_foot00",
    #"r_foot01",
    #"r_foot02",
    #"r_foot03",
    #"r_foot04",
    #"r_foot05",
    #"r_foot06",
    #"r_foot07",
    #"r_foot08",
    #"r_foot09",
    #"r_foot10",
    #"r_foot11",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "pelvis_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
