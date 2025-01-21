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
"""Constants for ALOHA."""

from mujoco_playground._src import mjx_env

XML_PATH = mjx_env.ROOT_PATH / "manipulation" / "aloha" / "xmls"

ARM_JOINTS = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
]

FINGER_GEOMS = [
    "left/left_finger_top",
    "left/left_finger_bottom",
    "left/right_finger_top",
    "left/right_finger_bottom",
    "right/left_finger_top",
    "right/left_finger_bottom",
    "right/right_finger_top",
    "right/right_finger_bottom",
]

FINGER_JOINTS = [
    "left/left_finger",
    "left/right_finger",
    "right/left_finger",
    "right/right_finger",
]
