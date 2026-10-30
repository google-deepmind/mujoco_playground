# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trossen WXAI Arm constants for simulation."""

# Single arm constants
JOINTS = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "right_carriage_joint",
]

STIFFNESS = {
    "joint_0": 65.092177,
    "joint_1": 83.698740,
    "joint_2": 115.808333,
    "joint_3": 45.972562,
    "joint_4": 11.364627,
    "joint_5": 4.388846,
    "right_carriage_joint": 339.018069,
}

FORCERANGE = {
    "joint_0": 35.0,
    "joint_1": 144.0,
    "joint_2": 59.0,
    "joint_3": 22.0,
    "joint_4": 35.0,
    "joint_5": 35.0,
    "right_carriage_joint": 35.0,
}

CTRLRANGE = {
    "joint_0": (-3.05433, 3.05433),
    "joint_1": (0.0, 3.14159),
    "joint_2": (0.0, 2.35619),
    "joint_3": (-1.5708, 1.5708),
    "joint_4": (-1.5708, 1.5708),
    "joint_5": (-3.14159, 3.14159),
    "right_carriage_joint": (0.0, 0.044),
}

INIT_CTRL = [0.0, 1.57, 1.57, 0.0, 0.0, 0.0, 0.0]
INIT_QPOS = [0.0, 1.57, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0]

PICKUP_QPOS = [0.0, 0.785, 1.25, -1.25, 0.0, 0.0, 0.02728, 0.0272195]
PICKUP_CTRL = [0.0, 0.785, 1.25, -1.25, 0.0, 0.0, 0.02728]

# Gripper midpoint and half-range.
GRIPPER_MID = (
    CTRLRANGE["right_carriage_joint"][0] + CTRLRANGE["right_carriage_joint"][1]
) / 2.0
GRIPPER_HALF = (
    CTRLRANGE["right_carriage_joint"][1] - CTRLRANGE["right_carriage_joint"][0]
) / 2.0

RIGHT_ARM_JOINTS = [f"right/{j}" for j in JOINTS]

# Action scale and offset for the controlled arm.
ARM_SCALES = [0.5 * FORCERANGE[j] / STIFFNESS[j] for j in JOINTS[:-1]]
RIGHT_ARM_ACTION_SCALE = ARM_SCALES + [GRIPPER_HALF]
RIGHT_ARM_ACTION_OFFSET = list(PICKUP_CTRL[:-1]) + [GRIPPER_MID]
