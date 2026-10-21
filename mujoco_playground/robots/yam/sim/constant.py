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

"""Simulation constants for the YAM: names in the MJCF model."""

# Scene attachment prefix and element names in the MJCF model.
ATTACHMENT_PREFIX = "robot/"
BASE_BODY = "arm"
EE_SITE_NAME = "grasp_site"
EE_SITE = f"{ATTACHMENT_PREFIX}{EE_SITE_NAME}"
WRIST_CAM_NAME = "wrist_cam"
WRIST_CAM_FULLNAME = f"{ATTACHMENT_PREFIX}{WRIST_CAM_NAME}"
WRIST_CAM_RGB = f"{WRIST_CAM_FULLNAME}_rgb"
# The gripper is two slide joints, coupled to move together (0 closed) and
# driven through the left one; the normalized gripper command maps linearly
# onto their range.
FINGER_JOINTS = ("left_finger", "right_finger")
GRIPPER_ACTUATOR = "gripper"
DESK_GEOM = "floor"

# Nominal `linear_4310` finger travel in meters (from the menagerie MJCF slide
# joint range).
GRIPPER_M_RANGE = (0.0, 0.0475)

# The MJCF gripper actuator's ctrlrange reaches GRIPPER_SQUEEZE_M past the
# closed stop, where the pads touch. A closed command (normalized 0) therefore
# targets a position the fingers cannot reach, so the position actuator keeps
# pressing the pads together or onto a grasped object, as the real motor does
# under its force limiter; with a target exactly at the stop there would be no
# grip force at all. Commands map linearly onto this range (as the Aloha model
# in menagerie does), measurements onto the physical travel GRIPPER_M_RANGE.
GRIPPER_SQUEEZE_M = 0.005
GRIPPER_CTRL_M_RANGE = (-GRIPPER_SQUEEZE_M, GRIPPER_M_RANGE[1])


def gripper_norm_to_m(
    norm: float,
    low: float = GRIPPER_M_RANGE[0],
    high: float = GRIPPER_M_RANGE[1],
) -> float:
  """Converts gripper position from normalized [0, 1] to MJCF meters."""
  from mujoco_playground.robots.yam.real import constant

  return float(low + (norm / constant.GRIPPER_MAX) * (high - low))


def gripper_m_to_norm(
    meters: float,
    low: float = GRIPPER_M_RANGE[0],
    high: float = GRIPPER_M_RANGE[1],
) -> float:
  """Converts gripper position from MJCF meters to normalized [0, 1]."""
  from mujoco_playground.robots.yam.real import constant
  import numpy as np

  norm = (meters - low) / (high - low) * constant.GRIPPER_MAX
  return float(np.clip(norm, constant.GRIPPER_MIN, constant.GRIPPER_MAX))


def gripper_norm_to_ctrl(norm: float) -> float:
  """Converts a normalized [0, 1] gripper command to MJCF actuator meters."""
  return gripper_norm_to_m(norm, *GRIPPER_CTRL_M_RANGE)


def gripper_ctrl_to_norm(meters: float) -> float:
  """Converts an MJCF actuator gripper command to normalized [0, 1]."""
  return gripper_m_to_norm(meters, *GRIPPER_CTRL_M_RANGE)
