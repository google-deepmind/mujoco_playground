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

# Joint and position-actuator dynamics, per actuator (the six arm joints, then
# the gripper, whose values apply to both finger joints). Identified with
# `tasks/sysid/yam` on a real YAM Pro driven through I2RT's SDK (50 Hz
# position commands into its 250 Hz PD with gravity compensation): they cut
# the simulated arm's held-out tracking error from 11.5 to 4.8 mrad RMS.
# These are effective values of that closed loop, not motor settings.
# Values the excitation data couldn't identify keep Menagerie's: the kp of
# joints 1 and 4 (the SDK's gains), the armature of joints 2-4, joint 4's
# friction, joint 6 (whose tracking depends on the wrist camera's load, which
# the MJCF doesn't model) and the gripper.

# Reflected rotor inertia, in kg m^2.
ARMATURE = {
    "joint1": 0.175,
    "joint2": 0.032,
    "joint3": 0.032,
    "joint4": 0.0018,
    "joint5": 0.00959,
    "joint6": 0.0018,
    "gripper": 0.1,
}
# MuJoCo dry (Coulomb) friction, in N m (N for the gripper's slide joints).
FRICTIONLOSS = {
    "joint1": 0.567,
    "joint2": 1.26,
    "joint3": 1.07,
    "joint4": 0.1,
    "joint5": 0.0805,
    "joint6": 0.1,
    "gripper": 0.1,
}
# Position actuator stiffness (kp), in N m/rad (N/m for the gripper).
STIFFNESS = {
    "joint1": 80.0,
    "joint2": 60.9,
    "joint3": 37.4,
    "joint4": 40.0,
    "joint5": 7.34,
    "joint6": 10.0,
    "gripper": 2000.0,
}
# Position actuator damping (kv), in N m s/rad (N s/m for the gripper).
DAMPING = {
    "joint1": 6.01,
    "joint2": 3.82,
    "joint3": 2.45,
    "joint4": 2.28,
    "joint5": 1.21,
    "joint6": 1.5,
    "gripper": 150.0,
}

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
