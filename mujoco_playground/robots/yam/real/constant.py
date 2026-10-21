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

"""Hardware constants for the I2RT YAM Pro, shared by the real arm and the sim.

A pose is a 7-vector in canonical joint order: six arm joints in radians, then
the gripper normalized to [0, 1] (0 closed, 1 open), which is the convention
of I2RT's SDK.
"""

import types

import numpy as np


def _pose(*joints: float) -> np.ndarray:
  """Returns a read-only pose vector, one entry per joint."""
  pose = np.array(joints, dtype=float)
  pose.setflags(write=False)
  return pose


ROBOT = "yam"
FOLLOWER = "follower"

# The I2RT SDK's names for our hardware (`i2rt.robots.utils.ArmType` and
# `GripperType` values). The SDK checks every motor's type over the bus when it
# connects and refuses a chain that does not match, which is what verifies the
# gripper motor (DM3507 vs DM4310). YAM, YAM Pro and YAM Ultra v1 share motors,
# so the arm variant cannot be verified that way: it comes from the order.
ARM_TYPE = "yam_pro"
GRIPPER_TYPE = "linear_4310"

# SocketCAN interface of the USB-CAN adapter, and the DM motors' bus rate.
CAN_CHANNEL = "can0"
CAN_BITRATE = 1_000_000

# Canonical joint order, which is also the CAN id order (1-based).
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT = "gripper"
JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)
NUM_ARM_JOINTS = len(ARM_JOINTS)
MOTOR_IDS = types.MappingProxyType({j: i + 1 for i, j in enumerate(JOINTS)})
MOTOR_TYPES = types.MappingProxyType({
    "joint1": "DM4340",
    "joint2": "DM4340",
    "joint3": "DM4340",
    "joint4": "DM4310",
    "joint5": "DM4310",
    "joint6": "DM4310",
    "gripper": "DM4310",
})

# REAF device name for the measured arm state.
ROBOT_STATE = "robot_state"

# Measurement and command keys for sim2real deployment.
#  JOINT_POS / JOINT_VEL / HOME_QPOS: standard articulated measurement channels.
JOINT_POS = "joint_pos"
JOINT_VEL = "joint_vel"
HOME_QPOS = "home_qpos"

# Stream names and keys emitted by the REAF devices.
WRIST_CAMERA = "wrist_camera"
TIMESTAMP = "timestamp"
FOLLOWER_COMMAND_TIMESTAMP = f"{FOLLOWER}/command_timestamp"
JOINT_POSITION_KEYS = {j: f"{FOLLOWER}/joint_position/{j}" for j in JOINTS}
GOAL_POSITION_KEYS = {j: f"{FOLLOWER}/goal_position/{j}" for j in JOINTS}

# Frame geometry the real camera and the sim camera entity must agree on.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Arm joint limits in radians, from I2RT's `yam_pro/v1` URDF. The SDK clips
# commands to these plus a 0.15 rad buffer; we stay strictly inside them.
JOINT_LIMITS = types.MappingProxyType({
    "joint1": (-2.61799, 3.14159),
    "joint2": (0.0, 3.66519),
    "joint3": (0.0, 3.14159),
    "joint4": (-1.69297, 1.5708),
    "joint5": (-1.5708, 1.5708),
    "joint6": (-2.0944, 2.0944),
})
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
GRIPPER_CLOSED = GRIPPER_MIN
GRIPPER_OPEN = GRIPPER_MAX
JOINT_BOUNDS = types.MappingProxyType(
    {**JOINT_LIMITS, GRIPPER_JOINT: (GRIPPER_MIN, GRIPPER_MAX)}
)

# The joint-space PD gains the SDK runs on the motors for `yam_pro` +
# `linear_4310`, on top of gravity compensation.
KP = _pose(80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0)
KD = _pose(5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5)

# REST: the folded pose the arm sits in, statically stable on the desk, where
# it is powered on and off. All arm joints at their zero.
REST_POSE = _pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSED)
FOLLOWER_HOME = REST_POSE
# READY: gripper pointing straight down, ~19 cm above the desk and ~27 cm in
# front of the base.
READY_POSE = _pose(0.0, 1.6, 1.6, -1.57, 0.0, 0.0, GRIPPER_CLOSED)
# HOVER: gripper pointing straight down ~10 cm above the desk and ~28 cm in
# front of the base.
HOVER_POSE = _pose(0.0, 1.6, 1.23, -1.19, 0.0, 0.0, GRIPPER_OPEN)

# Control period for host commands.
CONTROL_PERIOD = 0.02  # s between commands (50 Hz); the SDK runs PD at 250 Hz.

assert all(
    len(p) == len(JOINTS) for p in (REST_POSE, READY_POSE, HOVER_POSE, KP, KD)
)
assert set(JOINT_LIMITS) == set(ARM_JOINTS)
