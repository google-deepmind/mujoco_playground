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

"""Hardware constants for the SO101, shared by the real arm and the sim."""

import types

import numpy as np


def _pose(*joints: float) -> np.ndarray:
  """Returns a read-only pose vector, one entry per joint."""
  pose = np.array(joints, dtype=float)
  pose.setflags(write=False)
  return pose


# The robot's identity: its name, which prefixes every machine-local config
# this robot generates, and the two arms it is made of.
ROBOT = "so101"
FOLLOWER = "follower"
LEADER = "leader"
ROLES = (FOLLOWER, LEADER)

# Canonical joint order for the physical arm.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT = JOINTS[-1]

# Feetech bus ids are 1-based in the canonical joint order on both arms.
MOTOR_IDS = types.MappingProxyType(
    {name: i + 1 for i, name in enumerate(JOINTS)}
)

# Feetech STS3215 servos on a 12-bit magnetic encoder: integer positions in
# 0..ENCODER_MAX ticks, with the calibrated zero at the center tick. The center
# and maximum follow from the resolution; a different servo changes one line.
MOTOR_MODEL = "sts3215"
TICKS_PER_TURN = 4096
TICKS_CENTER = TICKS_PER_TURN // 2
ENCODER_MAX = TICKS_PER_TURN - 1


def relative_ticks(raw: float, zero: int) -> float:
  """Signed ticks from `zero`, unwrapped across the encoder boundary."""
  return (raw - zero + TICKS_CENTER) % TICKS_PER_TURN - TICKS_CENTER


# REAF device name for the measured arm state.
ROBOT_STATE = "robot_state"

# Measurement and command keys for sim2real deployment.
#  JOINT_POS / JOINT_VEL / HOME_QPOS: standard articulated measurement channels.
JOINT_POS = "joint_pos"
JOINT_VEL = "joint_vel"
HOME_QPOS = "home_qpos"

# Stream names and hardware roles used for teleoperation data collection.
#  WRIST_CAMERA: the camera device stream name in teleop logs.
#  TIMESTAMP: the per-step wall-clock time (nanoseconds).
#  LEADER_RAW: the leader's raw (unclipped) pose, carried as a policy extra.
#  LEADER_TIMESTAMP: timestamp (ns) when leader pose is sampled (t0).
#  FOLLOWER_COMMAND_TIMESTAMP: timestamp (ns) when follower receives command
#    (t1).
WRIST_CAMERA = "wrist_camera"
TIMESTAMP = "timestamp"
LEADER_RAW = "leader_raw"
LEADER_TIMESTAMP = "leader_timestamp"
FOLLOWER_COMMAND_TIMESTAMP = f"{FOLLOWER}/command_timestamp"

# The joint keys emitted by the REAF devices.
JOINT_POSITION_KEYS = {j: f"{FOLLOWER}/joint_position/{j}" for j in JOINTS}
GOAL_POSITION_KEYS = {j: f"{FOLLOWER}/goal_position/{j}" for j in JOINTS}
LEADER_RAW_KEYS = {j: f"{LEADER_RAW}/{j}" for j in JOINTS}

# Frame geometry the real camera and the sim camera entity must agree on.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# The sim/RL episode-start pose, hovering over the table: radians in the shared
# calibrated/MJCF frame. Deliberately not a teleop home — gravity would collapse
# a torque-free arm here (see LEADER_HOME/FOLLOWER_HOME below).
HOVER_POSE = _pose(-0.192, -0.262, 0.253, 1.13, 0.0793, 0.15)
# Hover pose with open gripper (0.80 rad) matching pick_cube trained policy.
HOVER_POSE_OPEN_GRIPPER = tuple(list(HOVER_POSE[:5]) + [0.80])

# The gripper is reported and commanded as a percentage of its calibrated
# range — lerobot's RANGE_0_100 normalization (see `get_calibration._make_bus`),
# rather than the raw ticks the arm joints use. So these bounds *are* its
# calibrated sweep, already normalized, and the sim converts to and from them
# with the MJCF joint limits.
GRIPPER_MIN = 0.0
GRIPPER_MAX = 100.0

# The gripper's position at rest (nearly closed), on that same scale.
GRIPPER_HOME = 2.0

# The HOME pose the arms drive to and start teleop from: the folded REST pose,
# measured once on the physical arm. It is statically
# stable, so the leader's torque can release there without the arm falling.
LEADER_HOME = _pose(0.0, -1.741, 1.668, 0.872, -0.074, GRIPPER_HOME)
# The same pose on both arms.
FOLLOWER_HOME = LEADER_HOME


# Nominal SO101 gripper limits in radians (from the menagerie MJCF joint range).
GRIPPER_RAD_RANGE = (-0.174533, 1.7453292)


def gripper_pct_to_rad(
    pct: float,
    low: float = GRIPPER_RAD_RANGE[0],
    high: float = GRIPPER_RAD_RANGE[1],
) -> float:
  """Converts gripper position from 0..100 percentage to radians."""
  return float(low + (pct / GRIPPER_MAX) * (high - low))


def gripper_rad_to_pct(
    rad: float,
    low: float = GRIPPER_RAD_RANGE[0],
    high: float = GRIPPER_RAD_RANGE[1],
) -> float:
  """Converts gripper position from radians to 0..100 percentage."""
  pct = (rad - low) / (high - low) * GRIPPER_MAX
  return float(np.clip(pct, GRIPPER_MIN, GRIPPER_MAX))


# Runtime assertions on the constants above.
assert all(
    len(pose) == len(JOINTS)
    for pose in (HOVER_POSE, LEADER_HOME, FOLLOWER_HOME)
)
assert GRIPPER_MIN <= GRIPPER_HOME <= GRIPPER_MAX
