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

"""Unitree G1 robot constants."""

import dataclasses

from reaf.simulation.torch import mujoco_types


def _reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
  """Compute reflected inertia of a two-stage planetary gearbox."""
  assert gear_ratio[0] == 1
  r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
  r2 = rotor_inertia[1] * gear_ratio[2] ** 2
  r3 = rotor_inertia[2]
  return r1 + r2 + r3


# Motor specs (from Unitree).
ARMATURE_5020 = _reflected_inertia_from_two_stage_planetary(
    (0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16))
)
ARMATURE_7520_14 = _reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22))
)
ARMATURE_7520_22 = _reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)
)
ARMATURE_4010 = _reflected_inertia_from_two_stage_planetary(
    (0.068e-4, 0.0, 0.0), (1, 5, 5)
)

EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


@dataclasses.dataclass(frozen=True)
class PositionActuatorGroup:
  """Position actuator configuration for a group of joints."""

  target_names_expr: tuple[str, ...]
  stiffness: float
  damping: float
  effort_limit: float
  armature: float


# Declaration order matters: position actuators are created group by
# group, so model ctrl order follows this ordering.
G1_ACTUATOR_GROUPS: tuple[PositionActuatorGroup, ...] = (
    PositionActuatorGroup(
        target_names_expr=(
            ".*_elbow_joint",
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_wrist_roll_joint",
        ),
        stiffness=STIFFNESS_5020,
        damping=DAMPING_5020,
        effort_limit=EFFORT_5020,
        armature=ARMATURE_5020,
    ),
    PositionActuatorGroup(
        target_names_expr=(
            ".*_hip_pitch_joint",
            ".*_hip_yaw_joint",
            "waist_yaw_joint",
        ),
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        effort_limit=EFFORT_7520_14,
        armature=ARMATURE_7520_14,
    ),
    PositionActuatorGroup(
        target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
        stiffness=STIFFNESS_7520_22,
        damping=DAMPING_7520_22,
        effort_limit=EFFORT_7520_22,
        armature=ARMATURE_7520_22,
    ),
    PositionActuatorGroup(
        target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
        stiffness=STIFFNESS_4010,
        damping=DAMPING_4010,
        effort_limit=EFFORT_4010,
        armature=ARMATURE_4010,
    ),
    # Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
    # Assuming a nominal 1:1 gear ratio, joint armature is approximated as
    # the sum of the 2 actuators' armatures.
    PositionActuatorGroup(
        target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
        stiffness=STIFFNESS_5020 * 2,
        damping=DAMPING_5020 * 2,
        effort_limit=EFFORT_5020 * 2,
        armature=ARMATURE_5020 * 2,
    ),
    PositionActuatorGroup(
        target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        stiffness=STIFFNESS_5020 * 2,
        damping=DAMPING_5020 * 2,
        effort_limit=EFFORT_5020 * 2,
        armature=ARMATURE_5020 * 2,
    ),
)

# Joints in g1_mjlab.xml document order. This is the canonical ordering for
# joint_pos/joint_vel vectors, actions, and motion npz columns.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NUM_JOINTS = len(JOINT_NAMES)

# action_scale[joint] = 0.25 * effort_limit / stiffness.
G1_ACTION_SCALE: dict[str, float] = {}
for _group in G1_ACTUATOR_GROUPS:
  for _name in mujoco_types.match_names(_group.target_names_expr, JOINT_NAMES):
    G1_ACTION_SCALE[_name] = 0.25 * _group.effort_limit / _group.stiffness

ACTION_SCALE_VECTOR: list[float] = [G1_ACTION_SCALE[n] for n in JOINT_NAMES]

##
# Default bent-knees keyframe config.
##

KNEES_BENT_ROOT_POS = (0.0, 0.0, 0.76)
KNEES_BENT_ROOT_QUAT = (1.0, 0.0, 0.0, 0.0)
KNEES_BENT_JOINT_POS_PATTERNS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
KNEES_BENT_JOINT_POS: list[float] = mujoco_types.resolve_values(
    KNEES_BENT_JOINT_POS_PATTERNS, JOINT_NAMES
)

##
# Collision and sensor config.
##

COLLISION_GEOM_REGEX = r".*_collision"
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"
FOOT_CONDIM = 3
FOOT_PRIORITY = 1
FOOT_FRICTION = 0.6
OTHER_CONDIM = 1

SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

ROOT_BODY_NAME = "pelvis"
IMU_LIN_VEL_SENSOR = "imu_lin_vel"
IMU_ANG_VEL_SENSOR = "imu_ang_vel"
ENCODER_BIAS_RANGE: tuple[float, float] = (-0.01, 0.01)

G1_XML_URI = "github:google-deepmind/mujoco_menagerie/unitree_g1/g1_mjlab.xml"
