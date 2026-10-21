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

"""Simulation constants for the SO101: the servo model as tuned for MuJoCo."""

# Scene attachment prefix and element names in the MJCF model.
ATTACHMENT_PREFIX = "robot/"
EE_SITE_NAME = "gripperframe"
EE_SITE = f"{ATTACHMENT_PREFIX}{EE_SITE_NAME}"
WRIST_CAM_NAME = "wrist_cam"
WRIST_CAM_FULLNAME = f"{ATTACHMENT_PREFIX}{WRIST_CAM_NAME}"
WRIST_CAM_RGB = f"{WRIST_CAM_FULLNAME}_rgb"

# Per-joint armature (reflected rotor and gearbox inertia) identified from
# real teleop data.
ARMATURE = {
    "shoulder_pan": 0.1088,
    "shoulder_lift": 0.1472,
    "elbow_flex": 0.1224,
    "wrist_flex": 0.1133,
    "wrist_roll": 0.1146,
    "gripper": 0.1985,
}
ARMATURE_STS3215 = 0.1088  # Nominal single-servo armature.

# Per-joint frictionloss identified from real teleop data.
FRICTIONLOSS = {
    "shoulder_pan": 0.0001,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.2517,
}

# Position servo PD gains identified from real teleop data.
STIFFNESS = {
    "shoulder_pan": 251.5896,
    "shoulder_lift": 265.0958,
    "elbow_flex": 243.9844,
    "wrist_flex": 223.2804,
    "wrist_roll": 218.4066,
    "gripper": 223.3508,
}
DAMPING = {
    "shoulder_pan": 17.5096,
    "shoulder_lift": 28.0620,
    "elbow_flex": 25.6669,
    "wrist_flex": 25.4543,
    "wrist_roll": 23.1058,
    "gripper": 34.1633,
}
FORCERANGE = 30 * 9.81 / 100  # STS3215 stall torque: 30 kg-cm @ 12V.

# Integrated action space: max delta per 10Hz step (~0.5 rad/s arm,
# 2.5 rad/s gripper).
DELTA_ACTION_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.25)
ACTION_SMOOTHING = 0.5
ACTION_CLIP_MIN = (
    -3.14159,
    -1.74533,
    -0.17453,
    -1.65806,
    -2.79253,
    -0.174533,
)
ACTION_CLIP_MAX = (
    3.14159,
    1.74533,
    2.79253,
    1.65806,
    2.79253,
    1.7453292,
)
