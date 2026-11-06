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

"""Scene configuration for SO101 single arm.

Loads SO101 robot assets from mujoco_menagerie.
"""

import mujoco
from mujoco_playground import asset
from mujoco_playground._src.objects import furniture
from mujoco_playground._src.objects import primitive
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
import numpy as np


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Ensure all geoms have names so they receive prefixes when attached."""
  for body in spec.bodies:
    for i, geom in enumerate(body.geoms):
      if not geom.name:
        geom.name = f"{body.name}_geom_{i}"
  return spec


def so101() -> mujoco.MjSpec:
  """The SO101 from menagerie, actuators are overwritten using sim_constant."""
  a = asset.get(
      "github:google-deepmind/mujoco_menagerie/robotstudio_so101/so101.xml",
  )
  spec = _name_unnamed_geoms(a.spec())
  spec = _position_actuators(spec)
  return spec


def _position_actuators(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add position actuators for each joint."""
  for joint in constant.JOINTS:
    if spec.joint(joint) is None:
      raise ValueError(f"Joint not found: {joint}")
    actuator = spec.actuator(joint)
    if actuator is None:  # MjSpec.actuator lookup bug; see ActuatorDevice.
      actuator = next((a for a in spec.actuators if a.name == joint), None)
    if actuator is None:
      actuator = spec.add_actuator(name=joint, target=joint)
    kp, kd = sim_constant.STIFFNESS[joint], sim_constant.DAMPING[joint]
    actuator.target = joint
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kd
    actuator.forcelimited = True
    actuator.forcerange = [-sim_constant.FORCERANGE, sim_constant.FORCERANGE]
    spec.joint(joint).armature = sim_constant.ARMATURE[joint]
    spec.joint(joint).frictionloss = sim_constant.FRICTIONLOSS[joint]
  return spec


# Height of the table surface above the ground plane.
TABLE_HEIGHT = 0.75
# Where the arm is bolted down on the table: centred along the table's length,
# near its front edge.
ALOHA_TABLE_ATTACHMENT_POS = (0.0, -0.27, TABLE_HEIGHT)
# Yawed a quarter turn about z so the arm faces across the table's width.
ALOHA_TABLE_ATTACHMENT_QUAT = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))


def so101_aloha_table() -> mujoco.MjSpec:
  """Places the SO101 arm on the aloha table with a ground plane."""
  spec = primitive.plane()
  spec.stat.center = [0.0, -0.1, TABLE_HEIGHT + 0.17]
  spec.stat.extent = 0.6
  spec.visual.global_.azimuth = 90
  spec.visual.global_.elevation = -45

  table = furniture.aloha_table()
  spec.attach(table, prefix="table/", frame=spec.worldbody.add_frame())

  arm = so101()
  frame = spec.worldbody.add_frame(
      pos=np.array(ALOHA_TABLE_ATTACHMENT_POS),
      quat=np.array(ALOHA_TABLE_ATTACHMENT_QUAT),
  )
  frame.attach_body(
      arm.worldbody.first_body(), sim_constant.ATTACHMENT_PREFIX, ""
  )

  left_light = spec.worldbody.add_light(name="left_side_light")
  left_light.pos = [-0.8, -0.2, 1.2]
  left_light.dir = [0.8, 0.2, -0.45]
  left_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  left_light.diffuse = [0.35, 0.35, 0.35]
  left_light.specular = [0.1, 0.1, 0.1]
  left_light.castshadow = False

  right_light = spec.worldbody.add_light(name="right_side_light")
  right_light.pos = [0.8, -0.2, 1.2]
  right_light.dir = [-0.8, 0.2, -0.45]
  right_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  right_light.diffuse = [0.35, 0.35, 0.35]
  right_light.specular = [0.1, 0.1, 0.1]
  right_light.castshadow = False

  return spec
