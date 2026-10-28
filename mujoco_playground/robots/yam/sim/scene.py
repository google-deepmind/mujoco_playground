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

"""Scene configuration for the YAM single arm.

Loads the YAM robot asset from mujoco_menagerie and attaches it to the
ALOHA table.
"""

import mujoco
from mujoco_playground import asset
from mujoco_playground._src.objects import furniture
from mujoco_playground._src.objects import primitive
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
import numpy as np

YAM_XML_URI = (
    "github:google-deepmind/mujoco_menagerie/i2rt_yam/yam_linear_4310.xml"
)


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Ensure all geoms have names so they receive prefixes when attached."""
  for body in spec.bodies:
    for i, geom in enumerate(body.geoms):
      if not geom.name:
        geom.name = f"{body.name}_geom_{i}"
  return spec


def yam_menagerie() -> mujoco.MjSpec:
  """The YAM from menagerie, with Menagerie's joint and actuator values."""
  spec = asset.get(YAM_XML_URI).spec()
  return _name_unnamed_geoms(spec)


def yam() -> mujoco.MjSpec:
  """The YAM from menagerie, dynamics are overwritten using sim_constant.

  The YAM Pro with the linear_4310 gripper and wrist D405 camera, with the
  armature, frictionloss and position-actuator gains identified on the real
  arm (see `sim_constant`).
  """
  return _joint_dynamics(yam_menagerie())


def _joint_dynamics(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Sets each actuator's joint armature, frictionloss and kp / kv."""
  for name in constant.JOINTS:
    actuator = spec.actuator(name)
    if actuator is None:  # MjSpec.actuator lookup bug; see ActuatorDevice.
      actuator = next((a for a in spec.actuators if a.name == name), None)
    if actuator is None:
      raise ValueError(f"Actuator not found: {name}")
    kp, kv = sim_constant.STIFFNESS[name], sim_constant.DAMPING[name]
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kv
    # The gripper actuator drives the left finger; the right one mirrors it
    # through an equality constraint, so both get the gripper's values.
    joints = (
        sim_constant.FINGER_JOINTS
        if name == sim_constant.GRIPPER_ACTUATOR
        else (name,)
    )
    for joint in joints:
      spec.joint(joint).armature = sim_constant.ARMATURE[name]
      spec.joint(joint).frictionloss = sim_constant.FRICTIONLOSS[name]
  return spec


# Height of the table surface above the ground plane.
TABLE_HEIGHT = 0.75
# Where the arm is bolted down on the table: centred along the table's length,
# near its front edge. `arm` has its origin 1 cm above the base plate bottom.
ALOHA_TABLE_ATTACHMENT_POS = (0.0, -0.27, TABLE_HEIGHT + 0.01)
# Yawed a quarter turn about z so the arm faces across the table's width.
ALOHA_TABLE_ATTACHMENT_QUAT = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))


def yam_aloha_table() -> mujoco.MjSpec:
  """Places the YAM arm on the aloha table with a ground plane."""
  spec = primitive.plane()
  spec.stat.center = [0.0, -0.1, TABLE_HEIGHT + 0.17]
  spec.stat.extent = 0.6
  spec.visual.global_.azimuth = 90
  spec.visual.global_.elevation = -45

  table = furniture.aloha_table()
  spec.attach(table, prefix="table/", frame=spec.worldbody.add_frame())

  arm = yam()
  frame = spec.worldbody.add_frame(
      pos=np.array(ALOHA_TABLE_ATTACHMENT_POS),
      quat=np.array(ALOHA_TABLE_ATTACHMENT_QUAT),
  )
  frame.attach_body(
      arm.worldbody.first_body(), sim_constant.ATTACHMENT_PREFIX, ""
  )

  return spec


# Alias for backwards compatibility.
yam_desk = yam_aloha_table
