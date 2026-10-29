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
"""Simulated pre-flight check of the YAM excitation trajectories.

Replays each episode in the YAM desk scene, kinematically and with `mj_step`,
and checks that the arm never touches the scene, stays above the desk and
inside its joint limits. The default scene only holds the desk: add your own
workcell's obstacles (cameras, fixtures, objects) before trusting the result.
"""

import dataclasses

import mujoco
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
from mujoco_playground.robots.yam.sim import scene
from mujoco_playground.tasks.sysid.yam import excitation
import numpy as np

# Bodies whose frames must stay this far above the desk during the active
# excitation. Contact with the desk itself is checked separately.
CLEARANCE_BODIES = (
    "link_4",
    "link_5",
    "link_6",
    "left_finger",
    "right_finger",
    "wrist_camera",
)
MIN_CLEARANCE_M = 0.15  # Above the desk surface.
TIMESTEP = 0.002


@dataclasses.dataclass(frozen=True)
class PreflightResult:
  name: str
  contacts: int
  min_clearance_m: float
  limit_errors: tuple[str, ...]
  max_tracking_error_rad: float

  @property
  def ok(self) -> bool:
    return (
        self.contacts == 0
        and self.min_clearance_m >= MIN_CLEARANCE_M
        and not self.limit_errors
    )


def _robot_bodies(model: mujoco.MjModel) -> set[int]:
  root = model.body(sim_constant.ATTACHMENT_PREFIX + "arm").id
  bodies = set()
  for b in range(model.nbody):
    p = b
    while p not in (0, root):
      p = model.body_parentid[p]
    if p == root:
      bodies.add(b)
  return bodies


def _robot_scene_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, robot: set[int]
) -> int:
  return sum(
      (model.geom_bodyid[c.geom1] in robot)
      != (model.geom_bodyid[c.geom2] in robot)
      for c in data.contact[: data.ncon]
  )


def check_episode(
    episode: excitation.Episode, spec: mujoco.MjSpec | None = None
) -> PreflightResult:
  """Replays one episode and reports contacts, clearance and limits."""
  spec = spec or scene.yam_aloha_table()
  spec.option.timestep = TIMESTEP
  model = spec.compile()
  data = mujoco.MjData(model)
  robot = _robot_bodies(model)
  prefix = sim_constant.ATTACHMENT_PREFIX
  qadr = [model.joint(prefix + j).qposadr[0] for j in constant.ARM_JOINTS]
  fingers = [
      model.joint(prefix + j).qposadr[0] for j in sim_constant.FINGER_JOINTS
  ]
  acts = [model.actuator(prefix + j).id for j in constant.ARM_JOINTS]
  gripper = model.actuator(prefix + sim_constant.GRIPPER_ACTUATOR).id
  clear = [model.body(prefix + b).id for b in CLEARANCE_BODIES]
  table_z = scene.TABLE_HEIGHT

  contacts = 0
  min_clearance = np.inf
  # Kinematic replay of the commanded joint angles.
  for k, q in enumerate(episode.q_cmd):
    data.qpos[qadr] = q[: constant.NUM_ARM_JOINTS]
    data.qpos[fingers] = sim_constant.gripper_norm_to_m(q[-1])
    mujoco.mj_forward(model, data)
    contacts += _robot_scene_contacts(model, data, robot)
    if episode.active[k]:
      low = float(data.xpos[clear, 2].min())
      min_clearance = min(min_clearance, low - table_z)

  # Dynamic replay: 50 Hz commands held over the physics substeps.
  mujoco.mj_resetData(model, data)
  data.qpos[qadr] = episode.q_cmd[0, : constant.NUM_ARM_JOINTS]
  substeps = int(round(excitation.DT / model.opt.timestep))
  max_err = 0.0
  for q in episode.q_cmd:
    data.ctrl[acts] = q[: constant.NUM_ARM_JOINTS]
    data.ctrl[gripper] = sim_constant.gripper_norm_to_ctrl(q[-1])
    for _ in range(substeps):
      mujoco.mj_step(model, data)
      contacts += _robot_scene_contacts(model, data, robot)
    err = np.abs(data.qpos[qadr] - q[: constant.NUM_ARM_JOINTS]).max()
    max_err = max(max_err, float(err))

  return PreflightResult(
      name=episode.name,
      contacts=contacts,
      min_clearance_m=min_clearance,
      limit_errors=tuple(excitation.limit_violations(episode)),
      max_tracking_error_rad=max_err,
  )
