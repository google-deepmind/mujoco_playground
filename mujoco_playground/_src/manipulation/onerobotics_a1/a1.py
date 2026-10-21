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

"""OneRobotics A1 7-DoF fixed-base right arm model."""

from typing import Any, Dict, Optional, Union

import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

ARM_JOINTS = (
    "joint1-a1_r",
    "joint2-a1_r",
    "joint3-a1_r",
    "joint4-a1_r",
    "joint5-a1_r",
    "joint6-a1_r",
    "joint7-a1_r",
)
XML_PATH = (
    mjx_env.ROOT_PATH / "manipulation" / "onerobotics_a1" / "xmls" / "a1.xml"
)


class A1Base(mjx_env.MjxEnv):
  """Loads the fixed-base A1 MJCF with a visual-only Reach target."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)

    if self._mj_model.nq != 7 or self._mj_model.nv != 7:
      raise ValueError("The A1 Reach model must have exactly 7 arm DoFs")
    if self._mj_model.nu != 7:
      raise ValueError("The A1 Reach model must have 7 position actuators")

    joint_ids = np.array(
        [self._mj_model.joint(name).id for name in ARM_JOINTS], dtype=int
    )
    qpos_ids = self._mj_model.jnt_qposadr[joint_ids]
    qvel_ids = self._mj_model.jnt_dofadr[joint_ids]
    self._arm_qpos_ids = jp.asarray(qpos_ids)
    self._arm_qvel_ids = jp.asarray(qvel_ids)

    # The source model's position actuators follow the seven joint names.
    actuator_ids = np.array(
        [self._mj_model.actuator(name).id for name in ARM_JOINTS],
        dtype=int,
    )
    self._actuator_ids = jp.asarray(actuator_ids)
    self._control_low = jp.asarray(
        self._mj_model.actuator_ctrlrange[actuator_ids, 0]
    )
    self._control_high = jp.asarray(
        self._mj_model.actuator_ctrlrange[actuator_ids, 1]
    )
    self._joint_low = jp.asarray(self._mj_model.jnt_range[joint_ids, 0])
    self._joint_high = jp.asarray(self._mj_model.jnt_range[joint_ids, 1])

    home = self._mj_model.keyframe("home")
    self._home_qpos = jp.asarray(home.qpos)
    self._home_ctrl = jp.asarray(home.ctrl)
    self._ee_site_id = self._mj_model.site("end_effector").id
    self._target_mocap_id = int(
        np.asarray(self._mj_model.body("mocap_target").mocapid).item()
    )
    if self._target_mocap_id < 0:
      raise ValueError("mocap_target must be a mocap body")

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return len(ARM_JOINTS)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
