# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Franka Emika Panda base class."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
_MENAGERIE_FRANKA_DIR = "franka_emika_panda"


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.ROOT_PATH / "manipulation" / "franka_emika_panda" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  path = mjx_env.MENAGERIE_PATH / _MENAGERIE_FRANKA_DIR
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      impl='jax',
      nconmax=12 * 8192,
      njmax=44,
  )


class PandaBase(mjx_env.MjxEnv):
  """Base environment for Franka Emika Panda."""

  def __init__(
      self,
      xml_path: epath.Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path.as_posix()
    xml = xml_path.read_text()
    self._model_assets = get_assets()
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
    self._action_scale = config.action_scale

  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = _ARM_JOINTS + _FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("gripper").id
    self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
    self._hand_geom = self._mj_model.geom("hand_capsule").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self._mj_model.body("mocap_target").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
