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
"""Base class for ALOHA."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import aloha_constants as consts


def get_assets() -> Dict[str, bytes]:
  """Returns a dictionary of all assets used by the environment."""
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "aloha"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  path = mjx_env.ROOT_PATH / "manipulation" / "aloha" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class AlohaEnv(mjx_env.MjxEnv):
  """Base class for ALOHA environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def _post_init_aloha(self, keyframe: str = "home"):
    """Initializes helpful robot properties."""
    self._left_gripper_site = self._mj_model.site("left/gripper").id
    self._right_gripper_site = self._mj_model.site("right/gripper").id
    self._table_geom = self._mj_model.geom("table").id
    self._finger_geoms = [
        self._mj_model.geom(geom_id).id for geom_id in consts.FINGER_GEOMS
    ]
    self._init_q = jp.array(self._mj_model.keyframe(keyframe).qpos)
    self._init_ctrl = jp.array(self._mj_model.keyframe(keyframe).ctrl)
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
    arm_joint_ids = [self._mj_model.joint(j).id for j in consts.ARM_JOINTS]
    self._arm_qadr = jp.array(
        [self._mj_model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids]
    )
    self._finger_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in consts.FINGER_JOINTS
    ])

    # Contact sensor IDs.
    self._table_finger_found_sensor = [
        self._mj_model.sensor("table_" + geom + "_found").id
        for geom in consts.FINGER_GEOMS
    ]

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def hand_table_collision(self, data) -> jp.ndarray:
    # Check for collisions with the floor.
    hand_table_collisions = [
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._table_finger_found_sensor
    ]
    return (sum(hand_table_collisions) > 0).astype(float)
