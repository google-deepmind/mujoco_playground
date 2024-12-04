# Copyright 2024 DeepMind Technologies Limited
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
"""Base class for Go1 environments."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mj_utils as utils
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
import numpy as np


def get_assets() -> Dict[str, bytes]:
  assets = {}
  utils.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_go1" / "assets")
  utils.update_assets(assets, epath.Path(__file__).parent / "xmls", "*.xml")
  return assets


class Go1Env(mjx_env.MjxEnv):
  """Base class for Go1 environments."""

  def __init__(
      self,
      xml_path: str,
      Kp: np.ndarray,
      Kd: np.ndarray,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    xml = epath.Path(xml_path).read_text()
    self._mj_model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())
    self._mj_model.opt.timestep = self.sim_dt

    self._mj_model.dof_damping[6:] = 0.0  # Remove existing joint-level damping.
    utils.modify_pd_gains(self._mj_model, Kp, Kd)

    # Increase offscreen framebuffer size to render at higher resolutions.
    # TODO(kevin): Consider moving this somewhere else.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  # Sensor readings.

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return utils.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return utils.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return utils.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return utils.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return utils.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return utils.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  # Accessors.

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
