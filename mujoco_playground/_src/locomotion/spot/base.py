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
"""Base classes for Spot."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import spot_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.ROOT_PATH / "locomotion" / "spot" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  path = mjx_env.MENAGERIE_PATH / "boston_dynamics_spot"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class SpotEnv(mjx_env.MjxEnv):
  """Base class for Spot environments."""

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
    self._mj_model.opt.timestep = config.sim_dt

    # Modify PD gains.
    self._mj_model.dof_damping[6:] = config.Kd
    self._mj_model.actuator_gainprm[:, 0] = config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp

    # Increase offscreen framebuffer size to render at higher resolutions.
    # TODO(kevin): Consider moving this somewhere else.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

    # Contact sensor IDs.
    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in consts.FEET_GEOMS
    ]

  # Sensor readings.

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    """Return the position of the feet relative to the trunk."""
    return jp.vstack([
        mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])

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
