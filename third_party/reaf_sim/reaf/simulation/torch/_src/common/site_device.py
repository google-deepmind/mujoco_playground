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

"""Site entity for extracting MuJoCo site positions and orientations."""

from collections.abc import Mapping
from typing import Union

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import math as reaf_math
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class SiteDevice(device.Device):
  """Read-only entity for extracting site position and orientation."""

  def __init__(
      self,
      *,
      name: str = "",
      site_name: str,
      reference_site_name: str | None = None,
  ):
    """Initialize SiteDevice.

    Args:
      name: Entity name for lookup and feature prefix.
      site_name: Name of the site in the MjSpec (with prefix).
      reference_site_name: Optional name of the site to measure relative to.
    """
    self._name = name
    self._site_name = site_name
    self._reference_site_name = reference_site_name
    self._site_id: int | None = None
    self._sensor_pos_adr: int | None = None
    self._sensor_pos_dim: int | None = None
    self._sensor_quat_adr: int | None = None
    self._sensor_quat_dim: int | None = None

  @property
  def name(self) -> str:
    return self._name

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
  ) -> mujoco.MjSpec:
    spec = super().configure_spec(spec)
    if self._reference_site_name is not None:
      self._sensor_pos_name = f"{self._name}_framepos"
      spec.add_sensor(
          name=self._sensor_pos_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
          objtype=mujoco.mjtObj.mjOBJ_SITE,
          objname=self._site_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
      self._sensor_quat_name = f"{self._name}_framequat"
      spec.add_sensor(
          name=self._sensor_quat_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
          objtype=mujoco.mjtObj.mjOBJ_SITE,
          objname=self._site_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
    return spec

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Resolve site ID from compiled model."""
    mj_model = simulator.mj_model
    self._site_id = mj_model.site(self._site_name).id

    if self._reference_site_name is not None:
      sensor_pos = mj_model.sensor(self._sensor_pos_name)
      self._sensor_pos_adr = int(sensor_pos.adr[0])
      self._sensor_pos_dim = int(sensor_pos.dim[0])

      sensor_quat = mj_model.sensor(self._sensor_quat_name)
      self._sensor_quat_adr = int(sensor_quat.adr[0])
      self._sensor_quat_dim = int(sensor_quat.dim[0])

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """No-op. Sites are read-only."""
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Spec for site xpos and xmat measurements."""
    return {
        f"{self._name}/xpos": specs.Array(shape=(3,), dtype=np.float32),
        f"{self._name}/xmat": specs.Array(shape=(9,), dtype=np.float32),
    }

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Return site xpos and xmat."""
    data = simulator.data
    if self._reference_site_name is not None:
      adr_pos = self._sensor_pos_adr
      dim_pos = self._sensor_pos_dim
      pos = data.sensordata[:, adr_pos : adr_pos + dim_pos]

      adr_quat = self._sensor_quat_adr
      dim_quat = self._sensor_quat_dim
      quat = data.sensordata[:, adr_quat : adr_quat + dim_quat]
      xmat = reaf_math.quat_to_matrix(quat).reshape(-1, 9)
      return {
          f"{self._name}/xpos": pos,
          f"{self._name}/xmat": xmat,
      }

    return {
        f"{self._name}/xpos": data.site_xpos[:, self._site_id],
        f"{self._name}/xmat": data.site_xmat[:, self._site_id].reshape(-1, 9),
    }
