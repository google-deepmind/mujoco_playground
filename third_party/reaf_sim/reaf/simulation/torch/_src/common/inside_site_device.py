# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""InsideSiteDevice for checking whether a body's CoM is inside a site volume."""

from collections.abc import Mapping
from typing import override
from typing import Union

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class InsideSiteDevice(device.Device):
  """Containment sensor entity wrapping mjSENS_INSIDESITE."""

  def __init__(
      self,
      *,
      name: str = "",
      body_name: str,
      site_name: str,
      dark: bool = False,
  ):
    """Initialize InsideSiteDevice.

    Args:
      name: Entity name for lookup and feature prefix.
      body_name: Name of the body whose CoM is tested.
      site_name: Name of the site defining the containment volume.
      dark: If True, hide measurements from Task Layer.
    """
    super().__init__()
    self._name = name
    self._body_name = body_name
    self._site_name = site_name
    self._dark = dark
    self._sensor_name = f"{name}_sensor"
    self._sensor_adr: int = 0

  @property
  def name(self) -> str:
    return self._name

  @override
  def configure_spec(
      self,
      spec: mujoco.MjSpec,
  ) -> mujoco.MjSpec:
    """Add an mjSENS_INSIDESITE sensor to the spec.

    The sensor evaluates to 1.0 when the body's centre of mass lies inside
    the site's axis-aligned bounding volume, and 0.0 otherwise.

    obj (body)  — the body whose CoM is tested for containment.
    ref (site)  — the site that defines the bounding volume.

    See https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor
    for full mjSENS_INSIDESITE documentation.
    """
    spec = super().configure_spec(spec)
    # intprm encodes [dataspec, reduce, dim]:
    #   dataspec=1 (bit 0 = "found"): output whether the body is inside.
    #   reduce=0  ("none"): no reduction across multiple contacts.
    #   dim=1     : scalar output.
    spec.add_sensor(
        name=self._sensor_name,
        type=mujoco.mjtSensor.mjSENS_INSIDESITE,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname=self._body_name,
        reftype=mujoco.mjtObj.mjOBJ_SITE,
        refname=self._site_name,
    )
    return spec

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    sensor = simulator.mj_model.sensor(self._sensor_name)
    self._sensor_adr = int(sensor.adr[0])

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    if self._dark:
      return {}
    return {
        f"{self._name}/data": specs.Array(shape=(1,), dtype=np.float32),
    }

  @override
  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      *,
      force: bool = False,
  ) -> Mapping[str, torch.Tensor]:
    if self._dark and not force:
      return {}
    return {
        f"{self._name}/data": simulator.data.sensordata[
            :, self._sensor_adr : self._sensor_adr + 1
        ]
    }
