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
"""ContactSensorDevice for reading contact sensor data.

Supports pattern matching with configurable fields (found, force, etc).
Uses MuJoCo naming: objtype/objname for primary, reftype/refname for secondary.
"""

from collections.abc import Mapping
from typing import Literal
from typing import override

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import mujoco_types
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch

# Contact data field bit indices (from MuJoCo contact sensor).
_CONTACT_DATA_MAP = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
}

# Dimensions of each field.
_CONTACT_DATA_DIMS = {
    "found": 1,
    "force": 3,
    "torque": 3,
    "dist": 1,
    "pos": 3,
    "normal": 3,
    "tangent": 3,
}

# Reduction modes.
_REDUCE_MAP = {
    "none": 0,
    "mindist": 1,
    "maxforce": 2,
    "netforce": 3,
}

# Object type mapping (string to mujoco.mjtObj).
_OBJTYPE_MAP = {
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "xbody": mujoco.mjtObj.mjOBJ_XBODY,
}

ObjType = Literal["geom", "body", "xbody"]
ReduceMode = Literal["none", "mindist", "maxforce", "netforce"]
ContactField = Literal[
    "found", "force", "torque", "dist", "pos", "normal", "tangent"
]


class ContactSensorDevice(device.Device):
  """Contact sensors with pattern matching and configurable fields.

  Uses MuJoCo naming:
  - objtype/objname: What to sense contact on
  - reftype/refname: What to sense contact against (optional)
  - fields: Which data to extract (found, force, etc.) - stacked together
  """

  @override
  def configure_spec(  # pyrefly: ignore[bad-override]
      self,
      spec: mujoco.MjSpec,
      *,
      objtype: ObjType,
      objname: str | tuple[str, ...],
      reftype: ObjType | None = None,
      refname: str | None = None,
      reduce: ReduceMode = "none",
      num: int = 1,
  ) -> mujoco.MjSpec:
    """Add contact sensors to the spec via pattern matching.

    Args:
      spec: MjSpec to modify.
      objtype: "geom", "body", or "xbody".
      objname: Name(s) to match on obj side (exact or regex pattern).
      reftype: "geom", "body", or "xbody" (None = any).
      refname: Name to match on ref side (None = any).
      reduce: Reduction mode for multiple contacts.
      num: How many contacts to report.
    """
    spec = super().configure_spec(spec)
    fields = self._fields

    mj_objtype = _OBJTYPE_MAP[objtype]
    obj_names = mujoco_types.resolve_names(spec, mj_objtype, objname)

    field_bits = 0
    for field in fields:
      field_bits |= 1 << _CONTACT_DATA_MAP[field]

    reduce_mode = _REDUCE_MAP[reduce]
    intprm = [field_bits, reduce_mode, num]
    self._sensor_names = []

    for obj_name in obj_names:
      sensor_name = f"{self._name}/{obj_name}"
      self._sensor_names.append(sensor_name)
      kwargs = {
          "name": sensor_name,
          "type": mujoco.mjtSensor.mjSENS_CONTACT,
          "objtype": mj_objtype,
          "objname": obj_name,
          "intprm": intprm,
      }
      if reftype is not None and refname is not None:
        kwargs["reftype"] = _OBJTYPE_MAP[reftype]
        # TODO: maybe support pattern matching for refname
        kwargs["refname"] = refname
      spec.add_sensor(**kwargs)

    return spec

  def __init__(
      self,
      *,
      name: str = "",
      fields: tuple[ContactField, ...] = ("found",),
      history_length: int = 0,
      dark: bool = False,
  ):
    """Initialize ContactSensorDevice.

    Measurements (per world), for each requested field `f`:

    - `{name}/data`: `(n_sensors, total_dims)`, all fields stacked.
    - `{name}/{f}`: `(n_sensors, dim(f))`, a view into `{name}/data`.
    - `{name}/{f}_history`: `(n_sensors, history_length, dim(f))`, only when
      `history_length > 0`. Index 0 is the most recent reading.

    The history advances once per physics substep. After a reset, every slot
    holds the reset reading.

    Args:
      name: Entity name for measurement keys.
      fields: Fields that were passed to configure_spec.
      history_length: Number of past readings to buffer per field. 0 disables
        the `*_history` measurements.
      dark: If True, hide measurements from Task Layer (simulation-only sensor).
    """
    if history_length < 0:
      raise ValueError(f"history_length must be >= 0, got {history_length}")
    self._name = name
    self._fields = fields
    self._history_length = history_length
    self._dark = dark

    self._field_slices: dict[str, slice] = {}
    self._total_dims = 0
    for field, dim in _CONTACT_DATA_DIMS.items():
      if field in fields:
        self._field_slices[field] = slice(
            self._total_dims, self._total_dims + dim
        )
        self._total_dims += dim

    # Set in initialize().
    self._sensor_indices: torch.Tensor | None = None
    self._n_sensors: int = 0
    # [nworld, n_sensors, history_length, total_dims]
    self._history: torch.Tensor | None = None

  @property
  def name(self) -> str:
    return self._name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Find sensors by prefix and build index tensor for fast gather."""
    mj_model = simulator.mj_model
    data = simulator.data
    # Collect all sensordata indices for matching sensors.
    indices = []
    self._n_sensors = 0
    for sensor_name in self._sensor_names:
      sensor = mj_model.sensor(sensor_name)
      adr = int(sensor.adr[0])
      indices.extend(range(adr, adr + self._total_dims))
      self._n_sensors += 1

    # Build index tensor on same device as data.
    device = data.sensordata.device
    self._sensor_indices = torch.tensor(
        indices, dtype=torch.long, device=device
    )
    if self._history_length > 0:
      self._history = torch.zeros(
          simulator.nworld,
          self._n_sensors,
          self._history_length,
          self._total_dims,
          dtype=torch.float32,
          device=device,
      )

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator
    if self._history is None:
      return
    if env_mask is None:
      self._history.zero_()
    else:
      self._history[:] = torch.where(
          env_mask.view(-1, 1, 1, 1), 0.0, self._history
      )

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Spec for contact sensor readings. Returns empty if dark."""
    if self._dark:
      return {}  # Dark entity - hidden from Task Layer
    spec = {
        f"{self._name}/data": specs.Array(
            shape=(self._n_sensors, self._total_dims), dtype=np.float32
        ),
    }
    for field, s in self._field_slices.items():
      dim = s.stop - s.start
      spec[f"{self._name}/{field}"] = specs.Array(
          shape=(self._n_sensors, dim), dtype=np.float32
      )
      if self._history_length > 0:
        spec[f"{self._name}/{field}_history"] = specs.Array(
            shape=(self._n_sensors, self._history_length, dim),
            dtype=np.float32,
        )
    return spec

  @override
  def on_substep(self, simulator: simulator_module.Simulator) -> None:
    """Pushes the current contact reading into the substep history."""
    if self._history is None:
      return
    self._history[:] = torch.roll(self._history, shifts=1, dims=2)
    self._history[:, :, 0] = self._read_sensor_data(simulator)

  @override
  def on_begin_stepping(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Fills the history of `env_mask` worlds with the reset reading."""
    if self._history is None:
      return
    reading = self._read_sensor_data(simulator).unsqueeze(2)
    if env_mask is None:
      self._history[:] = reading
    else:
      self._history[:] = torch.where(
          env_mask.view(-1, 1, 1, 1), reading, self._history
      )

  @override
  def get_measurements(  # pyrefly: ignore[bad-override]
      self,
      simulator: simulator_module.Simulator,
      *,
      force: bool = False,
  ) -> Mapping[str, torch.Tensor]:
    """Return contact sensor readings.

    Args:
      simulator: The simulator instance.
      force: If True, return measurements even if dark (for resetter access).
    """
    if self._dark and not force:
      return {}
    sensor_data = self._read_sensor_data(simulator)
    out = {f"{self._name}/data": sensor_data}
    for field, s in self._field_slices.items():
      out[f"{self._name}/{field}"] = sensor_data[..., s]
      if self._history is not None:
        out[f"{self._name}/{field}_history"] = self._history[..., s]
    return out

  def _read_sensor_data(
      self, simulator: simulator_module.Simulator
  ) -> torch.Tensor:
    flat = simulator.data.sensordata[:, self._sensor_indices]
    return flat.view(-1, self._n_sensors, self._total_dims)
