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
"""Utility functions for MuJoCo Playground."""

from typing import Any, Dict, Tuple, Union

from etils import epath
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import numpy.typing as npt


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  if geom1 > geom2:
    geom1, geom2 = geom2, geom1
  mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
  idx = jnp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  return get_collision_info(state.contact, geom1, geom2)[0] < 0


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
  """Gets sensor data given sensor name."""
  sensor_id = model.sensor(sensor_name).id
  sensor_adr = model.sensor_adr[sensor_id]
  sensor_dim = model.sensor_dim[sensor_id]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def update_assets(
    assets: Dict[str, Any],
    path: Union[str, epath.Path],
    glob: str = "*",
):
  for f in epath.Path(path).glob(glob):
    if not f.is_file():
      continue
    assets[f.name] = f.read_bytes()


def modify_pd_gains(
    model: mujoco.MjModel,
    kp: npt.ArrayLike,
    kd: npt.ArrayLike,
) -> None:
  """Modify the PD gains of the position actuators in the model.

  Args:
      model: The MuJoCo model.
      kp: The proportional gains.
      kd: The derivative gains.

  Raises:
      ValueError: If the dimensions of kp and kd are invalid.
  """
  kp = np.atleast_1d(kp)
  kd = np.atleast_1d(kd)
  if kp.ndim != 1 or kp.shape[0] not in (1, model.nu):
    raise ValueError(f"kp must be a scalar or a vector of length {model.nu}")
  if kd.ndim != 1 or kd.shape[0] not in (1, model.nu):
    raise ValueError(f"kd must be a scalar or a vector of length {model.nu}")
  # TODO(kevin): Assert that biastype="affine".
  # Remember <position kp kv> gets rewritten as:
  # <general biastype="affine" gainprm="kp 0 0" biasprm="0 -kp -kv"/>
  model.actuator_gainprm[:, 0] = kp
  model.actuator_biasprm[:, 1] = -kp
  model.actuator_biasprm[:, 2] = -kd
