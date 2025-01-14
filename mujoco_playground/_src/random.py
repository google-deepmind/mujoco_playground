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
"""Utilities for randomization."""
import jax
from jax import numpy as jp
import mujoco
import numpy as np


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])


# Reference: https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/arenas/bowl.py
def random_hfield(
    model: mujoco.MjModel,
    heightfield_id: int = 0,
    terrain_bump_scale: float = 1.0,
    terrain_smoothness: float = 0.4,
):
  """Randomize the heightfield."""
  from scipy import ndimage  # pylint: disable=g-import-not-at-top

  res = model.hfield_nrow[heightfield_id]

  # Sinusoidal bowl shape.
  row_grid, col_grid = np.ogrid[-1 : 1 : res * 1j, -1 : 1 : res * 1j]
  radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), 0.1, 1)
  bowl_shape = 0.5 - np.cos(2 * np.pi * radius) / 2

  # Random smooth bumps.
  terrain_size = 2 * model.hfield_size[heightfield_id, 0]
  bump_res = int(terrain_size / terrain_bump_scale)
  bumps = np.random.uniform(
      size=(bump_res, bump_res), low=terrain_smoothness, high=1
  )
  smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))

  # Terrain is elementwise product.
  terrain = bowl_shape * smooth_bumps

  start_idx = model.hfield_adr[heightfield_id]
  model.hfield_data[start_idx : start_idx + res**2] = terrain.ravel()
