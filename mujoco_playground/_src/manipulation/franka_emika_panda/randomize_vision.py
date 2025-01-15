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
"""Randomization functions."""
from typing import Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

FLOOR_GEOM_ID = 0
BOX_GEOM_ID = 81


def domain_randomize(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'geom_friction': 0,
      'cam_pos': 0,
      'cam_quat': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_directional': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  rng = jax.random.key(0)

  @jax.vmap
  def rand(rng):
    _, key = jax.random.split(rng, 2)
    # friction
    friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
    friction = mjx_model.geom_friction.at[:, 0].set(friction)
    key_r, key_g, key_b, key = jax.random.split(key, 4)
    rgba = jp.array([
        jax.random.uniform(key_r, (), minval=0.5, maxval=1.0),
        jax.random.uniform(key_g, (), minval=0.0, maxval=0.5),
        jax.random.uniform(key_b, (), minval=0.0, maxval=0.5),
        1.0,
    ])
    geom_rgba = mjx_model.geom_rgba.at[BOX_GEOM_ID].set(rgba)

    # Sample a shade of gray
    key_gs, key = jax.random.split(key)
    gray_scale = jax.random.uniform(key_gs, (), minval=0.0, maxval=0.8)
    geom_rgba = geom_rgba.at[FLOOR_GEOM_ID].set(
        jp.array([gray_scale, gray_scale, gray_scale, 1.0])
    )

    # Set unrandomized and randomized matID's to -1 and -2.
    geom_matid = jp.ones_like(mjx_model.geom_matid) * -1
    geom_matid = geom_matid.at[BOX_GEOM_ID].set(-2)
    geom_matid = geom_matid.at[FLOOR_GEOM_ID].set(-2)

    key_pos, key = jax.random.split(key)
    cam_offset = jax.random.uniform(key_pos, (3,), minval=-0.05, maxval=0.05)
    cam_pos = mjx_model.cam_pos.at[0].set(mjx_model.cam_pos[0] + cam_offset)

    key_axis, key_theta, key = jax.random.split(key, 3)
    perturb_axis = jax.random.uniform(key_axis, (3,), minval=-1, maxval=1)
    perturb_axis = perturb_axis / jp.linalg.norm(perturb_axis)
    perturb_theta = jax.random.uniform(
        key_theta, shape=(1,), maxval=np.deg2rad(10)
    )
    camera_rot_offset = math.axis_angle_to_quat(perturb_axis, perturb_theta)
    cam_quat = mjx_model.cam_quat.at[0].set(
        math.quat_mul(camera_rot_offset, mjx_model.cam_quat[0])
    )

    return friction, geom_rgba, geom_matid, cam_pos, cam_quat

  friction, geom_rgba, geom_matid, cam_pos, cam_quat = rand(
      jax.random.split(rng, num_worlds)
  )

  mjx_model = mjx_model.tree_replace({
      'geom_rgba': geom_rgba,
      'geom_matid': geom_matid,
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'geom_friction': friction,
      'cam_pos': cam_pos,
      'cam_quat': cam_quat,
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_directional': jp.repeat(
          jp.expand_dims(mjx_model.light_directional, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })

  return mjx_model, in_axes
