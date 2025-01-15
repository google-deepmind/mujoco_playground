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
# pylint: disable=line-too-long
"""Gait-related utilities.

References:
- https://arxiv.org/pdf/2201.00206
- https://github.com/google-deepmind/mujoco_mpc/blob/main/mjpc/tasks/quadruped/quadruped.h#L88
- https://colab.research.google.com/drive/1Rwu7fu9Wmko_GygRrHvxb8A_0zRDhhSp?usp=sharing
"""
# pylint: enable=line-too-long

from typing import Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np


def get_rz(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)


# Foot order:"FR", "FL", "RR", "RL".
GAIT_PHASES = {
    # trot (diagonals together).
    0: np.array([0, np.pi, np.pi, 0]),
    # walk (staggered diagonals).
    1: np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi]),
    # pace (same side legs together).
    2: np.array([0, np.pi, 0, np.pi]),
    # bound (front and back legs together).
    3: np.array([0, 0, np.pi, np.pi]),
    # pronk (all legs together).
    4: np.array([0, 0, 0, 0]),
}


def draw_joystick_command(
    scn,
    cmd,
    xyz,
    theta,
    rgba=None,
    radius=0.02,
    scl=1.0,
):
  if rgba is None:
    rgba = [0.2, 0.2, 0.6, 0.3]
  scn.ngeom += 1
  scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR

  vx, vy, vtheta = cmd

  angle = theta + vtheta
  rotation_matrix = np.array(
      [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
  )

  arrow_from = xyz
  rotated_velocity = rotation_matrix @ np.array([vx, vy])
  to = np.asarray([rotated_velocity[0], rotated_velocity[1], 0])
  to = to / (np.linalg.norm(to) + 1e-6)
  arrow_to = arrow_from + to * scl

  mujoco.mjv_initGeom(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
  )
  mujoco.mjv_connector(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=radius,
      from_=arrow_from,
      to=arrow_to,
  )
