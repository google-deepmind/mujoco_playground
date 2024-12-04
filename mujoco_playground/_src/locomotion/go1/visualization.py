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
import mujoco
import numpy as np


def draw_joystick_command(
    scn,
    cmd,
    xyz,
    theta,
    rgba=[0.2, 0.2, 0.6, 0.3],
    radius=0.01,
    scl=1.0,
):
  scn.ngeom += 1
  scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DYNAMIC

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

  # pytype: disable=wrong-arg-types
  mujoco.mjv_initGeom(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
  )
  mujoco.mjv_connector(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      width=radius,
      from_=arrow_from,
      to=arrow_to,
  )
  # pytype: enable=wrong-arg-types
