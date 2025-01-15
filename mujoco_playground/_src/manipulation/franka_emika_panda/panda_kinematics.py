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
"""Compute the inverse kinematics of the Franka Emika Panda robot.

IK is adapted from the analytical solution found in:
https://github.com/ffall007/franka_analytical_ik/blob/main/franka_ik_He.hpp
"""

# pylint: disable=all
from typing import Union

import jax
import jax.numpy as jp

PI_2 = jp.pi / 2
PI_4 = jp.pi / 4


def mat_from_dh_revolute(
    theta: Union[jax.Array, float],
    alpha: Union[jax.Array, float],
    a: Union[jax.Array, float],
    d: Union[jax.Array, float],
    offset: Union[jax.Array, float],
) -> jax.Array:
  """Generate a transformation matrix from DH parameters for a revolute joint."""
  ca = jp.cos(alpha)
  sa = jp.sin(alpha)
  th = theta + offset
  ct = jp.cos(th)
  st = jp.sin(th)
  mat = jp.asarray([
      [ct, -st, 0, a],
      [st * ca, ct * ca, -sa, -d * sa],
      [st * sa, ct * sa, ca, d * ca],
      [0, 0, 0, 1],
  ])
  return mat


def compute_franka_fk(joint_pos: jax.Array) -> jax.Array:
  """Compute the forward kinematics of the Franka Emika Panda robot."""

  l1 = 0.333
  l3 = 0.316
  l4 = 0.0825
  l5a = -0.0825
  l5d = 0.384
  l7 = 0.088

  # Flange is 0.107, hand is 0.2104
  # ee = 0.107
  ee = 0.2104

  t_ee_0 = jp.identity(4)
  t_1_0 = mat_from_dh_revolute(joint_pos[0], 0, 0, l1, 0)
  t_2_1 = mat_from_dh_revolute(joint_pos[1], -PI_2, 0, 0, 0)
  t_3_2 = mat_from_dh_revolute(joint_pos[2], PI_2, 0, l3, 0)
  t_4_3 = mat_from_dh_revolute(joint_pos[3], PI_2, l4, 0, 0)
  t_5_4 = mat_from_dh_revolute(joint_pos[4], -PI_2, l5a, l5d, 0)
  t_6_5 = mat_from_dh_revolute(joint_pos[5], PI_2, 0, 0, 0)
  t_7_6 = mat_from_dh_revolute(joint_pos[6], PI_2, l7, 0, 0)
  t_ee_7 = mat_from_dh_revolute(
      -jp.pi / 4, 0, 0, ee, 0
  )  # Hand is 45 degrees rotated

  t_ee_0 = jp.matmul(t_ee_0, t_1_0)
  t_ee_0 = jp.matmul(t_ee_0, t_2_1)
  t_ee_0 = jp.matmul(t_ee_0, t_3_2)
  t_ee_0 = jp.matmul(t_ee_0, t_4_3)
  t_ee_0 = jp.matmul(t_ee_0, t_5_4)
  t_ee_0 = jp.matmul(t_ee_0, t_6_5)
  t_ee_0 = jp.matmul(t_ee_0, t_7_6)
  t_ee_0 = jp.matmul(t_ee_0, t_ee_7)

  return t_ee_0


def compute_franka_ik(
    t_ee_0: jax.Array, q7: jax.Array, q_actual: jax.Array
) -> jax.Array:
  """Compute the inverse kinematics of the Franka Emika Panda robot.

  IK is adapted from the analytical solution found in:
  https://github.com/ffall007/franka_analytical_ik/blob/main/franka_ik_He.hpp

  Args:
    t_ee_0: The end-effector transformation matrix in base frame reference.
    q7: The joint 7 angle.
    q_actual: The actual current joint angles of the robot."""

  q = jp.array([0.0] * 7, dtype=jp.float32)

  d1 = 0.3330
  d3 = 0.3160
  d5 = 0.3840

  # d7e is based on if hand or flange is used
  d7e = 0.2104
  # d7e = 0.107

  a4 = 0.0825
  a7 = 0.0880

  LL24 = 0.10666225
  LL46 = 0.15426225
  L24 = 0.326591870689
  L46 = 0.392762332715

  thetaH46 = 1.35916951803
  theta342 = 1.31542071191
  theta46H = 0.211626808766

  q_min = jp.array(
      [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
  )
  q_max = jp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

  q = q.at[6].set(q7)

  c1_a = jp.cos(q_actual[0])
  s1_a = jp.sin(q_actual[0])
  c2_a = jp.cos(q_actual[1])
  s2_a = jp.sin(q_actual[1])
  c3_a = jp.cos(q_actual[2])
  s3_a = jp.sin(q_actual[2])
  c4_a = jp.cos(q_actual[3])
  s4_a = jp.sin(q_actual[3])
  c5_a = jp.cos(q_actual[4])
  s5_a = jp.sin(q_actual[4])
  c6_a = jp.cos(q_actual[5])
  s6_a = jp.sin(q_actual[5])

  As_a = []

  As_a.append(
      jp.array([
          [c1_a, -s1_a, 0.0, 0.0],
          [s1_a, c1_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, d1],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [c2_a, -s2_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [-s2_a, -c2_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [c3_a, -s3_a, 0.0, 0.0],
          [0.0, 0.0, -1.0, -d3],
          [s3_a, c3_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [c4_a, -s4_a, 0.0, a4],
          [0.0, 0.0, -1.0, 0.0],
          [s4_a, c4_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [1.0, 0.0, 0.0, -a4],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [c5_a, -s5_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, d5],
          [-s5_a, -c5_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  As_a.append(
      jp.array([
          [c6_a, -s6_a, 0.0, 0.0],
          [0.0, 0.0, -1.0, 0.0],
          [s6_a, c6_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
  )

  Ts_a = []
  Ts_a.append(As_a[0])
  for j in range(1, 7):
    Ts_a.append(jp.matmul(Ts_a[j - 1], As_a[j]))

  # identify q6 case
  V62_a = Ts_a[1][:3, 3] - Ts_a[6][:3, 3]
  V6H_a = Ts_a[4][:3, 3] - Ts_a[6][:3, 3]
  Z6_a = Ts_a[6][:3, 2]
  is_case6_0 = jp.sum(jp.matmul(jp.cross(V6H_a, V62_a), Z6_a)) <= 0

  # identify q1 case
  is_case1_1 = q_actual[1] < 0

  # IK: compute p_6
  R_EE = t_ee_0[:3, :3]
  z_EE = t_ee_0[:3, 2]
  p_EE = t_ee_0[:3, 3]
  p_7 = p_EE - (d7e * z_EE)

  x_EE_6 = jp.array([jp.cos(q7 - PI_4), -jp.sin(q7 - PI_4), 0.0])
  x_6 = jp.matmul(R_EE, x_EE_6)
  x_6 /= jp.linalg.norm(x_6)
  p_6 = p_7 - a7 * x_6

  # IK: compute q4
  p_2 = jp.array([0.0, 0.0, d1])
  V26 = p_6 - p_2

  LL26 = jp.sum(V26 * V26)
  L26 = jp.sqrt(LL26)

  theta246 = jp.arccos((LL24 + LL46 - LL26) / 2.0 / L24 / L46)
  q = q.at[3].set(theta246 + thetaH46 + theta342 - 2.0 * jp.pi)

  # IK: compute q6
  theta462 = jp.arccos((LL26 + LL46 - LL24) / 2.0 / L26 / L46)
  theta26H = theta46H + theta462
  D26 = -L26 * jp.cos(theta26H)

  Z_6 = jp.cross(z_EE, x_6)
  Y_6 = jp.cross(Z_6, x_6)
  R_6 = jp.column_stack(
      (x_6, Y_6 / jp.linalg.norm(Y_6), Z_6 / jp.linalg.norm(Z_6))
  )
  V_6_62 = jp.matmul(R_6.T, -V26)

  Phi6 = jp.arctan2(V_6_62[1], V_6_62[0])
  Theta6 = jp.arcsin(
      D26 / jp.sqrt((V_6_62[0] * V_6_62[0]) + (V_6_62[1] * V_6_62[1]))
  )

  q = jp.where(
      is_case6_0, q.at[5].set(jp.pi - Theta6 - Phi6), q.at[5].set(Theta6 - Phi6)
  )

  q = jp.where(q[5] <= q_min[5], q.at[5].set(q[5] + 2.0 * jp.pi), q)
  q = jp.where(q[5] >= q_max[5], q.at[5].set(q[5] - 2.0 * jp.pi), q)

  # IK: compute q1 & q2
  thetaP26 = 3.0 * jp.pi / 2 - theta462 - theta246 - theta342
  thetaP = jp.pi - thetaP26 - theta26H
  LP6 = L26 * jp.sin(thetaP26) / jp.sin(thetaP)

  z_6_5 = jp.array([jp.sin(q[5]), jp.cos(q[5]), 0.0])
  z_5 = jp.matmul(R_6, z_6_5)
  V2P = p_6 - LP6 * z_5 - p_2

  L2P = jp.linalg.norm(V2P)

  V2P_L2P = jp.abs(V2P[2] / L2P)
  greater_than_1 = jp.where(V2P_L2P > 0.999, True, False)

  q = jp.where(
      greater_than_1,
      q.at[0].set(q_actual[0]),
      q.at[0].set(jp.arctan2(V2P[1], V2P[0])),
  )

  q = jp.where(
      greater_than_1, q.at[1].set(0.0), q.at[1].set(jp.arccos(V2P[2] / L2P))
  )

  is_q_less_than_0 = jp.where(q[0] < 0.0, True, False)
  q = jp.where(
      is_case1_1,
      jp.where(
          is_q_less_than_0, q.at[0].set(q[0] + jp.pi), q.at[0].set(q[0] - jp.pi)
      ),
      q,
  )

  q = jp.where(greater_than_1, q, jp.where(is_case1_1, q.at[1].set(-q[1]), q))

  # IK: compute q3
  z_3 = V2P / L2P
  Y_3 = -jp.cross(V26, V2P)
  y_3 = Y_3 / jp.linalg.norm(Y_3)
  x_3 = jp.cross(y_3, z_3)
  c1 = jp.cos(q[0])
  s1 = jp.sin(q[0])
  R_1 = jp.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])

  c2 = jp.cos(q[1])
  s2 = jp.sin(q[1])
  R_1_2 = jp.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
  R_2 = jp.matmul(R_1, R_1_2)
  x_2_3 = jp.matmul(R_2.T, x_3)
  q = q.at[2].set(jp.arctan2(x_2_3[2], x_2_3[0]))

  # IK: compute q4
  VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5
  c6 = jp.cos(q[5])
  s6 = jp.sin(q[5])
  R_5_6 = jp.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])
  R_5 = jp.matmul(R_6, R_5_6.T)
  V_5_H4 = jp.matmul(R_5.T, VH4)

  q = q.at[4].set(-jp.arctan2(V_5_H4[1], V_5_H4[0]))

  return q
