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
"""Peg insertion task for ALOHA."""

from typing import Any, Dict, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.aloha import aloha_constants as consts
from mujoco_playground._src.manipulation.aloha import base as aloha_base


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.0025,
      sim_dt=0.0025,
      episode_length=1000,
      action_repeat=2,
      action_scale=0.005,
      reward_config=config_dict.create(
          scales=config_dict.create(
              left_reward=1,
              right_reward=1,
              left_target_qpos=0.3,
              right_target_qpos=0.3,
              no_table_collision=0.3,
              socket_z_up=0.5,
              peg_z_up=0.5,
              socket_entrance_reward=4,
              peg_end2_reward=4,
              peg_insertion_reward=8,
          )
      ),
      impl='jax',
      nconmax=24 * 8192,
      njmax=256,
  )


class SinglePegInsertion(aloha_base.AlohaEnv):
  """Single peg insertion task for ALOHA."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=(consts.XML_PATH / "mjx_single_peg_insertion.xml").as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self):
    self._post_init_aloha(keyframe="home")
    self._socket_entrance_site = self._mj_model.site("socket_entrance").id
    self._socket_rear_site = self._mj_model.site("socket_rear").id
    self._peg_end2_site = self._mj_model.site("peg_end2").id
    self._socket_body = self._mj_model.body("socket").id
    self._peg_body = self._mj_model.body("peg").id

    self._socket_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._socket_body]
    ]
    self._peg_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._peg_body]
    ]

    # Lift goal: both in the air.
    self._socket_entrance_goal_pos = jp.array([-0.05, 0, 0.15])
    self._peg_end2_goal_pos = jp.array([0.05, 0, 0.15])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_peg, rng_socket = jax.random.split(rng, 3)

    peg_xy = jax.random.uniform(rng_peg, (2,), minval=-0.1, maxval=0.1)
    socket_xy = jax.random.uniform(rng_socket, (2,), minval=-0.1, maxval=0.1)
    init_q = self._init_q.at[self._peg_qadr : self._peg_qadr + 2].add(peg_xy)
    init_q = init_q.at[self._socket_qadr : self._socket_qadr + 2].add(socket_xy)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    info = {"rng": rng}
    obs = self._get_obs(data)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "peg_end2_dist_to_line": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    socket_rear_pos = data.site_xpos[self._socket_rear_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    # Insertion reward: if peg end2 is aligned with hole entrance, then reward
    # distance from peg end to socket interior.
    socket_ab = socket_entrance_pos - socket_rear_pos
    socket_t = jp.dot(peg_end2_pos - socket_rear_pos, socket_ab)
    socket_t /= jp.dot(socket_ab, socket_ab) + 1e-6
    nearest_pt = data.site_xpos[self._socket_rear_site] + socket_t * socket_ab
    peg_end2_dist_to_line = jp.linalg.norm(peg_end2_pos - nearest_pt)

    out_of_bounds = jp.any(jp.abs(data.xpos[self._socket_body]) > 1.0)
    out_of_bounds |= jp.any(jp.abs(data.xpos[self._peg_body]) > 1.0)

    raw_rewards = self._get_reward(
        data, use_peg_insertion_reward=(peg_end2_dist_to_line < 0.005)
    )
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = sum(rewards.values()) / sum(
        self._config.reward_config.scales.values()
    )

    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    state.metrics.update(
        **rewards,
        peg_end2_dist_to_line=peg_end2_dist_to_line,
        out_of_bounds=out_of_bounds.astype(float),
    )
    obs = self._get_obs(data)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    left_gripper_pos = data.site_xpos[self._left_gripper_site]
    socket_pos = data.xpos[self._socket_body]
    right_gripper_pos = data.site_xpos[self._right_gripper_site]
    peg_pos = data.xpos[self._peg_body]
    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    socket_z = data.xmat[self._socket_body].ravel()[6:]
    peg_z = data.xmat[self._peg_body].ravel()[6:]

    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        left_gripper_pos,
        socket_pos,
        right_gripper_pos,
        peg_pos,
        socket_entrance_pos,
        peg_end2_pos,
        socket_z,
        peg_z,
    ])

    return obs

  def _get_reward(
      self, data: mjx.Data, use_peg_insertion_reward: bool
  ) -> Dict[str, jax.Array]:
    left_socket_dist = jp.linalg.norm(
        data.xpos[self._socket_body] - data.site_xpos[self._left_gripper_site]
    )
    left_reward = reward_util.tolerance(
        left_socket_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )
    right_peg_dist = jp.linalg.norm(
        data.xpos[self._peg_body] - data.site_xpos[self._right_gripper_site]
    )
    right_reward = reward_util.tolerance(
        right_peg_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )

    robot_qpos_diff = data.qpos[self._arm_qadr] - self._init_q[self._arm_qadr]
    left_pose = jp.linalg.norm(robot_qpos_diff[:6])
    left_pose = reward_util.tolerance(left_pose, (0, 0.01), margin=2.0)
    right_pose = jp.linalg.norm(robot_qpos_diff[6:])
    right_pose = reward_util.tolerance(right_pose, (0, 0.01), margin=2.0)

    socket_dist = jp.linalg.norm(
        self._socket_entrance_goal_pos - data.xpos[self._socket_body]
    )
    socket_lift = reward_util.tolerance(
        socket_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    peg_dist = jp.linalg.norm(
        self._peg_end2_goal_pos - data.xpos[self._peg_body]
    )
    peg_lift = reward_util.tolerance(
        peg_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    table_collision = self.hand_table_collision(data)

    socket_orientation = jp.dot(
        data.xmat[self._socket_body][2], jp.array([0.0, 0.0, 1.0])
    )
    socket_orientation = reward_util.tolerance(
        socket_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )
    peg_orientation = jp.dot(
        data.xmat[self._peg_body][2], jp.array([0.0, 0.0, 1.0])
    )
    peg_orientation = reward_util.tolerance(
        peg_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )

    peg_insertion_dist = jp.linalg.norm(
        data.site_xpos[self._peg_end2_site]
        - data.site_xpos[self._socket_rear_site]
    )
    peg_insertion_reward = (
        reward_util.tolerance(
            peg_insertion_dist, (0, 0.001), margin=0.1, sigmoid="linear"
        )
        * use_peg_insertion_reward
    )

    return {
        "left_reward": left_reward,
        "right_reward": right_reward,
        "left_target_qpos": left_pose * left_reward * right_reward,
        "right_target_qpos": right_pose * left_reward * right_reward,
        "no_table_collision": 1 - table_collision,
        "socket_entrance_reward": socket_lift,
        "peg_end2_reward": peg_lift,
        "socket_z_up": socket_orientation * socket_lift,
        "peg_z_up": peg_orientation * peg_lift,
        "peg_insertion_reward": peg_insertion_reward,
    }
