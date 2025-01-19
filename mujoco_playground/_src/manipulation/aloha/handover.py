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
from mujoco_playground._src.manipulation.aloha import aloha_constants as consts
from mujoco_playground._src.manipulation.aloha import base as aloha_base


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=250,  # 5 sec.
      action_repeat=1,
      action_scale=0.015,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_box=1,
              box_handover=4,
              handover_target=8,
              no_table_collision=0.3,
              transfer_hands=0.5,
              sparse_handover=10.0,
          ),
          table_penalty=-10.0
      ),
  )


class HandOver(aloha_base.AlohaEnv):
  """Single peg insertion task for ALOHA."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=(consts.XML_PATH / 'mjx_hand_over.xml').as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
    self.grip_alpha = 0.1

  def _post_init(self):
    self._post_init_aloha(keyframe="home")
    self._mocap_target = self._mj_model.body('mocap_target').mocapid
    self._box_body = self._mj_model.body('box').id
    self._box_top_site = self._mj_model.site('box_top').id
    self._box_bottom_site = self._mj_model.site('box_bottom').id

    self._box_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._box_body]
    ]

    # Used for reward calculation.
    self._left_thresh = -0.1
    self._right_thresh = 0.0
    self._handover_pos = jp.array([0.0, 0.0, 0.24])

    self._box_geom = self._mj_model.geom('box').id
    self._picked_q = self._mj_model.keyframe('picked').qpos
    self._picked_ctrl = self._mj_model.keyframe('picked').ctrl
    self._transferred_q = self._mj_model.keyframe('transferred').qpos
    self._transferred_ctrl = self._mj_model.keyframe('transferred').ctrl

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_box_x, rng_box_y = jax.random.split(rng, 3)

    box_xy = jp.array([
        jax.random.uniform(rng_box_x, (), minval=-0.05, maxval=0.05),
        jax.random.uniform(rng_box_y, (), minval=-0.2, maxval=0.2),
    ])
    init_q = self._init_q.at[self._box_qadr : self._box_qadr + 2].add(box_xy)

    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
    )

    rng, rng_target = jax.random.split(rng)
    target_pos = jp.array([0.18, 0.0, 0.25])
    target_pos += jax.random.uniform(
        rng_target, (3,), minval=-0.15, maxval=0.15
    )

    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target].set(target_pos)
    )
    info = {
        'rng': rng,
        'target_pos': target_pos,
        '_steps': jp.array(0, dtype=int),
        's_l_gripped': jp.array(0.0, dtype=float),  # smoothed grip signal
        's_r_gripped': jp.array(0.0, dtype=float),
        'f_sparse_handover': jp.array(0, dtype=bool), # whether transfer already awarded
        'picks': jp.array(0, dtype=int), # If still learning how to pick up the block, guide that. Else, guide the transfer.
        'episode_picked': jp.array(0, dtype=bool),  # To help count above.
    }

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)

    metrics = {
        'out_of_bounds': jp.array(0.0, dtype=float),
        'l_gripped': jp.array(0.0, dtype=float),
        'r_gripped': jp.array(0.0, dtype=float),
        'transfer_sampled': jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    newly_reset = state.info['_steps'] == 0
    state.info['s_l_gripped'] = jp.where(
        newly_reset, 0.0, state.info['s_l_gripped']
    )
    state.info['s_r_gripped'] = jp.where(
        newly_reset, 0.0, state.info['s_r_gripped']
    )
    state.info['episode_picked'] = jp.where(
        newly_reset, 0, state.info['episode_picked']
    )

    # Ocassionally steer exploration.
    state.info['rng'], key_swap = jax.random.split(state.info['rng'])
    to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
    picked, transferred = (self._picked_q, self._picked_ctrl), (self._transferred_q, self._transferred_ctrl)

    def choose_tuple(x, y, f: bool):
      # If f, choose x. Else, y.
      return jax.tree_util.tree_map(
        lambda x, y: f * x + (1-f) * y, x, y
      )
    guiding = choose_tuple(transferred, picked, (state.info['picks'] >= 2).astype(float))
    _qpos, _ctrl = choose_tuple(guiding, (state.data.qpos, state.data.ctrl), to_sample.astype(float))
    data = state.data.replace(qpos=_qpos, ctrl=_ctrl)

    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)

    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = sum(rewards.values()) / sum(
        self._config.reward_config.scales.values()
    )

    # # Table collisions can destabilize simulation.
    # table_collision = self.hand_table_collision(data)
    # reward += table_collision * self._config.reward_config.table_penalty
 
    box_pos = data.xpos[self._box_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )
    picked = jp.linalg.norm(box_pos - self._handover_pos) < 0.05
    state.info['picks'] += jp.logical_not(state.info['episode_picked']).astype(int) * picked.astype(int)
    state.info['episode_picked'] = jp.logical_or(
        state.info['episode_picked'], picked
    )

    state.metrics.update(**rewards, 
        out_of_bounds=out_of_bounds.astype(float),
        l_gripped = state.info['s_l_gripped'],
        r_gripped = state.info['s_r_gripped'],
        transfer_sampled = (to_sample * (state.info['picks'] >= 2).astype(float)))

    obs = self._get_obs(data, state.info)
    return mjx_env.State(
        data, obs, reward, done.astype(float), state.metrics, state.info
    )

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    def distance(x, y):
      return jp.exp(-10 * jp.linalg.norm(x - y))

    box_top = data.site_xpos[self._box_top_site]
    box_bottom = data.site_xpos[self._box_bottom_site]
    box = data.xpos[self._box_body]
    l_gripper = data.site_xpos[self._left_gripper_site]
    r_gripper = data.site_xpos[self._right_gripper_site]
    l_gripped = self.is_grasped(data, 'left')
    r_gripped = self.is_grasped(data, 'right')
    info['s_l_gripped'] = l_gripped * self.grip_alpha + info['s_l_gripped'] * (
        1 - self.grip_alpha
    )
    info['s_r_gripped'] = r_gripped * self.grip_alpha + info['s_r_gripped'] * (
        1 - self.grip_alpha
    )
    l_gripped, r_gripped = info['s_l_gripped'], info['s_r_gripped']

    pre = jp.where(box[0] < self._left_thresh, 1.0, 0.0)
    past = jp.where(box[0] >= self._right_thresh, 1.0, 0.0)
    btwn = (1 - pre) * (1 - past)

    #### Gripper Box
    r_lg = distance(box_top, l_gripper) * (pre + btwn)
    # If you're past the left threshold, also reward the right gripper.
    r_rg = distance(box_bottom, r_gripper) * (btwn + past)
    # Maintain reward level after left out of range.
    r_rg_bias = distance(box_bottom, r_gripper) * past

    #### Box Handover to handover point
    box_handover = distance(box, self._handover_pos)
    # Maintain this term after RH takes box away.
    hand_handover = distance(l_gripper, self._handover_pos) * past
    box_handover = jp.maximum(box_handover, hand_handover)

    #### Facilitate transfer between hands
    ## Let the right hand take the box
    transfer_hands = (1 - l_gripped) * past * r_gripped

    ## Big reward for the risk
    hand_over = ((1 - l_gripped) * past * r_gripped) > 0.8
    hand_over *= jp.logical_not(info['f_sparse_handover'])
    info['f_sparse_handover'] = jp.logical_or(
        info['f_sparse_handover'], hand_over > 0
    )

    #### Bring box to target
    box_target = distance(info['target_pos'], box) * (r_rg + r_rg_bias)
    # Don't let the left hand do it.
    def logistic_barrier(x: jax.Array):
      k = 100 # About 12 cm between 1 and 0, centered around 0.
      return 1 - (1/(1 + jp.exp(-k * x)))
    box_target *= logistic_barrier(l_gripper[0])

    #### Avoid table collision - unstable simulation.
    table_collision = self.hand_table_collision(data)

    return {
        'gripper_box': r_lg + r_rg + r_rg_bias,
        'box_handover': box_handover,
        'handover_target': box_target,
        'transfer_hands': transfer_hands,
        'sparse_handover': hand_over.astype(float),
        'no_table_collision': 1 - table_collision,
    }

  def _get_obs(self, data: mjx.Data, info: Dict) -> jax.Array:
    left_gripper_pos = data.site_xpos[self._left_gripper_site]
    left_gripper_mat = data.site_xmat[self._left_gripper_site]
    right_gripper_pos = data.site_xpos[self._right_gripper_site]
    right_gripper_mat = data.site_xmat[self._right_gripper_site]
    box_mat = data.xmat[self._box_body]
    box_top = data.site_xpos[self._box_top_site]
    box_bottom = data.site_xpos[self._box_bottom_site]
    finger_qposadr = data.qpos[self._finger_qposadr]
    l_gripped, r_gripped = info['s_l_gripped'], info['s_r_gripped']
    box_width = self.mjx_model.geom_size[self._box_geom][1]

    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        (finger_qposadr - box_width),
        box_top,
        box_bottom,
        left_gripper_pos,
        left_gripper_mat.ravel()[3:],
        right_gripper_pos,
        right_gripper_mat.ravel()[3:],
        box_mat.ravel()[3:],
        data.xpos[self._box_body] - info['target_pos'],
        (info['_steps'].reshape((1,)) / self._config.episode_length).astype(
            float
        ),
        jp.array([l_gripped, r_gripped]),
    ])

    return obs

  @property
  def observation_size(self) -> int:
    return 85
