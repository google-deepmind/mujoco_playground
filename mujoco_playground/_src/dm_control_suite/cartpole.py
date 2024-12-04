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
"""Cartpole environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import rewards
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
  )


class Balance(mjx_env.MjxEnv):
  """Cartpole environment with balance task."""

  _CART_RANGE = (-0.25, 0.25)
  _ANGLE_COSINE_RANGE = (0.995, 1)

  def __init__(
      self,
      swing_up: bool,
      sparse: bool,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides=config_overrides)
    if swing_up:
      self._reset_randomize = self._reset_swing_up
    else:
      self._reset_randomize = self._reset_balance
    if sparse:
      self._get_reward = self._sparse_reward
    else:
      self._get_reward = self._dense_reward

    self._xml_path = _XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), common.get_assets()
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

  def _post_init(self) -> None:
    slider_jid = self._mj_model.joint("slider").id
    self._slider_qposadr = self._mj_model.jnt_qposadr[slider_jid]
    hinge_1_jid = self._mj_model.joint("hinge_1").id
    self._hinge_1_qposadr = self._mj_model.jnt_qposadr[hinge_1_jid]

  def _reset_swing_up(self, rng: jax.Array) -> jax.Array:
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._slider_qposadr].set(0.01 * jax.random.normal(rng1))
    qpos = qpos.at[self._hinge_1_qposadr].set(
        jp.pi + 0.01 * jax.random.normal(rng2)
    )
    qpos = qpos.at[2:].set(
        0.1 * jax.random.uniform(rng3, (self.mjx_model.nq - 2,))
    )

    return qpos

  def _reset_balance(self, rng: jax.Array) -> jax.Array:
    rng1, rng2 = jax.random.split(rng, 2)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._slider_qposadr].set(
        jax.random.uniform(rng1, (), minval=-0.1, maxval=0.1)
    )
    qpos = qpos.at[1:].set(
        jax.random.uniform(
            rng2, (self.mjx_model.nq - 1,), minval=-0.034, maxval=0.034
        )
    )

    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._reset_randomize(rng)

    rng, rng1 = jax.random.split(rng, 2)
    qvel = 0.01 * jax.random.normal(rng1, (self.mjx_model.nv,))

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

    metrics = {
        "reward/upright": jp.zeros(()),
        "reward/centered": jp.zeros(()),
        "reward/small_control": jp.zeros(()),
        "reward/small_velocity": jp.zeros(()),
        "reward/cart_in_bounds": jp.zeros(()),
        "reward/angle_in_bounds": jp.zeros(()),
    }
    info = {"rng": rng}

    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    cart_position = data.qpos[self._slider_qposadr]
    pole_angle_cos = data.xmat[2:, 2, 2]  # zz.
    pole_angle_sin = data.xmat[2:, 0, 2]  # xz.
    return jp.concatenate([
        cart_position.reshape(1),
        pole_angle_cos,
        pole_angle_sin,
        data.qvel,
    ])

  def _dense_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    pole_angle_cos = data.xmat[2, 2, 2]
    upright = (pole_angle_cos + 1) / 2
    metrics["reward/upright"] = upright

    cart_position = data.qpos[self._slider_qposadr]
    centered = rewards.tolerance(cart_position, margin=2)
    centered = (1 + centered) / 2
    metrics["reward/centered"] = centered

    small_control = rewards.tolerance(
        action[0], margin=1, value_at_margin=0, sigmoid="quadratic"
    )
    small_control = (4 + small_control) / 5
    metrics["reward/small_control"] = small_control

    angular_vel = data.qvel[1:]
    small_velocity = rewards.tolerance(angular_vel, margin=5).min()
    small_velocity = (1 + small_velocity) / 2
    metrics["reward/small_velocity"] = small_velocity

    return upright * small_control * small_velocity * centered

  def _sparse_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    cart_position = data.qpos[self._slider_qposadr]
    cart_in_bounds = rewards.tolerance(cart_position, self._CART_RANGE)
    metrics["reward/cart_in_bounds"] = cart_in_bounds

    pole_angle_cos = data.xmat[2, 2, 2]
    angle_in_bounds = rewards.tolerance(
        pole_angle_cos, self._ANGLE_COSINE_RANGE
    ).prod()
    metrics["reward/angle_in_bounds"] = angle_in_bounds

    return cart_in_bounds * angle_in_bounds

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
