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
"""Hopper environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "hopper.xml"
# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2.0


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=50_000,
      njmax=50,
  )


class Hopper(mjx_env.MjxEnv):
  """Hopper environment."""

  def __init__(
      self,
      hopping: bool,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    if hopping:
      self._get_reward = self._hop_reward
      self._metric_keys = [
          "reward/standing",
          "reward/hopping",
      ]
    else:
      self._get_reward = self._stand_reward
      self._metric_keys = [
          "reward/standing",
          "reward/small_control",
      ]

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_id = self.mj_model.body("torso").id
    self._foot_id = self.mj_model.body("foot").id
    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2 = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(
        jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi)
    )
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng2,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    metrics = {k: jp.zeros(()) for k in self._metric_keys}
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        data.qpos[1:],
        data.qvel,
        self._touch(data),
    ])

  def _hop_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    standing = reward.tolerance(self._height(data), (_STAND_HEIGHT, 2))
    metrics["reward/standing"] = standing

    hopping = reward.tolerance(
        self._speed(data),
        bounds=(_HOP_SPEED, float("inf")),
        margin=_HOP_SPEED / 2,
        value_at_margin=0.5,
        sigmoid="linear",
    )
    metrics["reward/hopping"] = hopping

    return standing * hopping

  def _stand_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.

    standing = reward.tolerance(self._height(data), (_STAND_HEIGHT, 2))
    metrics["reward/standing"] = standing

    small_control = reward.tolerance(
        action, margin=1, value_at_margin=0, sigmoid="quadratic"
    ).mean()
    small_control = (small_control + 4) / 5
    metrics["reward/small_control"] = small_control

    return standing * small_control

  def _height(self, data: mjx.Data) -> jax.Array:
    torso_z = data.xipos[self._torso_id, -1]
    foot_z = data.xipos[self._foot_id, -1]
    return torso_z - foot_z

  def _speed(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")[
        0
    ]  # x component.

  def _touch(self, data: mjx.Data) -> jax.Array:
    toe = mjx_env.get_sensor_data(self.mj_model, data, "touch_toe")
    heel = mjx_env.get_sensor_data(self.mj_model, data, "touch_heel")
    touch = jp.hstack([toe, heel])
    return jp.log1p(touch)

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
