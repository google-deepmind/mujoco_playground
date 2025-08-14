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
"""Walker environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "walker.xml"
# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
WALK_SPEED = 1
RUN_SPEED = 8


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.025,
      sim_dt=0.0025,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=50_000,
      njmax=100,
  )


class PlanarWalker(mjx_env.MjxEnv):
  """A planar walker task."""

  def __init__(
      self,
      move_speed: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._move_speed = move_speed
    if self._move_speed == 0.0:
      self._get_reward = self._get_stand_reward
    else:
      self._get_reward = self._get_move_reward

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

    metrics = {
        "reward/standing": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "reward/stand": jp.zeros(()),
        "reward/move": jp.zeros(()),
    }
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
    orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
    height = data.xmat[self._torso_id, 2, 2]
    velocity = data.qvel
    return jp.concatenate([
        orientations,
        height.reshape(1),
        velocity,
    ])

  def _get_stand_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    torso_height = data.xpos[self._torso_id, -1]
    standing = reward.tolerance(
        torso_height,
        bounds=(_STAND_HEIGHT, float("inf")),
        margin=_STAND_HEIGHT / 2,
    )
    metrics["reward/standing"] = standing

    torso_upright = data.xmat[self._torso_id, 2, 2]  # zz component.
    upright = (1 + torso_upright) / 2
    metrics["reward/upright"] = upright

    stand_reward = (3 * standing + upright) / 4
    metrics["reward/stand"] = stand_reward

    return stand_reward

  def _get_move_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    stand_reward = self._get_stand_reward(data, action, info, metrics)

    horizontal_velocity = mjx_env.get_sensor_data(
        self.mj_model, data, "torso_subtreelinvel"
    )[0]
    move_reward = reward.tolerance(
        horizontal_velocity,
        bounds=(self._move_speed, float("inf")),
        margin=self._move_speed / 2,
        value_at_margin=0.5,
        sigmoid="linear",
    )
    metrics["reward/move"] = move_reward

    return stand_reward * (5 * move_reward + 1) / 6

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
