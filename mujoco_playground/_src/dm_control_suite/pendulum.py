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
"""Pendulum environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "pendulum.xml"
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=0,
  )


class SwingUp(mjx_env.MjxEnv):
  """Swingup environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._pole_body_id = self.mj_model.body("pole").id
    hinge_joint_id = self.mj_model.joint("hinge").id
    self._hinge_qposadr = self.mj_model.jnt_qposadr[hinge_joint_id]
    self._hinge_qveladr = self.mj_model.jnt_dofadr[hinge_joint_id]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1 = jax.random.split(rng)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._hinge_qposadr].set(jax.random.uniform(rng1) * jp.pi)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    metrics = {}
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
        self._pole_orientation(data),
        self._angular_velocity(data).reshape(1),
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info, metrics  # Unused.
    return reward.tolerance(self._pole_vertical(data), (_COSINE_BOUND, 1))

  def _pole_vertical(self, data: mjx.Data) -> jax.Array:
    """Returns vertical (z) component of pole frame."""
    return data.xmat[self._pole_body_id, 2, 2]

  def _angular_velocity(self, data: mjx.Data) -> jax.Array:
    """Returns the angular velocity of the pole."""
    return data.qvel[self._hinge_qveladr]

  def _pole_orientation(self, data: mjx.Data) -> jax.Array:
    """Returns both horizontal and vertical components of pole frame."""
    xz = data.xmat[self._pole_body_id, 0, 2]
    zz = data.xmat[self._pole_body_id, 2, 2]
    return jp.array([xz, zz])

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
