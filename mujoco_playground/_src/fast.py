# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiny analytic environment for testing Playground and Brax wrappers.

Fast does not create or step a MuJoCo model. Its State.data is simply a JAX
array, making it useful for integration tests that do not need physics.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
  """Return a small fixed-step configuration for wrapper smoke tests."""
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.05,
      episode_length=5,
      action_repeat=1,
  )


class Fast(mjx_env.MjxEnv):
  """Deterministic, model-free environment with two observations and one action."""

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config or default_config(), config_overrides)

  def _get_obs(self, data: jax.Array, info: Dict[str, Any]) -> jax.Array:
    del info
    return data

  def reset(self, rng: jax.Array) -> mjx_env.State:
    del rng
    data = jp.zeros((2,), dtype=jp.float32)
    reward = jp.zeros((), dtype=jp.float32)
    done = jp.zeros((), dtype=jp.float32)
    return mjx_env.State(
        data=data,
        obs=self._get_obs(data, {}),
        reward=reward,
        done=done,
        metrics={'reward/fast': reward},
        info={},
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # A small, cheap state update. Episode length and done/truncation are
    # managed by the existing Brax EpisodeWrapper, not a physics backend.
    displacement = jp.clip(jp.reshape(action, (-1,))[0], -1.0, 1.0)
    data = state.data + jp.array([displacement, 1.0], dtype=jp.float32)
    reward = -jp.square(data[0])
    return state.replace(
        data=data,
        obs=self._get_obs(data, state.info),
        reward=reward,
        done=jp.zeros_like(state.done),
        metrics={'reward/fast': reward},
    )

  @property
  def action_size(self) -> int:
    return 1

  @property
  def xml_path(self) -> str:
    raise NotImplementedError('Fast has no MJCF model or XML file.')

  @property
  def mj_model(self) -> mujoco.MjModel:
    raise NotImplementedError('Fast has no MuJoCo model.')

  @property
  def mjx_model(self) -> mjx.Model:
    raise NotImplementedError('Fast has no MJX model.')
