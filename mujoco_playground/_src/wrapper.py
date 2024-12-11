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
"""Wrappers for MuJoCo Playground environments."""

from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env


class Wrapper(mjx_env.MjxEnv):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Any):  # pylint: disable=super-init-not-called
    self.env = env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self.env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.env.mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self.env.mjx_model

  @property
  def xml_path(self) -> str:
    return self.env.xml_path


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  return env


class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    data = jax.tree.map(
        where_done, state.info['first_state'], state.data
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)


class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

  def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model  # pylint: disable=protected-access
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      env = self._env_fn(mjx_model=mjx_model)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res
