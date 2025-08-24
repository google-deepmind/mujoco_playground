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
"""Wrappers for MuJoCo Playground environments."""

import contextlib
import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np


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

  def render(
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return self.env.render(
        trajectory, height, width, camera, scene_option, modify_scene_fns
    )


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate, should
      equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over
    full_reset: whether to call `env.reset` during `env.step` on done rather
      than resetting to a cached first state. Setting full_reset=True may
      increase wallclock time because it forces full resets to random states.

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if vision:
    env = MadronaWrapper(env, num_vision_envs, randomization_fn)
  elif randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env, full_reset=full_reset)
  return env


class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done.

  If `full_reset` is disabled (default):
    * the environment will reset to a cached first state.
    * only data and obs are reset, not the environment info.

  If `full_reset` is enabled:
    * the environment will call env.reset during env.step on done.
    * `full_reset` will thus incur a penalty in wallclock time depending on the
      complexity of the reset function.
    * info is fully reset, except for info under the key
      `AutoResetWrapper_preserve_info`, which is passed through from the prior
      step. This can be used for curriculum learning.

  Attributes:
    env: The wrapped environment.
    full_reset: Whether to call `env.reset` during `env.step` on done.
  """

  def __init__(self, env: Any, full_reset: bool = False):
    super().__init__(env)
    self._full_reset = full_reset
    self._info_key = 'AutoResetWrapper'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[f'{self._info_key}_done_count'] = jp.zeros(
        key.shape[:-1], dtype=int
    )
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # grab the reset state.
    reset_state = None
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
    if self._full_reset:
      reset_state = self.reset(reset_key)
      reset_data = reset_state.data
      reset_obs = reset_state.obs
    else:
      reset_data = state.info[f'{self._info_key}_first_data']
      reset_obs = state.info[f'{self._info_key}_first_obs']

    if 'steps' in state.info:
      # reset steps to 0 if done.
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    done_count_key = f'{self._info_key}_done_count'
    if self._full_reset and reset_state:
      next_info = jax.tree.map(where_done, reset_state.info, state.info)
      next_info[done_count_key] = state.info[done_count_key]

      if 'steps' in next_info:
        next_info['steps'] = state.info['steps']
      preserve_info_key = f'{self._info_key}_preserve_info'
      if preserve_info_key in next_info:
        next_info[preserve_info_key] = state.info[preserve_info_key]

    next_info[done_count_key] += state.done.astype(int)
    next_info[f'{self._info_key}_rng'] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)


class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

  @contextlib.contextmanager
  def v_env_fn(self, mjx_model: mjx.Model):
    env = self.env.unwrapped
    old_mjx_model = env._mjx_model
    try:
      env.unwrapped._mjx_model = mjx_model
      yield env
    finally:
      env.unwrapped._mjx_model = old_mjx_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res


def _identity_vision_randomization_fn(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  mjx_model = mjx_model.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(mjx_model.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(jp.repeat(-1, mjx_model.geom_matid.shape[0], 0), 0),
          num_worlds,
          axis=0,
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(mjx_model.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return mjx_model, in_axes


def _supplement_vision_randomization_fn(
    mjx_model: mjx.Model,
    randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    num_worlds: int,
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  mjx_model, in_axes = randomization_fn(mjx_model)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_type',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = -1 if field == 'geom_matid' else getattr(mjx_model, field)
      mjx_model = mjx_model.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return mjx_model, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    return self._env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Run one timestep of the environment's dynamics."""
    return self._env.step(state, action)

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)
