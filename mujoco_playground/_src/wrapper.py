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

from typing import Any, List

from brax import base as brax_base
from brax.envs import base as brax_env
from brax.mjx import base as brax_mjx_base
import jax
from mujoco_playground._src import mjx_env


class BraxEnvWrapper(brax_env.Env):
  """Wraps a MuJoCo Playground environment as a Brax environment."""

  def __init__(self, env: mjx_env.MjxEnv):
    self._env = env
    self.sys = env.mjx_model

  def reset(self, rng: jax.Array) -> brax_env.State:
    """Resets the environment to an initial state."""
    state = self._env.reset(rng)
    x = brax_base.Transform(pos=state.data.xpos[1:], rot=state.data.xquat[1:])
    # pytype: disable=wrong-arg-types
    return brax_env.State(
        pipeline_state=brax_mjx_base.State(
            **state.data.__dict__, x=x, xd=None, q=None, qd=None),
        obs=state.obs,
        reward=state.reward,
        done=state.done,
        metrics=state.metrics,
        info=state.info,
    )
    # pytype: enable=wrong-arg-types

  def step(self, state: brax_env.State, action: jax.Array) -> brax_env.State:
    """Run one timestep of the environment's dynamics."""
    # pytype: disable=wrong-arg-types
    state_ = mjx_env.State(
        data=state.pipeline_state,
        obs=state.obs,
        reward=state.reward,
        done=state.done,
        metrics=state.metrics,
        info=state.info,
    )
    state_ = self._env.step(state_, action)
    x = brax_base.Transform(pos=state_.data.xpos[1:], rot=state_.data.xquat[1:])
    return brax_env.State(
        pipeline_state=state_.data.replace(x=x),
        obs=state_.obs,
        reward=state_.reward,
        done=state_.done,
        metrics=state_.metrics,
        info=state_.info,
    )
    # pytype: enable=wrong-arg-types

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    return self._env.observation_size

  @property
  def action_size(self) -> int:
    """The size of the action vector expected by step."""
    return self._env.action_size

  @property
  def backend(self) -> str:
    """The physics backend that this env was instantiated with."""
    return "mjx"

  @property
  def dt(self) -> float:
    """The env timestep."""
    return self._env.dt

  def render(self, trajectory: List[brax_env.State], *args, **kwargs) -> Any:
    """Renders the env."""
    new_traj = []
    for t in trajectory:
      new_traj.append(
          mjx_env.State(
              data=t.pipeline_state,
              obs=t.obs,
              reward=t.reward,
              done=t.done,
              metrics=t.metrics,
              info=t.info,
          )  # pytype: disable=wrong-arg-types
      )
    return self._env.render(new_traj, *args, **kwargs)
