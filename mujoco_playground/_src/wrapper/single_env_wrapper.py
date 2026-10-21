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

"""Wrapper to squeeze batch dimension from vectorized environments."""

from typing import Any

import dm_env
from dm_env import specs
from mujoco_playground._src.wrapper import wrapper
import numpy as np
import tree


class SingleEnvWrapper(wrapper.Wrapper):
  """Wraps a batched environment (nworld=1) to make it look unbatched."""

  def reset_with_options(self, *, options: Any) -> dm_env.TimeStep:
    return _squeeze_timestep(self._env.reset_with_options(options=options))

  def step(self, action: Any) -> dm_env.TimeStep:
    def add_batch_dim(x: Any) -> Any:
      """Adds a leading batch dimension to a single leaf array or tensor."""
      return (
          x.unsqueeze(0)
          if hasattr(x, "unsqueeze")
          else np.expand_dims(x, axis=0)
      )

    batched_action = tree.map_structure(add_batch_dim, action)
    return _squeeze_timestep(self._env.step(batched_action))

  def timestep_spec(self) -> Any:
    if not hasattr(self._env, "timestep_spec"):
      raise AttributeError("Underlying environment must have timestep_spec()")
    spec = self._env.timestep_spec()

    return spec._replace(
        reward=_squeeze_spec(spec.reward),
        discount=_squeeze_spec(spec.discount),
        observation=_squeeze_spec(spec.observation),
    )


def _squeeze_timestep(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  """Removes the leading batch dimension from components of a TimeStep."""

  def squeeze_leaf(x: Any) -> Any:
    """Squeezes the leading batch dimension of an array leaf if size is 1."""
    if hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == 1:
      if hasattr(x, "squeeze"):
        return x.squeeze(0)
      else:
        return x[0]
    return x

  reward = (
      tree.map_structure(squeeze_leaf, timestep.reward)
      if timestep.reward is not None
      else None
  )
  discount = (
      tree.map_structure(squeeze_leaf, timestep.discount)
      if timestep.discount is not None
      else None
  )
  observation = tree.map_structure(squeeze_leaf, timestep.observation)

  return timestep._replace(
      reward=reward, discount=discount, observation=observation
  )


def _squeeze_spec_leaf(s: Any) -> Any:
  """Squeezes the leading batch dimension of a single spec leaf if size is 1."""
  if hasattr(s, "shape") and s.shape and s.shape[0] == 1:
    new_shape = s.shape[1:]
    if isinstance(s, specs.BoundedArray):
      return specs.BoundedArray(
          shape=new_shape,
          dtype=s.dtype,
          minimum=s.minimum,
          maximum=s.maximum,
          name=s.name,
      )
    elif isinstance(s, specs.StringArray):
      return specs.StringArray(
          shape=new_shape,
          name=s.name,
      )
    else:
      return specs.Array(
          shape=new_shape,
          dtype=s.dtype,
          name=s.name,
      )
  return s


def _squeeze_spec(s: Any) -> Any:
  """Squeezes leading batch dimensions across a nested spec structure."""
  return tree.map_structure(_squeeze_spec_leaf, s)
