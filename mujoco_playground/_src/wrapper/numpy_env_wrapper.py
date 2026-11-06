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
"""Wrapper converting NumPy actions to torch and torch timesteps to NumPy."""

from typing import Any

import dm_env
from mujoco_playground._src.wrapper import wrapper
import numpy as np
import torch
import tree


class NumpyEnvWrapper(wrapper.Wrapper):
  """Converts NumPy actions to torch and torch observations to NumPy arrays."""

  def reset_with_options(self, *, options: Any) -> dm_env.TimeStep:
    return _to_numpy_timestep(self._env.reset_with_options(options=options))

  def step(self, action: Any) -> dm_env.TimeStep:
    device = getattr(self._env, "device", None)
    action = tree.map_structure(lambda x: _to_torch_leaf(x, device), action)
    return _to_numpy_timestep(self._env.step(action))


def _to_numpy_leaf(x: Any) -> Any:
  """Converts torch.Tensor to np.ndarray, leaving other types unchanged."""
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  return x


def _to_torch_leaf(x: Any, device: str | None = None) -> Any:
  """Converts np.ndarray to torch.Tensor, leaving other types unchanged."""
  if isinstance(x, np.ndarray):
    return torch.as_tensor(x, device=device)
  return x


def _to_numpy_timestep(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  """Converts all tensor components in TimeStep to NumPy arrays."""
  reward = (
      tree.map_structure(_to_numpy_leaf, timestep.reward)
      if timestep.reward is not None
      else None
  )
  discount = (
      tree.map_structure(_to_numpy_leaf, timestep.discount)
      if timestep.discount is not None
      else None
  )
  observation = tree.map_structure(_to_numpy_leaf, timestep.observation)

  return timestep._replace(
      reward=reward, discount=discount, observation=observation
  )
