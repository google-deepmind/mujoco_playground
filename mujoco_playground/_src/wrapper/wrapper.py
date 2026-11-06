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

"""Base environment wrapper for MuJoCo Playground."""

from typing import Any

import dm_env
from gdm_robotics.interfaces import environment as gdmr_env
from mujoco_playground._src.wrapper.device_layer_wrapper import DeviceLayer as DeviceLayer
from mujoco_playground._src.wrapper.device_layer_wrapper import DeviceLayerWrapper as DeviceLayerWrapper


class Wrapper(gdmr_env.Environment):
  """Base class for environment wrappers."""

  def __init__(self, env: Any):
    self._env = env

  def __getattr__(self, name: str) -> Any:
    if name.startswith("__"):
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    return getattr(self._env, name)

  @property
  def unwrapped(self) -> Any:
    """Returns the base environment (recursively unwrapping wrappers)."""
    if hasattr(self._env, "unwrapped"):
      return self._env.unwrapped
    return self._env

  def reset_with_options(self, *, options: Any) -> dm_env.TimeStep:
    return self._env.reset_with_options(options=options)

  def default_reset_options(self) -> Any:
    return self._env.default_reset_options()

  def step(self, action: Any) -> dm_env.TimeStep:
    return self._env.step(action)

  def action_spec(self) -> Any:
    return self._env.action_spec()

  def timestep_spec(self) -> Any:
    return self._env.timestep_spec()

  def close(self) -> None:
    self._env.close()
