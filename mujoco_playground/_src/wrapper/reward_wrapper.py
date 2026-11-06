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

"""Reward wrappers for MuJoCo Playground environments."""

from typing import Any

import dm_env
from mujoco_playground._src.wrapper import wrapper


class RewardDictToFloatWrapper(wrapper.Wrapper):
  """Replaces a dict-valued ``ts.reward`` with the scalar ``reward`` entry.

  This is only a wrapper around the dict-valued reward produced by
  ``reaf.simulation.torch.FunctionalRewardProvider``, which returns
  a dict of the form ``{"reward": Tensor, "terms": {...}}``.  The wrapper
  extracts the ``"reward"`` value so that ``ts.reward`` is a plain tensor.
  """

  def reset_with_options(self, *, options: Any) -> dm_env.TimeStep:
    ts = self._env.reset_with_options(options=options)
    return ts._replace(reward=_float_reward_from_timestep(ts))

  def step(self, action: Any) -> dm_env.TimeStep:
    ts = self._env.step(action)
    return ts._replace(reward=_float_reward_from_timestep(ts))


def _float_reward_from_timestep(timestep: Any) -> Any:
  reward = timestep.reward
  assert reward is not None, "reward must not be None"
  assert isinstance(reward, dict), f"reward must be a dict, got {type(reward)}"
  return reward["reward"]
