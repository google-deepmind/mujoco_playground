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

"""Reward provider which provides a zero reward.

TODO(idanly): This pattern is cumbersome and repeating, consider removing
ZeroRewardProvider in favor of the FunctionalRewardProvider.
"""

from collections.abc import Mapping

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.core import reward_provider
import torch
import tree


class ZeroRewardProvider(reward_provider.RewardProvider):
  """Reward provider which provides a zero reward."""

  def __init__(
      self,
      nworld: int,
      xpu_device: torch.device | str,
      name: str = 'zero_reward_provider',
  ):
    self._name = name
    self._reward = torch.zeros(
        nworld, 1, dtype=torch.float32, device=xpu_device
    )

  def name(self) -> str:
    return self._name

  def compute_reward(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    """Returns a zero reward."""
    return self._reward

  def reward_spec(self) -> tree.Structure[specs.Array]:  # pyrefly: ignore[invalid-type-var]
    """Returns the spec for a constant zero reward."""
    return specs.Array(shape=(1,), dtype=np.float32)

  def required_features_keys(self) -> set[str]:
    """Returns empty set.

    There are no feature keys that are required to compute the reward.
    """
    return set()

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    pass
