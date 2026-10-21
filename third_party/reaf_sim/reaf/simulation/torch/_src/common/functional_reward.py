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

"""FunctionalRewardProvider that wraps callable functions."""

from collections.abc import Mapping
from collections.abc import Set
from typing import TypedDict

from dm_env import specs
from reaf.simulation.torch._src.core import reward_provider
from reaf.simulation.torch._src.core import types
from reaf.simulation.torch._src.core.types import RewardTerm
from reaf.simulation.torch._src.core.types import Term
import torch


class RewardSpecDict(TypedDict):
  reward: specs.Array
  terms: dict[str, specs.Array]


class RewardDict(TypedDict):
  reward: torch.Tensor
  terms: dict[str, torch.Tensor]


class FunctionalRewardProvider(reward_provider.RewardProvider):
  """Wraps callable functions as a reward provider."""

  def __init__(
      self,
      *,
      name: str = "rewards",
      terms: Mapping[str, "RewardTerm"],
      required_keys: list[str],
  ):
    """Initialize FunctionalRewardProvider.

    Args:
      name: Provider name.
      terms: Dict mapping term names to RewardTerm objects with weight field.
      required_keys: Measurement keys required by the reward functions.
    """
    super().__init__()
    self._name = name
    self._terms = terms
    self._term_names = list(terms.keys())
    self._required_keys = frozenset(required_keys)
    self._weights: torch.Tensor | None = None
    self._dtype: torch.dtype | None = None
    self._dt: float | None = None

  def initialize(
      self, device: torch.device, dtype: torch.dtype, dt: float = 1.0
  ) -> None:
    """Initialize weight tensor on device. Call before compute_reward."""
    weights = [self._terms[n].weight for n in self._term_names]
    self._weights = torch.tensor(weights, device=device, dtype=dtype)
    self._dtype = dtype
    self._dt = dt

  @property
  def name(self) -> str:  # pyrefly: ignore[bad-override]
    return self._name

  def required_features_keys(self) -> Set[str]:  # pyrefly: ignore[bad-override]
    return self._required_keys

  def reward_spec(self) -> RewardSpecDict:
    if self._dtype is None:
      raise RuntimeError("Must call initialize() before reward_spec().")
    dtype = types.TORCH_TO_NUMPY_DTYPE[self._dtype]
    term_specs = {
        n: specs.Array(shape=(), dtype=dtype) for n in self._term_names
    }
    for n in self._term_names:
      term_specs["weighted/" + n] = specs.Array(shape=(), dtype=dtype)
    return {"reward": specs.Array(shape=(), dtype=dtype), "terms": term_specs}

  def compute_reward(
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> RewardDict:
    """Compute weighted sum of reward terms, scaled by dt at the end."""
    if self._weights is None or self._dt is None:
      raise RuntimeError("Must call initialize() before compute_reward().")
    components = [
        self._terms[n].func(required_features, **self._terms[n].params)
        for n in self._term_names
    ]
    stacked = torch.stack(components, dim=0)  # (nterms, nworld)
    # Sanitize NaNs, and consider them as failures (set to 0).
    stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    terms = dict(zip(self._term_names, stacked))
    for name, comp, w in zip(self._term_names, stacked, self._weights):
      terms["weighted/" + name] = comp * w * self._dt
    reward = (self._weights @ stacked) * self._dt
    return {"reward": reward, "terms": terms}

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    pass


class PassthroughRewardProvider(reward_provider.RewardProvider):
  """Passes a single reward term through without scaling or weighting.

  Returns the term's raw output directly from ``compute_reward`` as a
  ``torch.Tensor``.
  """

  def __init__(
      self,
      *,
      name: str = "rewards",
      term: "Term",
      required_keys: list[str],
  ):
    """Initialize PassthroughRewardProvider.

    Args:
      name: Provider name.
      term: A single Term whose func output is used as the reward.
      required_keys: Measurement keys required by the reward function.
    """
    super().__init__()
    self._name = name
    self._term = term
    self._required_keys = frozenset(required_keys)
    self._dtype: torch.dtype | None = None

  def initialize(
      self, device: torch.device, dtype: torch.dtype, dt: float = 1.0
  ) -> None:
    """Store dtype for reward_spec. Device and dt are unused."""
    self._dtype = dtype

  @property
  def name(self) -> str:  # pyrefly: ignore[bad-override]
    return self._name

  def required_features_keys(self) -> Set[str]:  # pyrefly: ignore[bad-override]
    return self._required_keys

  def reward_spec(self) -> specs.Array:
    """Returns a scalar reward spec."""
    if self._dtype is None:
      raise RuntimeError("Must call initialize() before reward_spec().")
    return specs.Array(shape=(), dtype=types.TORCH_TO_NUMPY_DTYPE[self._dtype])

  def compute_reward(
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> torch.Tensor:
    """Returns the term's reward value directly."""
    if self._dtype is None:
      raise RuntimeError("Must call initialize() before compute_reward().")
    reward = self._term.func(required_features, **self._term.params)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    pass
