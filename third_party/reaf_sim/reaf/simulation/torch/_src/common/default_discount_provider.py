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

"""Computes a constant discount given the termination state.

This provider returns a discount of 0.0 in case of termination and 1.0
otherwise (i.e. for truncation and not termination).

It is usually safe to use this discount provider for environments that return
strictly positive rewards.
"""

from collections.abc import Mapping

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.core import discount_provider
from reaf.simulation.torch._src.core import termination_checker
import torch
import tree


# TODO(reaf): add test.
class DefaultDiscountProvider(discount_provider.DiscountProvider):
  """Computes a constant discount given the termination state.

  This provider returns a discount of 0.0 in case of termination and 1.0
  otherwise (i.e. for truncation and not termination).

  It is usually safe to use this discount provider for environments that return
  strictly positive rewards.
  """

  def __init__(self, nworld: int, xpu_device: str):
    self._discount = torch.zeros(nworld, dtype=torch.float32, device=xpu_device)

  def name(self) -> str:
    return 'default_discount'

  def compute_discount(
      self,
      required_features: Mapping[str, torch.Tensor],
      termination_state: torch.Tensor,
  ) -> torch.Tensor:
    self._discount[:] = ~termination_checker.TerminationResult.is_terminated(
        termination_state  # pyrefly: ignore[bad-argument-type]
    )
    return self._discount

  def discount_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=np.float32, name='discount')

  def required_features_keys(self) -> set[str]:
    return set()
