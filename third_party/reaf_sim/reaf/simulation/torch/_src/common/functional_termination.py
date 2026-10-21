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

"""FunctionalTerminationChecker that wraps callable functions."""

from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Set

from reaf.simulation.torch._src.core import termination_checker
from reaf.simulation.torch._src.core.types import Term
import torch

TerminationResult = termination_checker.TerminationResult


class FunctionalTerminationChecker(termination_checker.TerminationChecker):
  """Wraps callable functions as a termination checker."""

  def __init__(
      self,
      *,
      name: str,
      terms: Sequence["Term"],
      required_keys: list[str],
  ):
    """Initialize FunctionalTerminationChecker.

    Args:
      name: Checker name.
      terms: List of Term objects with termination check functions.
      required_keys: Measurement keys required by the check functions.
    """
    self._name = name
    self._terms = list(terms)
    self._required_keys = frozenset(required_keys)
    self._result_buffer: torch.Tensor | None = None

  def initialize(self, nworld: int, device: torch.device) -> None:
    """Initialize result buffer. Call before check_termination."""
    self._result_buffer = torch.full(
        (nworld,),
        TerminationResult.DO_NOT_TERMINATE,
        dtype=torch.int32,
        device=device,
    )

  @property
  def name(self) -> str:  # pyrefly: ignore[bad-override]
    return self._name

  def required_features_keys(self) -> Set[str]:  # pyrefly: ignore[bad-override]
    return self._required_keys

  def check_termination(
      self, features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    """Run all check functions and combine results."""
    assert self._result_buffer is not None
    results = torch.stack(
        [term.func(features, **term.params) for term in self._terms]
    )
    torch.amax(results.to(torch.int32), dim=0, out=self._result_buffer)
    return self._result_buffer

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    pass
