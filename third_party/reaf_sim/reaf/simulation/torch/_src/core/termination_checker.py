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
"""Checks if the episode should terminate."""

from collections.abc import Mapping

from reaf.core import termination_checker
from reaf.core.termination_checker import TerminationResult  # pylint: disable=unused-import
import torch


class TerminationChecker(termination_checker.TerminationChecker):
  """Checks if the episode should terminate."""

  def check_termination(  # pyrefly: ignore[bad-override]
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> torch.Tensor:
    """Checks if the episode should terminate."""
    ...

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets the internal state of the termination checker."""
    ...
