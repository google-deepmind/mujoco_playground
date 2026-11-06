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
"""reaf-sim specific ObservationSpaceAdapter interface."""

from collections.abc import Mapping

from reaf.core import observation_space_adapter as core_adapter
import torch


class ObservationSpaceAdapter(core_adapter.ObservationSpaceAdapter):
  """reaf-sim specific ObservationSpaceAdapter interface.

  Overrides the `reset` method to provide torch-specific type annotations for
  environment indices.
  """

  def observations_from_features(  # pyrefly: ignore[bad-override]
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> Mapping[str, torch.Tensor]:
    """Computes the observation from the required features."""
    ...

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset any internal state for the given environment mask.

    Called by the environment on episode reset. Default is a no-op.
    Override this method to support stateful buffers (e.g. history).

    Args:
      env_mask: Boolean environment mask to reset. If None, resets all.
    """
