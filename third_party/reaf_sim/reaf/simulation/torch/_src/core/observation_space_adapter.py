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

import abc
from collections.abc import Mapping

from reaf.core import observation_space_adapter as core_adapter
import torch


class ObservationSpaceAdapter(core_adapter.ObservationSpaceAdapter):
  """reaf-sim specific ObservationSpaceAdapter interface.

  Overrides the `reset` method to provide torch-specific type annotations for
  environment indices.

  Lifecycle: hooks advance state, reads are pure.
    - `reset(env_mask)`: clears state of the `env_mask` worlds.
    - `on_begin_stepping(features, env_mask)`: after a reset, records the reset
      state of the `env_mask` worlds (e.g. fills history buffers).
    - `on_step(features)`: once per control step, after the task layer features
      are computed (e.g. pushes history, draws per-step noise).
    - `observations_from_features(features)`: pure read. The environment may
      call it any number of times, e.g. again for all worlds after a partial
      reset, so it must not advance state.
  """

  @abc.abstractmethod
  def observations_from_features(  # pyrefly: ignore[bad-override]
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> Mapping[str, torch.Tensor]:
    """Computes the observation from the required features.

    Should not advance any state: see the class docstring.

    Args:
      required_features: The task layer features.
    """

  def on_step(self, features: Mapping[str, torch.Tensor]) -> None:
    """Advances per-control-step state.

    Runs once per `Environment.step()`, after the task layer features are
    computed and before `observations_from_features()`. Not called after
    resets: `on_begin_stepping()` initializes the per-step state of the reset
    worlds instead.

    Args:
      features: All task layer features, after the physics step.
    """
    del features

  def on_begin_stepping(
      self,
      features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Records the reset state of the `env_mask` worlds.

    Runs after `reset()` and the task layer features of the reset state are
    computed, before `observations_from_features()`. Worlds outside `env_mask`
    must keep their state.

    Args:
      features: All task layer features, holding the reset state.
      env_mask: Boolean mask of the reset worlds. None: all worlds.
    """
    del features, env_mask

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset any internal state for the given environment mask.

    Called by the environment on episode reset. Default is a no-op.
    Override this method to support stateful buffers (e.g. history).

    Args:
      env_mask: Boolean environment mask to reset. If None, resets all.
    """
