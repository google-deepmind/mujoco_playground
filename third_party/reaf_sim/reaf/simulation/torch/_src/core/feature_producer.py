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
"""Torch implementation of FeaturesProducer base class."""

import abc
from collections.abc import Mapping

from reaf.core import features_producer as core_features_producer
import torch


class FeatureProducer(core_features_producer.FeaturesProducer):
  """Produces additional features to be exposed by the task layer.

  This is the torch version of the base FeaturesProducer class.

  Lifecycle: hooks advance state, reads are pure.
    - `reset(env_mask)`: clears state of the `env_mask` worlds.
    - `on_begin_stepping(features, env_mask)`: after a reset, records the reset
      state of the `env_mask` worlds.
    - `on_step(features)`: once per control step, after the physics step.
    - `produce_features(features)`: pure read. `TaskLayer` may call it any
      number of times, e.g. again for all worlds after a partial reset, so it
      must not advance state.

  Producers that wrap another producer must forward both hooks to it.
  """

  @abc.abstractmethod
  def produce_features(  # pyrefly: ignore[bad-override]
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Produces additional features for the environment.

    Should not advance any state: see the class docstring.

    Args:
      required_features: The features listed in `required_features_keys()`.

    Returns:
      The produced features, batched over worlds.
    """

  def on_step(self, required_features: Mapping[str, torch.Tensor]) -> None:
    """Advances per-control-step state.

    Runs once per control step, in `TaskLayer.compute_all_features()`, right
    before `produce_features()`. Not called after resets:
    `on_begin_stepping()` initializes the per-step state of the reset worlds
    instead.

    Args:
      required_features: The features listed in `required_features_keys()`,
        after the physics step.
    """
    del required_features

  def on_begin_stepping(
      self,
      required_features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Records the reset state of the `env_mask` worlds.

    Runs after `reset()` and the device layer's `begin_stepping()`, right before
    `produce_features()`, e.g. to fill history buffers with the reset value.
    Worlds outside `env_mask` must keep their state.

    Args:
      required_features: The features listed in `required_features_keys()`,
        holding the reset state.
      env_mask: Boolean mask of the reset worlds. None: all worlds.
    """
    del required_features, env_mask

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets the internal state of the feature producer.

    Args:
      env_mask: Optional boolean environment mask to reset. If None, resets all
        environments.
    """
