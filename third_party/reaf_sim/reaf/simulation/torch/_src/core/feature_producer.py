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
  """

  @abc.abstractmethod
  def produce_features(  # pyrefly: ignore[bad-override]
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Produces additional features for the environment."""

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets the internal state of the feature producer.

    Args:
      env_mask: Optional boolean environment mask to reset. If None, resets all
        environments.
    """
