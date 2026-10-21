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

"""REAF feature producer for the task instruction.

The safari episodic logger requires an `instruction` key in every timestep's
observations — instruction is part of the data stream (what VLA training
conditions on), not just session metadata.
"""

from typing import Any
from typing import Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import features_producer
from typing_extensions import override

INSTRUCTION_KEY = "instruction"


class InstructionFeaturesProducer(features_producer.FeaturesProducer):
  """Produces the `instruction` feature from a string."""

  def __init__(self, instruction: str):
    self._instruction = instruction

  @property
  @override
  def name(self) -> str:
    return "instruction_features_producer"

  def set_instruction(self, instruction: str) -> None:
    self._instruction = instruction

  @override
  def produce_features(
      self, required_features: Mapping[str, Any]
  ) -> Mapping[str, Any]:
    del required_features
    return {INSTRUCTION_KEY: np.array(self._instruction)}

  @override
  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {INSTRUCTION_KEY: specs.StringArray(shape=())}

  @override
  def required_features_keys(self) -> set[str]:
    return set()
