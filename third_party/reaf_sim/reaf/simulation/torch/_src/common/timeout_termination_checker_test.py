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

from collections.abc import Mapping
import itertools
from typing import TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import discount_provider as base_discount_provider
from reaf.core import reward_provider as base_reward_provider
from reaf.core import task_layer
from reaf.core import termination_checker as base_termination_checker
from reaf.simulation.torch._src.common import timeout_termination_checker
from reaf.simulation.torch._src.core import termination_checker
import torch

TerminationResult: TypeAlias = base_termination_checker.TerminationResult


class MockTerminationChecker(base_termination_checker.TerminationChecker):

  def __init__(
      self,
      required_features_keys: set[str],
      termination_state: termination_checker.TerminationResult,
  ):
    self._required_features_keys = required_features_keys
    self._termination_state = termination_state

  def name(self) -> str:
    return "test_termination_checker"

  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> termination_checker.TerminationResult:
    assert set(required_features.keys()) == self._required_features_keys
    return self._termination_state

  def required_features_keys(self) -> set[str]:
    return self._required_features_keys


class TerminationCheckerTest(parameterized.TestCase):

  @parameterized.named_parameters(*[
      dict(
          testcase_name="_".join([state1.name, state2.name, state3.name]),  # pyrefly: ignore[no-matching-overload]
          termination_state1=state1,
          termination_state2=state2,
          termination_state3=state3,
      )
      for state1, state2, state3 in itertools.product(
          *[[
              TerminationResult.DO_NOT_TERMINATE,
              TerminationResult.TRUNCATE,
              TerminationResult.TERMINATE,
          ]]
          * 3
      )
  ])
  def test_works_with_task_layer(
      self,
      termination_state1: TerminationResult,
      termination_state2: TerminationResult,
      termination_state3: TerminationResult,
  ):
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
        "feature2": np.array([4, 5, 6], dtype=np.int32),
        "feature3": np.array([0.7, -1.0, 12.0]),
    }

    termination_checker1 = mock.create_autospec(
        base_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker1",
    )
    termination_checker1.required_features_keys.return_value = set(
        ("feature1", "feature3")
    )
    termination_checker1.check_termination.return_value = termination_state1

    termination_checker2 = MockTerminationChecker(
        required_features_keys=set(("feature2", "feature3")),
        termination_state=termination_state2,
    )

    termination_checker3 = mock.create_autospec(
        base_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker3",
    )
    termination_checker3.required_features_keys.return_value = set(
        ("feature1", "feature2")
    )
    termination_checker3.check_termination.return_value = termination_state3

    task_layer_instance = task_layer.TaskLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=mock.create_autospec(
            base_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(
            termination_checker1,
            termination_checker2,
            termination_checker3,
        ),
        discount_provider=mock.create_autospec(
            base_discount_provider.DiscountProvider, instance=True
        ),
        features_observers=(),
        loggers=(),
    )

    self.assertEqual(
        task_layer_instance.check_for_termination(features),
        termination_state1.combine(termination_state2).combine(
            termination_state3
        ),
    )


class TimeoutTerminationCheckerTest(parameterized.TestCase):

  def test_timeout_termination_checker(self):
    checker = timeout_termination_checker.TimeoutTerminationChecker(
        timeout=10.0,
        time_feature_name="time",
        xpu_device="cpu",
    )
    self.assertEqual(checker.name(), "timeout_termination")
    self.assertEqual(checker.timeout, 10.0)
    self.assertEqual(checker.required_features_keys(), {"time"})

    features = {"time": torch.tensor([5.0, 10.0, 10.1, 15.0])}
    result = checker.check_termination(features)
    expected = torch.tensor([
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
        termination_checker.TerminationResult.TRUNCATE,
        termination_checker.TerminationResult.TRUNCATE,
    ])
    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
  absltest.main()
