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

"""Tests for declarative curriculum schedules."""

from absl.testing import absltest
from reaf.simulation.torch._src.common import curriculum
import torch


class CurriculumTest(absltest.TestCase):

  def test_stages_evaluates_thresholds_and_default(self):
    sched = curriculum.Stages([(10, -0.02), (20, -0.04)])

    torch.testing.assert_close(
        curriculum.evaluate(
            sched, {"global_step": torch.tensor([0, 9])}, default=-0.01
        ),
        torch.tensor([-0.01, -0.01]),
    )
    torch.testing.assert_close(
        curriculum.evaluate(
            sched, {"global_step": torch.tensor([10, 19])}, default=-0.01
        ),
        torch.tensor([-0.02, -0.02]),
    )
    torch.testing.assert_close(
        curriculum.evaluate(
            sched, {"global_step": torch.tensor([20, 100])}, default=-0.01
        ),
        torch.tensor([-0.04, -0.04]),
    )

  def test_stages_supports_custom_per_world_feature_key(self):
    sched = curriculum.Stages([(0.5, 1.0), (0.8, 2.0)], key="voc/success_rate")
    features = {"voc/success_rate": torch.tensor([0.2, 0.5, 0.9])}
    torch.testing.assert_close(
        curriculum.evaluate(sched, features, default=0.0),
        torch.tensor([0.0, 1.0, 2.0]),
    )

  def test_stages_squeezes_trailing_unit_dim_and_rejects_higher_rank(self):
    sched = curriculum.Stages([(10, 2.0)])
    torch.testing.assert_close(
        curriculum.evaluate(
            sched, {"global_step": torch.tensor([[0], [10]])}, default=1.0
        ),
        torch.tensor([1.0, 2.0]),
    )
    with self.assertRaisesRegex(ValueError, "Curriculum trigger"):
      curriculum.evaluate(
          sched, {"global_step": torch.zeros(2, 3)}, default=1.0
      )

  def test_resolve_params_rejects_non_numeric_base(self):
    sched = curriculum.Stages([(10, 2.0)])
    with self.assertRaisesRegex(TypeError, "numeric base value"):
      curriculum.resolve_params(
          {"flag": True}, {"flag": sched}, {"global_step": torch.tensor([0])}
      )
    with self.assertRaisesRegex(TypeError, "numeric base value"):
      curriculum.resolve_params(
          {"name": "a"}, {"name": sched}, {"global_step": torch.tensor([0])}
      )

  def test_stages_rejects_decreasing_thresholds(self):
    with self.assertRaisesRegex(ValueError, "nondecreasing"):
      curriculum.Stages([(20, -0.02), (10, -0.04)])

  def test_evaluate_rejects_unknown_schedule(self):
    with self.assertRaisesRegex(TypeError, "Unsupported curriculum schedule"):
      curriculum.evaluate(
          object(),  # pyrefly: ignore[bad-argument-type]
          {"global_step": torch.tensor([0])},
          default=1.0,
      )


if __name__ == "__main__":
  absltest.main()
