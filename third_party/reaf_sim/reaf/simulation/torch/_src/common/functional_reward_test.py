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

"""Tests for FunctionalRewardProvider."""

from absl.testing import absltest
import numpy as np
from reaf.simulation.torch._src.builders.flat_scene.config import Reward
from reaf.simulation.torch._src.builders.flat_scene.config import Term
from reaf.simulation.torch._src.common import curriculum
from reaf.simulation.torch._src.common import functional_reward
import torch


class FunctionalRewardProviderTest(absltest.TestCase):

  def test_weighted_sum(self):
    provider = functional_reward.FunctionalRewardProvider(
        name="test",
        terms={
            "a": Reward(lambda m, key: m[key], weight=1.0, key="x"),
            "b": Reward(lambda m, key: m[key], weight=2.0, key="y"),
            "c": Reward(lambda m, key: m[key], weight=0.0, key="z"),
            "d": Reward(lambda m, key: m[key], key="w"),
        },
        required_keys=["x", "y", "z", "w"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)

    m = {
        "x": torch.tensor([1.0, 2.0]),
        "y": torch.tensor([3.0, 4.0]),
        "z": torch.tensor([5.0, 6.0]),
        "w": torch.tensor([7.0, 8.0]),
    }
    result = provider.compute_reward(m)

    # reward = (1.0 * x + 2.0 * y + 0.0 * z + 1.0 * w) * 0.5
    #        = ([1.0+6.0+0.0+7.0, 2.0+8.0+0.0+8.0]) * 0.5
    #        = ([14.0, 18.0]) * 0.5 = [7.0, 9.0]
    torch.testing.assert_close(result["reward"], torch.tensor([7.0, 9.0]))
    # term a is raw x = [1.0, 2.0]
    torch.testing.assert_close(result["terms"]["a"], torch.tensor([1.0, 2.0]))
    # weighted/a = x * weight_a * dt = x * 1.0 * 0.5 = [0.5, 1.0]
    torch.testing.assert_close(
        result["terms"]["weighted/a"], torch.tensor([0.5, 1.0])
    )
    # term b is raw y = [3.0, 4.0]
    torch.testing.assert_close(result["terms"]["b"], torch.tensor([3.0, 4.0]))
    # weighted/b = y * weight_b * dt = y * 2.0 * 0.5 = [3.0, 4.0]
    torch.testing.assert_close(
        result["terms"]["weighted/b"], torch.tensor([3.0, 4.0])
    )
    # term c is raw z = [5.0, 6.0]
    torch.testing.assert_close(result["terms"]["c"], torch.tensor([5.0, 6.0]))
    # weighted/c = z * weight_c * dt = z * 0.0 * 0.5 = [0.0, 0.0]
    torch.testing.assert_close(
        result["terms"]["weighted/c"], torch.tensor([0.0, 0.0])
    )
    # term d is raw w = [7.0, 8.0]
    torch.testing.assert_close(result["terms"]["d"], torch.tensor([7.0, 8.0]))
    # weighted/d = w * weight_d * dt = w * 1.0 * 0.5 = [3.5, 4.0]
    torch.testing.assert_close(
        result["terms"]["weighted/d"], torch.tensor([3.5, 4.0])
    )

  def test_reward_spec_requires_initialize(self):
    provider = functional_reward.FunctionalRewardProvider(
        name="test",
        terms={"a": Reward(lambda m, key: m[key], weight=1.0, key="x")},
        required_keys=["x"],
    )
    with self.assertRaises(RuntimeError):
      provider.reward_spec()

    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)
    spec = provider.reward_spec()
    self.assertEqual(spec["reward"].dtype, np.dtype("float32"))

  def test_nan_and_inf_rewards_are_sanitized(self):
    provider = functional_reward.FunctionalRewardProvider(
        name="test",
        terms={
            "r1": Reward(lambda m, key: m[key], weight=2.0, key="x"),
            "r2": Reward(lambda m, key: m[key], weight=1.0, key="y"),
            "r_zero": Reward(lambda m, key: m[key], weight=0.0, key="z"),
        },
        required_keys=["x", "y", "z"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)

    m = {
        "x": torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0]),
        "y": torch.tensor([3.0, 4.0, float("nan"), 1.0]),
        "z": torch.tensor([float("nan"), float("inf"), 1.0, float("nan")]),
    }
    result = provider.compute_reward(m)

    torch.testing.assert_close(
        result["terms"]["r1"], torch.tensor([0.0, 0.0, 0.0, 2.0])
    )
    torch.testing.assert_close(
        result["terms"]["weighted/r1"], torch.tensor([0.0, 0.0, 0.0, 2.0])
    )
    torch.testing.assert_close(
        result["terms"]["r2"], torch.tensor([3.0, 4.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        result["terms"]["weighted/r2"], torch.tensor([1.5, 2.0, 0.0, 0.5])
    )
    torch.testing.assert_close(
        result["terms"]["r_zero"], torch.tensor([0.0, 0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        result["terms"]["weighted/r_zero"], torch.tensor([0.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        result["reward"], torch.tensor([1.5, 2.0, 0.0, 2.5])
    )

  def test_staged_curriculum_replaces_weight_and_params(self):
    provider = functional_reward.FunctionalRewardProvider(
        name="test",
        terms={
            "static": Reward(lambda m: m["x"], weight=2.0),
            "ramped": (
                Reward(
                    lambda m, scale: m["y"] * scale,
                    weight=-0.5,
                    scale=1.0,
                ).curriculum(
                    weight=curriculum.Stages([(10, -1.0), (20, -2.0)]),
                    scale=curriculum.Stages([(20, 3.0)]),
                )
            ),
        },
        required_keys=["x", "y", "global_step"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)

    base = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([1.0, 1.0])}
    # Before threshold: uses default weight (-0.5) and scale (1.0).
    r0 = provider.compute_reward({**base, "global_step": torch.tensor([0, 0])})
    torch.testing.assert_close(r0["reward"], torch.tensor([0.75, 1.75]))
    torch.testing.assert_close(
        r0["terms"]["weighted/ramped"], torch.tensor([-0.25, -0.25])
    )

    # Stage 1: weight=-1.0, scale=1.0.
    r1 = provider.compute_reward(
        {**base, "global_step": torch.tensor([10, 10])}
    )
    torch.testing.assert_close(r1["reward"], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(
        r1["terms"]["weighted/ramped"], torch.tensor([-0.5, -0.5])
    )

    # Stage 2: weight=-2.0, scale=3.0.
    r2 = provider.compute_reward(
        {**base, "global_step": torch.tensor([20, 20])}
    )
    torch.testing.assert_close(r2["reward"], torch.tensor([-2.0, -1.0]))
    torch.testing.assert_close(
        r2["terms"]["weighted/ramped"], torch.tensor([-3.0, -3.0])
    )

  def test_curriculum_rejects_unknown_key(self):
    with self.assertRaisesRegex(ValueError, "Unknown curriculum parameter"):
      Reward(lambda m: m["x"], weight=1.0).curriculum(
          bogus=curriculum.Stages([(10, 2.0)])
      )

  def test_passthrough_returns_scalar_tensor(self):
    provider = functional_reward.PassthroughRewardProvider(
        name="test",
        term=Term(lambda m, key: m[key], key="x"),
        required_keys=["x"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)

    result = provider.compute_reward({"x": torch.tensor([3.0])})
    # Passthrough: returns x directly, no weighting or dt scaling.
    self.assertIsInstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([3.0]))

  def test_passthrough_batched(self):
    provider = functional_reward.PassthroughRewardProvider(
        name="test",
        term=Term(lambda m, key: m[key], key="x"),
        required_keys=["x"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)

    result = provider.compute_reward({"x": torch.tensor([1.0, 2.0, 3.0])})
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))

  def test_passthrough_reward_spec(self):
    provider = functional_reward.PassthroughRewardProvider(
        name="test",
        term=Term(lambda m, key: m[key], key="x"),
        required_keys=["x"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)
    spec = provider.reward_spec()
    self.assertEqual(spec.shape, ())
    self.assertEqual(spec.dtype, np.dtype("float32"))

  def test_passthrough_requires_initialize(self):
    provider = functional_reward.PassthroughRewardProvider(
        name="test",
        term=Term(lambda m, key: m[key], key="x"),
        required_keys=["x"],
    )
    with self.assertRaises(RuntimeError):
      provider.reward_spec()
    with self.assertRaises(RuntimeError):
      provider.compute_reward({"x": torch.tensor([1.0])})

  def test_passthrough_nan_and_inf_rewards_are_sanitized(self):
    provider = functional_reward.PassthroughRewardProvider(
        name="test",
        term=Term(lambda m, key: m[key], key="x"),
        required_keys=["x"],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.5)
    result = provider.compute_reward(
        {"x": torch.tensor([float("nan"), float("inf"), float("-inf"), 2.5])}
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 2.5]))


if __name__ == "__main__":
  absltest.main()
