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
