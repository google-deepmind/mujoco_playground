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

"""Tests for reward_wrapper."""

from absl.testing import absltest
import dm_env
from mujoco_playground._src.wrapper import reward_wrapper
import numpy as np


class FakeEnvWithDictReward:

  def reset_with_options(self, *, options=None):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward={
            "reward": np.array([0.0]),
            "terms": {"progress_score": np.array([0.0])},
        },
        discount=np.array([1.0]),
        observation={"obs": np.array([1.0, 2.0])},
    )

  def default_reset_options(self):
    return {}

  def step(self, action):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward={
            "reward": np.array([0.7]),
            "terms": {
                "progress_score": np.array([0.7]),
                "weighted/progress_score": np.array([0.014]),
            },
        },
        discount=np.array([0.9]),
        observation={"obs": np.array([3.0, 4.0])},
    )


class FakeEnvWithNoneReward:

  def step(self, action):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=None,
        discount=np.array([0.9]),
        observation={"obs": np.array([3.0, 4.0])},
    )


class FakeEnvWithFloatReward:

  def step(self, action):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=np.array([1.0]),
        discount=np.array([0.9]),
        observation={"obs": np.array([3.0, 4.0])},
    )


class RewardDictToFloatWrapperTest(absltest.TestCase):

  def test_extracts_reward_from_dict(self):
    env = FakeEnvWithDictReward()
    wrapped = reward_wrapper.RewardDictToFloatWrapper(env)
    ts = wrapped.step(np.array([0.0]))
    np.testing.assert_array_equal(ts.reward, np.array([0.7]))

  def test_reset_extracts_reward_from_dict(self):
    env = FakeEnvWithDictReward()
    wrapped = reward_wrapper.RewardDictToFloatWrapper(env)
    ts = wrapped.reset_with_options(options=None)
    np.testing.assert_array_equal(ts.reward, np.array([0.0]))

  def test_asserts_on_none_reward(self):
    env = FakeEnvWithNoneReward()
    wrapped = reward_wrapper.RewardDictToFloatWrapper(env)
    with self.assertRaises(AssertionError):
      wrapped.step(np.array([0.0]))

  def test_asserts_on_non_dict_reward(self):
    env = FakeEnvWithFloatReward()
    wrapped = reward_wrapper.RewardDictToFloatWrapper(env)
    with self.assertRaises(AssertionError):
      wrapped.step(np.array([0.0]))

  def test_unwrapped_returns_base(self):
    env = FakeEnvWithDictReward()
    wrapped = reward_wrapper.RewardDictToFloatWrapper(env)
    self.assertIs(wrapped.unwrapped, env)


if __name__ == "__main__":
  absltest.main()
