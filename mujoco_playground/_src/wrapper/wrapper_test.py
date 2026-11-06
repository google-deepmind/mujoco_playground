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

"""Tests for base Wrapper class."""

from absl.testing import absltest
import dm_env
from dm_env import specs
from mujoco_playground._src.wrapper import wrapper
import numpy as np


class FakeEnv:

  def __init__(self):
    self.custom_attr = "custom_value"
    self.closed = False

  def reset_with_options(self, *, options=None):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=1.0,
        observation={"obs": np.array([1.0, 2.0])},
    )

  def default_reset_options(self):
    return {"opt": 123}

  def step(self, action):
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=1.0,
        discount=0.9,
        observation={"obs": np.array([3.0, 4.0])},
    )

  def action_spec(self):
    return specs.Array(shape=(2,), dtype=np.float32, name="action")

  def timestep_spec(self):
    return dm_env.TimeStep(
        step_type=None,
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.Array(shape=(), dtype=np.float32),
        observation={"obs": specs.Array(shape=(2,), dtype=np.float32)},
    )

  def close(self):
    self.closed = True


class WrapperTest(absltest.TestCase):

  def test_base_wrapper_unwrapped(self):
    base_env = FakeEnv()
    wrapped_env = wrapper.Wrapper(base_env)
    self.assertIs(wrapped_env.unwrapped, base_env)

    double_wrapped = wrapper.Wrapper(wrapped_env)
    self.assertIs(double_wrapped.unwrapped, base_env)

  def test_base_wrapper_delegation(self):
    base_env = FakeEnv()
    wrapped_env = wrapper.Wrapper(base_env)

    self.assertEqual(wrapped_env.custom_attr, "custom_value")
    self.assertEqual(wrapped_env.default_reset_options(), {"opt": 123})
    self.assertEqual(wrapped_env.action_spec().name, "action")

    wrapped_env.close()
    self.assertTrue(base_env.closed)


if __name__ == "__main__":
  absltest.main()
