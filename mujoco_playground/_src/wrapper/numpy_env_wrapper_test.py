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

"""Tests for NumpyEnvWrapper."""

from absl.testing import absltest
import dm_env
from dm_env import specs
from mujoco_playground._src.wrapper import numpy_env_wrapper
import numpy as np
import torch


class DummyTorchEnv:

  def reset_with_options(self, *, options=None):
    return dm_env.restart({
        "camera": torch.zeros((10, 10, 3), dtype=torch.uint8),
        "joint_pos": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "scalar_val": 42,
    })

  def default_reset_options(self):
    return {}

  def step(self, action):
    self.last_action = action
    return dm_env.transition(
        reward=torch.tensor(1.0, dtype=torch.float32),
        observation={
            "camera": torch.ones((10, 10, 3), dtype=torch.uint8),
            "joint_pos": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            "scalar_val": 100,
        },
        discount=torch.tensor(0.99, dtype=torch.float32),
    )

  def action_spec(self):
    return specs.Array(shape=(3,), dtype=np.float32)

  def timestep_spec(self):
    return dm_env.TimeStep(
        step_type=specs.Array(shape=(), dtype=np.int32),
        reward=specs.Array(shape=(), dtype=np.float32),
        discount=specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
        ),
        observation={
            "camera": specs.Array(shape=(10, 10, 3), dtype=np.uint8),
            "joint_pos": specs.Array(shape=(3,), dtype=np.float32),
        },
    )


class NumpyEnvWrapperTest(absltest.TestCase):

  def test_reset_and_step_return_numpy_arrays(self):
    base_env = DummyTorchEnv()
    wrapped_env = numpy_env_wrapper.NumpyEnvWrapper(base_env)

    timestep = wrapped_env.reset_with_options(options={})
    self.assertIsInstance(timestep.observation["camera"], np.ndarray)
    self.assertIsInstance(timestep.observation["joint_pos"], np.ndarray)
    self.assertEqual(timestep.observation["scalar_val"], 42)
    np.testing.assert_array_equal(
        timestep.observation["joint_pos"], np.array([1.0, 2.0, 3.0])
    )

    step_timestep = wrapped_env.step(np.zeros(3))
    self.assertIsInstance(step_timestep.observation["camera"], np.ndarray)
    self.assertIsInstance(step_timestep.observation["joint_pos"], np.ndarray)
    self.assertIsInstance(step_timestep.reward, (float, np.ndarray, np.number))
    self.assertIsInstance(
        step_timestep.discount, (float, np.ndarray, np.number)
    )
    np.testing.assert_array_equal(
        step_timestep.observation["joint_pos"], np.array([4.0, 5.0, 6.0])
    )

  def test_step_converts_action_to_torch(self):
    base_env = DummyTorchEnv()
    wrapped_env = numpy_env_wrapper.NumpyEnvWrapper(base_env)

    wrapped_env.step(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    self.assertIsInstance(base_env.last_action, torch.Tensor)
    torch.testing.assert_close(
        base_env.last_action,
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )

  def test_specs_are_forwarded(self):
    base_env = DummyTorchEnv()
    wrapped_env = numpy_env_wrapper.NumpyEnvWrapper(base_env)
    self.assertEqual(wrapped_env.action_spec(), base_env.action_spec())
    self.assertEqual(wrapped_env.timestep_spec(), base_env.timestep_spec())


if __name__ == "__main__":
  absltest.main()
