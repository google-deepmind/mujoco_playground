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

"""Tests for the locomotion environments."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp

from mujoco_playground._src import locomotion


class TestSuite(parameterized.TestCase):
  """Tests for the locomotion environments."""

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in locomotion.ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = locomotion.load(env_name, config_overrides={"impl": "jax"})
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)
    obs_shape = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    self.assertEqual(obs_shape, env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())

  def test_g1_joint_deviation_hip_weights(self) -> None:
    env = locomotion.load("G1JoystickFlatTerrain")
    self.assertEqual(env._hip_indices.shape, (4,))
    error = jp.array([0.1, 0.2, 0.3, 0.4])
    qpos = env._default_pose.at[env._hip_indices].add(error)
    cost = env._cost_joint_deviation_hip(qpos, jp.array([0.5, 0.0, 0.0]))
    self.assertAlmostEqual(float(cost), 1.0, places=5)
    # Roll deviation (indices 0 and 2) is not penalized for lateral commands.
    cost = env._cost_joint_deviation_hip(qpos, jp.array([0.5, 0.5, 0.0]))
    self.assertAlmostEqual(float(cost), 0.6, places=5)


if __name__ == "__main__":
  absltest.main()
