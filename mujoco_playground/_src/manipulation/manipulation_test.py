# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
from mujoco_playground._src import manipulation

_ALL_ENVS = manipulation.ALL


class TestSuite(parameterized.TestCase):

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in _ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = manipulation.load(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    self.assertFalse(jp.isnan(state.data.qpos).any())


if __name__ == "__main__":
  absltest.main()
