# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for the registry module."""

from absl.testing import absltest
from ml_collections import config_dict

from mujoco_playground._src import dm_control_suite
from mujoco_playground._src import locomotion
from mujoco_playground._src import manipulation
from mujoco_playground._src import registry


class RegistryTest(absltest.TestCase):

  def test_new_env(self):
    class DemoEnv:

      def __init__(self, config, config_overrides):
        pass

    def demo_default_config() -> None:
      return config_dict.ConfigDict()

    dm_control_suite.register_environment(
        'DemoEnv', DemoEnv, demo_default_config
    )
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())

    manipulation.register_environment('DemoEnv', DemoEnv, demo_default_config)
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())

    locomotion.register_environment('DemoEnv', DemoEnv, demo_default_config)
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())

  def test_constants(self) -> None:
    self.assertNotEmpty(dm_control_suite.ALL_ENVS)
    self.assertNotEmpty(locomotion.ALL_ENVS)
    self.assertNotEmpty(manipulation.ALL_ENVS)
    self.assertNotEmpty(registry.ALL_ENVS)
    self.assertEqual(
        registry.ALL_ENVS,
        dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS + manipulation.ALL_ENVS,
    )


if __name__ == '__main__':
  absltest.main()
