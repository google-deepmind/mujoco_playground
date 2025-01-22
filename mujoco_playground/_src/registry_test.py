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
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import registry
from mujoco_playground._src.dm_control_suite import register_environment as register_dm_control_suite
from mujoco_playground._src.locomotion import register_environment as register_locomotion
from mujoco_playground._src.manipulation import register_environment as register_manipulation


class RegistryTest(absltest.TestCase):

  def test_new_env(self):
    class DemoEnv:

      def __init__(self, config, config_overrides):
        pass

    def demo_default_config():
      return config_dict.ConfigDict()

    register_dm_control_suite('DemoEnv', DemoEnv, demo_default_config)
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())

    register_manipulation('DemoEnv', DemoEnv, demo_default_config)
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())

    register_locomotion('DemoEnv', DemoEnv, demo_default_config)
    env = registry.load('DemoEnv')
    self.assertIsInstance(env, DemoEnv)
    config = registry.get_default_config('DemoEnv')
    self.assertEqual(config, config_dict.ConfigDict())


if __name__ == '__main__':
  absltest.main()
