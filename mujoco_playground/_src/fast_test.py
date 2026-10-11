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

"""Tests for the physics-free Fast environment and its Brax wrappers."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest

from mujoco_playground._src import registry, wrapper


class FastTest(absltest.TestCase):

  def test_registered_environment_has_no_physics_model(self):
    self.assertIn('Fast', registry.ALL_ENVS)
    self.assertEqual(registry.get_default_config('Fast').episode_length, 5)
    env = registry.load('Fast')
    self.assertEqual(env.action_size, 1)
    self.assertEqual(env.observation_size, 2)
    with self.assertRaises(NotImplementedError):
      _ = env.mjx_model

  def test_jit_reset_and_step_are_deterministic(self):
    env = registry.load('Fast')
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(state.obs, jp.zeros(2))
    state = step(state, jp.array([0.5]))
    np.testing.assert_array_equal(state.obs, jp.array([0.5, 1.0]))
    self.assertAlmostEqual(float(state.reward), -0.25)
    self.assertEqual(float(state.done), 0)

  def test_brax_wrapper_composition_and_episode_truncation(self):
    for full_reset in (False, True):
      with self.subTest(full_reset=full_reset):
        env = wrapper.wrap_for_brax_training(
            registry.load('Fast'), episode_length=2, action_repeat=1,
            full_reset=full_reset)
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = reset(keys)
        self.assertEqual(state.obs.shape, (2, 2))
        state = step(state, jp.ones((2, 1)))
        np.testing.assert_array_equal(state.done, jp.zeros(2))
        state = step(state, jp.ones((2, 1)))
        np.testing.assert_array_equal(state.done, jp.ones(2))
        # A full autoreset replaces the episode info with the freshly reset
        # info, including truncation=0. Cached resets retain the final flag.
        expected_truncation = jp.zeros(2) if full_reset else jp.ones(2)
        np.testing.assert_array_equal(
            state.info['truncation'], expected_truncation)
        np.testing.assert_array_equal(state.obs, jp.zeros((2, 2)))
        state = step(state, jp.zeros((2, 1)))
        np.testing.assert_array_equal(state.done, jp.zeros(2))


if __name__ == '__main__':
  absltest.main()
