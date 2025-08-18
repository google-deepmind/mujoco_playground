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
"""Tests for the wrapper module."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from brax.envs.wrappers import training as brax_training
import jax
import jax.numpy as jp
from mujoco_playground._src import dm_control_suite
from mujoco_playground._src import wrapper
import numpy as np


class WrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('full_reset', True),
      ('cache_reset', False),
  )
  def test_auto_reset_wrapper(self, full_reset):
    """Tests the AutoResetWrapper."""
    class DoneEnv:

      def __init__(self, env):
        self._env = env

      def reset(self, key):
        state = self._env.reset(key)
        state.info['AutoResetWrapper_preserve_info'] = 1
        state.info['other_info'] = 1
        return state

      def step(self, state, action):
        state = self._env.step(state, jp.ones_like(action))
        state = state.replace(done=action[0] > 0)
        state.info['AutoResetWrapper_preserve_info'] = 2
        state.info['other_info'] = 2
        return state

    env = wrapper.BraxAutoResetWrapper(
        brax_training.VmapWrapper(
            DoneEnv(dm_control_suite.load('CartpoleBalance'))
        ),
        full_reset=full_reset,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0)[None])
    first_qpos = state.data.qpos

    # First step should not be done.
    state = jit_step(state, -jp.ones(env._env.action_size)[None])
    np.testing.assert_allclose(state.info['AutoResetWrapper_done_count'], 0)
    self.assertGreater(np.linalg.norm(state.data.qpos - first_qpos), 1e-3)
    self.assertEqual(state.info['AutoResetWrapper_preserve_info'], 2)
    self.assertEqual(state.info['other_info'], 2)

    for i in range(1, 3):
      state = jit_step(state, jp.ones(env._env.action_size)[None])
      jax.tree.map(lambda x: x.block_until_ready(), state)
      if full_reset:
        self.assertTrue((state.data.qpos != first_qpos).all())
      else:
        np.testing.assert_allclose(state.data.qpos, first_qpos, atol=1e-6)
      np.testing.assert_allclose(state.info['AutoResetWrapper_done_count'], i)
      self.assertEqual(state.info['AutoResetWrapper_preserve_info'], 2)
      expected_other_info = 1 if full_reset else 2
      self.assertEqual(state.info['other_info'], expected_other_info)

  @parameterized.named_parameters(
      ('full_reset', True),
      ('cache_reset', False),
  )
  def test_evalwrapper_with_reset(self, full_reset):
    """Tests EvalWrapper with reset in the AutoResetWrapper."""
    episode_length = 10
    num_envs = 4

    env = dm_control_suite.load('CartpoleBalance')
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        full_reset=full_reset,
    )
    env = brax_training.EvalWrapper(env)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, num_envs)
    state = jit_reset(rng)
    first_obs = state.obs
    action = jp.zeros((num_envs, env.action_size))

    for _ in range(episode_length):
      state = jit_step(state, action)

    # All episodes should finish at episode_length.
    avg_episode_length = state.info['eval_metrics'].episode_steps.mean()
    np.testing.assert_allclose(avg_episode_length, episode_length, atol=1e-6)
    active_episodes = state.info['eval_metrics'].active_episodes
    self.assertTrue(np.all(active_episodes == 0))

    np.testing.assert_array_equal(state.info['steps'], 10 * np.ones(num_envs))
    if full_reset:
      self.assertTrue((state.obs != first_obs).all())
    else:
      np.testing.assert_allclose(state.obs, first_obs, rtol=1e-6)

  def test_domain_randomization_wrapper(self):
    def randomization_fn(model, rng):
      @jax.vmap
      def get_gravity(rng):
        dg = jax.random.uniform(rng, shape=(3,), minval=-10.0, maxval=10.0)
        return model.opt.gravity + dg

      model_v = model.tree_replace({'opt.gravity': get_gravity(rng)})
      in_axes = jax.tree.map(lambda x: None, model)
      in_axes = in_axes.tree_replace({'opt.gravity': 0})
      return model_v, in_axes

    env = dm_control_suite.load('CartpoleBalance')
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 256)
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=200,
        randomization_fn=functools.partial(randomization_fn, rng=rng),
    )

    # set the same key across the batch for env.reset so that only the
    # randomization wrapper creates variability in the env.step
    key = jp.zeros((256, 2), dtype=jp.uint32)
    state = jax.jit(env.reset)(key)
    self.assertEqual(state.data.qpos[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.data.qpos[:, 0]).shape[0], 1)

    # test that the DomainRandomizationWrapper creates variability in env.step
    state = jax.jit(env.step)(state, jp.zeros((256, env.action_size)))
    self.assertEqual(state.data.qpos[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.data.qpos[:, 0]).shape[0], 256)


if __name__ == '__main__':
  absltest.main()
