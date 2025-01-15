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
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import dm_control_suite
from mujoco_playground._src import wrapper


class WrapperTest(absltest.TestCase):

  def test_auto_reset_wrapper(self):
    class DoneEnv:

      def __init__(self, env):
        self._env = env

      def reset(self, key):
        return self._env.reset(key)

      def step(self, state, action):
        state = self._env.step(state, jp.ones_like(action))
        state = state.replace(done=action[0] > 0)
        return state

    env = wrapper.BraxAutoResetWrapper(
        DoneEnv(dm_control_suite.load('CartpoleBalance'))
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0))
    first_qpos = state.info['first_state'].qpos

    state = jit_step(state, -jp.ones(env._env.action_size))
    self.assertGreater(np.linalg.norm(state.data.qpos - first_qpos), 1e-3)
    state = jit_step(state, jp.ones(env._env.action_size))
    np.testing.assert_allclose(state.data.qpos, first_qpos, atol=1e-6)

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
