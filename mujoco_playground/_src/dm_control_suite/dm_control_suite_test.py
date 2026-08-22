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

"""Tests for the DM Control Suite."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp

from mujoco_playground._src import dm_control_suite
from mujoco_playground._src import mjx_env


class TestSuite(parameterized.TestCase):
  """Tests for the DM Control Suite."""

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in dm_control_suite.ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = dm_control_suite.load(env_name, config_overrides={"impl": "jax"})
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    self.assertEqual(state.obs.shape[0], env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())

  def test_warn_overflow_filtered_warp(self) -> None:
    import mujoco_warp as mjw  # pylint: disable=g-import-not-at-top

    env = dm_control_suite.load(
        "CartpoleBalance", config_overrides={"impl": "warp"}
    )
    warn_overflow = int(env.mjx_model.opt._impl.warn_overflow)
    self.assertEqual(warn_overflow & int(mjw.OverflowType.ITERATIONS), 0)
    self.assertEqual(warn_overflow & int(mjw.OverflowType.LS_ITERATIONS), 0)

  def test_put_model_warn_overflow_filtered_warp(self) -> None:
    import mujoco  # pylint: disable=g-import-not-at-top
    import mujoco_warp as mjw  # pylint: disable=g-import-not-at-top
    from mujoco_playground._src import mjx_env  # pylint: disable=g-import-not-at-top

    mj_model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    mjx_model = mjx_env.put_model(mj_model, impl="warp")
    warn_overflow = int(mjx_model.opt._impl.warn_overflow)
    self.assertEqual(warn_overflow & int(mjw.OverflowType.ITERATIONS), 0)
    self.assertEqual(warn_overflow & int(mjw.OverflowType.LS_ITERATIONS), 0)

  @parameterized.parameters("qpos", "qvel")
  def test_humanoid_nan_state_zeroes_reward(self, invalid_field: str) -> None:
    env = dm_control_suite.load(
        "HumanoidStand", config_overrides={"impl": "jax"}
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    invalid_data = state.data.replace(
        **{invalid_field: getattr(state.data, invalid_field).at[0].set(jp.nan)},
        xpos=state.data.xpos.at[env.mj_model.body("head").id, -1].set(jp.nan),
    )

    with mock.patch.object(mjx_env, "step", return_value=invalid_data):
      state = env.step(state, jp.zeros(env.action_size))

    self.assertEqual(float(state.done), 1.0)
    self.assertEqual(float(state.reward), 0.0)


if __name__ == "__main__":
  absltest.main()
