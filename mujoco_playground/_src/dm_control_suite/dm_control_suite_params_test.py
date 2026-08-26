# Copyright 2026 DeepMind Technologies Limited
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
"""Tests for DM Control Suite training configs."""

from absl.testing import absltest

from mujoco_playground.config import dm_control_suite_params


class DmControlSuiteParamsTest(absltest.TestCase):

  def test_rsl_rl_config_reuses_brax_ppo_hyperparameters(self):
    for env_name in ("CartpoleBalance", "BallInCup", "FingerSpin"):
      with self.subTest(env_name=env_name):
        brax_config = dm_control_suite_params.brax_ppo_config(env_name)
        rsl_config = dm_control_suite_params.rsl_rl_config(env_name)

        self.assertEqual(
            rsl_config.algorithm.learning_rate, brax_config.learning_rate
        )
        self.assertEqual(rsl_config.algorithm.gamma, brax_config.discounting)
        self.assertEqual(
            rsl_config.algorithm.entropy_coef, brax_config.entropy_cost
        )
        self.assertEqual(rsl_config.num_steps_per_env, brax_config.unroll_length)

  def test_rsl_rl_config_has_required_runner_fields(self):
    config = dm_control_suite_params.rsl_rl_config("CartpoleBalance")

    self.assertEqual(config.runner_class_name, "OnPolicyRunner")
    self.assertEqual(config.policy.class_name, "ActorCritic")
    self.assertEqual(config.algorithm.class_name, "PPO")
    self.assertGreater(config.max_iterations, 0)

  def test_rsl_rl_config_rejects_unknown_environment(self):
    with self.assertRaisesRegex(ValueError, "not found in default configs"):
      dm_control_suite_params.rsl_rl_config("NotAnEnvironment")


if __name__ == "__main__":
  absltest.main()
