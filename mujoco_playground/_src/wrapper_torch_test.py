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
"""Tests for the Torch/RSL-RL wrapper."""

from unittest import mock

from absl.testing import absltest

from mujoco_playground._src import wrapper_torch


class RSLRLBraxWrapperTest(absltest.TestCase):

  def test_get_observations_does_not_reset_initialized_env(self):
    env = object.__new__(wrapper_torch.RSLRLBraxWrapper)
    env.env_state = object()
    expected_obs = object()
    env._current_observations = mock.Mock(return_value=expected_obs)
    env.reset = mock.Mock()

    obs = env.get_observations()

    self.assertIs(obs, expected_obs)
    env.reset.assert_not_called()
    env._current_observations.assert_called_once_with()

  def test_get_observations_initializes_uninitialized_env(self):
    env = object.__new__(wrapper_torch.RSLRLBraxWrapper)
    env.env_state = None
    expected_obs = object()
    env.reset = mock.Mock(return_value=expected_obs)

    obs = env.get_observations()

    self.assertIs(obs, expected_obs)
    env.reset.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
