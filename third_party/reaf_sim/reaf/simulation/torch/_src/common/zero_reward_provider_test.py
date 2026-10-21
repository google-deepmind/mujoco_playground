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

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.common import zero_reward_provider
import torch


class ZeroRewardProviderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.nworld = 4
    self.xpu_device = torch.device('cpu')
    self.provider = zero_reward_provider.ZeroRewardProvider(
        nworld=self.nworld, xpu_device=self.xpu_device
    )

  def test_name(self):
    self.assertEqual(self.provider.name(), 'zero_reward_provider')

  def test_required_features_keys(self):
    self.assertEmpty(self.provider.required_features_keys())

  def test_reward_spec(self):
    spec = self.provider.reward_spec()
    self.assertIsInstance(spec, specs.Array)
    self.assertEqual(spec.shape, (1,))
    self.assertEqual(spec.dtype, np.float32)

  def test_compute_reward(self):
    # It doesn't use required_features, so passing empty dict is fine.
    reward = self.provider.compute_reward({})
    expected_reward = torch.zeros(
        self.nworld, 1, dtype=torch.float32, device=self.xpu_device
    )
    torch.testing.assert_close(reward, expected_reward)

  def test_reset(self):
    # Check reset (should accept arguments but do nothing visible for this
    # provider)
    self.provider.reset(torch.tensor([0, 1]))


if __name__ == '__main__':
  absltest.main()
