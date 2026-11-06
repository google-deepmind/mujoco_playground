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

"""Tests for tensor utility functions."""

from absl.testing import absltest
from reaf.simulation.torch._src.common import tensors
import torch


class TensorsTest(absltest.TestCase):

  def test_masked_fill_with_none_mask_scalar(self):
    target = torch.zeros((4, 3))
    tensors.masked_fill(target, None, 5.0)
    self.assertTrue(torch.all(target == 5.0))

  def test_masked_fill_with_boolean_mask_scalar(self):
    target = torch.zeros((4, 3))
    env_mask = torch.tensor([True, False, True, False])
    tensors.masked_fill(target, env_mask, 2.5)
    expected = torch.tensor([
        [2.5, 2.5, 2.5],
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 2.5],
        [0.0, 0.0, 0.0],
    ])
    torch.testing.assert_close(target, expected)

  def test_masked_fill_1d_target(self):
    target = torch.zeros(4, dtype=torch.long)
    env_mask = torch.tensor([False, True, True, False])
    tensors.masked_fill(target, env_mask, 10)
    expected = torch.tensor([0, 10, 10, 0], dtype=torch.long)
    torch.testing.assert_close(target, expected)

  def test_masked_fill_tensor_val(self):
    target = torch.zeros((4, 3))
    env_mask = torch.tensor([True, False, False, True])
    val = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ])
    tensors.masked_fill(target, env_mask, val)
    expected = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0, 11.0, 12.0],
    ])
    torch.testing.assert_close(target, expected)

  def test_masked_fill_multidimensional(self):
    target = torch.zeros((4, 2, 3))
    env_mask = torch.tensor([True, False, True, False])
    tensors.masked_fill(target, env_mask, 7.0)
    self.assertTrue(torch.all(target[0] == 7.0))
    self.assertTrue(torch.all(target[1] == 0.0))
    self.assertTrue(torch.all(target[2] == 7.0))
    self.assertTrue(torch.all(target[3] == 0.0))

  def test_masked_fill_slice_view(self):
    target = torch.zeros((4, 6))
    env_mask = torch.tensor([True, False, True, False])
    tensors.masked_fill(target[:, 2:5], env_mask, 3.0)
    self.assertTrue(torch.all(target[0, 2:5] == 3.0))
    self.assertTrue(torch.all(target[0, :2] == 0.0))
    self.assertTrue(torch.all(target[1, :] == 0.0))


if __name__ == "__main__":
  absltest.main()
