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

"""Unit tests for DelayBuffer multi-environment latency buffering."""

from absl.testing import absltest
from reaf.simulation.torch._src.common import delay_buffer
import torch


def _delay_buf(min_lag, max_lag, nworld=2):
  return delay_buffer.DelayBuffer(
      min_lag=min_lag, max_lag=max_lag, nworld=nworld, device="cpu"
  )


class DelayBufferTest(absltest.TestCase):
  """Tests for the circular delay buffer."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)

  def test_zero_lag_passthrough(self):
    buf = _delay_buf(0, 0, nworld=3)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    buf.append(x)
    torch.testing.assert_close(buf.read(), x)

  def test_fixed_lag_reads_correct_historical_frame(self):
    """Append frames [0, 1, 2, 3, 4]; with lag=3 the read should be 3 behind."""
    buf = _delay_buf(3, 3, nworld=1)  # capacity = 4
    for i in range(5):
      buf.append(torch.tensor([[float(i)]]))
    # Most recent is 4; lag 3 should read frame 1.
    torch.testing.assert_close(buf.read(), torch.tensor([[1.0]]))

  def test_sliding_window_advances(self):
    """Verify that each successive append shifts the delayed read forward."""
    buf = _delay_buf(2, 2, nworld=1)
    actual = []
    for i in range(1, 7):
      buf.append(torch.tensor([[float(i)]]))
      actual.append(buf.read().item())
    self.assertEqual(actual, [1.0, 1.0, 1.0, 2.0, 3.0, 4.0])

  def test_ring_wraps_correctly(self):
    """Many appends >capacity should still read the correct lag."""
    buf = _delay_buf(3, 3, nworld=1)
    actual = []
    for i in range(1, 11):
      buf.append(torch.tensor([[float(i)]]))
      actual.append(buf.read().item())
    self.assertEqual(actual, [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

  def test_different_envs_get_different_lags(self):
    """With min!=max, envs should get different lags."""
    buf = _delay_buf(0, 10, nworld=200)
    self.assertGreater(buf._lags.unique().numel(), 1)

  def test_per_env_lag_reads_correct_frame(self):
    """Force lags [0, 2] and verify each env reads the correct frame."""
    buf = _delay_buf(0, 2, nworld=2)  # capacity = 3
    buf._lags = torch.tensor([0, 2])  # override stochastic sampling
    for i in range(3):
      buf.append(torch.full((2, 1), float(i)))
    # env 0: lag=0 returns 2. env 1: lag=2 returns 0.
    torch.testing.assert_close(buf.read(), torch.tensor([[2.0], [0.0]]))

  def test_first_append_fills_history(self):
    """Before enough real frames exist, delayed reads return first frame."""
    buf = _delay_buf(3, 3, nworld=1)
    buf.append(torch.tensor([[7.0]]))
    torch.testing.assert_close(buf.read(), torch.tensor([[7.0]]))
    torch.testing.assert_close(buf.read(), torch.tensor([[7.0]]))

  def test_reset_fills_history(self):
    buf = _delay_buf(1, 1, nworld=2)
    for i in range(3):
      buf.append(torch.ones(2, 4) * (i + 1))
    buf.reset()
    buf.append(torch.ones(2, 4) * 99.0)
    torch.testing.assert_close(buf.read(), torch.ones(2, 4) * 99.0)

  def test_partial_reset_only_affects_selected_envs(self):
    buf = _delay_buf(1, 1, nworld=4)
    buf.append(torch.arange(4).unsqueeze(1).float())  # [0,1,2,3]
    buf.append(torch.arange(4).unsqueeze(1).float() + 10)  # [10,11,12,13]

    buf.reset(torch.tensor([False, True, False, True]))  # reset envs 1 and 3
    buf.append(torch.ones(4, 1) * 99)  # write one more frame
    # env 0: not reset, lag=1 reads previous = 10
    # env 1: reset, lag=1 reads first frame after reset = 99
    # env 2: not reset, lag=1 reads previous = 12
    # env 3: reset, lag=1 reads first frame after reset = 99
    torch.testing.assert_close(
        buf.read(), torch.tensor([[10.0], [99.0], [12.0], [99.0]])
    )

  def test_reset_resamples_lags(self):
    """After reset, per-env lags should be resampled."""
    buf = _delay_buf(0, 10, nworld=100)
    assert buf._lags is not None
    old_lags = buf._lags.clone()
    buf.reset()
    self.assertFalse(torch.equal(buf._lags, old_lags))

  def test_nd_shape_preserved(self):
    buf = _delay_buf(1, 1, nworld=4)
    buf.append(torch.randn(4, 3, 5))
    self.assertEqual(buf.read().shape, (4, 3, 5))

  def test_read_before_append_raises(self):
    with self.assertRaises(RuntimeError):
      _delay_buf(0, 0).read()

  def test_negative_min_lag_raises(self):
    with self.assertRaises(ValueError):
      _delay_buf(-1, 2)

  def test_max_less_than_min_raises(self):
    with self.assertRaises(ValueError):
      _delay_buf(3, 1)

  def test_in_place_storage(self):
    """append() and reset() mutate preallocated tensors in-place."""
    buf = _delay_buf(1, 3, nworld=4)
    buf.append(torch.ones(4, 2))
    assert buf._lags is not None
    assert buf._steps_since_reset is not None
    buf_ptr = buf._buffer.data_ptr()
    lags_ptr = buf._lags.data_ptr()
    steps_ptr = buf._steps_since_reset.data_ptr()

    buf.append(torch.full((4, 2), 2.0))
    self.assertEqual(buf._buffer.data_ptr(), buf_ptr)

    buf.reset(torch.tensor([True, False, True, False]))
    self.assertEqual(buf._buffer.data_ptr(), buf_ptr)
    self.assertEqual(buf._lags.data_ptr(), lags_ptr)
    self.assertEqual(buf._steps_since_reset.data_ptr(), steps_ptr)

  def test_reset_zeroes_history_and_resamples_masked_lags(self):
    """reset() zeroes history and resamples lags only for masked envs."""
    buf = _delay_buf(0, 2, nworld=4)
    buf.append(torch.ones(4, 1))
    buf._lags.fill_(-1)  # Out-of-range sentinel; read() clamps it to 0.

    buf.reset(torch.tensor([True, False, True, False]))
    torch.testing.assert_close(
        buf.read(), torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    )
    self.assertTrue((buf._lags[[0, 2]] >= 0).all())
    torch.testing.assert_close(buf._lags[[1, 3]], torch.tensor([-1, -1]))

    buf.reset()
    torch.testing.assert_close(buf.read(), torch.zeros(4, 1))


if __name__ == "__main__":
  absltest.main()
