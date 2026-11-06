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
"""Tests for Clock."""

from absl.testing import absltest
from reaf.simulation.torch._src.core import clock
import torch


class ClockTest(absltest.TestCase):

  def test_initialization(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=4,
    )
    self.assertEqual(clk.global_step, 0)
    self.assertEqual(clk.global_substep, 0)
    self.assertEqual(clk.global_time, 0.0)
    torch.testing.assert_close(
        clk.episode_step, torch.zeros(4, dtype=torch.int64)
    )
    torch.testing.assert_close(
        clk.episode_substep, torch.zeros(4, dtype=torch.int64)
    )
    torch.testing.assert_close(
        clk.episode_time, torch.zeros(4, dtype=torch.float32)
    )

  def test_tick_substep(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=2,
    )
    clk.tick_substep()
    self.assertEqual(clk.global_substep, 1)
    self.assertEqual(clk.global_step, 0)
    self.assertAlmostEqual(clk.global_time, 0.002)
    torch.testing.assert_close(
        clk.episode_substep, torch.ones(2, dtype=torch.int64)
    )
    torch.testing.assert_close(
        clk.episode_step, torch.zeros(2, dtype=torch.int64)
    )

  def test_tick(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=2,
    )
    clk.tick_step()
    self.assertEqual(clk.global_step, 1)
    self.assertEqual(clk.global_substep, 0)
    torch.testing.assert_close(
        clk.episode_step, torch.ones(2, dtype=torch.int64)
    )

  def test_full_reset(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=2,
    )
    for _ in range(5):
      clk.tick_substep()
    clk.tick_step()

    self.assertEqual(clk.global_substep, 5)
    self.assertEqual(clk.global_step, 1)

    clk.reset()

    # Episode counters are reset, global counters are preserved
    torch.testing.assert_close(
        clk.episode_step, torch.zeros(2, dtype=torch.int64)
    )
    torch.testing.assert_close(
        clk.episode_substep, torch.zeros(2, dtype=torch.int64)
    )
    self.assertEqual(clk.global_substep, 5)
    self.assertEqual(clk.global_step, 1)

  def test_masked_reset(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=3,
    )
    for _ in range(5):
      clk.tick_substep()
    clk.tick_step()

    # Reset only env 1
    mask = torch.tensor([False, True, False])
    clk.reset(mask)

    expected_steps = torch.tensor([1, 0, 1], dtype=torch.int64)
    expected_substeps = torch.tensor([5, 0, 5], dtype=torch.int64)
    torch.testing.assert_close(clk.episode_step, expected_steps)
    torch.testing.assert_close(clk.episode_substep, expected_substeps)

  def test_integer_index_reset(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=3,
    )
    for _ in range(5):
      clk.tick_substep()
    clk.tick_step()

    # Reset only env 1 using int tensor index
    indices = torch.tensor([1], dtype=torch.int32)
    clk.reset(indices)

    expected_steps = torch.tensor([1, 0, 1], dtype=torch.int64)
    expected_substeps = torch.tensor([5, 0, 5], dtype=torch.int64)
    torch.testing.assert_close(clk.episode_step, expected_steps)
    torch.testing.assert_close(clk.episode_substep, expected_substeps)

  def test_xpu_device_and_no_cpu_sync_on_reset(self):
    clk = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=2,
        xpu_device="meta",
    )
    self.assertEqual(clk.episode_step.device, torch.device("meta"))
    self.assertEqual(clk.episode_substep.device, torch.device("meta"))

    clk_cpu = clock.Clock(
        sim_dt=0.002,
        control_dt=0.02,
        n_substeps=10,
        nworld=2,
        xpu_device="cpu",
    )
    clk_cpu.tick_step()
    clk_cpu.tick_substep()

    class NoCpuSyncTensor(torch.Tensor):

      def cpu(self, *args, **kwargs):
        raise RuntimeError("Unexpected .cpu() call on env_mask")

    mask = torch.tensor([True, False]).as_subclass(NoCpuSyncTensor)
    clk_cpu.reset(mask)
    torch.testing.assert_close(
        clk_cpu.episode_step, torch.tensor([0, 1], dtype=torch.int64)
    )
    torch.testing.assert_close(
        clk_cpu.episode_substep, torch.tensor([0, 1], dtype=torch.int64)
    )


if __name__ == "__main__":
  absltest.main()
