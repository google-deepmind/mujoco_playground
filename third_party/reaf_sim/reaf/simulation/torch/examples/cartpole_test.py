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
"""Tests for the cartpole example."""

from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reaf.simulation.torch.examples import cartpole
import torch


class CartpoleTest(parameterized.TestCase):

  def test_tile_grid(self):
    dummy_images = torch.zeros((4, 60, 80, 3), dtype=torch.uint8)
    grid = cartpole.tile_grid(dummy_images, cols=2)
    self.assertEqual(grid.shape, (120, 160, 3))
    self.assertIsInstance(grid, np.ndarray)

  @parameterized.named_parameters(
      ("cpu_cpp", "cpu", "cpp", 1),
      ("gpu_warp", "cuda:0", "warp", 4),
  )
  def test_cartpole_env_rollout(
      self,
      xpu_device: str,
      impl: Literal["cpp", "warp"],
      nworld: int,
  ):
    if "cuda" in xpu_device and not torch.cuda.is_available():
      self.skipTest("CUDA is not available.")

    try:
      env = cartpole.create_cartpole_env(
          nworld=nworld,
          xpu_device=xpu_device,
          impl=impl,
      )
      timestep = env.reset()
    except (torch.AcceleratorError, RuntimeError) as e:
      if "CUDA" in str(e) or "kernel image" in str(e):
        self.skipTest(f"CUDA accelerator not supported on runner: {e}")
      raise

    self.assertEqual(timestep.reward.shape[0], nworld)
    self.assertEqual(timestep.step_type.shape, (nworld,))
    self.assertIn("slider_actuator/joint_pos", timestep.observation)
    self.assertIn("camera/fixed_rgb", timestep.observation)
    self.assertEqual(
        timestep.observation["camera/fixed_rgb"].shape,
        (nworld, 120, 160, 3),
    )

    # Step the environment with a control command
    commands = torch.ones((nworld, 1), device=torch.device(xpu_device)) * 0.5
    next_timestep = env.step({"slider_actuator/ctrl": commands})
    self.assertEqual(next_timestep.reward.shape[0], nworld)
    self.assertEqual(
        next_timestep.observation["camera/fixed_rgb"].shape,
        (nworld, 120, 160, 3),
    )


if __name__ == "__main__":
  absltest.main()
