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

"""Tests for compute device resolution helpers."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground._src import xpu
import torch


class XpuTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("default_with_cuda", None, 0, True, 4, "cuda:0"),
      ("default_with_index", None, 2, True, 4, "cuda:2"),
      ("default_without_cuda", None, 0, False, 0, "cpu"),
      ("int_index", 0, 0, True, 1, "cuda:0"),
      ("str_index", "1", 0, True, 2, "cuda:1"),
      ("bare_cuda", "cuda", 2, True, 4, "cuda:2"),
      ("explicit_cuda", "cuda:1", 0, True, 2, "cuda:1"),
      ("explicit_cpu", "cpu", 0, True, 2, "cpu"),
  )
  def test_resolve_device(
      self,
      raw_device: str | int | None,
      default_index: int,
      cuda_available: bool,
      num_gpus: int,
      expected: str,
  ) -> None:
    with (
        mock.patch.object(
            torch.cuda, "is_available", return_value=cuda_available
        ),
        mock.patch.object(torch.cuda, "device_count", return_value=num_gpus),
    ):
      self.assertEqual(
          xpu.resolve_device(raw_device, default_index=default_index),
          expected,
      )

  @parameterized.named_parameters(
      ("unknown_backend", "mps", 0, False, 0, "Invalid device"),
      ("malformed_cuda", "cuda0", 0, True, 2, "Invalid device"),
      ("negative_index", -1, 0, True, 2, "Invalid device"),
      ("no_cuda", 0, 0, False, 0, "not available"),
      ("out_of_range", "cuda:2", 0, True, 2, "out of range"),
      ("negative_default_index", None, -1, True, 2, "out of range"),
  )
  def test_resolve_device_invalid(
      self,
      raw_device: str | int | None,
      default_index: int,
      cuda_available: bool,
      num_gpus: int,
      error_regex: str,
  ) -> None:
    with (
        mock.patch.object(
            torch.cuda, "is_available", return_value=cuda_available
        ),
        mock.patch.object(torch.cuda, "device_count", return_value=num_gpus),
    ):
      with self.assertRaisesRegex(ValueError, error_regex):
        xpu.resolve_device(raw_device, default_index=default_index)

  def test_get_device_info(self) -> None:
    with (
        mock.patch.dict(
            os.environ, {"LOCAL_RANK": "2", "RANK": "5"}, clear=False
        ),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=4),
    ):
      self.assertEqual(
          xpu.get_device_info(),
          xpu.DeviceInfo(device="cuda:2", rank=5, local_rank=2),
      )
      self.assertEqual(
          xpu.get_device_info("cpu"),
          xpu.DeviceInfo(device="cpu", rank=5, local_rank=2),
      )
    env = {
        k: v for k, v in os.environ.items() if k not in ("RANK", "LOCAL_RANK")
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(torch.cuda, "is_available", return_value=False),
    ):
      self.assertEqual(
          xpu.get_device_info(),
          xpu.DeviceInfo(device="cpu", rank=0, local_rank=0),
      )
    env["LOCAL_RANK"] = "3"
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=4),
    ):
      self.assertEqual(
          xpu.get_device_info(),
          xpu.DeviceInfo(device="cuda:3", rank=3, local_rank=3),
      )

  def test_configure_torch_backends(self) -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    xpu.configure_torch_backends()
    self.assertEqual(torch.get_float32_matmul_precision(), "high")
    self.assertTrue(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
  absltest.main()
