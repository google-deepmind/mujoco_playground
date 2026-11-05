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

"""Compute device and process rank helpers."""

import dataclasses
import os

import torch


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
  """Resolved compute device and distributed process rank info."""

  device: str
  rank: int = 0
  local_rank: int = 0


def resolve_device(
    device: str | int | None = None, default_index: int = 0
) -> str:
  """Resolves a device spec to 'cpu' or 'cuda:<index>'."""
  raw = "" if device is None else str(device).strip()
  if not raw:
    if not torch.cuda.is_available():
      return "cpu"
    idx = default_index
  elif raw == "cpu":
    return "cpu"
  elif raw == "cuda":
    idx = default_index
  elif raw.removeprefix("cuda:").isdigit():
    idx = int(raw.removeprefix("cuda:"))
  else:
    raise ValueError(f"Invalid device {device!r}.")

  if not torch.cuda.is_available():
    raise ValueError(
        f"CUDA device 'cuda:{idx}' requested, but CUDA is not available."
    )
  num_gpus = torch.cuda.device_count()
  if not 0 <= idx < num_gpus:
    raise ValueError(
        f"CUDA device 'cuda:{idx}' out of range ({num_gpus} GPUs available)."
    )
  return f"cuda:{idx}"


def get_device_info(device: str | int | None = None) -> DeviceInfo:
  """Returns resolved device and distributed rank info for this process."""
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  rank = int(os.environ.get("RANK", str(local_rank)))
  return DeviceInfo(
      device=resolve_device(device, default_index=local_rank),
      rank=rank,
      local_rank=local_rank,
  )


def configure_torch_backends() -> None:
  """Enables TF32 matmuls and cuDNN kernel autotuning."""
  torch.set_float32_matmul_precision("high")
  torch.backends.cudnn.benchmark = True
