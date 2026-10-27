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
"""Types for reaf-sim."""

from collections.abc import Callable
import enum
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypeVar

import immutabledict
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.core import torch_wrapper
import torch

Model = torch_wrapper.TorchWrapper[mjw.Model | mujoco.MjModel]
Data = torch_wrapper.TorchWrapper[mjw.Data | mujoco.MjData]
ArrayType = TypeVar("ArrayType", bound=np.typing.ArrayLike)


class Term(Protocol):
  """Protocol for term objects with function and params."""

  func: Callable[..., Any]
  params: dict[str, Any]
  output_shape: tuple[int, ...] | None


class RewardTerm(Protocol):
  """Protocol for reward term objects with function, params, and weight."""

  func: Callable[..., Any]
  params: dict[str, Any]
  weight: float


MetricReduceMode = Literal["last", "max", "mean", "sum"]


class MetricTerm(Protocol):
  """Protocol for metric term objects with function, params, and reduce."""

  func: Callable[..., Any]
  params: dict[str, Any]
  reduce: MetricReduceMode


class Impl(enum.Enum):
  """Simulation implementation to use."""

  CPP = "cpp"
  WARP = "warp"

  @classmethod
  def _missing_(cls, value):
    if not isinstance(value, str):
      return None
    for member in cls:
      if member.value == value.lower():
        return member
    return None


# Define the mapping between NumPy dtypes and PyTorch dtypes
# This is needed because reaf specs are defined in terms of NumPy dtypes,
# but the simulation is performed using PyTorch tensors.
NUMPY_TO_TORCH_DTYPE = immutabledict.immutabledict({
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
})

TORCH_TO_NUMPY_DTYPE = immutabledict.immutabledict(
    {v: k for k, v in NUMPY_TO_TORCH_DTYPE.items()}
)
