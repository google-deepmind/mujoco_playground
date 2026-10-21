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

"""Tensor utility functions for simulation."""

from typing import Union

import torch


def masked_fill(
    target: torch.Tensor,
    env_mask: torch.Tensor | None,
    val: Union[torch.Tensor, float, int],
) -> None:
  """In-place updates target with val for environments in env_mask.

  If env_mask is None, target is completely filled with val.

  Args:
    target: Target tensor to update in-place. The leading dimension is assumed
      to be the environment/batch dimension (nworld).
    env_mask: Optional boolean tensor of shape (nworld,) indicating which
      environments to update. If None, all environments are updated.
    val: Value to fill into target for masked environments. Can be a scalar
      (e.g., float, int) or a tensor that broadcasts with the masked slice of
      target.
  """
  if env_mask is None:
    target[:] = val
  else:
    mask = env_mask.to(device=target.device).view(
        -1, *([1] * (target.ndim - 1))
    )
    target[:] = torch.where(mask, val, target)
