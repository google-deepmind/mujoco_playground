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

from collections.abc import Mapping
from typing import overload

import torch


@overload
def masked_where(
    env_mask: torch.Tensor | None,
    val: torch.Tensor | float | int,
    other: torch.Tensor,
) -> torch.Tensor:
  ...


@overload
def masked_where(
    env_mask: torch.Tensor | None,
    val: Mapping[str, torch.Tensor],
    other: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  ...


def masked_where(
    env_mask: torch.Tensor | None,
    val: torch.Tensor | float | int | Mapping[str, torch.Tensor],
    other: torch.Tensor | Mapping[str, torch.Tensor],
) -> torch.Tensor | dict[str, torch.Tensor]:
  """Out-of-place selects `val` for environments in `env_mask`, `other` else.

  Supports both single tensors and `Mapping[str, torch.Tensor]` (skipping
  unbatched entries whose leading dimension does not match `env_mask`, and
  returning `dict(val)` when `other` is empty). If `env_mask` is None, returns
  `val` (or `torch.full_like(other, val)` for scalars). Broadcasts `env_mask`
  across trailing dimensions of `other` and moves it to `other.device`.

  Args:
    env_mask: Optional boolean tensor of shape `(nworld,)` indicating which
      environments select `val`. If None, selects `val` for all environments.
    val: Value (scalar, tensor, or mapping of tensors) selected where `env_mask`
      is True.
    other: Tensor or mapping of tensors selected where `env_mask` is False.
      Leading dimension of batched tensors is assumed to be `nworld`.

  Returns:
    A new tensor or dict of tensors with `val` on `env_mask` rows and `other`
    elsewhere.
  """
  if isinstance(val, Mapping):
    assert isinstance(other, Mapping)
    if env_mask is None or not other:
      return dict(val)
    nworld = env_mask.shape[0]
    # Only mask per-world tensors; pass unbatched entries through unchanged.
    return {
        k: (
            masked_where(env_mask, v, other[k])
            if v.ndim > 0 and v.shape[0] == nworld
            else v
        )
        for k, v in val.items()
    }
  assert not isinstance(other, Mapping)
  if env_mask is None:
    return val if isinstance(val, torch.Tensor) else torch.full_like(other, val)
  mask = env_mask.to(device=other.device).view(-1, *([1] * (other.ndim - 1)))
  return torch.where(mask, val, other)


def masked_fill(
    target: torch.Tensor,
    env_mask: torch.Tensor | None,
    val: torch.Tensor | float | int,
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
    target[:] = masked_where(env_mask, val, target)
