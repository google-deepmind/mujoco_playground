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

"""Circular buffer for stochastic per-environment delay."""

import torch


class DelayBuffer:
  """A batched circular buffer that returns stochastically delayed values.

  Maintains a rolling history of `(nworld, *shape)` tensors on the GPU and
  serves values from `T - lag` timesteps ago, where `lag` is sampled
  independently per environment from `[min_lag, max_lag]`.

  The buffer is lazily initialized on the first `append()` call so it adapts
  to arbitrary tensor shapes without upfront configuration.

  Example usage::

    buf = DelayBuffer(min_lag=1, max_lag=3, nworld=64, device="cuda:0")
    buf.append(obs)          # obs shape: (64, 6)
    delayed = buf.read()     # delayed shape: (64, 6) — each env gets a
                             # different historical frame

  Args:
    min_lag: Minimum delay in substeps (inclusive). 0 means no delay.
    max_lag: Maximum delay in substeps (inclusive). Must be >= min_lag.
    nworld: Number of parallel environments.
    device: Torch device for all tensor allocations.
  """

  def __init__(
      self,
      min_lag: int,
      max_lag: int,
      nworld: int,
      device: torch.device | str,
  ):
    if min_lag < 0:
      raise ValueError(f"min_lag must be >= 0, got {min_lag}")
    if max_lag < min_lag:
      raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")

    self._min_lag = min_lag
    self._max_lag = max_lag
    self._nworld = nworld
    self._xpu_device = torch.device(device)

    self._capacity = max_lag + 1
    self._buffer: torch.Tensor | None = None  # (capacity, nworld, *shape)
    self._write_idx = 0

    # Per-environment lag, sampled on reset.
    self._lags = torch.randint(
        min_lag,
        max_lag + 1,
        (nworld,),
        dtype=torch.long,
        device=self._xpu_device,
    )
    self._batch_arange = torch.arange(nworld, device=self._xpu_device)
    self._steps_since_reset = torch.zeros(
        nworld, dtype=torch.long, device=self._xpu_device
    )

  @property
  def min_lag(self) -> int:
    return self._min_lag

  @property
  def max_lag(self) -> int:
    return self._max_lag

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Clears history and resamples lags for the given environments.

    Args:
      env_mask: Boolean environment mask to reset. If None, resets all.
    """
    new_lags = torch.randint(
        self._min_lag,
        self._max_lag + 1,
        (self._nworld,),
        dtype=torch.long,
        device=self._xpu_device,
    )
    if env_mask is None:
      self._write_idx = 0  # reset write index when resetting all environments.
      if self._buffer is not None:
        self._buffer.zero_()
      self._steps_since_reset.zero_()
      self._lags.copy_(new_lags)
      return

    if self._buffer is not None:
      mask = env_mask.view(1, -1, *([1] * (self._buffer.ndim - 2)))
      self._buffer.masked_fill_(mask, 0.0)
    self._steps_since_reset.masked_fill_(env_mask, 0)
    torch.where(env_mask, new_lags, self._lags, out=self._lags)

  def append(self, x: torch.Tensor) -> None:
    """Pushes a new frame into the buffer.

    Args:
      x: Tensor of shape `(nworld, *shape)` containing the new frame for all
        environments.
    """
    if self._buffer is None:
      data_shape = x.shape[1:]
      self._buffer = torch.zeros(
          (self._capacity, self._nworld, *data_shape),
          dtype=x.dtype,
          device=self._xpu_device,
      )
      self._steps_since_reset[:] = 0  # Force init on first append

    # Fill all history slots with the first frame for the first step after
    # reset.
    is_first_step = self._steps_since_reset == 0
    # Reshape is_first_step to (1, nworld, 1, ..., 1) to broadcast with the
    # buffer shape: (capacity, nworld, *data_shape)
    mask_shape = (1, -1) + (1,) * len(x.shape[1:])
    mask = is_first_step.view(mask_shape)  # pyrefly: ignore[missing-attribute]
    torch.where(mask, x.unsqueeze(0), self._buffer, out=self._buffer)

    self._buffer[self._write_idx] = x
    self._write_idx = (self._write_idx + 1) % self._capacity
    self._steps_since_reset += 1

  def read(self) -> torch.Tensor:
    """Returns the delayed frame for each environment.

    Returns:
      Tensor of shape `(nworld, *shape)`.

    Raises:
      RuntimeError: If no frames have been appended yet.
    """
    if self._buffer is None:
      raise RuntimeError("Buffer is empty. Call append() first.")

    # Clamp lags in case they are out of bounds.
    valid_lags = self._lags.clamp(max=self._capacity - 1).clamp(min=0)

    # Compute read positions: the most recent write is at write_idx - 1.
    read_positions = (self._write_idx - 1 - valid_lags) % self._capacity

    return self._buffer[read_positions, self._batch_arange]
