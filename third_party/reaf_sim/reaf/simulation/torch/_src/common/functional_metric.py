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

"""Functional metric provider with per-episode reductions."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import get_args

from reaf.simulation.torch._src.core import metric_provider
from reaf.simulation.torch._src.core.types import MetricReduceMode
from reaf.simulation.torch._src.core.types import MetricTerm
import torch


class FunctionalMetricProvider(metric_provider.MetricProvider):
  """Computes per-step metrics and accumulates episodic reductions."""

  def __init__(
      self,
      terms: Mapping[str, MetricTerm],
      *,
      nworld: int = 1,
      xpu_device: str | torch.device = "cpu",
      dtype: torch.dtype = torch.float32,
  ):
    by_mode: dict[
        MetricReduceMode,
        list[tuple[str, Callable[..., torch.Tensor], dict[str, object]]],
    ] = {"mean": [], "sum": [], "max": [], "last": []}
    for k, term in terms.items():
      params = dict(term.params) if term.params else {}
      reduce_mode = term.reduce
      if reduce_mode not in by_mode:
        raise ValueError(
            f"Unsupported metric reduce mode {reduce_mode!r} for metric {k!r}."
            f" Must be one of {get_args(MetricReduceMode)}."
        )
      by_mode[reduce_mode].append((k, term.func, params))

    self._key_order: tuple[str, ...] = tuple(terms.keys())
    self._mean_names: tuple[str, ...] = tuple(k for k, _, _ in by_mode["mean"])
    self._sum_names: tuple[str, ...] = tuple(k for k, _, _ in by_mode["sum"])
    self._max_names: tuple[str, ...] = tuple(k for k, _, _ in by_mode["max"])
    self._last_names: tuple[str, ...] = tuple(k for k, _, _ in by_mode["last"])
    self._ordered_items: tuple[
        tuple[str, Callable[..., torch.Tensor], dict[str, object]], ...
    ] = tuple(
        by_mode["mean"] + by_mode["sum"] + by_mode["max"] + by_mode["last"]
    )

    self._n_mean: int = len(self._mean_names)
    self._n_sum: int = len(self._sum_names)
    self._n_max: int = len(self._max_names)
    self._n_last: int = len(self._last_names)

    self._nworld: int = nworld
    self._device: torch.device = torch.device(xpu_device)
    self._dtype: torch.dtype = dtype
    self._step_counts: torch.Tensor = torch.zeros(
        nworld, device=self._device, dtype=torch.long
    )
    self._mean_sums: torch.Tensor | None = None
    if self._n_mean:
      self._mean_sums = torch.zeros(
          (self._n_mean, nworld), device=self._device, dtype=dtype
      )
    self._sums: torch.Tensor | None = None
    if self._n_sum:
      self._sums = torch.zeros(
          (self._n_sum, nworld), device=self._device, dtype=dtype
      )
    self._maxs: torch.Tensor | None = None
    if self._n_max:
      self._maxs = torch.full(
          (self._n_max, nworld), -torch.inf, device=self._device, dtype=dtype
      )
    self._lasts: torch.Tensor | None = None
    if self._n_last:
      self._lasts = torch.zeros(
          (self._n_last, nworld), device=self._device, dtype=dtype
      )

  def _eval_term(
      self,
      features: Mapping[str, torch.Tensor],
      name: str,
      func: Callable[..., torch.Tensor],
      params: dict[str, object],
  ) -> torch.Tensor:
    val = func(features, **params).to(device=self._device, dtype=self._dtype)
    if val.shape != (self._nworld,):
      raise ValueError(
          f"Metric {name!r} returned shape {tuple(val.shape)}, expected"
          f" ({self._nworld},)."
      )
    return val

  def update_metrics(
      self,
      features: Mapping[str, torch.Tensor],
  ) -> None:
    """Evaluates step metrics and updates internal episode accumulators.

    Modifies internal state (`_step_counts`, `_mean_sums`, `_sums`, `_maxs`,
    `_lasts`); should only be called once per control step by
    `TaskLayer.update_metrics` (via `Environment.step`).
    """
    if not self._ordered_items:
      return
    vals = torch.stack(
        [
            self._eval_term(features, name, func, params)
            for name, func, params in self._ordered_items
        ],
        dim=0,
    )
    self._step_counts.add_(1)
    vals.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    idx = 0
    if self._mean_sums is not None:
      self._mean_sums.add_(vals[idx : idx + self._n_mean])
      idx += self._n_mean
    if self._sums is not None:
      self._sums.add_(vals[idx : idx + self._n_sum])
      idx += self._n_sum
    if self._maxs is not None:
      torch.maximum(self._maxs, vals[idx : idx + self._n_max], out=self._maxs)
      idx += self._n_max
    if self._lasts is not None:
      self._lasts.copy_(vals[idx:])

  def get_episode_metrics(
      self,
      env_mask: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Returns 0-D mean reduced episode metrics over the masked worlds.

    Read-only; does not modify internal state.
    """
    if not self._ordered_items:
      return {}
    mask = env_mask.to(device=self._device, dtype=torch.bool)
    if mask.shape != (self._nworld,):
      raise ValueError(
          f"Expected env_mask of shape ({self._nworld},), got"
          f" {tuple(mask.shape)}."
      )
    n_env = mask.sum().clamp(min=1).to(self._dtype)
    by_name: dict[str, torch.Tensor] = {}
    if self._mean_sums is not None:
      safe_counts = self._step_counts.clamp(min=1).to(self._dtype)
      per_world = self._mean_sums / safe_counts
      means = torch.where(mask, per_world, 0.0).sum(dim=-1) / n_env
      by_name.update(zip(self._mean_names, means.unbind(0), strict=True))
    if self._sums is not None:
      sums = torch.where(mask, self._sums, 0.0).sum(dim=-1) / n_env
      by_name.update(zip(self._sum_names, sums.unbind(0), strict=True))
    if self._maxs is not None:
      has_steps = mask & (self._step_counts > 0)
      maxs = torch.where(has_steps, self._maxs, 0.0).sum(dim=-1) / n_env
      by_name.update(zip(self._max_names, maxs.unbind(0), strict=True))
    if self._lasts is not None:
      lasts = torch.where(mask, self._lasts, 0.0).sum(dim=-1) / n_env
      by_name.update(zip(self._last_names, lasts.unbind(0), strict=True))
    return {k: by_name[k] for k in self._key_order}

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    if not self._ordered_items:
      return
    if env_mask is None:
      self._step_counts.zero_()
      if self._mean_sums is not None:
        self._mean_sums.zero_()
      if self._sums is not None:
        self._sums.zero_()
      if self._maxs is not None:
        self._maxs.fill_(-torch.inf)
      if self._lasts is not None:
        self._lasts.zero_()
      return

    mask = env_mask.to(device=self._device, dtype=torch.bool)
    if mask.shape != (self._nworld,):
      raise ValueError(
          f"Expected env_mask of shape ({self._nworld},), got"
          f" {tuple(mask.shape)}."
      )
    self._step_counts.masked_fill_(mask, 0)
    if self._mean_sums is not None:
      self._mean_sums.masked_fill_(mask, 0.0)
    if self._sums is not None:
      self._sums.masked_fill_(mask, 0.0)
    if self._maxs is not None:
      self._maxs.masked_fill_(mask, -torch.inf)
    if self._lasts is not None:
      self._lasts.masked_fill_(mask, 0.0)
