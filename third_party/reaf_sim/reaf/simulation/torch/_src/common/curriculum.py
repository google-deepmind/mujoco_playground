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

"""Declarative stateless curriculum schedules."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses

from reaf.simulation.torch._src.core import clock
import torch


@dataclasses.dataclass(frozen=True)
class Stages:
  """Piecewise-constant schedule keyed on a feature (default: `global_step`).

  Each `(threshold, value)` pair replaces the value once
  `features[key] >= threshold`. Later stages take precedence.
  """

  stages: Sequence[tuple[int | float, float]]
  key: str = clock.GLOBAL_STEP_KEY

  def __post_init__(self) -> None:
    thresholds = [t for t, _ in self.stages]
    if thresholds != sorted(thresholds):
      raise ValueError("Curriculum stages must be in nondecreasing order.")


def eval_stages(
    schedule: Stages,
    features: Mapping[str, torch.Tensor],
    default: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Evaluates a `Stages` schedule against `features[schedule.key]`."""
  trigger = features[schedule.key]
  if trigger.ndim == 2 and trigger.shape[-1] == 1:
    trigger = trigger.squeeze(-1)
  elif trigger.ndim > 1:
    raise ValueError(
        f"Curriculum trigger {schedule.key!r} must have shape (), (nworld,), "
        f"or (nworld, 1), got {tuple(trigger.shape)}."
    )
  # Collapse to 0-D so it broadcasts cleanly; global_step is identical across
  # worlds.
  if schedule.key == clock.GLOBAL_STEP_KEY and trigger.ndim == 1:
    trigger = trigger[0]
  value = torch.full_like(trigger, default, dtype=dtype)
  for threshold, target in schedule.stages:
    value = torch.where(trigger >= threshold, target, value)
  return value


_SCHEDULE_EVALUATORS: dict[type[Stages], Callable[..., torch.Tensor]] = {
    Stages: eval_stages,
}


def evaluate(
    schedule: Stages,
    features: Mapping[str, torch.Tensor],
    default: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Evaluates a declarative curriculum schedule on device."""
  evaluator = _SCHEDULE_EVALUATORS.get(type(schedule))
  if evaluator is None:
    raise TypeError(
        f"Unsupported curriculum schedule: {type(schedule).__name__}."
    )
  return evaluator(schedule, features, default, dtype=dtype)


def resolve_params(
    params: Mapping[str, object],
    curricula: Mapping[str, Stages],
    features: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> Mapping[str, object]:
  """Evaluates any curriculum-scheduled entries in `params`."""
  if not curricula:
    return params
  resolved = dict(params)
  for key, schedule in curricula.items():
    base = params.get(key)
    if isinstance(base, bool) or not isinstance(base, (int, float)):
      raise TypeError(
          f"Curriculum parameter {key!r} requires a numeric base value, "
          f"got {type(base).__name__}."
      )
    resolved[key] = evaluate(schedule, features, float(base), dtype=dtype)
  return resolved
