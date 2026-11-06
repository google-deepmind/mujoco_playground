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

"""Smooth trajectory interpolation utilities for real and simulated robot arms."""

from collections.abc import Callable
from collections.abc import Sequence
import math
import time
from typing import Any

from absl import logging
import numpy as np


def cosine_interpolate(
    write_fn: Callable[[np.ndarray], Any],
    start_pose: np.ndarray | Sequence[float],
    target_pose: np.ndarray | Sequence[float],
    *,
    duration_secs: float,
    control_timestep: float,
    real_time: bool = True,
) -> None:
  """Interpolates from start_pose to target_pose using a cosine velocity profile."""
  start = np.asarray(start_pose, dtype=np.float32)
  target = np.asarray(target_pose, dtype=np.float32)
  steps = max(round(duration_secs / control_timestep), 1)

  for step in range(1, steps + 1):
    t_start = time.perf_counter()
    blend = 0.5 * (1.0 - math.cos(math.pi * step / steps))
    write_fn(start + blend * (target - start))

    if real_time:
      elapsed = time.perf_counter() - t_start
      if control_timestep > elapsed:
        time.sleep(control_timestep - elapsed)
      else:
        logging.warning(
            "Motion interpolation step %d/%d took %.4fs, exceeding control"
            " timestep of %.4fs (running slower than real-time).",
            step,
            steps,
            elapsed,
            control_timestep,
        )
