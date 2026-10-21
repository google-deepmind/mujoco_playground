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

"""Wall clock for the real robot, the counterpart of the simulation clock."""

from collections.abc import Mapping
import time
from typing import Any

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import device as reaf_device
from typing_extensions import override

# The feature name the task layer's timeout checker looks for, matching the
# name the simulated clock publishes under.
_TIME = "time"


class ClockDevice(reaf_device.Device):
  """Elapsed wall-clock seconds since the episode started."""

  def __init__(self):
    self._epoch = time.monotonic()

  @property
  @override
  def name(self) -> str:
    return "clock"

  def reset(self) -> None:
    self._epoch = time.monotonic()

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    return {_TIME: specs.Array(shape=(), dtype=np.float32)}

  @override
  def commands_spec(self) -> dict[str, gdmr_types.AnyArraySpec]:
    return {}

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    return {_TIME: np.float32(time.monotonic() - self._epoch)}

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    del commands
