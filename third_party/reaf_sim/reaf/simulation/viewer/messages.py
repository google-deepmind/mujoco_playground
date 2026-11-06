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
"""Messages and event definitions for REAF Studio Viewer."""

import dataclasses

import dm_env
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class TimestepEvent(messages.Event):
  """Sent from sim to viewer whenever the environment timestep changes.

  Sent for the initial timestep, after each step, and after each reset, so
  viewer-side handlers can derive episode boundaries (``timestep.first()``),
  step counts, rewards, and episode ends (``timestep.last()``) from this single
  stream.
  """

  timestep: dm_env.TimeStep


@dataclasses.dataclass(frozen=True)
class InstructionEvent(messages.Event):
  """Sent from viewer to sim when the user submits a new instruction."""

  instruction: str
