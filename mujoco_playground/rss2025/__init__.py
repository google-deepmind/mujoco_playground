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

"""MuJoCo Playground."""
import importlib

from mujoco_playground.rss2025._src import dm_control_suite
from mujoco_playground.rss2025._src import locomotion
from mujoco_playground.rss2025._src import manipulation
from mujoco_playground.rss2025._src import registry
from mujoco_playground.rss2025._src import wrapper

try:
  wrapper_torch = importlib.import_module(
      "mujoco_playground.rss2025._src.wrapper_torch"
  )
except (ImportError, AttributeError):
  wrapper_torch = None

# pylint: disable=g-importing-member
from mujoco_playground.rss2025._src.mjx_env import MjxEnv
from mujoco_playground.rss2025._src.mjx_env import put_model
from mujoco_playground.rss2025._src.mjx_env import render_array
from mujoco_playground.rss2025._src.mjx_env import State
from mujoco_playground.rss2025._src.mjx_env import step

# pylint: enable=g-importing-member

__all__ = [
    "dm_control_suite",
    "locomotion",
    "manipulation",
    "MjxEnv",
    "put_model",
    "registry",
    "render_array",
    "State",
    "step",
    "wrapper",
]
if wrapper_torch is not None:
  __all__.append("wrapper_torch")
