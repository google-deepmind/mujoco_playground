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

"""Playground-wide paths and constants."""

from mujoco_playground._src.paths import local_dir
from mujoco_playground._src.paths import LOCAL_DIR_ENV_VAR
from mujoco_playground._src.paths import PROJECT_ROOT
from mujoco_playground._src.paths import resolve_local_path
from mujoco_playground._src.paths import URI_PREFIX_TO_LOCAL_DIR

__all__ = [
    "LOCAL_DIR_ENV_VAR",
    "local_dir",
    "PROJECT_ROOT",
    "resolve_local_path",
    "URI_PREFIX_TO_LOCAL_DIR",
]
