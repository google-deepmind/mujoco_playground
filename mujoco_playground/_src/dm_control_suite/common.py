# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common functions for the DM Control Suite."""

from typing import Dict

from mujoco_playground._src import mjx_env

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "common"


def get_assets() -> Dict[str, bytes]:
  assets = {}
  for f in _XML_PATH.glob("*"):
    if f.is_file():
      assets[f.name] = f.read_bytes()
  return assets
