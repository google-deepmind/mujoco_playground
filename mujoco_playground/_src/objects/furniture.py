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

"""Furniture objects for simulation."""

import mujoco
from mujoco_playground import asset

_ALOHA_TABLE_XML = """
<mujoco model="aloha_table">
  <compiler meshdir="{meshdir}" texturedir="{texturedir}"/>

  <asset>
    <mesh name="tabletop" file="tabletop.obj"/>
    <mesh name="tablelegs" file="tablelegs.obj"/>
    <texture name="small_meta_table_diffuse" type="2d" file="small_meta_table_diffuse.png"/>
    <material name="table" texture="small_meta_table_diffuse"/>
  </asset>

  <worldbody>
    <body name="aloha_table" pos="0 0 0.75">
      <geom name="tabletop" type="mesh" mesh="tabletop" material="table" pos="0 0 -0.75" quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="tablelegs" type="mesh" mesh="tablelegs" material="table" pos="0 0 -0.75" quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="table_collision" type="box" size="0.61 0.37 0.1" pos="0 0 -0.1" group="3" condim="3" friction="1 0.005 0.0001" solref="0.01 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def aloha_table() -> mujoco.MjSpec:
  """Create Aloha table spec from embedded XML string."""
  aloha_asset = asset.get(
      "github:google-deepmind/mujoco_menagerie/aloha/scene.xml"
  )
  asset_dir = str(aloha_asset.path.parent / "assets")

  # Inject the resolved asset directories into the XML string
  formatted_xml = _ALOHA_TABLE_XML.format(
      meshdir=asset_dir, texturedir=asset_dir
  )
  spec = mujoco.MjSpec.from_string(formatted_xml)

  return spec
