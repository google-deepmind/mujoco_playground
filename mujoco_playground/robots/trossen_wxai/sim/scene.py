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

"""Scene configuration for Trossen WXAI bimanual arm.

Loads the biarm XML from mujoco_menagerie and attaches it to a plane or table.
"""

import mujoco
from mujoco_playground import asset
from mujoco_playground._src.objects import furniture
from mujoco_playground._src.objects import primitive

# Height of the table surface above the ground plane.
TABLE_HEIGHT = 0.75


def _load_biarm_trossen_wxai() -> mujoco.MjSpec:
  """Loads the biarm spec and removes keyframes."""
  uri = "github:google-deepmind/mujoco_menagerie/trossen_wxai/wxai_follower_biarm.xml"
  spec = asset.get(uri).spec()
  spec.option.timestep = 0.005

  while spec.keys:
    spec.delete(spec.keys[0])

  return spec


def trossen_wxai_biarm_plane() -> mujoco.MjSpec:
  """Places the biarm on a plane."""
  spec = primitive.plane()
  biarm = _load_biarm_trossen_wxai()
  spec.attach(biarm, prefix="", frame=spec.worldbody.add_frame())
  return spec


def trossen_wxai_biarm_table() -> mujoco.MjSpec:
  """Places the biarm on the aloha table with a ground plane."""
  spec = primitive.plane()

  # Attach table raised above the floor
  table = furniture.aloha_table()
  spec.attach(table, prefix="table/", frame=spec.worldbody.add_frame())

  # Attach biarm at the table surface height
  biarm = _load_biarm_trossen_wxai()
  biarm_frame = spec.worldbody.add_frame()
  biarm_frame.pos = [0, 0, TABLE_HEIGHT]
  spec.attach(biarm, prefix="", frame=biarm_frame)

  # Add d405 overhead and worms-eye cameras.
  d405_focal = [1.93e-3, 1.93e-3]
  d405_sensorsize = [3896e-6, 2140e-6]
  d405_resolution = [1280, 720]
  overhead_cam = spec.worldbody.add_camera(name="overhead")
  overhead_cam.pos = [0, -0.303794, 0.97524 + TABLE_HEIGHT]
  overhead_cam.quat = [0.976332, 0.216277, 0, 0]
  worms_eye_cam = spec.worldbody.add_camera(name="worms_eye")
  worms_eye_cam.pos = [0, -0.377167, 0.0316055 + TABLE_HEIGHT]
  worms_eye_cam.quat = [0.672659, 0.739953, 0, 0]
  worms_eye_cam.focal_length = overhead_cam.focal_length = d405_focal
  worms_eye_cam.sensor_size = overhead_cam.sensor_size = d405_sensorsize
  worms_eye_cam.resolution = overhead_cam.resolution = d405_resolution

  # Add side fill lights pointing to the workspace
  left_light = spec.worldbody.add_light(name="left_side_light")
  left_light.pos = [-0.8, -0.2, 1.2]
  left_light.dir = [0.8, 0.2, -0.45]
  left_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  left_light.diffuse = [0.35, 0.35, 0.35]
  left_light.specular = [0.1, 0.1, 0.1]
  left_light.castshadow = False

  right_light = spec.worldbody.add_light(name="right_side_light")
  right_light.pos = [0.8, -0.2, 1.2]
  right_light.dir = [-0.8, 0.2, -0.45]
  right_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  right_light.diffuse = [0.35, 0.35, 0.35]
  right_light.specular = [0.1, 0.1, 0.1]
  right_light.castshadow = False

  return spec
