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

"""Primitive shape configurations for playground environments."""

import importlib.resources

import mujoco


def cube(name: str, mocap: bool = False) -> mujoco.MjSpec:
  """Create a cube MjSpec."""
  spec = mujoco.MjSpec()
  spec.modelname = name

  box = spec.worldbody.add_body(name=name, mocap=mocap)

  if mocap:
    # Ghost cube - no physics
    box.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        rgba=[1.0, 0.0, 0.0, 0.3],
        contype=0,
        conaffinity=0,
        group=4,
    )
  else:
    # Dynamic cube with freejoint
    box.add_freejoint(name=f"{name}_freejoint")
    box.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        density=1000.0,
        condim=3,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.0, 1.0, 0.0, 1.0],
        contype=2,
        conaffinity=1,
        solref=[0.01, 1],
    )

  return spec


def plane() -> mujoco.MjSpec:
  xml_text = (
      importlib.resources.files(__package__)
      .joinpath("assets", "primitive", "plane.xml")
      .read_text(encoding="utf-8")
  )
  return mujoco.MjSpec.from_string(xml_text)
