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

"""Agentic pick-and-place task registry and environment configuration.

Tasks are robot-independent: `env_config(task, robot)` places any registered
task in front of any registered embodiment (see `embodiment.py`).

Each `Task` lists the free objects in the scene, a natural-language prompt, a
success check over simulator features, and an oracle that turns privileged
object poses into a plan in the primitive grammar understood by `agent.py`.
The oracle is what lets an agent practice in simulation before a VLM takes
over on the real robot.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import functools
from typing import Any
from typing import Literal

import mujoco
from mujoco_playground import asset
from mujoco_playground._src.objects import primitive
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
from mujoco_playground.tasks.agentic.tabletop_v0 import ik_controller
from mujoco_playground.tasks.agentic.tabletop_v0 import mdp
from mujoco_playground.tasks.agentic.tabletop_v0 import reset
from mujoco_playground.tasks.manipulation.pick_cube import mdp as pick_mdp
import numpy as np
from reaf.simulation.torch import CameraDevice
from reaf.simulation.torch import DomainRandomizationResetter
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import FreeDevice
from reaf.simulation.torch import model_randomization
from reaf.simulation.torch import MujocoCameraDevice
from reaf.simulation.torch import noise
import torch

ALOHA_ASSETS = "github:google-deepmind/aloha_sim/aloha_sim/assets"

CUBE_HALF = 0.015
BAR_HALF = (0.035, 0.012, 0.014)

RED = (0.85, 0.15, 0.15, 1.0)
GREEN = (0.15, 0.75, 0.2, 1.0)
BLUE = (0.15, 0.35, 0.9, 1.0)
YELLOW = (0.95, 0.85, 0.15, 1.0)
WHITE = (0.9, 0.9, 0.9, 1.0)

# Privileged pose of an object: [x, y, z, yaw] of its bounding-box center.
Pose = np.ndarray
Plan = list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Obj:
  """A free object in the scene: a colored box, a mesh asset, or a custom spec.

  The body frame is centered on the object's axis-aligned bounding box, so the
  `xpos` feature is the geometric center for boxes and meshes alike and the
  oracle, success checks, and spawn logic need no per-object offsets.
  """

  name: str
  rgba: tuple[float, float, float, float] = WHITE
  half_size: tuple[float, float, float] | None = None  # Box half extents.
  asset: str | None = None  # Mesh model path, see `ALOHA_ASSETS`.
  scale: float = 1.0
  container: bool = False  # Never moved by the oracle; things go into it.
  spec_fn: Callable[[str], mujoco.MjSpec] | None = None

  @functools.cached_property
  def half(self) -> np.ndarray:
    """Bounding-box half extents in the (yaw = 0) body frame."""
    if self.half_size is not None:
      return np.asarray(self.half_size, dtype=np.float64)
    if self.spec_fn is not None:
      return _model_bounds(self.spec().compile())[1]
    return _asset_bounds(self.asset, self.scale)[1]

  @property
  def rest_z(self) -> float:
    return float(self.half[2])

  @property
  def radius(self) -> float:
    """Footprint radius used to keep spawns apart."""
    return float(max(self.half[:2]))

  @property
  def long_axis_yaw(self) -> float:
    """Yaw of the object's longest horizontal axis in its body frame."""
    return np.pi / 2 if self.half[1] > self.half[0] else 0.0

  @functools.cached_property
  def grasp_offset(self) -> np.ndarray:
    """Grasp point minus bounding-box center, (x, y) in the body frame.

    For curved meshes (a banana) the bounding-box center can lie off the
    object; the grasp point is the centroid of the mid-length cross-section.
    """
    if self.asset is None or self.spec_fn is not None:
      return np.zeros(2)
    return _asset_grasp_offset(self.asset, self.scale)

  @property
  def xpos_key(self) -> str:
    return f"{self.name}/xpos"

  def spec(self) -> mujoco.MjSpec:
    """Builds and returns the MjSpec for this object."""
    if self.spec_fn is not None:
      return _center_spec(self.spec_fn(self.name))
    if self.asset is not None:
      return _asset_spec(self.name, self.asset, self.scale)
    assert self.half_size is not None
    spec = primitive.cube(self.name)
    geom = spec.geom(self.name)
    geom.size = list(self.half_size)
    geom.rgba = list(self.rgba)
    # `primitive.cube` only collides with the robot and floor; objects must
    # also collide with each other to be stackable.
    geom.contype = geom.conaffinity = 1
    return spec


@dataclasses.dataclass(frozen=True)
class Task:
  """A pick-and-place task: scene objects, prompt, success check, oracle."""

  name: str
  prompt: str
  objects: tuple[Obj, ...]
  success: Callable[..., torch.Tensor]
  success_params: Mapping[str, Any]
  oracle: Callable[[Mapping[str, Pose]], Plan]


def _raw_asset_spec(name: str, asset_path: str, scale: float) -> mujoco.MjSpec:
  """Loads a mesh model, scales it, and names its body, joint, and geoms."""
  spec = asset.get(asset_path).spec()
  spec.modelname = name
  body = spec.worldbody.first_body()
  body.name = name
  body.pos = [0.0, 0.0, 0.0]
  body.quat = [1.0, 0.0, 0.0, 0.0]
  joint = body.first_joint()
  if joint is not None:
    joint.name = f"{name}_freejoint"
  for mesh in spec.meshes:
    mesh.scale = [s * scale for s in mesh.scale]
  for i, geom in enumerate(body.geoms):
    if not geom.name:
      geom.name = f"{name}_geom_{i}"
    if geom.contype != 0 or geom.conaffinity != 0:
      geom.contype = geom.conaffinity = 1  # Collide with other objects too.
  return spec


def _model_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """Returns (center, half extents) of a compiled model's bounding box."""
  signs = np.array(
      [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
      dtype=np.float64,
  )
  lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
  for g in range(model.ngeom):
    if model.geom_group[g] > 3:  # Invisible helper geometry.
      continue
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    corners = model.geom_aabb[g, :3] + signs * model.geom_aabb[g, 3:]
    corners = model.geom_pos[g] + corners @ rot.reshape(3, 3).T
    lo, hi = np.minimum(lo, corners.min(0)), np.maximum(hi, corners.max(0))
  return (lo + hi) / 2, (hi - lo) / 2


def _center_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Shifts body geoms and sites so the body frame sits at the AABB center."""
  center, _ = _model_bounds(spec.compile())
  body = spec.worldbody.first_body()
  for geom in body.geoms:
    geom.pos = list(np.asarray(geom.pos) - center)
  for site in body.sites:
    site.pos = list(np.asarray(site.pos) - center)
  return spec


@functools.lru_cache(maxsize=None)
def _asset_bounds(asset_path: str, scale: float) -> tuple[np.ndarray, ...]:
  """Returns (center, half extents) of the model's bounding box."""
  return _model_bounds(_raw_asset_spec("obj", asset_path, scale).compile())


@functools.lru_cache(maxsize=None)
def _asset_grasp_offset(asset_path: str, scale: float) -> np.ndarray:
  """Centroid of the mid-length mesh slice relative to the bounding-box center."""
  model = _raw_asset_spec("obj", asset_path, scale).compile()
  center, half = _asset_bounds(asset_path, scale)
  verts = []
  for g in range(model.ngeom):
    mesh = model.geom_dataid[g]
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mesh < 0:
      continue
    if model.geom_group[g] > 3:
      continue
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    local = model.mesh_vert[adr : adr + num]
    verts.append(model.geom_pos[g] + local @ rot.reshape(3, 3).T)
  if not verts:
    return np.zeros(2)
  xy = np.concatenate(verts)[:, :2] - center[:2]
  long_ax = int(half[1] > half[0])
  near = np.abs(xy[:, long_ax]) < max(0.1 * half[long_ax], 0.002)
  if not near.any():
    return np.zeros(2)
  off = np.zeros(2)
  off[1 - long_ax] = float(xy[near, 1 - long_ax].mean())
  return off


def _asset_spec(name: str, asset_path: str, scale: float) -> mujoco.MjSpec:
  """Mesh model whose body frame sits at the bounding-box center."""
  return _center_spec(_raw_asset_spec(name, asset_path, scale))


def _aloha_socket_spec(name: str, *, vertical: bool = False) -> mujoco.MjSpec:
  """Canonical ALOHA 4-wall socket from `mjx_single_peg_insertion.xml`.

  In its horizontal orientation (`vertical=False`), the 36 mm x 36 mm tunnel
  runs along X with its white entrance collar at `-X` (x = -0.048 m) and red
  inner stop wall at `+X` (x = +0.044 m). When `vertical=True`, the same 4-wall
  socket stands upright with its white entrance at `+Z` and red floor at `-Z`.

  Args:
    name: Body and model name for the socket.
    vertical: Whether to rotate the socket so its opening faces `+Z`.

  Returns:
    An `MjSpec` for the socket body.
  """
  spec = mujoco.MjSpec()
  spec.modelname = name
  body = spec.worldbody.add_body(
      name=name, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
  )
  body.add_joint(name=f"{name}_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
  blue = list(BLUE)
  white = list(WHITE)
  red = list(RED)

  # Canonical `mjx_single_peg_insertion.xml` walls in the horizontal (-X
  # entrance) frame: 4 flush blue walls (x in [-0.044, +0.048] m) + 4 flush
  # white entrance rim bars (x in [-0.048, -0.044] m) + red rear stop wall.
  parts = (
      ("B", [0.046, 0.022, 0.002], [0.002, 0.0, -0.020], blue, 0.25),
      ("T", [0.046, 0.022, 0.002], [0.002, 0.0, 0.020], blue, 0.016),
      ("L", [0.046, 0.002, 0.018], [0.002, -0.020, 0.0], blue, 0.016),
      ("R", [0.046, 0.002, 0.018], [0.002, 0.020, 0.0], blue, 0.016),
      ("rim_B", [0.002, 0.022, 0.002], [-0.046, 0.0, -0.020], white, 0.004),
      ("rim_T", [0.002, 0.022, 0.002], [-0.046, 0.0, 0.020], white, 0.004),
      ("rim_L", [0.002, 0.002, 0.018], [-0.046, -0.020, 0.0], white, 0.004),
      ("rim_R", [0.002, 0.002, 0.018], [-0.046, 0.020, 0.0], white, 0.004),
      (
          "wall",
          [0.004, 0.018, 0.018],
          [0.044, 0.0, 0.0],
          red,
          0.25 if vertical else 0.016,
      ),
  )
  for suffix, sz, pos, rgba, mass in parts:
    if vertical:
      # Map (-X entrance -> +Z entrance, +X rear -> -Z floor).
      sz = [sz[2], sz[1], sz[0]]
      pos = [pos[2], pos[1], -pos[0]]
    g = body.add_geom(name=f"{name}_{suffix}", type=mujoco.mjtGeom.mjGEOM_BOX)
    g.size = sz
    g.pos = pos
    g.rgba = rgba
    g.mass = mass
    g.condim = 4
    g.friction = [1.0, 0.05, 0.001]
    g.contype = g.conaffinity = 1
  return spec


def _marker_stand_spec(name: str) -> mujoco.MjSpec:
  """Canonical Aloha `Marker/marker.xml` held upright in a base stand."""
  marker_asset = f"{ALOHA_ASSETS}/mujoco/Marker/marker.xml"
  spec = _raw_asset_spec(name, marker_asset, scale=0.85)
  body = spec.worldbody.first_body()
  g_base = body.add_geom(
      name=f"{name}_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER
  )
  g_base.size = [0.035, 0.005, 0.0]
  g_base.pos = [0.0, 0.0, 0.005]
  g_base.rgba = list(WHITE)
  g_base.mass = 0.5
  g_base.friction = [1.0, 0.05, 0.001]
  g_base.contype = g_base.conaffinity = 1
  return spec


def ycb(name: str, model: str, scale: float) -> Obj:
  return Obj(
      name,
      asset=f"{ALOHA_ASSETS}/ycb/{model}/google_64k/model.xml",
      scale=scale,
  )


# --- Oracle plans -------------------------------------------------------------


def _grasp(
    p: Pose, obj: Obj, poses: Mapping[str, Pose] | None = None
) -> dict[str, Any]:
  """Top-down grasp across the object's long axis at its center height.

  Objects with a square footprint (cubes) can be grasped across either pair of
  faces; with `poses` given, the pair whose jaw axis points away from nearby
  objects is chosen so an open jaw does not land on a neighbour.
  """
  x, y, z, yaw = (float(v) for v in p)
  c, s = np.cos(yaw), np.sin(yaw)
  dx, dy = obj.grasp_offset
  x, y = x + c * dx - s * dy, y + s * dx + c * dy
  yaw += obj.long_axis_yaw
  half = obj.half
  if poses is not None and abs(half[0] - half[1]) < 1e-3:

    def crowding(grasp_yaw: float) -> float:
      jaw = grasp_yaw + np.pi / 2  # Jaws open/close across the long axis.
      cost = 0.0
      for other in poses.values():
        d = np.array(other[:2], dtype=np.float64) - (x, y)
        dist = float(np.hypot(*d))
        if 1e-6 < dist < 0.1:
          cost += abs(np.cos(np.arctan2(d[1], d[0]) - jaw)) / dist
      return cost

    yaw = min((yaw, yaw + np.pi / 2), key=crowding)
  return {"op": "grasp", "x": x, "y": y, "z": z, "yaw": yaw}


def _place(x: float, y: float, z: float) -> dict[str, Any]:
  return {"op": "place", "x": float(x), "y": float(y), "z": float(z)}


def _oracle_into(
    poses: Mapping[str, Pose],
    objs: tuple[Obj, ...],
    container: Obj,
    xy_radius: float,
    drop_z: float,
) -> Plan:
  cx, cy = poses[container.name][:2]
  plan: Plan = []
  for obj in objs:
    x, y = poses[obj.name][:2]
    if np.hypot(x - cx, y - cy) < xy_radius:  # Already in the container.
      continue
    plan += [_grasp(poses[obj.name], obj, poses), _place(cx, cy, drop_z)]
  return plan + [{"op": "home"}]


def _oracle_insert(
    poses: Mapping[str, Pose],
    peg: Obj,
    socket: Obj,
) -> Plan:
  px, py, pz, pyaw = (float(v) for v in poses[peg.name])
  sx, sy, sz, _ = (float(v) for v in poses[socket.name])
  yaw = pyaw + peg.long_axis_yaw
  return [
      {"op": "grasp", "x": px, "y": py, "z": pz + 0.015, "yaw": yaw},
      {"op": "place", "x": sx, "y": sy, "z": sz + 0.055, "yaw": yaw},
      {"op": "home"},
  ]


def _oracle_cap_marker(
    poses: Mapping[str, Pose],
    cap: Obj,
    marker: Obj,
) -> Plan:
  cx, cy, cz, cyaw = (float(v) for v in poses[cap.name])
  mx, my, mz, _ = (float(v) for v in poses[marker.name])
  yaw = cyaw + cap.long_axis_yaw
  return [
      {"op": "grasp", "x": cx, "y": cy, "z": cz + 0.005, "yaw": yaw},
      {"op": "place", "x": mx, "y": my, "z": mz + 0.055, "yaw": yaw},
      {"op": "home"},
  ]


def _oracle_usb_in_port(
    poses: Mapping[str, Pose],
    usb: Obj,
    port: Obj,
) -> Plan:
  ux, uy, uz, uyaw = (float(v) for v in poses[usb.name])
  px, py, pz, _ = (float(v) for v in poses[port.name])
  yaw = uyaw + usb.long_axis_yaw
  tail_x = ux - 0.020 * float(np.cos(yaw))
  tail_y = uy - 0.020 * float(np.sin(yaw))
  return [
      {"op": "grasp", "x": tail_x, "y": tail_y, "z": uz, "yaw": yaw},
      {"op": "goto", "x": px - 0.090, "y": py, "z": pz, "yaw": 0.0},
      {"op": "goto", "x": px - 0.055, "y": py, "z": pz, "yaw": 0.0},
      {"op": "gripper", "gripper": 1.0},
      {"op": "goto", "x": px - 0.090, "y": py, "z": pz + 0.06, "yaw": 0.0},
      {"op": "home"},
  ]


def _oracle_stack(poses: Mapping[str, Pose], objs: tuple[Obj, ...]) -> Plan:
  base = poses[objs[0].name]
  plan: Plan = []
  for i, obj in enumerate(objs[1:], start=1):
    top_z = base[2] + 2 * CUBE_HALF * i
    x, y, z = poses[obj.name][:3]
    if np.hypot(x - base[0], y - base[1]) < 0.012 and abs(z - top_z) < 0.008:
      continue  # Already stacked.
    plan += [
        _grasp(poses[obj.name], obj, poses),
        _place(base[0], base[1], top_z),
    ]
  return plan + [{"op": "home"}]


# --- Task registry ------------------------------------------------------------


def _cube(name: str, rgba) -> Obj:
  return Obj(name, rgba, half_size=(CUBE_HALF,) * 3)


_BOWL = dataclasses.replace(ycb("bowl", "024_bowl", 0.7), container=True)
_PLATE = dataclasses.replace(ycb("plate", "029_plate", 0.6), container=True)
_BANANA = ycb("banana", "011_banana", 0.6)
_MARKER = Obj(
    "marker", asset=f"{ALOHA_ASSETS}/mujoco/Marker/marker.xml", scale=0.8
)
_STACK_CUBES = (_cube("red", RED), _cube("green", GREEN), _cube("blue", BLUE))
_PEG = Obj("peg", RED, half_size=(0.010, 0.010, 0.048))
_SOCKET = Obj(
    "socket",
    BLUE,
    container=True,
    spec_fn=functools.partial(_aloha_socket_spec, vertical=True),
)
_CAP = Obj(
    "cap",
    asset=f"{ALOHA_ASSETS}/mujoco/Marker/cap.xml",
    scale=0.85,
)
_FIXED_MARKER = Obj(
    "marker_stand",
    WHITE,
    asset=f"{ALOHA_ASSETS}/mujoco/Marker/marker.xml",
    scale=0.85,
    container=True,
    spec_fn=_marker_stand_spec,
)
_USB = Obj(
    "usb",
    asset=(
        f"{ALOHA_ASSETS}/gso/Kingston_DT4000MR_G2_Management_Ready_USB_64GB/model.xml"
    ),
    scale=0.9,
)
_USB_PORT = Obj(
    "usb_port",
    BLUE,
    container=True,
    spec_fn=functools.partial(_aloha_socket_spec, vertical=False),
)


def _into_task(
    name: str,
    prompt: str,
    objs: tuple[Obj, ...],
    container: Obj,
    xy_radius: float,
    z_range: tuple[float, float],
    drop_z: float,
) -> Task:
  """Task: every object rests within `xy_radius` of the container center.

  `z_range` bounds the object height relative to the container center; it must
  admit objects resting on top of others already inside. `drop_z` is the pinch
  height at which the oracle releases objects above the container.
  """
  return Task(
      name=name,
      prompt=prompt,
      objects=(*objs, container),
      success=mdp.check_contained,
      success_params={
          "object_keys": [o.xpos_key for o in objs],
          "anchor_key": container.xpos_key,
          "xy_radius": xy_radius,
          "z_min": z_range[0],
          "z_max": z_range[1],
      },
      oracle=functools.partial(
          _oracle_into,
          objs=objs,
          container=container,
          xy_radius=xy_radius,
          drop_z=drop_z,
      ),
  )


# XY containment is what separates "inside the bowl" from "on the rim".
_in_bowl_task = functools.partial(
    _into_task,
    container=_BOWL,
    xy_radius=0.045,
    z_range=(-0.03, 0.07),
    drop_z=0.075,
)
_on_plate_task = functools.partial(
    _into_task,
    container=_PLATE,
    xy_radius=0.05,
    z_range=(-0.01, 0.05),
    drop_z=0.045,
)

TASKS: dict[str, Task] = {
    t.name: t
    for t in (
        _in_bowl_task(
            "cube_in_bowl",
            "Put the green cube in the bowl.",
            (_cube("cube", GREEN),),
        ),
        _in_bowl_task(
            "bar_in_bowl",
            "Put the yellow bar in the bowl.",
            (Obj("bar", YELLOW, half_size=BAR_HALF),),
        ),
        _in_bowl_task(
            "cubes_in_bowl",
            "Put all three cubes in the bowl.",
            _STACK_CUBES,
        ),
        _in_bowl_task(
            "banana_in_bowl",
            "Put the banana in the bowl.",
            (_BANANA,),
        ),
        _on_plate_task(
            "marker_on_plate",
            "Put the marker on the plate.",
            (_MARKER,),
        ),
        Task(
            name="stack3",
            prompt=(
                "Stack the green cube on the red cube, then the blue cube on"
                " top."
            ),
            objects=_STACK_CUBES,
            success=mdp.check_stacked,
            success_params={"object_keys": [o.xpos_key for o in _STACK_CUBES]},
            oracle=functools.partial(_oracle_stack, objs=_STACK_CUBES),
        ),
        Task(
            name="peg_in_socket",
            prompt="Insert the red peg into the blue socket.",
            objects=(_PEG, _SOCKET),
            success=mdp.check_inserted,
            success_params={
                "peg_key": _PEG.xpos_key,
                "socket_key": _SOCKET.xpos_key,
                "xy_tol": 0.016,
                "z_max": 0.036,
            },
            oracle=functools.partial(_oracle_insert, peg=_PEG, socket=_SOCKET),
        ),
        Task(
            name="cap_fixed_marker",
            prompt="Place the marker cap onto the upright marker.",
            objects=(_CAP, _FIXED_MARKER),
            success=mdp.check_capped,
            success_params={
                "cap_key": _CAP.xpos_key,
                "marker_key": _FIXED_MARKER.xpos_key,
                "xy_tol": 0.015,
                "dz_min": 0.035,
                "dz_max": 0.075,
            },
            oracle=functools.partial(
                _oracle_cap_marker, cap=_CAP, marker=_FIXED_MARKER
            ),
        ),
        Task(
            name="usb_in_port",
            prompt="Plug the USB thumb drive into the horizontal socket.",
            objects=(_USB, _USB_PORT),
            success=mdp.check_usb_plugged,
            success_params={
                "usb_key": _USB.xpos_key,
                "port_key": _USB_PORT.xpos_key,
                "dx_range": (-0.048, 0.015),
                "y_tol": 0.016,
                "dz_range": (-0.016, 0.016),
            },
            oracle=functools.partial(
                _oracle_usb_in_port, usb=_USB, port=_USB_PORT
            ),
        ),
    )
}


# --- Environment configuration -----------------------------------------------


def _base_scene(emb: embodiment_lib.Embodiment) -> mujoco.MjSpec:
  scene = primitive.plane()
  scene.copy_during_attach = True
  floor = scene.geom("floor")
  floor.material = ""
  floor.rgba = [0.28, 0.28, 0.28, 1.0]
  scene.stat.extent = 4.0
  center_x = float(np.mean(emb.spawn_x))
  scene.worldbody.add_body(name="workspace_center", pos=[center_x, 0.0, 0.0])
  key_light = scene.light("light")
  key_light.pos = [0.2, 0.0, 1.4]
  key_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  key_light.targetbody = "workspace_center"
  key_light.diffuse = [0.6, 0.6, 0.6]
  key_light.castshadow = True
  for name, pos in (
      ("left_light", [-0.7, 0.1, 1.2]),
      ("right_light", [0.7, 0.1, 1.2]),
  ):
    light = scene.worldbody.add_light(name=name)
    light.pos = pos
    light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    light.targetbody = "workspace_center"
    light.diffuse = [0.35, 0.35, 0.35]
    light.specular = [0.05, 0.05, 0.05]
    light.castshadow = False
  return scene


def _single_term(func, output_shape, **params) -> config.ObsGroup:
  return config.ObsGroup(
      terms={
          "obs": config.Obs(func, **params).configure(output_shape=output_shape)
      },
      enable_noise=False,
  )


def env_config(
    task: str = "cube_in_bowl",
    robot: str = "so101",
    impl: Literal["cpp", "warp"] = "cpp",
) -> config.Environment:
  """Returns the environment configuration for a task on an embodiment.

  Args:
    task: Registered task name (see `TASKS`).
    robot: Registered embodiment name (see `embodiment.EMBODIMENTS`).
    impl: Simulator backend, "cpp" (CPU) or "warp" (CUDA).
  """
  spec = TASKS[task]
  emb = embodiment_lib.get(robot)
  joints = list(emb.joints)
  prefix = embodiment_lib.ATTACHMENT_PREFIX

  devices = {
      "follower": (
          config.Component(
              emb.sim_device_cls,
          ).attach(emb.spec_fn, prefix=prefix)
      ),
  }
  spawn_x = float(np.mean(emb.spawn_x))
  for i, obj in enumerate(spec.objects):
    devices[obj.name] = config.Component(
        FreeDevice,
        body_name=f"{obj.name}/{obj.name}",
        joint_name=f"{obj.name}/{obj.name}_freejoint",
        measurements=["xpos", "xquat"],
    ).attach(
        obj.spec,
        prefix=f"{obj.name}/",
        pos=(spawn_x, emb.spawn_y[0] + 0.07 * i, obj.rest_z),
    )

  sim_cfg = config.Sim(
      nworld=1,
      xpu_device="cpu" if impl == "cpp" else "cuda:0",
      impl=impl,
      simulation_timestep=0.002,
      control_timestep=0.02,
      ccd_iterations=300,
  )
  cam = emb.wrist_cam_fullname
  cam_res = emb.camera_size
  if impl == "cpp":
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    devices["wrist_cam"] = config.Component(
        MujocoCameraDevice,
        camera_name=cam,
        width=cam_res[0],
        height=cam_res[1],
        render_rate_hz=1.0 / sim_cfg.control_timestep,
        scene_option=scene_option,
        image_output_name_prefix=cam,
        rgb_output_name_suffix="_rgb",
    )
  else:
    devices["wrist_cam"] = config.Component(
        CameraDevice,
        cam_name=[cam],
        cam_resolution=[cam_res],
        render_rate_hz=1.0 / sim_cfg.control_timestep,
        render_rgb=True,
        render_depth=False,
        enabled_geom_groups=(0, 2),
        use_shadows=True,
    )

  rgb_key = emb.wrist_rgb_key
  observation_adapter = {
      "ee_pos": _single_term(mdp.EE_XPOS_KEY, (3,)),
      "ee_ori": _single_term(
          mdp.obs_ee_ori,
          (3,),
          base_xy=emb.base_xy,
          free_azimuth=emb.n_arm >= 6,
      ),
      "gripper_state": _single_term(
          mdp.obs_gripper_state,
          (1,),
          gripper_range=emb.gripper_obs_range,
      ),
      "joint_pos": _single_term(mdp.QPOS_KEY, (len(joints),)),
      "wrist_rgb": _single_term(
          lambda f, **_: f[rgb_key].to(torch.float32),
          (cam_res[1], cam_res[0], 3),
      ),
  }

  dr_terms = {
      "flat_materials": config.Term(
          pick_mdp.override_flat_materials,
          # Mesh assets keep their textures; boxes are flat colors.
          patterns=(
              f"{prefix}.*",
              *(f"{o.name}/.*" for o in spec.objects if o.asset is None),
              "floor",
          ),
      ),
  }
  light_rand: dict[str, tuple[float, float, noise.OperationType]] = {
      "light_pos": (-0.25, 0.25, "add"),
      "light_dir": (-0.35, 0.35, "add"),
      "light_diffuse": (0.70, 1.30, "scale"),
  }
  for field, (lo, hi, op) in light_rand.items():
    dr_terms[field] = config.Term(
        model_randomization.randomize_field,
        field=field,
        pattern=".*",
        noise=noise.uniform(lo, hi, op=op),
        trigger="reset",
    )

  # Domain randomization must run first: `mj_setConst` resets `qpos` to
  # `qpos0`, which would clobber the poses set by the scatter resetter.
  resetters = {
      "domain_randomization": config.Component(
          DomainRandomizationResetter,
          terms=dr_terms,
      ),
      "scatter": config.Component(
          reset.ScatterObjectsResetter,
          objects=[
              (o.name, o.rest_z, not o.container, o.radius)
              for o in spec.objects
          ],
          robot=emb.name,
      ),
  }

  success_params: dict[str, Any] = {
      "check": spec.success,
      "gripper_range": emb.gripper_obs_range,
      "released_norm": emb.gripper_released_norm,
      **spec.success_params,
  }
  task_layer = config.TaskLayer(
      command_feature_producers={
          "smooth_cartesian_ik": config.Component(
              ik_controller.SmoothCartesianIKProcessor,
              robot=emb.name,
              control_timestep=sim_cfg.control_timestep,
          ),
      },
      rewards={
          "success": config.Reward(
              mdp.success_reward,
              weight=1.0 / sim_cfg.control_timestep,
              **success_params,
          ),
      },
  )

  return config.Environment(
      base_scene=_base_scene(emb),
      device_layer=config.DeviceLayer(devices=devices),
      task_layer=task_layer,
      observation_adapter=observation_adapter,
      resetters=resetters,
      sim=sim_cfg,
      episode_length_secs=120.0,
  )
