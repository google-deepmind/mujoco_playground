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

"""A device that gives one body different geoms/meshes in each world.

Each asset's geoms (meshes or primitives) replace all of the target body's
geoms. At init, every world is assigned one asset, in proportion to the
assets' weights. Assets are interleaved across worlds. With
`resample_on_reset`, every reset draws a new asset for each reset world.

MultiAssetDevice can:
  - Target a body with joints (e.g. a freejoint). Joints are never touched.
  - Mix meshes and primitives across assets.
  - Vary geom size, pose, mesh, material, color, and body inertia per world.

MultiAssetDevice cannot:
  - Target the worldbody or a static body (no joints).
  - Change bodies, joints or the kinematic tree.
  - Add assets after build. Every asset is compiled into the model, so mesh
    and geom counts grow with the number of assets.

TODO(b/521643407): Support mjGEOM_HFIELD assets on static bodies.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types as reaf_types
import torch

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_GEOM_TYPES = tuple(
    int(t)
    for t in (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_MESH,
    )
)

# Geom fields shared by all worlds. Geoms of different assets can only stand in
# for each other when these match.
_SHARED_GEOM_FIELDS = (
    "type",
    "contype",
    "conaffinity",
    "condim",
    "priority",
    "group",
)
# Batch geom fields copied from the assets as compiled on their own.
_GEOM_FIELDS = (
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_rgba",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_solmix",
    "geom_margin",
    "geom_gap",
    "geom_surfacevel",
    "geom_adhesion",
)
_BODY_FIELDS = (
    "body_mass",
    "body_ipos",
    "body_iquat",
    "body_inertia",
)

# Unused geoms are shrunk and moved to the asset's center of mass. Moving them
# far away is unsafe since planes have infinite size.
_UNUSED_GEOM_SIZE = 1e-3
_UNUSED_GEOM_VERTICES = [
    0, 0, 0, _UNUSED_GEOM_SIZE, 0, 0, 0, _UNUSED_GEOM_SIZE, 0, 0, 0, _UNUSED_GEOM_SIZE,
]  # fmt: skip
# Unused geoms should not collide: a contact needs `dist < margin1 + margin2 +
# gap1 + gap2`, and contype/conaffinity cannot differ per world.
_UNUSED_GEOM_MARGIN = -1e6
# Fields of unused geoms that should be overwritten by the scene's
# compile-time values.
_UNUSED_GEOM_FIELDS = frozenset({
    "geom_quat",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_rgba",
    "geom_margin",
    "geom_gap",
})


@dataclasses.dataclass(frozen=True)
class MultiAsset:
  """An asset contains a spec with a single body with one or more geoms.

  Assets are independent: each is compiled on its own, with its own compiler
  settings, default classes and file paths, so assets can come from anywhere.

  Note:
    - `spec` should hold exactly one body under its worldbody.
    - Geoms can be meshes or primitives (sphere, capsule, ellipsoid, cylinder,
      box).
    - Inertia comes from the compiled asset.

  Attributes:
    spec: MjSpec with one body that may contain multiple geoms.
    weight: Relative weight to sample this asset; must be > 0.
  """

  spec: mujoco.MjSpec
  weight: float = 1.0

  def __post_init__(self) -> None:
    if not self.weight > 0:
      raise ValueError(f"Asset weight must be > 0, got {self.weight}.")


class MultiAssetDevice(device.Device):
  """Gives one body a different asset in each world."""

  def __init__(
      self,
      *,
      name: str = "",
      body_name: str,
      assets: Sequence[MultiAsset],
      resample_on_reset: bool = False,
  ):
    """Initializes the device and compiles every asset independently.

    Args:
      name: Device name. Also prefixes everything the device adds to the scene.
      body_name: Name of the target body in the MjSpec.
      assets: The MultiAsset objects.
      resample_on_reset: If True, every reset draws a new asset for each reset
        world, sampled by weight. Otherwise worlds keep their startup asset.
    """
    if not assets:
      raise ValueError(f"{name}: needs at least one asset.")
    self._name = name
    self._body_name = body_name
    self._assets = tuple(assets)
    self._resample_on_reset = resample_on_reset
    self._compiled = tuple(
        _CompiledAsset.build(a.spec, f"{name}: assets[{i}]")
        for i, a in enumerate(self._assets)
    )
    self._layout = _layout(self._compiled)
    self._unused_mesh = f"{name}/unused"
    self._geom_names = tuple(f"{name}/{j}" for j in range(len(self._layout)))
    self._asset_prefixes = tuple(
        f"{name}/{i}/" for i in range(len(self._assets))
    )
    self._asset_index: torch.Tensor | None = None
    self._weights: torch.Tensor | None = None
    self._values: list[
        tuple[torch.Tensor, int | torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []

  @property
  def name(self) -> str:
    return self._name

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
  ) -> mujoco.MjSpec:
    """Replaces the body's geoms and copies every asset's meshes into `spec`."""
    spec = super().configure_spec(spec)
    body = spec.body(self._body_name)
    if body is None:
      raise ValueError(f"{self._name}: body '{self._body_name}' not found.")
    if not body.joints:
      raise ValueError(
          f"{self._name}: body '{self._body_name}' must have at least one"
          " joint (worldbody and static bodies are not supported)."
      )
    body.simple = False
    for geom in list(body.geoms):
      spec.delete(geom)
    for asset_prefix, asset in zip(
        self._asset_prefixes, self._assets, strict=True
    ):
      _copy_assets(spec, asset_prefix, asset.spec)
    if any(key[0] == _MESH for key, _ in self._layout):
      spec.add_mesh(name=self._unused_mesh, uservert=_UNUSED_GEOM_VERTICES)

    for geom_name, (key, _) in zip(self._geom_names, self._layout, strict=True):
      geom = body.add_geom(name=geom_name)
      for field, value in zip(_SHARED_GEOM_FIELDS, key, strict=True):
        setattr(geom, field, type(getattr(geom, field))(value))
      if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        geom.meshname = self._unused_mesh
      else:
        geom.size = [_UNUSED_GEOM_SIZE] * 3
    return spec

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Assigns an asset to every world and writes its values."""
    if simulator.impl != reaf_types.Impl.WARP:
      raise ValueError(
          f"{self._name}: MultiAssetDevice requires Impl.WARP, got"
          f" {simulator.impl}."
      )
    mj = simulator.mj_model
    bid = mj.body(self._body_name).id
    gids = np.array([mj.geom(name).id for name in self._geom_names])
    model = simulator.model
    xpu_device = model.body_mass.device
    nworld = simulator.nworld
    weights = [a.weight for a in self._assets]
    self._weights = torch.tensor(weights, device=xpu_device)
    self._asset_index = torch.as_tensor(
        _assign(weights, nworld), dtype=torch.int64, device=xpu_device
    )

    gids_t = torch.as_tensor(gids, device=xpu_device)
    self._values = []

    def add_value(
        field: str, ids: int | torch.Tensor, table: np.ndarray
    ) -> None:
      dst = getattr(model, field)
      table_t = torch.as_tensor(table, dtype=dst.dtype, device=xpu_device)
      buf = torch.empty(
          (nworld, *table_t.shape[1:]), dtype=dst.dtype, device=xpu_device
      )
      self._values.append((dst, ids, table_t, buf))

    for field, table in self._geom_values(mj, gids).items():
      add_value(field, gids_t, table)
    for field in _BODY_FIELDS:
      table = np.stack([getattr(c.model, field)[1] for c in self._compiled])
      add_value(field, bid, table)
    self._write()

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Draws a new asset for every reset world if `resample_on_reset`."""
    del simulator
    if not self._resample_on_reset:
      return
    assert self._asset_index is not None and self._weights is not None
    # TODO(b/568879997): Avoid torch.multinomial.
    drawn = torch.multinomial(
        self._weights, self._asset_index.shape[0], replacement=True
    )
    if env_mask is not None:
      torch.where(env_mask, drawn, self._asset_index, out=self._asset_index)
    else:
      self._asset_index.copy_(drawn)
    self._write(env_mask)

  def _write(self, env_mask: torch.Tensor | None = None) -> None:
    """Writes every world's asset values; only worlds in `env_mask` if given."""
    # TorchWrapper tracks if set_const needs to be called after model writes
    assert self._asset_index is not None
    for dst, ids, table, buf in self._values:
      torch.index_select(table, 0, self._asset_index, out=buf)
      if env_mask is not None:
        mask = env_mask.reshape(-1, *([1] * (buf.ndim - 1)))
        torch.where(mask, buf, dst[:, ids], out=buf)
      dst[:, ids] = buf

  def _geom_values(
      self, mj: mujoco.MjModel, gids: np.ndarray
  ) -> dict[str, np.ndarray]:
    """Builds a lookup table per asset and geom field."""
    unused = _unused_model()
    names = (*_GEOM_FIELDS, "geom_dataid", "geom_matid")
    tables = {
        f: np.empty(
            (len(self._compiled), len(gids), *getattr(mj, f).shape[1:]),
            dtype=getattr(mj, f).dtype,
        )
        for f in names
    }
    for i, (asset_prefix, c) in enumerate(
        zip(self._asset_prefixes, self._compiled, strict=True)
    ):
      for j, (key, k) in enumerate(self._layout):
        geoms = c.geoms.get(key, ())
        if k >= len(geoms):
          ug = _GEOM_TYPES.index(key[0])
          for f in _GEOM_FIELDS:
            if f in _UNUSED_GEOM_FIELDS:
              tables[f][i, j] = getattr(unused, f)[ug]
            else:
              tables[f][i, j] = getattr(mj, f)[gids[j]]
          tables["geom_pos"][i, j] = c.model.body_ipos[1]
          tables["geom_dataid"][i, j] = (
              mj.mesh(self._unused_mesh).id if key[0] == _MESH else -1
          )
          tables["geom_matid"][i, j] = -1
          continue
        g = geoms[k]
        for f in _GEOM_FIELDS:
          tables[f][i, j] = getattr(c.model, f)[g]
        dataid = -1
        if key[0] == _MESH:
          mesh = c.model.mesh(c.model.geom_dataid[g]).name
          dataid = mj.mesh(asset_prefix + mesh).id
        tables["geom_dataid"][i, j] = dataid
        matid = -1
        if c.model.geom_matid[g] >= 0:
          mat = c.model.mat(c.model.geom_matid[g]).name
          matid = mj.mat(asset_prefix + mat).id
        tables["geom_matid"][i, j] = matid
    # MjModel stores geom_aabb as (ngeom, 6); mjw.Model stores (*, ngeom, 2, 3).
    tables["geom_aabb"] = tables["geom_aabb"].reshape(
        len(self._compiled), len(gids), 2, 3
    )
    return tables


@dataclasses.dataclass(frozen=True)
class _CompiledAsset:
  """One asset compiled on its own."""

  model: mujoco.MjModel
  geoms: dict[tuple[int, ...], list[int]]

  @classmethod
  def build(cls, spec: mujoco.MjSpec, what: str) -> "_CompiledAsset":
    """Compiles `spec` on its own and checks the `MultiAsset` contract."""
    if len(spec.worldbody.bodies) != 1 or spec.worldbody.bodies[0].bodies:
      raise ValueError(
          f"{what} must hold exactly one body (no child bodies; use frames"
          " to offset geoms)."
      )
    model = spec.copy().compile()
    for count, kind in (
        (model.njnt, "joints"),
        (model.nu, "actuators"),
        (model.nsensor, "sensors"),
        (model.ntendon, "tendons"),
        (model.neq, "equalities"),
        (model.nsite, "sites"),
        (model.ncam, "cameras"),
        (model.nlight, "lights"),
    ):
      if count:
        raise ValueError(f"{what} must not have {kind}, got {count}.")
    if model.body_geomnum[0]:
      raise ValueError(f"{what} must not have geoms on its worldbody.")
    if not model.ngeom:
      raise ValueError(f"{what} must have at least one geom.")
    if model.body_mass[1] <= mujoco.mjMINVAL:
      raise ValueError(f"{what} has no mass.")
    geoms: dict[tuple[int, ...], list[int]] = {}
    for g in range(model.ngeom):
      if model.geom_type[g] not in _GEOM_TYPES:
        raise ValueError(
            f"{what}: geom '{model.geom(g).name}' has unsupported type"
            f" {mujoco.mjtGeom(model.geom_type[g]).name}."
        )
      key = tuple(
          int(getattr(model, f"geom_{f}")[g]) for f in _SHARED_GEOM_FIELDS
      )
      geoms.setdefault(key, []).append(g)
    return cls(model=model, geoms=geoms)


def _layout(
    compiled: Sequence[_CompiledAsset],
) -> tuple[tuple[tuple[int, ...], int], ...]:
  """Scene geoms as `(key, k)`: the `k`-th geom with shared fields `key`."""
  counts: dict[tuple[int, ...], int] = {}
  for c in compiled:
    for key, geoms in c.geoms.items():
      counts[key] = max(counts.get(key, 0), len(geoms))
  layout = []
  for key in sorted(counts):
    for k in range(counts[key]):
      layout.append((key, k))
  return tuple(layout)


def _copy_assets(
    spec: mujoco.MjSpec, prefix: str, asset_spec: mujoco.MjSpec
) -> None:
  """Copies meshes, materials and textures from `asset_spec` into `spec`."""
  clone = asset_spec.copy()
  frame = spec.worldbody.add_frame()
  attached = frame.attach_body(clone.worldbody.first_body(), prefix=prefix)
  # We are only copying meshes, materials and textures, so we can delete the
  # temporary frame/body after attach.
  spec.delete(attached)
  spec.delete(frame)
  if clone.assets:
    tag = prefix.replace("/", "_")
    for name, data in clone.assets.items():
      head, sep, tail = name.rpartition("/")
      spec.assets[f"{head}{sep}{tag}{tail}"] = data
    for elem in (*spec.meshes, *spec.textures):
      if elem.name.startswith(prefix) and elem.file:
        head, sep, tail = elem.file.rpartition("/")
        elem.file = f"{head}{sep}{tag}{tail}"


def _unused_model() -> mujoco.MjModel:
  """One tiny transparent geom per type in `_GEOM_TYPES`, in that order."""
  spec = mujoco.MjSpec()
  spec.add_mesh(name="unused", uservert=_UNUSED_GEOM_VERTICES)
  for t in _GEOM_TYPES:
    geom = spec.worldbody.add_geom(
        type=mujoco.mjtGeom(t),
        rgba=[0, 0, 0, 0],
        margin=_UNUSED_GEOM_MARGIN,
        gap=0,
    )
    if t == _MESH:
      geom.meshname = "unused"
    else:
      geom.size = [_UNUSED_GEOM_SIZE] * 3
  return spec.compile()


def _assign(weights: Sequence[float], nworld: int) -> np.ndarray:
  """Assigns an asset index per world, by weight, with interleaving."""
  w = np.asarray(weights, dtype=np.float64)
  share = w / w.sum() * nworld
  counts = np.floor(share).astype(np.int64)
  order = np.argsort(-(share - counts), kind="stable")
  counts[order[: nworld - counts.sum()]] += 1
  spots = []
  for i, c in enumerate(counts):
    spots.extend(((k + 0.5) / c, i) for k in range(c))
  spots.sort()
  return np.array([i for _, i in spots], dtype=np.int64)
