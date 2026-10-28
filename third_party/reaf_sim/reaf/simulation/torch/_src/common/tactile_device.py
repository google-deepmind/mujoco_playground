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

"""TactileDevice for reading tactile sensor data."""

from collections.abc import Mapping
from typing import override

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class TactileDevice(device.Device):
  """Attaches a `mjSENS_TACTILE` sensor to a geom and mesh and reads depth.

  Exposes `f"{name}/depth"` with shape `(nworld, ntaxel)` containing raw
  penetration depth in meters (typically `~1e-3 m`). Downstream `flat_scene.Obs`
  terms should configure `scale=...` (e.g. `1 / max_depth`) for RL policy inputs
  and Studio `Obs 2D` visualization (which clips float maps to `[0, 1]`).
  """

  def __init__(
      self,
      *,
      name: str = "",
      postprocess: bool = False,
      dark: bool = False,
  ):
    """Initializes the tactile device.

    Args:
      name: Device name used for measurement keys and the added MJCF sensor.
      postprocess: If True, smooth raw penetration depths by solving a Dirichlet
        cotangent Laplacian system using a precomputed and cached inverse.
      dark: If True, hide measurements from Task Layer.
    """
    super().__init__()
    self._name = name
    self._sensor_name = f"{name}_sensor"
    self._postprocess = postprocess
    self._dark = dark
    self._cutoff = 0.0
    self._ntaxel = 0
    self._sensor_slice = slice(0, 0)
    self._laplacian_inv: torch.Tensor | None = None

  @property
  @override
  def name(self) -> str:
    return self._name

  @override
  def configure_spec(  # pyrefly: ignore[bad-override]
      self,
      spec: mujoco.MjSpec,
      *,
      geom: str,
      mesh: str = "",
      cutoff: float = 0.0,
  ) -> mujoco.MjSpec:
    """Attaches a `mjSENS_TACTILE` sensor to `geom` and `mesh` in `spec`.

    Args:
      spec: The MjSpec being configured.
      geom: Reference geom name in `spec` whose frame positions the taxels.
      mesh: Mesh name in `spec` whose vertices define the taxels. Defaults to
        `geom`'s mesh when `geom` is a mesh geom.
      cutoff: Optional non-negative sensor cutoff value.

    Returns:
      The configured MjSpec.
    """
    spec = super().configure_spec(spec)
    self._cutoff = cutoff
    geom_obj = spec.geom(geom)
    mesh_name = mesh or (geom_obj.meshname if geom_obj is not None else "")

    # Defer cutoff until after Laplacian smoothing when postprocess is enabled.
    spec.add_sensor(
        name=self._sensor_name,
        type=mujoco.mjtSensor.mjSENS_TACTILE,
        objtype=mujoco.mjtObj.mjOBJ_MESH,
        objname=mesh_name,
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname=geom,
        cutoff=cutoff if cutoff < 0.0 or not self._postprocess else 0.0,
    )

    # Enable SDF octrees on colliding mesh geoms so mjSENS_TACTILE detects them.
    for g in spec.geoms:
      if (g.contype != 0 or g.conaffinity != 0) and (
          g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname
      ):
        m = spec.mesh(g.meshname)
        if m is not None:
          m.needsdf = True
    return spec

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Resolves sensor address, taxel count, and optional Laplacian inverse."""
    model = simulator.mj_model
    sensor = model.sensor(self._sensor_name)
    adr = int(sensor.adr[0])
    mesh_id = int(sensor.objid[0])
    self._ntaxel = int(model.mesh_vertnum[mesh_id])
    # Depth occupies the first ntaxel entries of the 3 * ntaxel sensor block.
    self._sensor_slice = slice(adr, adr + self._ntaxel)
    if self._postprocess:
      vertadr = int(model.mesh_vertadr[mesh_id])
      faceadr = int(model.mesh_faceadr[mesh_id])
      nface = int(model.mesh_facenum[mesh_id])
      lap_inv = self.compute_laplacian_inv(
          model.mesh_vert[vertadr : vertadr + self._ntaxel],
          model.mesh_face[faceadr : faceadr + nface],
      )
      self._laplacian_inv = torch.from_numpy(lap_inv).to(
          dtype=simulator.data.sensordata.dtype,
          device=simulator.xpu_device,
      )
    else:
      self._laplacian_inv = None

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Returns the unbatched measurement spec (`{}` when `dark=True`)."""
    if self._dark:
      return {}
    return {
        f"{self._name}/depth": specs.Array(
            shape=(self._ntaxel,), dtype=np.float32
        )
    }

  @override
  def get_substep_measurements(  # pyrefly: ignore[bad-override]
      self,
      simulator: simulator_module.Simulator,
      *,
      force: bool = False,
  ) -> Mapping[str, torch.Tensor]:
    """Returns a batched `(nworld, ntaxel)` tensor of tactile depths."""
    if self._dark and not force:
      return {}
    depth = simulator.data.sensordata[:, self._sensor_slice]
    if self._laplacian_inv is not None:
      # Smooth raw depths via the cached inverse Laplacian and apply cutoff.
      depth = depth @ self._laplacian_inv
      if self._cutoff > 0.0:
        depth = depth.clamp(max=self._cutoff)
    return {f"{self._name}/depth": depth}

  def compute_laplacian_inv(
      self, verts: np.ndarray, faces: np.ndarray
  ) -> np.ndarray:
    """Precomputes the `(nvert, nvert)` inverse cotangent Laplacian."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    nvert = len(verts)

    # Assemble symmetric cotangent Laplacian weights across triangle half-edges.
    lap = np.zeros((nvert, nvert), dtype=np.float64)
    for k in range(3):
      a, u, v = faces[:, k], faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]
      ca = verts[u] - verts[a]
      cb = verts[v] - verts[a]
      cross_norm = np.linalg.norm(np.cross(ca, cb), axis=1)
      dot = np.sum(ca * cb, axis=1)
      w = np.where(
          cross_norm > 1e-12,
          0.5 * np.maximum(0.0, dot) / np.maximum(cross_norm, 1e-12),
          0.0,
      )
      np.add.at(lap, (u, u), w)
      np.add.at(lap, (v, v), w)
      np.add.at(lap, (u, v), -w)
      np.add.at(lap, (v, u), -w)

    # Identify boundary vertices lying on edges shared by a single triangle.
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    is_bndry = np.zeros(nvert, dtype=bool)
    is_bndry[unique_edges[counts == 1]] = True

    # Verify every active vertex reaches a boundary via positive weights.
    has_edge = np.zeros(nvert, dtype=bool)
    has_edge[faces] = True
    reach = is_bndry.copy()
    adj = lap < 0.0
    while not np.array_equal(next_reach := reach | (adj @ reach), reach):
      reach = next_reach
    if not is_bndry.any() or not np.all(reach[has_edge]):
      raise ValueError(
          "Laplacian postprocessing requires an open triangle mesh."
      )

    # Invert the interior block with zero Dirichlet boundary conditions.
    interior = np.flatnonzero(~is_bndry & has_edge)
    lap_inv = np.zeros((nvert, nvert), dtype=np.float32)
    if interior.size > 0:
      lap_int = lap[np.ix_(interior, interior)]
      lap_inv[np.ix_(interior, interior)] = np.linalg.inv(lap_int).astype(
          np.float32
      )
    return lap_inv
