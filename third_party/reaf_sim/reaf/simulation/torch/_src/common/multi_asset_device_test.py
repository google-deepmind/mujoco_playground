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

import dataclasses
import os
import tempfile
from typing import Any

from absl.testing import absltest
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.common import multi_asset_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_OBJ_POS = (0.0, 0.0, 0.3)


def _tetra_obj(edge: float) -> str:
  return (
      f"v 0 0 0\nv {edge} 0 0\nv 0 {edge} 0\nv 0 0 {edge}\n"
      "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
  )


def _tetra_vertices(edge: float) -> list[float]:
  return [0, 0, 0, edge, 0, 0, 0, edge, 0, 0, 0, edge]


def _boxes_asset() -> mujoco.MjSpec:
  """Two boxes, a cylinder, and a box on a rotated frame."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="mug")
  body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.01], pos=[0, 0, 0.01]
  )
  body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[0.01, 0.03, 0.04],
      pos=[0.03, 0, 0.05],
  )
  body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=[0.02, 0.01, 0],
      pos=[0, 0, 0.1],
      density=500,
  )
  handle = body.add_frame(
      pos=[0.05, 0, 0.05], quat=[np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
  )
  handle.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, 0.01, 0.02], pos=[0.01, 0, 0]
  )
  return spec


def _mesh_asset(edge: float, rgba: tuple[float, ...]) -> mujoco.MjSpec:
  """One tetrahedron mesh with a textured material."""
  spec = mujoco.MjSpec()
  spec.add_mesh(name="tetra", uservert=_tetra_vertices(edge))
  spec.add_texture(
      name="tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=8, height=8,
  )  # fmt: skip
  mat = spec.add_material(name="mat", rgba=rgba)
  textures = list(mat.textures)
  textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "tex"
  mat.textures = textures
  body = spec.worldbody.add_body(name="obj")
  body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tetra", material="mat",
      margin=0.01, gap=0.005,
  )  # fmt: skip
  return spec


def _scene(joint: bool = True) -> mujoco.MjSpec:
  """A floor and body `obj` with a placeholder box the device replaces."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
  )
  obj = spec.worldbody.add_body(name="obj", pos=_OBJ_POS)
  if joint:
    obj.add_freejoint(name="obj_joint")
  obj.add_geom(
      name="placeholder",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[0.05, 0.05, 0.05],
  )
  return spec


def _build(
    scene: mujoco.MjSpec,
    assets: list[multi_asset_device.MultiAsset],
    nworld: int,
    body_name: str = "obj",
    impl: types.Impl = types.Impl.WARP,
) -> simulator_module.Simulator:
  """Configures the spec, compiles it and initializes the device."""
  dev = multi_asset_device.MultiAssetDevice(
      name="obj_asset", body_name=body_name, assets=assets
  )
  spec = dev.configure_spec(scene)
  sim = simulator_module.Simulator(
      mj_spec=spec, nworld=nworld, xpu_device="cpu", impl=impl
  )
  dev.initialize(sim)
  return sim


class MultiAssetDeviceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.config.kernel_cache_dir = "/tmp/kernel_cache"

  def test_assign_is_exact_and_interleaved(self):
    np.testing.assert_array_equal(
        multi_asset_device._assign([1.0, 2.0], 12),
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    )
    idx = multi_asset_device._assign([1.0, 1.0, 1.0], 7)
    self.assertEqual(np.bincount(idx).tolist(), [3, 2, 2])
    # A prefix of the worlds already covers every asset.
    self.assertEqual(sorted(set(idx[:3].tolist())), [0, 1, 2])
    np.testing.assert_array_equal(multi_asset_device._assign([5.0], 3), 0)

  def test_mixed_primitive_and_mesh_assets(self):
    assets = [
        multi_asset_device.MultiAsset(spec=_boxes_asset()),
        multi_asset_device.MultiAsset(
            spec=_mesh_asset(0.1, (1, 0, 0, 1)), weight=2.0
        ),
    ]
    dev = multi_asset_device.MultiAssetDevice(
        name="obj_asset", body_name="obj", assets=assets
    )
    sim = simulator_module.Simulator(
        mj_spec=dev.configure_spec(_scene()),
        nworld=3,
        xpu_device="cpu",
        impl=types.Impl.WARP,
    )
    dev.initialize(sim)
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.FULL
    )
    sim.set_const()
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )
    mj = sim.mj_model
    bid = mj.body("obj").id

    # Placeholder replaced by 3 boxes + 1 cylinder + 1 mesh.
    with self.assertRaises(KeyError):
      mj.geom("placeholder")
    self.assertEqual(mj.body_geomnum[bid], 5)
    gids = mj.body_geomadr[bid] + np.arange(5)
    geom_types = [int(mj.geom_type[g]) for g in gids]
    self.assertCountEqual(geom_types, [_BOX] * 3 + [_CYLINDER, _MESH])
    unused_mesh = mj.mesh("obj_asset/unused").id

    # Worlds [1, 0, 1]: world 1 gets asset 0 (boxes), worlds 0 and 2 get asset 1
    # (mesh). Verify every _GEOM_FIELDS and _BODY_FIELDS entry against each
    # standalone compiled asset, and verify unused geom invariants.
    assert dev._asset_index is not None
    self.assertEqual(dev._asset_index.tolist(), [1, 0, 1])
    qvel = [0.1, -0.2, -0.5, 1.0, -2.0, 1.5]
    sim.data.qvel[:] = torch.tensor(qvel)
    sim.forward()
    for w, asset_idx in enumerate(dev._asset_index.tolist()):
      c = dev._compiled[asset_idx]
      ref = c.model
      for f in multi_asset_device._BODY_FIELDS:
        np.testing.assert_allclose(
            getattr(sim.model, f)[w, bid].numpy(),
            getattr(ref, f)[1],
            rtol=1e-5,
            atol=1e-6,
        )
      for gid, (key, k) in zip(gids, dev._layout, strict=True):
        geoms = c.geoms.get(key, ())
        if k < len(geoms):
          g = geoms[k]
          for f in multi_asset_device._GEOM_FIELDS:
            got = getattr(sim.model, f)[w, gid].numpy()
            want = getattr(ref, f)[g].reshape(got.shape)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
          if key[0] == _MESH:
            self.assertEqual(
                sim.model.geom_dataid[w, gid].item(),
                mj.mesh(f"obj_asset/{asset_idx}/tetra").id,
            )
            self.assertEqual(
                sim.model.geom_matid[w, gid].item(),
                mj.mat(f"obj_asset/{asset_idx}/mat").id,
            )
          else:
            self.assertEqual(sim.model.geom_dataid[w, gid].item(), -1)
            self.assertEqual(sim.model.geom_matid[w, gid].item(), -1)
        else:
          self.assertEqual(sim.model.geom_rgba[w, gid, 3].item(), 0.0)
          self.assertLess(sim.model.geom_rbound[w, gid].item(), 5e-3)
          self.assertEqual(
              sim.model.geom_margin[w, gid].item(),
              multi_asset_device._UNUSED_GEOM_MARGIN,
          )
          self.assertEqual(sim.model.geom_gap[w, gid].item(), 0.0)
          np.testing.assert_allclose(
              sim.model.geom_pos[w, gid].numpy(), ref.body_ipos[1], atol=1e-6
          )
          self.assertEqual(
              sim.model.geom_dataid[w, gid].item(),
              unused_mesh if key[0] == _MESH else -1,
          )
          self.assertEqual(sim.model.geom_matid[w, gid].item(), -1)
      # Unused geoms are inert and per-world runtime inertia + body.simple=False
      # give the same full mass matrix M and qacc as standalone compile.
      ref_spec = assets[asset_idx].spec.copy()
      ref_spec.worldbody.add_geom(
          type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
      )
      body = ref_spec.worldbody.first_body()
      body.pos = _OBJ_POS
      body.add_freejoint()
      ref_m = ref_spec.compile()
      ref_d = mujoco.MjData(ref_m)
      ref_d.qvel[:] = qvel
      mujoco.mj_forward(ref_m, ref_d)
      np.testing.assert_allclose(sim.data.qacc[w], ref_d.qacc, atol=1e-4)

    # set_const ran: subtree mass follows the written masses.
    np.testing.assert_allclose(
        sim.model.body_subtreemass[:, 0].numpy(),
        sim.model.body_mass[:, bid].numpy(),
        rtol=1e-5,
    )

  def test_unused_geoms_never_collide(self):
    scene = _scene()
    # A probe enclosing every asset's center of mass, where unused geoms sit.
    scene.worldbody.add_geom(
        name="probe",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.2, 0, 0],
        pos=_OBJ_POS,
    )
    assets = [
        multi_asset_device.MultiAsset(spec=_boxes_asset()),
        multi_asset_device.MultiAsset(spec=_mesh_asset(0.1, (1, 0, 0, 1))),
    ]
    sim = _build(scene, assets, nworld=2)
    sim.set_const()
    unused = (
        sim.model.geom_margin[:].numpy()
        == multi_asset_device._UNUSED_GEOM_MARGIN
    )
    self.assertTrue(unused.any())

    sim.forward()

    d: Any = sim.data.obj  # mjw.Data
    n = int(d.nacon.numpy()[0])
    worlds = d.contact.worldid.numpy()[:n]
    pairs = d.contact.geom.numpy()[:n]
    probe = sim.mj_model.geom("probe").id
    touching = []
    for w, (g1, g2) in zip(worlds, pairs):
      if g1 == probe:
        touching.append((w, g2))
      elif g2 == probe:
        touching.append((w, g1))
    # Used geoms touch the probe; unused ones never do.
    self.assertNotEmpty(touching)
    for w, g in touching:
      self.assertFalse(unused[w, g], (w, g))

  def test_shared_geom_fields(self):
    # Two primitive-only assets with different shared fields (contype, group)
    # get separate layout slots and do not allocate an unused mesh.
    contypes = [1, 2]
    groups = [0, 3]
    s0 = mujoco.MjSpec()
    s0.worldbody.add_body().add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        pos=[0.01, -0.02, 0.03],
        quat=[np.cos(0.3), np.sin(0.3), 0, 0],
        contype=contypes[0],
        group=groups[0],
    )
    s1 = mujoco.MjSpec()
    s1.worldbody.add_body().add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.04, 0.04, 0.04],
        pos=[-0.04, 0.05, 0.06],
        quat=[np.cos(0.5), 0, np.sin(0.5), 0],
        contype=contypes[1],
        group=groups[1],
    )
    assets = [
        multi_asset_device.MultiAsset(spec=s0),
        multi_asset_device.MultiAsset(spec=s1),
    ]
    sim = _build(_scene(), assets, nworld=2)
    sim.set_const()
    sim.forward()
    mj = sim.mj_model
    bid = mj.body("obj").id
    self.assertEqual(mj.body_simple[bid], 0)
    with self.assertRaises(KeyError):
      mj.mesh("obj_asset/unused")
    self.assertEqual(mj.body_geomnum[bid], 2)
    gids = mj.body_geomadr[bid] + np.arange(2)
    self.assertEqual(list(mj.geom_contype[gids]), contypes)
    self.assertEqual(list(mj.geom_group[gids]), groups)
    masses = sim.model.body_mass[:, bid].numpy()
    refs = [a.spec.copy().compile() for a in assets]
    np.testing.assert_allclose(
        masses, [r.body_mass[1] for r in refs], rtol=1e-5
    )
    for w, ref in enumerate(refs):
      ref_data = mujoco.MjData(ref)
      mujoco.mj_forward(ref, ref_data)
      np.testing.assert_allclose(
          sim.data.geom_xpos[w, gids[w]].numpy(),
          mj.body_pos[bid] + ref_data.geom_xpos[0],
          atol=1e-6,
      )
      np.testing.assert_allclose(
          sim.data.geom_xmat[w, gids[w]].numpy(),
          ref_data.geom_xmat[0].reshape(3, 3),
          atol=1e-6,
      )

  def test_resample_on_reset(self):
    assets = [
        multi_asset_device.MultiAsset(spec=_mesh_asset(0.1, (1, 0, 0, 1))),
        multi_asset_device.MultiAsset(
            spec=_mesh_asset(0.2, (0, 1, 0, 1)), weight=4
        ),
    ]
    dev = multi_asset_device.MultiAssetDevice(
        name="obj_asset", body_name="obj", assets=assets, resample_on_reset=True
    )
    sim = simulator_module.Simulator(
        mj_spec=dev.configure_spec(_scene()),
        nworld=64,
        xpu_device="cpu",
        impl=types.Impl.WARP,
    )
    dev.initialize(sim)
    sim.set_const()
    bid = sim.mj_model.body("obj").id
    ref = torch.tensor(
        [a.spec.copy().compile().body_mass[1] for a in assets],
        dtype=torch.float32,
    )
    assert dev._asset_index is not None
    startup = dev._asset_index.clone()
    # Domain randomization scaled the masses after the device wrote them.
    scaled = sim.model.body_mass[:, bid] * 2.0
    sim.model.body_mass[:, bid] = scaled
    sim.set_const()

    torch.manual_seed(0)
    env_mask = torch.arange(64) % 2 == 0
    dev.reset(sim, env_mask)
    # Reset worlds get a fresh asset, drawn by weight; others are untouched.
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.FULL
    )
    mass = sim.model.body_mass[:, bid]
    torch.testing.assert_close(mass[env_mask], ref[dev._asset_index[env_mask]])
    torch.testing.assert_close(mass[~env_mask], scaled[~env_mask])
    torch.testing.assert_close(dev._asset_index[~env_mask], startup[~env_mask])
    self.assertGreater((dev._asset_index[env_mask] == 1).float().mean(), 0.5)
    sim.set_const()

    dev.reset(sim)
    torch.testing.assert_close(
        sim.model.body_mass[:, bid], ref[dev._asset_index]
    )

    # Without resample_on_reset, reset is a no-op.
    sim.set_const()
    before = sim.model.body_mass[:, bid].clone()
    dev._resample_on_reset = False
    dev.reset(sim)
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )
    torch.testing.assert_close(sim.model.body_mass[:, bid], before)

  def test_mesh_files_resolve_against_each_asset(self):
    mesh_dir = "meshes"
    mesh_file = "tetra.obj"

    def make_mesh_spec(
        modelfiledir: str = "",
        meshdir: str = "",
        assets: dict[str, bytes] | None = None,
    ) -> mujoco.MjSpec:
      spec = mujoco.MjSpec()
      spec.modelfiledir = modelfiledir
      spec.meshdir = meshdir
      if assets is not None:
        spec.assets = assets
      spec.add_mesh(name="tetra", file=mesh_file)
      spec.worldbody.add_body().add_geom(
          type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tetra"
      )
      return spec

    with tempfile.TemporaryDirectory() as tmpdir:
      os.makedirs(os.path.join(tmpdir, mesh_dir))
      with open(os.path.join(tmpdir, mesh_dir, mesh_file), "w") as f:
        f.write(_tetra_obj(0.1))
      specs = [
          make_mesh_spec(modelfiledir=tmpdir, meshdir=mesh_dir),
          make_mesh_spec(assets={mesh_file: _tetra_obj(0.2).encode()}),
          make_mesh_spec(
              meshdir=mesh_dir, assets={mesh_file: _tetra_obj(0.3).encode()}
          ),
      ]
      scene = _scene()
      scene.meshdir = "unrelated_dir"
      assets = [multi_asset_device.MultiAsset(spec=s) for s in specs]
      mj = _build(scene, assets, nworld=len(specs)).mj_model
      for i, s in enumerate(specs):
        ref = s.copy().compile()
        mid = mj.mesh(f"obj_asset/{i}/tetra").id
        vadr = mj.mesh_vertadr[mid]
        vnum = mj.mesh_vertnum[mid]
        np.testing.assert_allclose(
            mj.mesh_vert[vadr : vadr + vnum], ref.mesh_vert
        )

  def test_validation_errors(self):
    with self.assertRaisesRegex(ValueError, "weight must be > 0"):
      multi_asset_device.MultiAsset(spec=_boxes_asset(), weight=0.0)

    def make(assets, scene=None, body="obj", cpp=False):
      impl = types.Impl.CPP if cpp else types.Impl.WARP
      return _build(
          scene or _scene(), assets, nworld=1, body_name=body, impl=impl
      )

    with self.assertRaisesRegex(ValueError, "at least one asset"):
      make([])

    jointed = _boxes_asset()
    jointed.body("mug").add_freejoint()
    two = _boxes_asset()
    two.worldbody.add_body().add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
    )
    child_body = _boxes_asset()
    child_body.body("mug").add_body().add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
    )
    with_site = _boxes_asset()
    with_site.body("mug").add_site()
    world_geom = _boxes_asset()
    world_geom.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0]
    )
    empty = mujoco.MjSpec()
    empty.worldbody.add_body()
    plane = _boxes_asset()
    plane.body("mug").add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1])
    massless = mujoco.MjSpec()
    massless.worldbody.add_body().add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], density=0
    )

    for bad_spec, pattern in (
        (jointed, "must not have joints"),
        (two, "exactly one body"),
        (child_body, "exactly one body"),
        (with_site, "must not have sites"),
        (world_geom, "must not have geoms on its worldbody"),
        (empty, "at least one geom"),
        (plane, "unsupported type"),
        (massless, "has no mass"),
    ):
      with self.assertRaisesRegex(ValueError, pattern):
        make([multi_asset_device.MultiAsset(spec=bad_spec)])

    asset = [multi_asset_device.MultiAsset(spec=_boxes_asset())]
    with self.assertRaisesRegex(ValueError, "not found"):
      make(asset, body="missing")
    with self.assertRaisesRegex(ValueError, "worldbody"):
      make(asset, body="world")
    with self.assertRaisesRegex(ValueError, "static bodies"):
      make(asset, scene=_scene(joint=False))
    with self.assertRaisesRegex(ValueError, "Impl.WARP"):
      make(asset, cpp=True)

  def test_geom_fields_match_mjw_model(self):
    mj_fields = set(dir(mujoco.MjModel))
    mjw_geom = {
        f.name: getattr(f.type, "shape")
        for f in dataclasses.fields(mjw.Model)
        if f.name.startswith("geom_") and f.name in mj_fields
    }
    shared = {k for k, s in mjw_geom.items() if s[0] == "ngeom"}
    batched = {k for k, s in mjw_geom.items() if s[:2] == ("*", "ngeom")}
    self.assertEqual(shared | batched, set(mjw_geom))
    self.assertEqual(
        shared,
        {f"geom_{f}" for f in multi_asset_device._SHARED_GEOM_FIELDS}
        | {"geom_bodyid", "geom_fluid"},
    )
    self.assertEqual(
        batched,
        set(multi_asset_device._GEOM_FIELDS) | {"geom_dataid", "geom_matid"},
    )


if __name__ == "__main__":
  absltest.main()
