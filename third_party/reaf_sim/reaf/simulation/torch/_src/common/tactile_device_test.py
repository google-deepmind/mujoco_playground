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

"""Unit tests for TactileDevice."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from reaf.simulation.torch._src.common import tactile_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


def _make_sphere_floor_spec() -> mujoco.MjSpec:
  """Creates a sphere mesh on a free body penetrating a floor plane."""
  spec = mujoco.MjSpec()
  mesh = spec.add_mesh(name="sphere_mesh")
  mesh.make_sphere(0)
  spec.worldbody.add_geom(
      name="floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[1.0, 1.0, 1.0],
  )
  ball = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 0.8])
  ball.add_freejoint(name="ball_joint")
  ball.add_geom(
      name="ball_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="sphere_mesh",
  )
  return spec


def _make_mesh_contact_spec(
    mesh_type: str = "plate",
    resolution: tuple[int, int] = (4, 4),
) -> mujoco.MjSpec:
  """Creates a scene with a tactile mesh geom touching a colliding mesh."""
  spec = mujoco.MjSpec()
  sensor_mesh = spec.add_mesh(name="sensor_mesh")
  if mesh_type == "plate":
    sensor_mesh.make_plate(list(resolution))
  else:
    sensor_mesh.make_wedge(
        resolution=list(resolution), fov=[45.0, 45.0], gamma=0.0
    )

  collide_mesh = spec.add_mesh(name="collide_mesh")
  collide_mesh.make_sphere(1)

  body = spec.worldbody.add_body(name="sensor_body")
  body.add_geom(
      name="col_box",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[0.5, 0.5, 0.5],
  )
  body.add_geom(
      name="sensor_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="sensor_mesh",
      contype=0,
      conaffinity=0,
      mass=0.0,
  )

  collider = spec.worldbody.add_body(name="collider", pos=[0.0, 0.0, -0.8])
  collider.add_freejoint(name="collider_joint")
  collider.add_geom(
      name="collider_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="collide_mesh",
  )
  return spec


class TactileDeviceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, 1, "cpu"),
      (
          "warp",
          types.Impl.WARP,
          2,
          "cuda:0" if torch.cuda.is_available() else "cpu",
      ),
  )
  def test_sphere_floor_penetration_and_cutoff(
      self, impl: types.Impl, nworld: int, xpu_device: str
  ) -> None:
    # Configure a tactile sensor on the icosahedron sphere mesh geom.
    spec = _make_sphere_floor_spec()
    dev = tactile_device.TactileDevice(name="tactile")
    dev.configure_spec(spec, geom="ball_geom")

    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=nworld, xpu_device=xpu_device, impl=impl
    )
    dev.initialize(sim)
    sim.forward()

    # Verify device specs, no-op command/reset hooks, and analytical depth.
    self.assertEqual(dev.name, "tactile")
    self.assertEqual(dev.substep_commands_spec(), {})
    dev.set_substep_commands(sim, {})
    dev.reset(
        sim,
        env_mask=torch.ones((nworld,), dtype=torch.bool, device=xpu_device),
    )
    self.assertEqual(
        dev.substep_measurements_spec()["tactile/depth"].shape, (12,)
    )
    depth = dev.get_substep_measurements(sim)["tactile/depth"]
    self.assertEqual(depth.shape, (nworld, 12))
    torch.testing.assert_close(
        depth.float().amax(dim=-1).cpu(),
        torch.full((nworld,), 0.0506508, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )

    # Verify positive sensor cutoff clamps raw penetration depths in the engine.
    spec_cutoff = _make_sphere_floor_spec()
    dev_cutoff = tactile_device.TactileDevice(name="tactile")
    dev_cutoff.configure_spec(spec_cutoff, geom="ball_geom", cutoff=0.02)
    sim_cutoff = simulator_module.Simulator(
        mj_spec=spec_cutoff, nworld=nworld, xpu_device=xpu_device, impl=impl
    )
    dev_cutoff.initialize(sim_cutoff)
    sim_cutoff.forward()
    depth_cutoff = dev_cutoff.get_substep_measurements(sim_cutoff)[
        "tactile/depth"
    ]
    torch.testing.assert_close(
        depth_cutoff.float().amax(dim=-1).cpu(),
        torch.full((nworld,), 0.02, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )

  @parameterized.named_parameters(
      ("plate", "plate", (3, 4)),
      ("wedge", "wedge", (3, 3)),
  )
  def test_builtin_mesh_parity_and_needsdf(
      self, builtin: str, expected_res: tuple[int, int]
  ) -> None:
    # Build a scene with a built-in tactile mesh touching a colliding mesh geom.
    spec = _make_mesh_contact_spec(mesh_type=builtin, resolution=expected_res)
    dev_cpp = tactile_device.TactileDevice(name="pad")
    dev_cpp.configure_spec(spec, geom="sensor_geom")

    # Only colliding mesh geoms have needsdf enabled.
    self.assertTrue(spec.mesh("collide_mesh").needsdf)
    self.assertFalse(spec.mesh("sensor_mesh").needsdf)

    # Evaluate on CPU MuJoCo C backend.
    sim_cpp = simulator_module.Simulator(
        mj_spec=spec, nworld=1, xpu_device="cpu", impl=types.Impl.CPP
    )
    dev_cpp.initialize(sim_cpp)
    sim_cpp.forward()
    ntaxel = expected_res[0] * expected_res[1]
    depth_cpp = dev_cpp.get_substep_measurements(sim_cpp)["pad/depth"]
    self.assertEqual(depth_cpp.shape, (1, ntaxel))
    self.assertEqual(
        dev_cpp.substep_measurements_spec()["pad/depth"].shape, (ntaxel,)
    )
    self.assertGreater(float(depth_cpp.max()), 0.0)

    # Evaluate on MuJoCo Warp backend and verify parity across worlds.
    dev_warp = tactile_device.TactileDevice(name="pad")
    sim_warp = simulator_module.Simulator(
        mj_spec=spec,
        nworld=2,
        xpu_device="cuda:0" if torch.cuda.is_available() else "cpu",
        impl=types.Impl.WARP,
    )
    dev_warp.initialize(sim_warp)
    sim_warp.forward()
    depth_warp = dev_warp.get_substep_measurements(sim_warp)["pad/depth"].cpu()

    torch.testing.assert_close(
        depth_cpp[0].float(), depth_warp[0], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        depth_cpp[0].float(), depth_warp[1], atol=1e-4, rtol=1e-4
    )

  def test_explicit_mesh_on_primitive_geom_and_multiple_sensors(self) -> None:
    # Attach two tactile sensors in the same spec to verify non-zero sensor adr.
    spec = _make_sphere_floor_spec()
    box = spec.worldbody.add_body(name="box", pos=[2.0, 0.0, 0.8])
    box.add_freejoint()
    box.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[1.0, 1.0, 1.0],
    )

    dev_ball = tactile_device.TactileDevice(name="ball_tactile")
    dev_ball.configure_spec(spec, geom="ball_geom")
    dev_box = tactile_device.TactileDevice(name="box_tactile")
    dev_box.configure_spec(spec, geom="box_geom", mesh="sphere_mesh")

    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=1, xpu_device="cpu", impl=types.Impl.CPP
    )
    dev_ball.initialize(sim)
    dev_box.initialize(sim)
    sim.forward()

    # Second sensor starts after the first sensor's 3 * ntaxel block (adr=36).
    self.assertEqual(dev_ball._sensor_slice, slice(0, 12))
    self.assertEqual(dev_box._sensor_slice, slice(36, 48))
    depth_ball = dev_ball.get_substep_measurements(sim)["ball_tactile/depth"]
    depth_box = dev_box.get_substep_measurements(sim)["box_tactile/depth"]
    torch.testing.assert_close(depth_ball, depth_box)

  def test_multiworld_contact_and_dark_mode(self) -> None:
    spec = _make_sphere_floor_spec()
    dev = tactile_device.TactileDevice(name="tactile")
    dev.configure_spec(spec, geom="ball_geom")
    dev_dark = tactile_device.TactileDevice(name="dark_tactile", dark=True)
    dev_dark.configure_spec(spec, geom="ball_geom")

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=2,
        xpu_device="cuda:0" if torch.cuda.is_available() else "cpu",
        impl=types.Impl.WARP,
    )
    dev.initialize(sim)
    dev_dark.initialize(sim)

    # Lift world 1 out of contact while keeping world 0 in contact.
    sim.data.qpos[1, 2] = 5.0
    sim.forward()
    depth = dev.get_substep_measurements(sim)["tactile/depth"]
    self.assertAlmostEqual(float(depth[0].max()), 0.0506508, places=5)
    self.assertEqual(float(depth[1].abs().max()), 0.0)

    # Dark devices hide measurements unless force=True is requested.
    self.assertEqual(dev_dark.substep_measurements_spec(), {})
    self.assertEqual(dev_dark.get_substep_measurements(sim), {})
    forced = dev_dark.get_substep_measurements(sim, force=True)[
        "dark_tactile/depth"
    ]
    self.assertEqual(
        forced.data_ptr(),
        sim.data.sensordata[:, dev_dark._sensor_slice].data_ptr(),
    )

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, 1, "cpu"),
      (
          "warp",
          types.Impl.WARP,
          2,
          "cuda:0" if torch.cuda.is_available() else "cpu",
      ),
  )
  def test_postprocess_laplacian_solve_and_cutoff(
      self, impl: types.Impl, nworld: int, xpu_device: str
  ) -> None:
    cutoff_val = 0.01
    spec = _make_mesh_contact_spec(mesh_type="plate", resolution=(4, 4))
    raw_dev = tactile_device.TactileDevice(name="raw", postprocess=False)
    raw_dev.configure_spec(spec, geom="sensor_geom")
    post_dev = tactile_device.TactileDevice(name="post", postprocess=True)
    post_dev.configure_spec(spec, geom="sensor_geom")
    cut_dev = tactile_device.TactileDevice(name="cut", postprocess=True)
    cut_dev.configure_spec(spec, geom="sensor_geom", cutoff=cutoff_val)

    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=nworld, xpu_device=xpu_device, impl=impl
    )
    raw_dev.initialize(sim)
    post_dev.initialize(sim)
    cut_dev.initialize(sim)
    sim.forward()

    d = raw_dev.get_substep_measurements(sim)["raw/depth"][0].float().cpu()
    phi_batch = (
        post_dev.get_substep_measurements(sim)["post/depth"].float().cpu()
    )
    self.assertEqual(phi_batch.shape, (nworld, 16))
    phi = phi_batch[0]

    # Boundary vertices of the 4x4 grid are clamped to zero.
    for b in (0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15):
      self.assertEqual(float(phi[b]), 0.0)

    # Interior vertices satisfy 0 < phi_i < d_i and discrete Poisson equation.
    for i in (5, 6, 9, 10):
      self.assertGreater(float(phi[i]), 0.0)
      self.assertLess(float(phi[i]), float(d[i]))
    self.assertAlmostEqual(
        float(4.0 * phi[5] - phi[6] - phi[9]), float(d[5]), places=5
    )

    # Cutoff clamps the postprocessed output rather than raw inputs.
    phi_cut = cut_dev.get_substep_measurements(sim)["cut/depth"].float().cpu()
    torch.testing.assert_close(
        phi_cut.amax(dim=-1),
        torch.full((nworld,), cutoff_val, dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )

  def test_postprocess_invalid_mesh_raises(self) -> None:
    # Closed sphere mesh has no boundary edges and raises ValueError.
    spec = _make_sphere_floor_spec()
    dev = tactile_device.TactileDevice(name="tactile", postprocess=True)
    dev.configure_spec(spec, geom="ball_geom")
    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=1, xpu_device="cpu", impl=types.Impl.CPP
    )
    with self.assertRaisesRegex(
        ValueError, "Laplacian postprocessing requires an open triangle mesh"
    ):
      dev.initialize(sim)

  @parameterized.named_parameters(
      (
          "missing_geom",
          "missing",
          "sphere_mesh",
          0.0,
          "unrecognized name 'missing' of object",
      ),
      (
          "missing_mesh",
          "floor",
          "",
          0.0,
          "missing name of sensorized object in sensor",
      ),
      (
          "bad_mesh",
          "ball_geom",
          "missing_mesh",
          0.0,
          "unrecognized name 'missing_mesh' of sensorized object",
      ),
      (
          "negative_cutoff",
          "ball_geom",
          "",
          -0.1,
          "negative cutoff in sensor",
      ),
  )
  def test_configure_spec_validation_errors(
      self,
      geom: str,
      mesh: str,
      cutoff: float,
      error_regex: str,
  ) -> None:
    spec = _make_sphere_floor_spec()
    dev = tactile_device.TactileDevice(name="tactile")
    dev.configure_spec(spec, geom=geom, mesh=mesh, cutoff=cutoff)
    with self.assertRaisesRegex(ValueError, error_regex):
      spec.compile()


if __name__ == "__main__":
  absltest.main()
