# Copyright 2026 Google LLC
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


"""Tests for camera_device."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from reaf.simulation.torch._src.common import camera_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_PENDULUM = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <camera name="cam" pos="0 -2 0" xyaxes="1 0 0 0 0 1"/>
    <camera name="cam2" pos="0 -2 0" xyaxes="1 0 0 0 0 1"/>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.04"/>
    </body>
  </worldbody>
</mujoco>
"""

_TWO_CAMERAS = """
<mujoco>
  <visual>
    <headlight ambient="0.8 0.8 0.8" diffuse="0.2 0.2 0.2"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 5"/>
    <camera name="cam_red" pos="0 -2 0" xyaxes="1 0 0 0 0 1"/>
    <camera name="cam_green" pos="2 0 0" xyaxes="0 1 0 0 0 1"/>
    <geom name="red_box" type="box" pos="0 -1 0" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    <geom name="green_box" type="box" pos="1 0 0" size="0.2 0.2 0.2" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""


class CameraDeviceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rgb_and_depth", True, True),
      ("rgb_only", True, False),
      ("depth_only", False, True),
  )
  def test_camera_entity_renders(self, render_rgb, render_depth):
    mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    nworld = 1
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    width, height = 320, 240
    width2, height2 = 160, 120
    cam_entity = camera_device.CameraDevice(
        name="cam",
        cam_resolution=[(width, height), (width2, height2)],
        render_rate_hz=500,
        render_rgb=[render_rgb, render_rgb],
        render_depth=[render_depth, render_depth],
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
    )
    cam_entity.initialize(sim)
    cam_entity.reset(sim)
    measurements = cam_entity.get_substep_measurements(sim)
    if render_rgb:
      self.assertIn("cam_rgb", measurements)
      self.assertEqual(
          measurements["cam_rgb"].shape, (nworld, height, width, 3)
      )
      self.assertEqual(measurements["cam_rgb"].dtype, torch.uint8)
      self.assertIn("cam2_rgb", measurements)
      self.assertEqual(
          measurements["cam2_rgb"].shape, (nworld, height2, width2, 3)
      )
      self.assertEqual(measurements["cam2_rgb"].dtype, torch.uint8)
    else:
      self.assertNotIn("cam_rgb", measurements)
      self.assertNotIn("cam2_rgb", measurements)
    if render_depth:
      self.assertIn("cam_depth", measurements)
      self.assertEqual(measurements["cam_depth"].shape, (nworld, height, width))
      self.assertEqual(measurements["cam_depth"].dtype, torch.uint8)
      self.assertIn("cam2_depth", measurements)
      self.assertEqual(
          measurements["cam2_depth"].shape, (nworld, height2, width2)
      )
      self.assertEqual(measurements["cam2_depth"].dtype, torch.uint8)
    else:
      self.assertNotIn("cam_depth", measurements)
      self.assertNotIn("cam2_depth", measurements)

  @parameterized.named_parameters(
      ("unnormalized_float32", None, torch.float32, False),
      ("normalized_float32", 5.0, torch.float32, True),
  )
  def test_camera_entity_depth_scale_and_dtype(
      self, depth_scale, depth_dtype, is_normalized
  ):
    mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    nworld = 1
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    width, height = 320, 240
    cam_entity = camera_device.CameraDevice(
        name="cam",
        cam_name="cam",
        cam_resolution=[(width, height)],
        render_rate_hz=500,
        render_rgb=[False],
        render_depth=[True],
        depth_scale=depth_scale,
        depth_dtype=depth_dtype,
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
    )
    cam_entity.initialize(sim)
    cam_entity.reset(sim)

    # Verify measurement specs are created for depth.
    specs_dict = cam_entity.substep_measurements_spec()
    self.assertIn("cam_depth", specs_dict)
    self.assertEqual(specs_dict["cam_depth"].shape, (height, width))

    # Verify measurements match expected dtype and scaling.
    measurements = cam_entity.get_substep_measurements(sim)
    self.assertIn("cam_depth", measurements)
    depth = measurements["cam_depth"]
    self.assertEqual(depth.shape, (nworld, height, width))
    self.assertEqual(depth.dtype, depth_dtype)
    if is_normalized:
      self.assertLessEqual(depth.max().item(), 255.0)
    self.assertGreater(depth.max().item(), 1.0)

  def test_mjwarp_model_tiling_invalidates_render_graphs(self):
    mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    nworld = 2
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    model = sim.model
    data = sim.data

    width, height = 32, 32
    cam_entity = camera_device.CameraDevice(
        name="cam_entity",
        cam_name="cam",
        cam_resolution=[(width, height)],
        render_rate_hz=500,
        render_rgb=[True],
        render_depth=[False],
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
    )
    cam_entity.initialize(sim)

    # Verify initial render graph was captured.
    self.assertIsNotNone(
        cam_entity._render_graph,
        "Render graph should be captured during initialization.",
    )
    original_render_graph = cam_entity._render_graph

    # Trigger on-demand tiling of a TorchWrapper Model field and check graph is
    # invalidated.
    _ = model.geom_pos
    self.assertIsNone(
        cam_entity._render_graph,
        "Render graph should be cleared after TorchWrapper reallocates memory.",
    )

    # Re-trigger graph capture.
    cam_entity.render(model, data)
    self.assertIsNotNone(
        cam_entity._render_graph,
        "Render graph should be recaptured during the next render call.",
    )
    self.assertNotEqual(
        cam_entity._render_graph,
        original_render_graph,
        "New render graph should be captured after memory reallocation.",
    )

  def test_camera_entity_reverse_order_renders_correct_views(self):
    mj_spec = mujoco.MjSpec.from_string(_TWO_CAMERAS)
    nworld = 1
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )

    # cam_green is requested first, cam_red second (reverse of mj_spec order)
    width_g, height_g = 160, 120
    width_r, height_r = 320, 240
    cam_entity = camera_device.CameraDevice(
        name="cam_entity",
        cam_name=["cam_green", "cam_red"],
        cam_resolution=[(width_g, height_g), (width_r, height_r)],
        render_rate_hz=500,
        render_rgb=[True, True],
        render_depth=[True, True],
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
    )
    cam_entity.initialize(sim)
    cam_entity.reset(sim)
    measurements = cam_entity.get_substep_measurements(sim)

    self.assertIn("cam_green_rgb", measurements)
    self.assertIn("cam_red_rgb", measurements)
    self.assertIn("cam_green_depth", measurements)
    self.assertIn("cam_red_depth", measurements)

    green_img = measurements["cam_green_rgb"]
    red_img = measurements["cam_red_rgb"]
    green_depth = measurements["cam_green_depth"]
    red_depth = measurements["cam_red_depth"]

    # Check shapes match requested resolutions for each camera
    self.assertEqual(green_img.shape, (nworld, height_g, width_g, 3))
    self.assertEqual(red_img.shape, (nworld, height_r, width_r, 3))
    self.assertEqual(green_depth.shape, (nworld, height_g, width_g))
    self.assertEqual(red_depth.shape, (nworld, height_r, width_r))

    # Center pixel of cam_green should see the green box (green channel > red channel)
    center_g = green_img[0, height_g // 2, width_g // 2]
    self.assertGreater(
        center_g[1].item(),
        center_g[0].item(),
        f"cam_green expected green > red, got RGB={center_g.tolist()}",
    )

    # Center pixel of cam_red should see the red box (red channel > green channel)
    center_r = red_img[0, height_r // 2, width_r // 2]
    self.assertGreater(
        center_r[0].item(),
        center_r[1].item(),
        f"cam_red expected red > green, got RGB={center_r.tolist()}",
    )

  def test_update_interval_via_render_rate_hz(self):
    mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    width, height = 32, 32
    # Pendulum default timestep is 0.002s (500 Hz).
    # 250 Hz render rate -> interval of 2 substeps.
    cam_entity = camera_device.CameraDevice(
        name="cam_entity",
        cam_name="cam",
        cam_resolution=[(width, height)],
        render_rate_hz=250,
        render_rgb=[True],
        render_depth=[False],
    )
    cam_entity.initialize(sim)
    cam_entity.reset(sim)

    # Substep 0: renders and returns measurements
    m0 = cam_entity.get_substep_measurements(sim)
    self.assertIn("cam_rgb", m0)

    # Calling again on substep 0 is idempotent: returns cached measurements
    m0_repeat = cam_entity.get_substep_measurements(sim)
    self.assertIs(m0["cam_rgb"], m0_repeat["cam_rgb"])

    # Substep 1: does not re-render, returns cached measurements
    sim.step()
    m1 = cam_entity.get_substep_measurements(sim)
    self.assertIs(m0["cam_rgb"], m1["cam_rgb"])

    # Substep 2: renders again, returns fresh measurements
    sim.step()
    m2 = cam_entity.get_substep_measurements(sim)
    self.assertIsNot(m0["cam_rgb"], m2["cam_rgb"])

  def test_render_rate_hz(self):
    mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    # Pendulum default timestep is 0.002s (500 Hz).
    # 50 Hz render rate -> interval of 10 substeps.
    cam_entity = camera_device.CameraDevice(
        name="cam_hz",
        cam_name="cam",
        cam_resolution=[(32, 32)],
        render_rate_hz=50,
    )
    cam_entity.initialize(sim)
    self.assertEqual(cam_entity._update_interval_in_substeps, 10)

  def test_render_rate_hz_validation(self):
    with self.assertRaises(ValueError):
      camera_device.CameraDevice(
          name="cam",
          cam_resolution=[(32, 32)],
          render_rate_hz=0.0,
      )
    with self.assertRaises(ValueError):
      camera_device.CameraDevice(
          name="cam",
          cam_resolution=[(32, 32)],
          render_rate_hz=-10.0,
      )

  def test_reset_with_env_mask_zeros_buffer(self):
    mj_spec = mujoco.MjSpec.from_string(_TWO_CAMERAS)
    nworld = 2
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=_DEVICE,
        impl=types.Impl.WARP,
    )
    width, height = 32, 32
    # 250 Hz render rate -> interval of 2 substeps (sim_dt = 0.002s).
    cam_entity = camera_device.CameraDevice(
        name="cam_entity",
        cam_name="cam_red",
        cam_resolution=[(width, height)],
        render_rate_hz=250,
        render_rgb=[True],
        render_depth=[True],
    )
    cam_entity.initialize(sim)
    cam_entity.reset(sim)

    # Substep 0: renders both worlds (both non-zero).
    m0 = cam_entity.get_substep_measurements(sim)
    self.assertGreater(m0["cam_red_rgb"][0].sum().item(), 0)
    self.assertGreater(m0["cam_red_rgb"][1].sum().item(), 0)
    self.assertGreater(m0["cam_red_depth"][0].sum().item(), 0)
    self.assertGreater(m0["cam_red_depth"][1].sum().item(), 0)
    world0_rgb_before = m0["cam_red_rgb"][0].clone()
    world0_depth_before = m0["cam_red_depth"][0].clone()

    # Step once to global_substep=1 (mid-interval) and reset only world 1.
    sim.step()
    env_mask = torch.tensor([False, True], device=_DEVICE)
    cam_entity.reset(sim, env_mask=env_mask)

    m1 = cam_entity.get_substep_measurements(sim)
    # World 1's buffer should be zeroed out; world 0's frame is preserved.
    self.assertEqual(m1["cam_red_rgb"][1].sum().item(), 0)
    self.assertEqual(m1["cam_red_depth"][1].sum().item(), 0)
    torch.testing.assert_close(m1["cam_red_rgb"][0], world0_rgb_before)
    torch.testing.assert_close(m1["cam_red_depth"][0], world0_depth_before)

    # Step to global_substep=2 (render tick): both worlds are re-rendered.
    sim.step()
    m2 = cam_entity.get_substep_measurements(sim)
    self.assertGreater(m2["cam_red_rgb"][1].sum().item(), 0)
    self.assertGreater(m2["cam_red_depth"][1].sum().item(), 0)


class CamNamesToMaskTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mj_spec = mujoco.MjSpec.from_string(_PENDULUM)
    self.ncam = len(self.mj_spec.cameras)
    self.assertEqual(self.ncam, 2)

  def test_none_selects_all(self):
    names, cam_active = camera_device._cam_names_to_mask(
        self.mj_spec, cam_name=None
    )
    self.assertSequenceEqual(names, ["cam", "cam2"])
    self.assertSequenceEqual(cam_active, [True, True])

  def test_single_camera_str(self):
    names, cam_active = camera_device._cam_names_to_mask(
        self.mj_spec, cam_name="cam"
    )
    self.assertSequenceEqual(names, ["cam"])
    self.assertSequenceEqual(cam_active, [True, False])

  def test_camera_list(self):
    names, cam_active = camera_device._cam_names_to_mask(
        self.mj_spec, cam_name=["cam2"]
    )
    self.assertSequenceEqual(names, ["cam2"])
    self.assertSequenceEqual(cam_active, [False, True])

  def test_both_cameras(self):
    names, cam_active = camera_device._cam_names_to_mask(
        self.mj_spec, cam_name=["cam", "cam2"]
    )
    self.assertSequenceEqual(names, ["cam", "cam2"])
    self.assertSequenceEqual(cam_active, [True, True])

  def test_reverse_order_normalizes_to_spec_order(self):
    names, cam_active = camera_device._cam_names_to_mask(
        self.mj_spec, cam_name=["cam2", "cam"]
    )
    self.assertSequenceEqual(names, ["cam", "cam2"])
    self.assertSequenceEqual(cam_active, [True, True])

  def test_unknown_camera_name_raises(self):
    with self.assertRaises(ValueError):
      camera_device._cam_names_to_mask(self.mj_spec, cam_name=["cam_unknown"])

  def test_duplicate_camera_name_raises(self):
    with self.assertRaises(ValueError):
      camera_device._cam_names_to_mask(self.mj_spec, cam_name=["cam", "cam"])


if __name__ == "__main__":
  absltest.main()
