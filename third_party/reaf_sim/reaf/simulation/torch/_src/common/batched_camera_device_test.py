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


"""Tests for BatchedMujocoCameraDevice."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import batched_camera_device
from reaf.simulation.torch._src.common import mujoco_camera_device
from reaf.simulation.torch._src.core import device_layer
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


_SIMPLE_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <body pos="0 0 0.1">
      <joint name="slide" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
    <camera name="test_cam" pos="0.5 0 0.3" xyaxes="0 1 0 -0.3 0 1"/>
    <camera name="top_cam" pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


class BatchedMujocoCameraDeviceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._cameras_to_close = []

  def tearDown(self):
    for camera in self._cameras_to_close:
      if camera._renderer is not None:
        camera._renderer.close()
    super().tearDown()

  def _create_camera(self, **kwargs):
    kwargs.setdefault("name", "test_camera")
    kwargs.setdefault(
        "camera_configs", [{"camera_name": "test_cam", "output_name": "test"}]
    )
    kwargs.setdefault("width", 64)
    kwargs.setdefault("height", 64)
    camera = batched_camera_device.BatchedMujocoCameraDevice(**kwargs)
    self._cameras_to_close.append(camera)
    return camera

  def _create_device_layer(self, sim, cameras):
    """Drives `cameras` like an environment, one physics substep per step."""
    return device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=sim.mj_model.opt.timestep,
        devices=cameras,
    )

  @parameterized.parameters(
      (types.Impl.CPP,),
      (types.Impl.WARP,),
  )
  def test_camera_entity_renders(self, impl):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("Warp camera requires CUDA.")

    xpu_device = _DEVICE if impl == types.Impl.WARP else "cpu"
    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        xpu_device=xpu_device,
        impl=impl,
    )
    sim.forward()

    camera = self._create_camera(render_rate_hz=500)
    with self.assertRaisesRegex(RuntimeError, "Device not initialized"):
      camera.render_now(sim)
    dl = self._create_device_layer(sim, [camera])
    dl.begin_stepping()

    measurements = camera.get_measurements(sim)

    self.assertIn("test_rgb", measurements)
    img = measurements["test_rgb"]
    self.assertEqual(img.shape, (1, 64, 64, 3))
    self.assertEqual(img.dtype, torch.uint8)
    self.assertEqual(img.device, torch.device(xpu_device))
    self.assertGreater(img.sum(), 0)
    if impl == types.Impl.WARP:
      with self.assertRaisesRegex(ValueError, "only supports Impl.CPP"):
        camera.render_now(sim)
    else:
      with self.assertRaisesRegex(ValueError, "not enabled on this device"):
        camera.render_now(
            sim, render_mode=mujoco_camera_device.RenderMode.DEPTH
        )

  def test_update_interval_via_render_rate_hz(self):
    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        impl=types.Impl.CPP,
    )
    sim.forward()

    # Default timestep is 0.002s (500 Hz). 250 Hz -> interval of 2 substeps.
    camera = self._create_camera(render_rate_hz=250)
    # None: renders only the first measurement after each reset.
    reset_only_camera = self._create_camera(
        name="reset_only_camera",
        camera_configs=[
            {"camera_name": "test_cam", "output_name": "reset_only"}
        ],
        render_rate_hz=None,
    )
    # On-demand only: publishes no measurements and never renders by itself.
    on_demand_camera = self._create_camera(
        name="on_demand_camera", render_rate_hz=None, on_demand_only=True
    )
    self.assertEqual(on_demand_camera.substep_measurements_spec(), {})
    dl = self._create_device_layer(
        sim, [camera, reset_only_camera, on_demand_camera]
    )

    # Reads do not render: nothing is cached until `begin_stepping()`.
    assert camera._renderer is not None
    with mock.patch.object(
        camera._renderer, "render", wraps=camera._renderer.render
    ) as read_spy:
      with self.assertRaisesRegex(RuntimeError, "nothing was rendered"):
        camera.get_measurements(sim)
      self.assertEqual(read_spy.call_count, 0)

    # Substep 0: `begin_stepping()` renders the first frame of each camera,
    # except the on-demand one.
    assert on_demand_camera._renderer is not None
    with mock.patch.object(
        on_demand_camera._renderer,
        "render",
        wraps=on_demand_camera._renderer.render,
    ) as on_demand_spy:
      dl.begin_stepping()
      self.assertEqual(on_demand_camera.get_measurements(sim), {})
      self.assertEqual(on_demand_spy.call_count, 0)
    m0 = camera.get_measurements(sim)
    self.assertIn("test_rgb", m0)
    r0 = reset_only_camera.get_measurements(sim)
    on_demand_frame = on_demand_camera.render_now(sim)
    self.assertTrue(
        torch.equal(on_demand_frame["test_rgb"], r0["reset_only_rgb"])
    )

    # Calling again on substep 0 is idempotent: returns cached measurements
    m0_repeat = camera.get_measurements(sim)
    self.assertIs(m0["test_rgb"], m0_repeat["test_rgb"])

    # Substep 1: does not re-render, returns cached measurements
    dl.step({})
    m1 = camera.get_measurements(sim)
    self.assertIs(m0["test_rgb"], m1["test_rgb"])

    # Substep 2: renders again, returns fresh measurements
    dl.step({})
    m2 = camera.get_measurements(sim)
    self.assertIsNot(m0["test_rgb"], m2["test_rgb"])
    # Without a render rate, measurements keep returning the first frame.
    r2 = reset_only_camera.get_measurements(sim)
    self.assertIs(r0["reset_only_rgb"], r2["reset_only_rgb"])

    # After a reset and moving the sphere, `begin_stepping()` renders the
    # current state.
    dl.reset()
    sim.data.qpos[0] = 0.3
    sim.forward()
    dl.begin_stepping()
    r3 = reset_only_camera.get_measurements(sim)
    self.assertFalse(torch.equal(r0["reset_only_rgb"], r3["reset_only_rgb"]))
    m3 = camera.get_measurements(sim)
    self.assertFalse(torch.equal(m3["test_rgb"], m2["test_rgb"]))

    # `render_now` leaves the cached frames and the render schedule untouched,
    # so it cannot suppress the next scheduled render.
    dl.step({})
    self.assertEqual(sim.clock.global_substep, 3)
    last_rendered_substep = camera._last_rendered_substep
    camera.render_now(sim)
    self.assertEqual(camera._last_rendered_substep, last_rendered_substep)
    self.assertIs(camera.get_measurements(sim)["test_rgb"], m3["test_rgb"])
    dl.step({})
    self.assertEqual(sim.clock.global_substep, 4)
    m4 = camera.get_measurements(sim)
    self.assertIsNot(m4["test_rgb"], m3["test_rgb"])
    self.assertTrue(
        torch.equal(m4["test_rgb"], camera.render_now(sim)["test_rgb"])
    )

  def test_render_now_renders_requested_cameras_without_side_effects(self):
    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        impl=types.Impl.CPP,
    )
    sim.forward()
    mode_all = (
        mujoco_camera_device.RenderMode.RGB
        | mujoco_camera_device.RenderMode.DEPTH
        | mujoco_camera_device.RenderMode.SEGMENTATION
    )
    camera = self._create_camera(
        camera_configs=[
            {"camera_name": "test_cam", "output_name": "a"},
            {"camera_name": "top_cam", "output_name": "b"},
        ],
        render_rate_hz=None,
        height=48,
        render_mode=mode_all,
    )
    spec = camera.substep_measurements_spec()
    self.assertEqual(spec["a_rgb"].shape, (48, 64, 3))
    self.assertEqual(spec["a_rgb"].dtype, np.uint8)
    self.assertEqual(spec["a_depth"].shape, (48, 64))
    self.assertEqual(spec["a_depth"].dtype, np.float32)
    self.assertEqual(spec["a_segmentation"].shape, (48, 64, 2))
    self.assertEqual(spec["a_segmentation"].dtype, np.int32)

    dl = self._create_device_layer(sim, [camera])

    # Side-channel subset render works before the first measurement without
    # populating cached substep measurements.
    pre_step = camera.render_now(
        sim,
        output_names=["a"],
        render_mode=mujoco_camera_device.RenderMode.RGB,
    )
    self.assertEqual(set(pre_step.keys()), {"a_rgb"})
    self.assertEqual(camera._last_measurements, {})

    dl.begin_stepping()
    first = dict(camera.get_measurements(sim))
    self.assertEqual(first["a_rgb"].shape, (1, 48, 64, 3))
    self.assertEqual(first["a_rgb"].dtype, torch.uint8)
    self.assertEqual(first["a_depth"].shape, (1, 48, 64))
    self.assertEqual(first["a_depth"].dtype, torch.float32)
    self.assertTrue(torch.isfinite(first["a_depth"]).all())
    self.assertGreater(first["a_depth"].min().item(), 0.0)
    self.assertEqual(first["a_segmentation"].shape, (1, 48, 64, 2))
    self.assertEqual(first["a_segmentation"].dtype, torch.int32)

    # Move the sphere: measurements keep returning the reset frames.
    sim.data.qpos[0] = 0.3
    sim.forward()
    dl.step({})
    self.assertIs(camera.get_measurements(sim)["a_rgb"], first["a_rgb"])

    # Rendering only RGB on camera `a` runs a single render pass and leaves
    # cached substep measurements untouched.
    assert camera._renderer is not None
    with mock.patch.object(
        camera._renderer, "render", wraps=camera._renderer.render
    ) as render_spy:
      rgb_only = camera.render_now(
          sim,
          output_names=["a"],
          render_mode=mujoco_camera_device.RenderMode.RGB,
      )
      self.assertEqual(render_spy.call_count, 1)
    self.assertEqual(set(rgb_only.keys()), {"a_rgb"})
    self.assertFalse(torch.equal(rgb_only["a_rgb"], first["a_rgb"]))
    self.assertIs(camera.get_measurements(sim)["a_rgb"], first["a_rgb"])
    self.assertIs(camera.get_measurements(sim)["b_rgb"], first["b_rgb"])

    # Rendering only RGB on camera `b` (after skipping `a` in `camera_configs`)
    # uses `b`'s own camera.
    rendered_b = camera.render_now(
        sim,
        output_names=["b"],
        render_mode=mujoco_camera_device.RenderMode.RGB,
    )
    self.assertEqual(set(rendered_b.keys()), {"b_rgb"})
    self.assertFalse(torch.equal(rendered_b["b_rgb"], rgb_only["a_rgb"]))
    self.assertIs(camera.get_measurements(sim)["b_rgb"], first["b_rgb"])

    # Rendering only DEPTH and then SEGMENTATION on camera `a` leaves cached
    # substep measurements untouched.
    depth_only = camera.render_now(
        sim,
        output_names=["a"],
        render_mode=mujoco_camera_device.RenderMode.DEPTH,
    )
    self.assertEqual(set(depth_only.keys()), {"a_depth"})
    self.assertFalse(torch.equal(depth_only["a_depth"], first["a_depth"]))
    self.assertIs(camera.get_measurements(sim)["a_depth"], first["a_depth"])

    seg_only = camera.render_now(
        sim,
        output_names=["a"],
        render_mode=mujoco_camera_device.RenderMode.SEGMENTATION,
    )
    self.assertEqual(set(seg_only.keys()), {"a_segmentation"})
    self.assertEqual(seg_only["a_segmentation"].dtype, torch.int32)

    # Rendering all cameras and modes returns all outputs without mutating
    # cached substep measurements.
    rendered_all = camera.render_now(sim)
    self.assertEqual(
        set(rendered_all.keys()),
        {
            "a_rgb",
            "a_depth",
            "a_segmentation",
            "b_rgb",
            "b_depth",
            "b_segmentation",
        },
    )
    self.assertTrue(torch.equal(rendered_all["a_rgb"], rgb_only["a_rgb"]))
    self.assertTrue(torch.equal(rendered_all["b_rgb"], rendered_b["b_rgb"]))
    self.assertIs(camera.get_measurements(sim)["a_rgb"], first["a_rgb"])

    with self.assertRaisesRegex(ValueError, "At least one render mode"):
      camera.render_now(sim, render_mode=mujoco_camera_device.RenderMode(0))
    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      camera.render_now(sim, output_names="a")
    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      camera.render_now(sim, output_names=[])
    with self.assertRaisesRegex(ValueError, "Unknown camera output names"):
      camera.render_now(sim, output_names=["missing"])

    dl.reset()
    multi_world_data = mock.create_autospec(types.Data, instance=True)
    multi_world_data.nworld = 2
    with mock.patch.object(
        type(sim),
        "data",
        new_callable=mock.PropertyMock,
        return_value=multi_world_data,
    ):
      with self.assertRaisesRegex(ValueError, "only supports nworld=1"):
        dl.begin_stepping()
      with self.assertRaisesRegex(ValueError, "only supports nworld=1"):
        camera.render_now(sim)

  def test_render_rate_hz_validation(self):
    with self.assertRaises(ValueError):
      batched_camera_device.BatchedMujocoCameraDevice(
          name="test_camera",
          camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
          render_rate_hz=0.0,
      )
    with self.assertRaises(ValueError):
      batched_camera_device.BatchedMujocoCameraDevice(
          name="test_camera",
          camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
          render_rate_hz=-10.0,
      )
    with self.assertRaisesRegex(ValueError, "on_demand_only requires"):
      batched_camera_device.BatchedMujocoCameraDevice(
          name="test_camera",
          camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
          render_rate_hz=10.0,
          on_demand_only=True,
      )
    with self.assertRaisesRegex(ValueError, "At least one render mode"):
      batched_camera_device.BatchedMujocoCameraDevice(
          name="test_camera",
          camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
          render_rate_hz=None,
          render_mode=mujoco_camera_device.RenderMode(0),
      )
    # None renders only after resets.
    batched_camera_device.BatchedMujocoCameraDevice(
        name="test_camera",
        camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
        render_rate_hz=None,
    )


if __name__ == "__main__":
  absltest.main()
