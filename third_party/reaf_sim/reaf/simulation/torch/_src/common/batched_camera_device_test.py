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
from reaf.simulation.torch._src.common import batched_camera_device
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
    camera.initialize(sim)

    measurements = camera.get_substep_measurements(sim)

    self.assertIn("test_rgb", measurements)
    img = measurements["test_rgb"]
    self.assertEqual(img.shape, (1, 64, 64, 3))
    self.assertEqual(img.dtype, torch.uint8)
    self.assertEqual(img.device, torch.device(xpu_device))
    self.assertGreater(img.sum(), 0)
    if impl == types.Impl.WARP:
      with self.assertRaisesRegex(ValueError, "only supports Impl.CPP"):
        camera.render_now(sim)

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
        name="reset_only_camera", render_rate_hz=None
    )
    for cam in (camera, reset_only_camera):
      cam.initialize(sim)
      cam.reset(sim)

    # Substep 0: renders and returns measurements
    m0 = camera.get_substep_measurements(sim)
    self.assertIn("test_rgb", m0)
    r0 = reset_only_camera.get_substep_measurements(sim)

    # Calling again on substep 0 is idempotent: returns cached measurements
    m0_repeat = camera.get_substep_measurements(sim)
    self.assertIs(m0["test_rgb"], m0_repeat["test_rgb"])

    # Substep 1: does not re-render, returns cached measurements
    sim.step()
    m1 = camera.get_substep_measurements(sim)
    self.assertIs(m0["test_rgb"], m1["test_rgb"])

    # Substep 2: renders again, returns fresh measurements
    sim.step()
    m2 = camera.get_substep_measurements(sim)
    self.assertIsNot(m0["test_rgb"], m2["test_rgb"])
    # Without a render rate, measurements keep returning the first frame.
    r2 = reset_only_camera.get_substep_measurements(sim)
    self.assertIs(r0["test_rgb"], r2["test_rgb"])

    # After the sphere moves and a reset, it renders the current state.
    sim.data.qpos[0] = 0.3
    sim.forward()
    reset_only_camera.reset(sim)
    r3 = reset_only_camera.get_substep_measurements(sim)
    self.assertFalse(torch.equal(r0["test_rgb"], r3["test_rgb"]))

    # Calling `render_now` on a scheduled substep does not suppress scheduled
    # `get_substep_measurements` renders.
    sim.step()
    sim.step()
    self.assertEqual(sim.clock.global_substep, 4)
    side = camera.render_now(sim)
    m4 = camera.get_substep_measurements(sim)
    self.assertFalse(torch.equal(m4["test_rgb"], m2["test_rgb"]))
    self.assertTrue(torch.equal(m4["test_rgb"], side["test_rgb"]))

  def test_render_now_renders_requested_cameras_without_side_effects(self):
    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        impl=types.Impl.CPP,
    )
    sim.forward()
    camera = self._create_camera(
        camera_configs=[
            {"camera_name": "test_cam", "output_name": "a"},
            {"camera_name": "top_cam", "output_name": "b"},
        ],
        render_rate_hz=None,
    )
    camera.initialize(sim)
    camera.reset(sim)

    # Side-channel subset render works immediately after reset without
    # populating cached substep measurements.
    pre_step = camera.render_now(sim, output_names=["a"])
    self.assertEqual(set(pre_step.keys()), {"a_rgb"})
    self.assertEqual(camera._last_measurements, {})

    first = dict(camera.get_substep_measurements(sim))

    # Move the sphere: measurements keep returning the reset frames.
    sim.data.qpos[0] = 0.3
    sim.forward()
    sim.step()
    self.assertIs(camera.get_substep_measurements(sim)["a_rgb"], first["a_rgb"])

    # Rendering only `a` returns only `a_rgb` and leaves measurements untouched.
    rendered = camera.render_now(sim, output_names=["a"])
    self.assertEqual(set(rendered.keys()), {"a_rgb"})
    self.assertFalse(torch.equal(rendered["a_rgb"], first["a_rgb"]))
    self.assertIs(camera.get_substep_measurements(sim)["a_rgb"], first["a_rgb"])
    self.assertIs(camera.get_substep_measurements(sim)["b_rgb"], first["b_rgb"])

    # Rendering only `b` (after skipping `a` in `camera_configs`) uses `b`'s own
    # camera.
    rendered_b = camera.render_now(sim, output_names=["b"])
    self.assertEqual(set(rendered_b.keys()), {"b_rgb"})
    self.assertFalse(torch.equal(rendered_b["b_rgb"], rendered["a_rgb"]))
    self.assertIs(camera.get_substep_measurements(sim)["b_rgb"], first["b_rgb"])

    # Rendering all cameras returns both without mutating measurements.
    rendered_all = camera.render_now(sim)
    self.assertEqual(set(rendered_all.keys()), {"a_rgb", "b_rgb"})
    self.assertTrue(torch.equal(rendered_all["a_rgb"], rendered["a_rgb"]))
    self.assertTrue(torch.equal(rendered_all["b_rgb"], rendered_b["b_rgb"]))
    self.assertIs(camera.get_substep_measurements(sim)["a_rgb"], first["a_rgb"])

    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      camera.render_now(sim, output_names="a")
    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      camera.render_now(sim, output_names=[])
    with self.assertRaisesRegex(ValueError, "Unknown camera output names"):
      camera.render_now(sim, output_names=["missing"])

    camera.reset(sim)
    multi_world_data = mock.create_autospec(types.Data, instance=True)
    multi_world_data.nworld = 2
    with mock.patch.object(
        type(sim),
        "data",
        new_callable=mock.PropertyMock,
        return_value=multi_world_data,
    ):
      with self.assertRaisesRegex(ValueError, "only supports nworld=1"):
        camera.get_substep_measurements(sim)
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
    # None renders only after resets.
    batched_camera_device.BatchedMujocoCameraDevice(
        name="test_camera",
        camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
        render_rate_hz=None,
    )


if __name__ == "__main__":
  absltest.main()
