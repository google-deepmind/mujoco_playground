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
  </worldbody>
</mujoco>
"""


class BatchedMujocoCameraDeviceTest(parameterized.TestCase):

  @parameterized.parameters(
      (types.Impl.CPP,),
      (types.Impl.WARP,),
  )
  def test_camera_entity_renders(self, impl):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("Warp camera requires CUDA.")

    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        impl=impl,
    )
    sim.forward()

    camera = batched_camera_device.BatchedMujocoCameraDevice(
        name="test_camera",
        camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
        render_rate_hz=500,
        width=64,
        height=64,
    )
    camera.initialize(sim)

    measurements = camera.get_substep_measurements(sim)

    self.assertIn("test_rgb", measurements)
    img = measurements["test_rgb"]
    self.assertEqual(img.shape, (1, 64, 64, 3))
    self.assertEqual(img.dtype, torch.uint8)
    self.assertGreater(img.sum(), 0)

  def test_update_interval_via_render_rate_hz(self):
    mj_spec = mujoco.MjSpec.from_string(_SIMPLE_XML)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        impl=types.Impl.CPP,
    )
    sim.forward()

    # Default timestep is 0.002s (500 Hz). 250 Hz -> interval of 2 substeps.
    camera = batched_camera_device.BatchedMujocoCameraDevice(
        name="test_camera",
        camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
        render_rate_hz=250,
        width=64,
        height=64,
    )
    # None: renders only the first measurement after each reset.
    reset_only_camera = batched_camera_device.BatchedMujocoCameraDevice(
        name="reset_only_camera",
        camera_configs=[{"camera_name": "test_cam", "output_name": "test"}],
        render_rate_hz=None,
        width=64,
        height=64,
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
