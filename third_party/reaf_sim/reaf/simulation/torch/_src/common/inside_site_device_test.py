# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for InsideSiteDevice."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.common import inside_site_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


def _make_spec_with_containment_site(
    box_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> mujoco.MjSpec:
  """Create a spec with a body and a bounding site."""
  spec = mujoco.MjSpec()

  # Target body
  body = spec.worldbody.add_body(name="target_box", pos=box_pos)
  body.add_freejoint(name="box_joint")
  body.add_geom(
      name="box_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.05, 0.05, 0.05),
  )

  # Container body carrying a box site at origin
  container = spec.worldbody.add_body(name="container", pos=(0.0, 0.0, 0.0))
  container.add_site(
      name="container_site",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.2, 0.2, 0.2),
      pos=(0.0, 0.0, 0.0),
  )
  return spec


class InsideSiteDeviceTest(parameterized.TestCase):

  def test_configure_spec(self):
    spec = _make_spec_with_containment_site()
    dev = inside_site_device.InsideSiteDevice(
        name="box_in_container",
        body_name="target_box",
        site_name="container_site",
    )
    dev.configure_spec(spec)
    mj_model = spec.compile()
    self.assertEqual(mj_model.nsensor, 1)

  @parameterized.parameters(
      dict(box_pos=(0.0, 0.0, 0.0), expected_contained=True),
      dict(box_pos=(0.1, 0.1, 0.1), expected_contained=True),
      dict(box_pos=(0.5, 0.5, 0.5), expected_contained=False),
      dict(box_pos=(0.0, 0.0, 2.0), expected_contained=False),
  )
  def test_containment_measurements(
      self, box_pos: tuple[float, float, float], expected_contained: bool
  ):
    spec = _make_spec_with_containment_site(box_pos=box_pos)
    dev = inside_site_device.InsideSiteDevice(
        name="box_in_container",
        body_name="target_box",
        site_name="container_site",
    )
    dev.configure_spec(spec)
    nworld = 2

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=nworld,
        impl=types.Impl.WARP,
    )
    sim.forward()

    dev.initialize(sim)

    # Check specs
    meas_spec = dev.substep_measurements_spec()
    self.assertIn("box_in_container/data", meas_spec)
    self.assertEqual(meas_spec["box_in_container/data"].shape, (1,))

    # Check measurements
    measurements = dev.get_substep_measurements(sim)
    self.assertEqual(measurements["box_in_container/data"].shape, (nworld, 1))
    expected_val = 1.0 if expected_contained else 0.0
    self.assertEqual(
        measurements["box_in_container/data"][0, 0].item(), expected_val
    )

  def test_dark_mode(self):
    if not torch.cuda.is_available():
      self.skipTest("CUDA is not available")
    spec = _make_spec_with_containment_site()
    dev = inside_site_device.InsideSiteDevice(
        name="box_in_container",
        body_name="target_box",
        site_name="container_site",
        dark=True,
    )
    dev.configure_spec(spec)
    nworld = 2

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=nworld,
        impl=types.Impl.WARP,
    )
    sim.forward()

    dev.initialize(sim)

    # Dark mode should return empty specs
    self.assertEqual(dev.substep_measurements_spec(), {})
    # Dark mode should return empty measurements unless force=True
    self.assertEqual(dev.get_substep_measurements(sim), {})
    forced_meas = dev.get_substep_measurements(sim, force=True)
    self.assertIn("box_in_container/data", forced_meas)
    self.assertEqual(forced_meas["box_in_container/data"].shape, (nworld, 1))


if __name__ == "__main__":
  absltest.main()
