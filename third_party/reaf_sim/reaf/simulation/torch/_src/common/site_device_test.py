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

"""Tests for SiteDevice."""

from absl.testing import absltest
import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.common import site_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


def _make_spec_with_site() -> mujoco.MjSpec:
  """Create a spec with a body and site."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="arm", pos=(0, 0, 1))
  body.add_geom(
      name="arm_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.1, 0.1, 0.1),
  )
  body.add_site(name="ee_site", pos=(0, 0, 0.15))
  return spec


class SiteDeviceTest(absltest.TestCase):

  def test_site_measurements(self):
    """Test that site device returns correct xpos and xmat shapes."""
    if not torch.cuda.is_available():
      self.skipTest("Warp put_model/put_data requires CUDA.")
    spec = _make_spec_with_site()
    nworld = 2

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=nworld,
        impl=types.Impl.WARP,
    )
    sim.forward()
    model = sim.model
    data = sim.data

    dev = site_device.SiteDevice(name="ee", site_name="ee_site")
    dev.initialize(sim)

    # Check spec
    meas_spec = dev.substep_measurements_spec()
    self.assertEqual(meas_spec["ee/xpos"].shape, (3,))
    self.assertEqual(meas_spec["ee/xmat"].shape, (9,))

    # Check measurements shape
    measurements = dev.get_substep_measurements(sim)
    self.assertEqual(measurements["ee/xpos"].shape, (nworld, 3))
    self.assertEqual(measurements["ee/xmat"].shape, (nworld, 9))

    device = measurements["ee/xpos"].device

    # Check measurements values
    expected_xpos = torch.tensor([0.0, 0.0, 1.15], device=device)
    torch.testing.assert_close(
        measurements["ee/xpos"][0], expected_xpos, atol=1e-5, rtol=1e-5
    )
    expected_xmat = torch.eye(3, device=device).reshape(9)
    torch.testing.assert_close(
        measurements["ee/xmat"][0], expected_xmat, atol=1e-5, rtol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
