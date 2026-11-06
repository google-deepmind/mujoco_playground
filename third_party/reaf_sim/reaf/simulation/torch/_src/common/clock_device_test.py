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

"""Tests for the ClockDevice."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from reaf.simulation.torch._src.common import clock_device
from reaf.simulation.torch._src.core import device_layer
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


_PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 1" size="0.05"/>
      <inertial pos="0 0 0.5" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge"/>
  </actuator>
</mujoco>
"""


@parameterized.named_parameters(
    dict(testcase_name="warp_backend", impl=types.Impl.WARP, device=_DEVICE),
)
class ClockDeviceTest(parameterized.TestCase):
  """Tests for ClockDevice with DeviceLayer integration."""

  def test_staggered_resets(self, impl, device):
    """Verifies that reset correctly resets data.time for specific env_mask."""
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    mj_spec = mujoco.MjSpec.from_string(_PENDULUM_XML)
    nworld = 4
    clock = clock_device.ClockDevice(
        name="clock",
        time_output_name="sim_time",
    )

    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=nworld,
        xpu_device=device,
        impl=impl,
    )
    layer = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.01,
        devices=[clock],
    )

    layer.begin_stepping()

    # Step 10 times: time should be 0.1 for all environments.
    for _ in range(10):
      measurements = layer.step({})
    expected_time = torch.full((nworld,), 0.1, device=device)
    torch.testing.assert_close(
        measurements["sim_time"].float(),
        expected_time,
        atol=1e-6,
        rtol=1e-6,
    )

    # Reset only environments 1 and 3.
    reset_mask = torch.tensor(
        [False, True, False, True], dtype=torch.bool, device=device
    )
    layer.reset(reset_mask)

    # Check that only envs 1 and 3 have time reset to 0.
    layer.update_measurements()
    measurements = layer.get_measurements()
    expected_time = torch.tensor([0.1, 0.0, 0.1, 0.0], device=device)
    torch.testing.assert_close(
        measurements["sim_time"].float(),
        expected_time,
        atol=1e-6,
        rtol=1e-6,
    )

    # Step 5 more times.
    for _ in range(5):
      measurements = layer.step({})
    measurements = layer.get_measurements()
    # Envs 0,2: 0.1 + 0.05 = 0.15; Envs 1,3: 0.0 + 0.05 = 0.05
    expected_time = torch.tensor([0.15, 0.05, 0.15, 0.05], device=device)
    torch.testing.assert_close(
        measurements["sim_time"].float(),
        expected_time,
        atol=1e-6,
        rtol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
