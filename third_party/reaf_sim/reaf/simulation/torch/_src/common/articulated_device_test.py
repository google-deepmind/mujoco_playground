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


"""Tests for ArticulatedDevice."""

from absl.testing import absltest
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.common import articulated_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


_DUMMY_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="robot/link1">
      <joint name="robot/joint1" type="hinge" range="-1.57 1.57"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="robot/link2">
      <joint name="robot/joint2" type="hinge" range="-3.14 3.14"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="object/link1">
      <joint name="object/joint1" type="hinge"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

_NWORLD = 2


def _setup(measurements=None):
  """Compile model, wrap in torch, and initialize entity."""
  wp.init()
  wp.set_device(_DEVICE)
  spec = mujoco.MjSpec.from_string(_DUMMY_XML)
  dev = articulated_device.ArticulatedDevice(
      name="robot_state",
      measurements=measurements or [],
      xpu_device=_DEVICE,
      joint_names=["robot/joint1", "robot/joint2"],
  )
  sim = simulator_module.Simulator(
      mj_spec=spec,
      nworld=_NWORLD,
      impl=types.Impl.WARP,
  )
  dev.initialize(sim)
  return dev, sim, spec


class ArticulatedDeviceTest(absltest.TestCase):

  def test_measurements_spec(self):
    dev, _, _ = _setup(measurements=["joint_pos", "joint_vel"])
    meas_spec = dev.substep_measurements_spec()
    self.assertEqual(meas_spec["robot_state/joint_pos"].shape, (2,))
    self.assertEqual(meas_spec["robot_state/joint_vel"].shape, (2,))

  def test_get_measurements(self):
    dev, sim, _ = _setup(measurements=["joint_pos", "joint_vel"])
    data = sim.data
    # Robot joints are the first 2 of 3 total qpos/qvel entries.
    data.qpos[:] = torch.tensor([[1.0, 2.0, 3.0]] * _NWORLD)
    data.qvel[:] = torch.tensor([[4.0, 5.0, 6.0]] * _NWORLD)
    m = dev.get_substep_measurements(sim)
    np.testing.assert_allclose(
        m["robot_state/joint_pos"].cpu().numpy()[0], [1.0, 2.0]
    )
    np.testing.assert_allclose(
        m["robot_state/joint_vel"].cpu().numpy()[0], [4.0, 5.0]
    )

  def test_joint_range(self):
    dev, sim, _ = _setup(measurements=["joint_range"])
    m = dev.get_substep_measurements(sim)
    np.testing.assert_allclose(
        m["robot_state/joint_range"].cpu().numpy()[0],
        [[-1.57, 1.57], [-3.14, 3.14]],
    )

  def test_no_measurements(self):
    dev, sim, _ = _setup()
    self.assertEqual(dev.substep_measurements_spec(), {})
    self.assertEqual(dev.get_substep_measurements(sim), {})

  def test_set_qpos(self):
    dev, sim, _ = _setup()
    data = sim.data
    new_qpos = torch.tensor([[10.0, 20.0]] * _NWORLD, device=data.qpos.device)
    dev.set_qpos(None, new_qpos)
    np.testing.assert_allclose(
        data.qpos[:, :2].cpu().numpy(), new_qpos.cpu().numpy()
    )

  def test_set_qvel(self):
    dev, sim, _ = _setup()
    data = sim.data
    new_qvel = torch.tensor([[7.0, 8.0]] * _NWORLD, device=data.qvel.device)
    dev.set_qvel(None, new_qvel)
    np.testing.assert_allclose(
        data.qvel[:, :2].cpu().numpy(), new_qvel.cpu().numpy()
    )


if __name__ == "__main__":
  absltest.main()
