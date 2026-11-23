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
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import articulated_device
from reaf.simulation.torch._src.core import device_layer
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
      xpu_device=_DEVICE,
  )
  dev.initialize(sim)
  return dev, sim, spec


class ArticulatedDeviceTest(parameterized.TestCase):

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
    m = dev.get_measurements(sim)
    np.testing.assert_allclose(
        m["robot_state/joint_pos"].cpu().numpy()[0], [1.0, 2.0]
    )
    np.testing.assert_allclose(
        m["robot_state/joint_vel"].cpu().numpy()[0], [4.0, 5.0]
    )

  def test_joint_range(self):
    dev, sim, _ = _setup(measurements=["joint_range"])
    sim.forward()
    recaptures = []
    sim.model.register_graph_capture_callback(lambda: recaptures.append(1))
    if torch.cuda.is_available():
      prev_mode = torch.cuda.get_sync_debug_mode()
      self.addCleanup(torch.cuda.set_sync_debug_mode, prev_mode)
      torch.cuda.set_sync_debug_mode("error")
      m = dev.get_measurements(sim)
      torch.cuda.set_sync_debug_mode(prev_mode)
    else:
      m = dev.get_measurements(sim)
    self.assertEmpty(recaptures)
    np.testing.assert_allclose(
        m["robot_state/joint_range"].cpu().numpy()[0],
        [[-1.57, 1.57], [-3.14, 3.14]],
    )

  def test_no_measurements(self):
    dev, sim, _ = _setup()
    self.assertEqual(dev.substep_measurements_spec(), {})
    self.assertEqual(dev.get_measurements(sim), {})

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

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_joint_range_shapes_with_unnamed_ball_joint(self, impl: types.Impl):
    xml = """
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body>
          <joint type="ball" range="0 1.2"/>
          <geom size="0.1"/>
          <body pos="0 0 -0.2">
            <joint name="elbow" type="hinge" range="-2.0 0.1"/>
            <geom size="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    nworld = 1 if impl == types.Impl.CPP else 2
    xpu_device = _DEVICE if impl == types.Impl.WARP else "cpu"
    sim = simulator_module.Simulator(
        mj_spec=mujoco.MjSpec.from_string(xml),
        nworld=nworld,
        impl=impl,
        xpu_device=xpu_device,
    )
    all_joints = articulated_device.ArticulatedDevice(
        name="all", measurements=["joint_range"], xpu_device=xpu_device
    )
    elbow = articulated_device.ArticulatedDevice(
        name="elbow",
        measurements=["joint_range"],
        xpu_device=xpu_device,
        joint_names=["elbow"],
    )
    all_joints.initialize(sim)
    elbow.initialize(sim)

    self.assertEqual(
        all_joints.substep_measurements_spec()["all/joint_range"].shape, (2, 2)
    )
    self.assertEqual(
        elbow.substep_measurements_spec()["elbow/joint_range"].shape, (1, 2)
    )
    np.testing.assert_allclose(
        all_joints.get_measurements(sim)["all/joint_range"].cpu().numpy(),
        [[[0.0, 1.2], [-2.0, 0.1]]] * nworld,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        elbow.get_measurements(sim)["elbow/joint_range"].cpu().numpy(),
        [[[-2.0, 0.1]]] * nworld,
        rtol=1e-6,
    )

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_single_joint_pos_vel_and_partial_reset(self, impl: types.Impl):
    nworld = 1 if impl == types.Impl.CPP else _NWORLD
    xpu_device = _DEVICE if impl == types.Impl.WARP else "cpu"
    sim = simulator_module.Simulator(
        mj_spec=mujoco.MjSpec.from_string(_DUMMY_XML),
        nworld=nworld,
        impl=impl,
        xpu_device=xpu_device,
    )
    dev = articulated_device.ArticulatedDevice(
        name="robot_state",
        measurements=["joint_pos", "joint_vel"],
        xpu_device=xpu_device,
        joint_names=["robot/joint1"],
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        devices=[dev],
        control_timestep=float(sim.mj_model.opt.timestep),
    )

    sim.data.qpos[:, 0] = torch.arange(1.0, 1.0 + nworld, device=xpu_device)
    sim.data.qvel[:, 0] = torch.arange(4.0, 4.0 + nworld, device=xpu_device)
    m = dev.get_measurements(sim)
    self.assertEqual(m["robot_state/joint_pos"].shape, (nworld, 1))
    self.assertEqual(m["robot_state/joint_vel"].shape, (nworld, 1))

    if nworld > 1:
      dl.begin_stepping()
      sim.data.qpos[:, 0] = 9.0
      sim.data.qvel[:, 0] = 8.0
      mask = torch.tensor([True, False], device=xpu_device)
      out = dl.begin_stepping(env_mask=mask)
      np.testing.assert_allclose(
          out["robot_state/joint_pos"].cpu().numpy(), [[9.0], [2.0]]
      )
      np.testing.assert_allclose(
          out["robot_state/joint_vel"].cpu().numpy(), [[8.0], [5.0]]
      )


if __name__ == "__main__":
  absltest.main()
