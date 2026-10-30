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

"""Tests for ActuatorDevice runtime behavior."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import actuator_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

MINIMAL_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
  <actuator>
    <position name="robot/j1" joint="robot/j1" kp="100" kv="10" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""

MIXED_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
      <body pos="0 0 0.3">
        <joint name="robot/j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
        <site name="robot/s1" pos="0 0 0.3"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="robot/t1">
      <joint joint="robot/j1" coef="1"/>
      <joint joint="robot/j2" coef="-1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="robot/j1" joint="robot/j1" kp="100" kv="10" forcerange="-30 30"/>
    <position name="robot/t1" tendon="robot/t1" kp="100" kv="10" forcerange="-30 30"/>
    <motor name="robot/s1" site="robot/s1" gear="1 0 0 0 0 0"/>
  </actuator>
</mujoco>
"""

DIFF_NAMES_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/my_joint" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
  <actuator>
    <position name="robot/my_actuator" joint="robot/my_joint" kp="100"/>
  </actuator>
</mujoco>
"""


class RuntimeTest(parameterized.TestCase):
  """Tests for ActuatorDevice runtime behavior."""

  def _make_sim_with_actuator(
      self,
      xml: str,
      impl: types.Impl,
      targets: list[str],
      measurements: list[str] | None = None,
      nworld: int = 1,
  ) -> tuple[actuator_device.ActuatorDevice, simulator_module.Simulator]:
    """Builds and initializes an ActuatorDevice and Simulator from XML."""
    xpu_device = (
        "cpu"
        if impl == types.Impl.CPP or not torch.cuda.is_available()
        else "cuda:0"
    )
    spec = mujoco.MjSpec.from_string(xml)
    dev = actuator_device.ActuatorDevice(
        name="dc",
        xpu_device=xpu_device,
        targets=targets,
        measurements=measurements or ["actuator_force"],
    )
    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=nworld, impl=impl, xpu_device=xpu_device
    )
    dev.initialize(sim)
    return dev, sim

  def test_measurements_match_mj_data(self):
    """Measurements from entity match raw mj_data values."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)

    mj_data.qpos[0] = 0.5
    mj_data.qvel[0] = 1.0
    mj_data.ctrl[0] = 0.3
    mujoco.mj_forward(mj_model, mj_data)

    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/j1"],
        measurements=["joint_pos", "joint_vel", "actuator_force"],
    )

    sim = simulator_module.Simulator(mj_spec=spec, impl=types.Impl.CPP)
    sim.data.obj.qpos[:] = mj_data.qpos[:]
    sim.data.obj.qvel[:] = mj_data.qvel[:]
    sim.data.obj.actuator_force[:] = mj_data.actuator_force[:]
    dev.initialize(sim)
    m = dev.get_substep_measurements(sim)

    np.testing.assert_allclose(
        m["robot/joint_pos"].numpy(), mj_data.qpos[None, :], rtol=1e-5
    )
    np.testing.assert_allclose(
        m["robot/joint_vel"].numpy(), mj_data.qvel[None, :], rtol=1e-5
    )
    np.testing.assert_allclose(
        m["robot/actuator_force"].numpy(),
        mj_data.actuator_force[None, :],
        rtol=1e-5,
    )

  def test_set_substep_commands(self):
    """set_substep_commands writes to data.ctrl correctly."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/j1"],
    )
    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=1, impl=types.Impl.CPP
    )
    dev.initialize(sim)

    commands = {"robot/ctrl": torch.tensor([[0.5]])}
    dev.set_substep_commands(sim, commands)

    np.testing.assert_allclose(sim.data.ctrl.numpy(), [[0.5]])

  def test_tendon_measurements(self):
    """Tendon measurements use correct indices."""
    spec = mujoco.MjSpec.from_string(MIXED_XML)
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[0] = 0.5
    mj_data.qpos[1] = 0.2
    mujoco.mj_forward(mj_model, mj_data)

    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/t1"],
        measurements=["tendon_length", "tendon_velocity"],
    )
    sim = simulator_module.Simulator(mj_spec=spec, impl=types.Impl.CPP)
    sim.data.obj.qpos[:] = mj_data.qpos[:]
    sim.forward()
    dev.initialize(sim)

    m = dev.get_substep_measurements(sim)

    np.testing.assert_allclose(
        m["robot/tendon_length"].numpy()[0, 0], mj_data.ten_length[0], rtol=1e-5
    )
    np.testing.assert_allclose(
        m["robot/tendon_velocity"].numpy()[0, 0],
        mj_data.ten_velocity[0],
        rtol=1e-5,
    )

  @parameterized.product(
      impl=[types.Impl.CPP, types.Impl.WARP],
      case=[
          # Zero-input passive mode (ctrlnum=0).
          ('input="none" damping="0.1"', []),
          # Single-input voltage mode (ctrlnum=1).
          ("", [12.0]),
          # Multi-input position + velocity mode (ctrlnum=2).
          ('controller="10 0 5 0 0 0" input="pos vel"', [1.0, 0.2]),
      ],
  )
  def test_dcmotor(self, impl: types.Impl, case: tuple[str, list[float]]):
    """ActuatorDevice maps ctrl inputs and force outputs across dcmotor spans."""
    attrs, ctrl_vals = case
    num_ctrl = len(ctrl_vals)
    nworld = 1 if impl == types.Impl.CPP else 2
    xml = f"""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j0" type="hinge"/>
          <geom size="0.1"/>
        </body>
        <body>
          <joint name="j1" type="hinge"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="j0" joint="j0"/>
        <dcmotor name="j1" joint="j1" motorconst="0.05" resistance="2.0"
                 {attrs}/>
      </actuator>
    </mujoco>
    """
    dev, sim = self._make_sim_with_actuator(
        xml,
        impl,
        ["j1"],
        measurements=["ctrl", "home_ctrl", "actuator_force"],
        nworld=nworld,
    )
    xpu_device = sim.data.ctrl.device

    # Verify command and measurement specs match resolved control/output spans.
    self.assertEqual(dev.substep_commands_spec()["dc/ctrl"].shape, (num_ctrl,))
    self.assertEqual(
        dev.substep_measurements_spec()["dc/ctrl"].shape, (num_ctrl,)
    )
    self.assertEqual(
        dev.substep_measurements_spec()["dc/home_ctrl"].shape, (num_ctrl,)
    )
    self.assertEqual(
        dev.substep_measurements_spec()["dc/actuator_force"].shape, (1,)
    )

    # Verify set_substep_commands writes to ctrl_adr=1..1+num_ctrl only.
    ctrl = torch.tensor(
        [ctrl_vals] * nworld, dtype=torch.float32, device=xpu_device
    )
    dev.set_substep_commands(sim, {"dc/ctrl": ctrl})
    sim.data.qvel[:, 1] = 1.0
    sim.step()
    sim.forward()

    # Verify get_substep_measurements reads mapped ctrl and force slices.
    m = dev.get_substep_measurements(sim)
    np.testing.assert_allclose(m["dc/ctrl"].cpu().numpy(), ctrl.cpu().numpy())
    np.testing.assert_allclose(
        sim.data.ctrl[:, 1 : 1 + num_ctrl].cpu().numpy(), ctrl.cpu().numpy()
    )
    np.testing.assert_allclose(
        m["dc/actuator_force"].cpu().numpy(),
        sim.data.actuator_force[:, [1]].cpu().numpy(),
    )
    self.assertTrue(m["dc/actuator_force"].ne(0.0).all())
    self.assertTrue((sim.data.ctrl[:, 0] == 0.0).all())

    # Verify set_ctrl updates the mapped ctrl span with and without env_mask.
    dev.set_ctrl(None, dev.home_ctrl)
    self.assertTrue((sim.data.ctrl[:, 1 : 1 + num_ctrl] == 0.0).all())
    if nworld > 1 and num_ctrl > 0:
      mask = torch.tensor([True, False], device=xpu_device)
      dev.set_ctrl(mask, ctrl)
      np.testing.assert_allclose(
          sim.data.ctrl[0, 1 : 1 + num_ctrl].cpu().numpy(), ctrl_vals
      )
      self.assertTrue((sim.data.ctrl[1, 1 : 1 + num_ctrl] == 0.0).all())

  def test_dcmotor_multi_output_offset(self):
    """ActuatorDevice after a 3-output <orientation> actuator maps out_adr."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="ball" type="ball"/>
          <geom size="0.1"/>
        </body>
        <body>
          <joint name="hinge" type="hinge"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <orientation name="so3" joint="ball" kp="1"/>
        <dcmotor name="dc" joint="hinge" motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    dev, sim = self._make_sim_with_actuator(xml, types.Impl.CPP, ["dc"])

    # Actuator id is 1, but output address is 3 (<orientation> has 3 outputs).
    self.assertEqual(sim.mj_model.actuator_outadr[1], 3)
    dev.set_substep_commands(sim, {"dc/ctrl": torch.tensor([[12.0]])})
    sim.forward()
    m = dev.get_substep_measurements(sim)
    self.assertNotEqual(float(m["dc/actuator_force"][0, 0]), 0.0)
    np.testing.assert_allclose(
        m["dc/actuator_force"].numpy(),
        sim.data.actuator_force[:, [3]].numpy(),
    )

    # Mismatched custom home_ctrl shape raises ValueError in initialize().
    bad_dev = actuator_device.ActuatorDevice(
        name="dc", xpu_device="cpu", targets=["dc"], home_ctrl=[0.0, 0.0]
    )
    with self.assertRaises(ValueError):
      bad_dev.initialize(sim)

  def test_different_actuator_joint_names(self):
    """ActuatorDevice resolves joint indices when actuator and joint names differ."""
    dev, sim = self._make_sim_with_actuator(
        DIFF_NAMES_XML,
        types.Impl.CPP,
        ["robot/my_actuator"],
        measurements=["joint_pos"],
    )
    sim.data.qpos[:, 0] = 0.7
    m = dev.get_substep_measurements(sim)
    np.testing.assert_allclose(m["dc/joint_pos"].numpy(), [[0.7]], rtol=1e-5)

  def test_mixed_transmissions(self):
    """ActuatorDevice supports joint, tendon, and site actuators together."""
    measurements = [
        "joint_pos",
        "joint_vel",
        "joint_range",
        "tendon_length",
        "tendon_velocity",
        "actuator_force",
        "ctrl",
        "home_ctrl",
    ]
    dev, sim = self._make_sim_with_actuator(
        MIXED_XML,
        types.Impl.CPP,
        ["robot/j1", "robot/t1", "robot/s1"],
        measurements=measurements,
    )
    dev.set_substep_commands(sim, {"dc/ctrl": torch.tensor([[0.4, 0.2, 1.5]])})
    sim.forward()

    specs_dict = dev.substep_measurements_spec()
    m = dev.get_substep_measurements(sim)
    self.assertEqual(specs_dict["dc/joint_pos"].shape, (1,))
    self.assertEqual(specs_dict["dc/joint_vel"].shape, (1,))
    self.assertEqual(specs_dict["dc/joint_range"].shape, (1, 2))
    self.assertEqual(specs_dict["dc/tendon_length"].shape, (1,))
    self.assertEqual(specs_dict["dc/tendon_velocity"].shape, (1,))
    self.assertEqual(specs_dict["dc/actuator_force"].shape, (3,))
    self.assertEqual(specs_dict["dc/ctrl"].shape, (3,))
    for key, spec in specs_dict.items():
      self.assertEqual(tuple(m[key].shape[-len(spec.shape) :]), spec.shape)


if __name__ == "__main__":
  absltest.main()
