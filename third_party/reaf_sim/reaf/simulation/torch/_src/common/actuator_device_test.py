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

"""Tests for ActuatorDevice - compares MjSpec config vs reference XML."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import actuator_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

# Minimal XML for configure_spec to modify
MINIMAL_XML = """
<mujoco>
  <compiler autolimits="true"/>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

# XML with tendon for tendon transmission test
TENDON_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="robot/t1">
      <joint joint="robot/j1" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

# XML with site for site transmission test
SITE_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <site name="robot/s1" pos="0.1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

MIXED_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="robot/j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
      <body pos="0 0 0.3">
        <joint name="robot/j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="robot/t1">
      <joint joint="robot/j1" coef="1"/>
      <joint joint="robot/j2" coef="-1"/>
    </fixed>
  </tendon>
</mujoco>
"""

DIFF_NAMES_XML = """
<mujoco>
  <compiler autolimits="true"/>
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


GRIPPER_XML = """
<mujoco>
  <worldbody>
    <body name="gripper">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
  </worldbody>
  <actuator>
    <position name="my_actuator" joint="slider" kp="100"/>
  </actuator>
</mujoco>
"""


class ConfigureSpecTest(absltest.TestCase):
  """Tests for ActuatorDevice.configure_spec."""

  def test_position_mode(self):
    """Position actuator: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/j1"]
    )
    dev.configure_spec(
        spec,
        mode="position",
        stiffness=100.0,
        damping=10.0,
        forcerange=30.0,
    )

    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/j1"
    )

    self.assertEqual(model.actuator_dyntype[aid], mujoco.mjtDyn.mjDYN_NONE)
    self.assertEqual(model.actuator_gaintype[aid], mujoco.mjtGain.mjGAIN_FIXED)
    self.assertEqual(model.actuator_biastype[aid], mujoco.mjtBias.mjBIAS_AFFINE)
    self.assertEqual(model.actuator_gainprm[aid, 0], 100.0)
    self.assertEqual(model.actuator_biasprm[aid, 1], -100.0)
    self.assertEqual(model.actuator_biasprm[aid, 2], -10.0)
    self.assertFalse(model.actuator_ctrllimited[aid])
    self.assertTrue(model.actuator_forcelimited[aid])
    np.testing.assert_allclose(model.actuator_forcerange[aid], [-30, 30])

  def test_velocity_mode(self):
    """Velocity actuator: gainprm[0]=kd, biasprm[2]=-kd."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/j1"]
    )
    dev.configure_spec(
        spec,
        mode="velocity",
        damping=20.0,
        forcerange=40.0,
    )

    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/j1"
    )

    self.assertEqual(model.actuator_dyntype[aid], mujoco.mjtDyn.mjDYN_NONE)
    self.assertEqual(model.actuator_gaintype[aid], mujoco.mjtGain.mjGAIN_FIXED)
    self.assertEqual(model.actuator_biastype[aid], mujoco.mjtBias.mjBIAS_AFFINE)
    self.assertEqual(model.actuator_gainprm[aid, 0], 20.0)
    self.assertEqual(model.actuator_biasprm[aid, 2], -20.0)
    self.assertFalse(model.actuator_ctrllimited[aid])  # TODO: inheritrange
    self.assertTrue(model.actuator_forcelimited[aid])
    np.testing.assert_allclose(model.actuator_forcerange[aid], [-40, 40])

  def test_motor_mode(self):
    """Motor actuator: biastype=NONE, ctrl=force so both limited."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/j1"]
    )
    dev.configure_spec(
        spec,
        mode="motor",
        forcerange=50.0,
    )

    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/j1"
    )

    self.assertEqual(model.actuator_dyntype[aid], mujoco.mjtDyn.mjDYN_NONE)
    self.assertEqual(model.actuator_gaintype[aid], mujoco.mjtGain.mjGAIN_FIXED)
    self.assertEqual(model.actuator_biastype[aid], mujoco.mjtBias.mjBIAS_NONE)
    self.assertEqual(model.actuator_gainprm[aid, 0], 1.0)
    self.assertTrue(model.actuator_ctrllimited[aid])
    np.testing.assert_allclose(model.actuator_ctrlrange[aid], [-50, 50])
    self.assertTrue(model.actuator_forcelimited[aid])
    np.testing.assert_allclose(model.actuator_forcerange[aid], [-50, 50])

  def test_tendon_transmission(self):
    """Tendon transmission: trntype=TENDON, armature/frictionloss set."""
    spec = mujoco.MjSpec.from_string(TENDON_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/t1"],
        transmission="tendon",
    )
    dev.configure_spec(
        spec,
        mode="position",
        stiffness=50.0,
        damping=5.0,
        forcerange=20.0,
        armature=0.1,
        frictionloss=0.5,
    )

    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/t1"
    )
    tid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_TENDON), "robot/t1")

    self.assertGreaterEqual(aid, 0)
    self.assertEqual(model.actuator_trntype[aid], mujoco.mjtTrn.mjTRN_TENDON)
    self.assertEqual(model.tendon_frictionloss[tid], 0.5)
    self.assertEqual(model.actuator_gainprm[aid, 0], 50.0)

  def test_site_transmission(self):
    """Site transmission: trntype=SITE, 6-element gear."""
    spec = mujoco.MjSpec.from_string(SITE_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/s1"],
        transmission="site",
    )
    dev.configure_spec(
        spec,
        mode="motor",
        forcerange=10.0,
        gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/s1"
    )

    self.assertGreaterEqual(aid, 0)
    self.assertEqual(model.actuator_trntype[aid], mujoco.mjtTrn.mjTRN_SITE)
    self.assertEqual(model.actuator_biastype[aid], mujoco.mjtBias.mjBIAS_NONE)
    self.assertTrue(model.actuator_ctrllimited[aid])
    np.testing.assert_allclose(model.actuator_gear[aid, :6], [1, 0, 0, 0, 0, 0])

  def test_different_actuator_joint_names(self):
    """Spec with actuator name different than joint name compiles successfully."""
    spec = mujoco.MjSpec.from_string(DIFF_NAMES_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/my_actuator"]
    )
    # Should compile successfully without "Joint not found" ValueError
    dev.configure_spec(
        spec,
        use_existing=True,
    )
    model = spec.compile()
    aid = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), "robot/my_actuator"
    )
    self.assertGreaterEqual(aid, 0)

  def test_configure_spec_existing_dynamic_actuator(self):
    """ActuatorDevice skips re-adding dynamically added actuator when use_existing=True."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)

    # Simulate another component adding the actuator dynamically first
    act = spec.add_actuator(name="robot/j1", target="robot/j1")
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/j1"]
    )
    # If the bug is present (no fallback search), this will try to re-add
    # "robot/j1" and fail with ValueError: repeated name.
    # With the fallback search, it should find it, skip it, and succeed.
    dev.configure_spec(
        spec,
        use_existing=True,
    )
    model = spec.compile()
    self.assertIsNotNone(model)

  def test_configure_spec_existing_velocity_actuator(self):
    """ActuatorDevice finds dynamically added velocity actuator with _vel suffix."""
    # Base scene spec
    scene_spec = mujoco.MjSpec.from_string(MINIMAL_XML)

    # Child gripper spec loaded from XML
    gripper_spec = mujoco.MjSpec.from_string(GRIPPER_XML)

    # Attach gripper to scene with prefix "gripper/"
    frame = scene_spec.worldbody.add_frame(name="gripper_mount")
    scene_spec.attach(gripper_spec, prefix="gripper/", frame=frame)

    # Simulate VelocityActuatorSetup adding the velocity actuator
    # gripper/slider exists in scene_spec after attach.
    vel_act = scene_spec.add_actuator(
        name="gripper/slider_vel", target="gripper/slider"
    )
    vel_act.trntype = mujoco.mjtTrn.mjTRN_JOINT

    dev = actuator_device.ActuatorDevice(
        name="gripper", xpu_device="cpu", targets=["gripper/slider_vel"]
    )
    # This should fail with "repeated name" if it fails to find the existing.
    dev.configure_spec(
        scene_spec,
        use_existing=True,
    )
    model = scene_spec.compile()
    self.assertIsNotNone(model)

  def test_configure_spec_nonstandard_actuator_requires_use_existing(self):
    """Overwriting a <dcmotor> with use_existing=False raises NotImplementedError."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor name="j1" joint="j1" motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    dev = actuator_device.ActuatorDevice(
        name="dc", xpu_device="cpu", targets=["j1"]
    )
    with self.assertRaises(NotImplementedError):
      dev.configure_spec(spec, use_existing=False)


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
        nworld=nworld,
        xpu_device=xpu_device,
        targets=targets,
        measurements=measurements or ["actuator_force"],
    )
    dev.configure_spec(spec, use_existing=True)
    sim = simulator_module.Simulator(
        mj_spec=spec, nworld=nworld, impl=impl, xpu_device=xpu_device
    )
    dev.initialize(sim)
    return dev, sim

  def test_measurements_match_mj_data(self):
    """Measurements from entity match raw mj_data values."""
    spec = mujoco.MjSpec.from_string(MINIMAL_XML)
    dev = actuator_device.ActuatorDevice(
        name="robot", xpu_device="cpu", targets=["robot/j1"]
    )
    dev.configure_spec(spec, mode="position", stiffness=100.0)
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
        transmission="joint",
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
        transmission="joint",
    )
    dev.configure_spec(spec, mode="position")

    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/j1"],
        transmission="joint",
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
    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/t1"],
        transmission="tendon",
    )
    dev.configure_spec(
        spec,
        mode="position",
        stiffness=100.0,
    )

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[0] = 0.5
    mj_data.qpos[1] = 0.2
    mujoco.mj_forward(mj_model, mj_data)

    dev = actuator_device.ActuatorDevice(
        name="robot",
        xpu_device="cpu",
        targets=["robot/t1"],
        transmission="tendon",
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


if __name__ == "__main__":
  absltest.main()
