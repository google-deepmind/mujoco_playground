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
"""Tests for FreeDevice and MocapDevice."""

import math

from absl.testing import absltest
import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.common import free_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_free_body_spec() -> mujoco.MjSpec:
  """Create a spec with a freejoint body."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
      name="cube_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.02, 0.02, 0.02),
      mass=0.05,
  )
  return spec


def _make_mocap_body_spec() -> mujoco.MjSpec:
  """Create a spec with a mocap body."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="target", mocap=True)
  body.add_geom(
      name="target_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.02, 0.02, 0.02),
      rgba=(0, 1, 0, 0.5),
      contype=0,
      conaffinity=0,
  )
  return spec


def _setup_free_device(pos=None, quat=None):
  """Set up a FreeDevice with compiled model and data."""
  wp.init()
  wp.set_device(_DEVICE)
  spec = _make_free_body_spec()
  nworld = 4

  sim = simulator_module.Simulator(
      mj_spec=spec,
      nworld=nworld,
      impl=types.Impl.WARP,
  )
  if pos is not None:
    sim.data.qpos[:, :3] = torch.as_tensor(pos, device=sim.data.qpos.device)
  if quat is not None:
    sim.data.qpos[:, 3:7] = torch.as_tensor(quat, device=sim.data.qpos.device)
  if pos is not None or quat is not None:
    sim.forward()

  dev = free_device.FreeDevice(
      name="cube",
      body_name="cube",
      joint_name="cube_joint",
  )
  dev.initialize(sim)

  return dev, sim, nworld


def _setup_mocap_device():
  """Set up a MocapDevice with compiled model and data."""
  wp.init()
  wp.set_device(_DEVICE)
  spec = _make_mocap_body_spec()
  nworld = 4

  sim = simulator_module.Simulator(
      mj_spec=spec,
      nworld=nworld,
      impl=types.Impl.WARP,
  )

  dev = free_device.MocapDevice(
      name="target",
      body_name="target",
  )
  dev.initialize(sim)

  return dev, sim, nworld


class FreeDeviceTest(absltest.TestCase):
  """Tests for FreeDevice."""

  def test_name(self):
    dev, _, _ = _setup_free_device()
    self.assertEqual(dev.name, "cube")

  def test_set_pos(self):
    dev, sim, nworld = _setup_free_device()
    data = sim.data
    pos = torch.tensor([[0.1, 0.2, 0.3]] * nworld, device=data.qpos.device)
    dev.set_pos(None, pos)
    qpos = data.qpos[:, dev._qpos_adr : dev._qpos_adr + 3]
    self.assertTrue(torch.allclose(qpos, pos))

  def test_set_quat(self):
    dev, sim, nworld = _setup_free_device()
    data = sim.data
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * nworld, device=data.qpos.device
    )
    dev.set_quat(None, quat)
    data_quat = data.qpos[:, dev._qpos_adr + 3 : dev._qpos_adr + 7]
    self.assertTrue(torch.allclose(data_quat, quat))

  def test_set_vel(self):
    dev, sim, nworld = _setup_free_device()
    data = sim.data
    vel = torch.tensor(
        [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]] * nworld, device=data.qvel.device
    )
    dev.set_vel(None, vel)
    data_vel = data.qvel[:, dev._qvel_adr : dev._qvel_adr + 6]
    self.assertTrue(torch.allclose(data_vel, vel))

  def test_reset_is_noop(self):
    """Entity reset() is a no-op."""
    dev, sim, nworld = _setup_free_device()
    data = sim.data
    # Set some values
    pos = torch.tensor([[0.5, 0.5, 0.5]] * nworld)
    dev.set_pos(None, pos)
    initial_qpos = data.qpos[:, dev._qpos_adr : dev._qpos_adr + 7].clone()

    # Reset should not change anything
    dev.reset(sim)
    self.assertTrue(
        torch.allclose(
            data.qpos[:, dev._qpos_adr : dev._qpos_adr + 7], initial_qpos
        )
    )

  def test_substep_measurements_spec(self):
    dev, _, _ = _setup_free_device()
    spec = dev.substep_measurements_spec()
    self.assertIn("cube/xpos", spec)
    self.assertIn("cube/xquat", spec)

  def test_get_substep_measurements(self):
    pos = [0.5, -0.3, 1.2]
    quat_90z = [math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2]
    dev, sim, nworld = _setup_free_device(pos=pos, quat=quat_90z)

    measurements = dev.get_substep_measurements(sim)
    self.assertIn("cube/xpos", measurements)
    self.assertIn("cube/xquat", measurements)
    self.assertEqual(measurements["cube/xpos"].shape, (nworld, 3))
    self.assertEqual(measurements["cube/xquat"].shape, (nworld, 4))

    device = measurements["cube/xpos"].device
    expected_xpos = torch.tensor(pos, device=device)
    torch.testing.assert_close(
        measurements["cube/xpos"][0], expected_xpos, atol=1e-5, rtol=1e-5
    )
    expected_xquat = torch.tensor(quat_90z, device=device)
    torch.testing.assert_close(
        measurements["cube/xquat"][0], expected_xquat, atol=1e-5, rtol=1e-5
    )

  def test_configure_spec(self):
    spec = _make_free_body_spec()
    dev = free_device.FreeDevice(
        name="cube", body_name="cube", joint_name="cube_joint"
    )
    # configure_spec is mostly a no-op for FreeDevice currently but should be callable
    spec = dev.configure_spec(spec)
    self.assertIsInstance(spec, mujoco.MjSpec)


class MocapDeviceTest(absltest.TestCase):
  """Tests for MocapDevice."""

  def test_name(self):
    dev, _, _ = _setup_mocap_device()
    self.assertEqual(dev.name, "target")

  def test_set_pos(self):
    dev, sim, nworld = _setup_mocap_device()
    data = sim.data
    pos = torch.tensor([[0.1, 0.2, 0.3]] * nworld, device=data.mocap_pos.device)
    dev.set_pos(None, pos)
    mocap_pos = data.mocap_pos[:, dev._mocap_id]
    self.assertTrue(torch.allclose(mocap_pos, pos))

  def test_set_quat(self):
    dev, sim, nworld = _setup_mocap_device()
    data = sim.data
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * nworld, device=data.mocap_quat.device
    )
    dev.set_quat(None, quat)
    mocap_quat = data.mocap_quat[:, dev._mocap_id]
    self.assertTrue(torch.allclose(mocap_quat, quat))

  def test_reset_is_noop(self):
    """Entity reset() is a no-op."""
    dev, sim, nworld = _setup_mocap_device()
    data = sim.data
    pos = torch.tensor([[0.5, 0.5, 0.5]] * nworld)
    dev.set_pos(None, pos)
    initial_pos = data.mocap_pos[:, dev._mocap_id].clone()

    dev.reset(sim)
    self.assertTrue(
        torch.allclose(data.mocap_pos[:, dev._mocap_id], initial_pos)
    )

  def test_substep_measurements_spec(self):
    dev, _, _ = _setup_mocap_device()
    spec = dev.substep_measurements_spec()
    self.assertIn("target/mocap_pos", spec)
    self.assertIn("target/mocap_quat", spec)

  def test_get_substep_measurements(self):
    dev, sim, nworld = _setup_mocap_device()
    data = sim.data

    pos = torch.tensor([[0.1, 0.2, 0.3]] * nworld, device=data.mocap_pos.device)
    dev.set_pos(None, pos)
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * nworld, device=data.mocap_quat.device
    )
    dev.set_quat(None, quat)

    measurements = dev.get_substep_measurements(sim)
    self.assertIn("target/mocap_pos", measurements)
    self.assertIn("target/mocap_quat", measurements)
    self.assertEqual(measurements["target/mocap_pos"].shape, (4, 3))
    self.assertEqual(measurements["target/mocap_quat"].shape, (4, 4))

    torch.testing.assert_close(
        measurements["target/mocap_pos"][0], pos[0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        measurements["target/mocap_quat"][0], quat[0], atol=1e-5, rtol=1e-5
    )

  def test_configure_spec(self):
    spec = _make_mocap_body_spec()
    dev = free_device.MocapDevice(name="target", body_name="target")
    spec = dev.configure_spec(spec, geom_group=1)
    self.assertIsInstance(spec, mujoco.MjSpec)
    body = spec.body("target")
    self.assertEqual(body.geoms[0].group, 1)


if __name__ == "__main__":
  absltest.main()
