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
"""Tests wrapping MuJoCo Warp structs with TorchTensor."""

import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


XML = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="floor">
      <geom name="floor" type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
    </body>
    <body name="sphere" pos="0 0 1">
      <joint type="free"/>
      <geom name="sphere" type="sphere" size="0.1" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class TorchTensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.nworld = 4
    wp.init()
    wp.set_device(_DEVICE)
    self.device = _DEVICE
    with wp.ScopedDevice(self.device):
      qpos: Any = wp.from_torch(
          torch.randn(
              (self.nworld, 17), device=self.device, dtype=torch.float32
          ),
          dtype=wp.float32,
      )
    self.qpos = torch_wrapper.TorchTensor(qpos)

  def test_operators(self):
    qpos = self.qpos
    qpos_tensor = qpos._tensor
    torch.testing.assert_close(qpos + 1, qpos_tensor + 1)
    torch.testing.assert_close(1 + qpos, 1 + qpos_tensor)
    torch.testing.assert_close(qpos - 1, qpos_tensor - 1)
    torch.testing.assert_close(1 - qpos, 1 - qpos_tensor)
    torch.testing.assert_close(qpos * 2, qpos_tensor * 2)
    torch.testing.assert_close(2 * qpos, 2 * qpos_tensor)
    torch.testing.assert_close(qpos / 2, qpos_tensor / 2)
    torch.testing.assert_close(2 / qpos, 2 / qpos_tensor)
    torch.testing.assert_close(-qpos, -qpos_tensor)
    torch.testing.assert_close(abs(qpos), abs(qpos_tensor))

  def test_torch_function(self):
    qpos: Any = self.qpos
    qpos_tensor = qpos._tensor
    torch.testing.assert_close(torch.mean(qpos), torch.mean(qpos_tensor))
    torch.testing.assert_close(
        torch.all(qpos > -100), torch.all(qpos_tensor > -100)
    )

  def test_set_raises(self):
    qpos = self.qpos
    m = mujoco.MjModel.from_xml_string(XML)
    mw = mjw.put_model(m)
    with self.assertRaises(AttributeError):
      qpos._tensor = torch.randn(self.nworld, mw.nq)  # pyrefly: ignore[read-only]

  def test_numpy_to_torch_tensor(self):
    np_array = np.random.randn(10, 10)
    torch_tensor = torch_wrapper.TorchTensor(np_array)
    torch.testing.assert_close(torch_tensor._tensor, torch.from_numpy(np_array))


class TorchTensorCPUTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.nworld = 4
    self.np_array = np.random.randn(self.nworld, 17).astype(np.float32)

  def test_numpy_array_preserved(self):
    qpos = torch_wrapper.TorchTensor(self.np_array)
    torch.testing.assert_close(qpos._tensor, torch.from_numpy(self.np_array))

  def test_set_raises(self):
    qpos = torch_wrapper.TorchTensor(self.np_array)
    with self.assertRaises(AttributeError):
      qpos._tensor = torch.randn(self.nworld, 17)  # pyrefly: ignore[read-only]

  def test_setitem(self):
    qpos = torch_wrapper.TorchTensor(self.np_array)
    new_val = np.random.randn(17).astype(np.float32)
    qpos[0] = torch.from_numpy(new_val)
    torch.testing.assert_close(qpos[0], torch.from_numpy(new_val))
    torch.testing.assert_close(qpos[1:], torch.from_numpy(self.np_array[1:]))
    np.testing.assert_array_equal(qpos._array[0], new_val)
    np.testing.assert_array_equal(qpos._array[1:], qpos[1:])


class ScalarTensorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    class MockParent:
      time: float = 0.002

    self.parent = MockParent()

  def test_scalar_tensor_init(self):
    scalar = torch_wrapper.ScalarTensor(self.parent, "time", add_batch_dim=True)
    self.assertEqual(scalar.shape, (1,))
    self.assertEqual(scalar[0].item(), 0.002)

  def test_scalar_tensor_write_back(self):
    scalar = torch_wrapper.ScalarTensor(self.parent, "time", add_batch_dim=True)
    scalar[0] = 99.0
    self.assertEqual(self.parent.time, 99.0)
    self.assertEqual(scalar[0].item(), 99.0)  # pyrefly: ignore[missing-attribute]

  def test_scalar_tensor_operators(self):
    scalar = torch_wrapper.ScalarTensor(self.parent, "time", add_batch_dim=True)
    torch.testing.assert_close(
        scalar + 1, torch.tensor([1.002], dtype=torch.float64)
    )
    torch.testing.assert_close(
        scalar * 2, torch.tensor([0.004], dtype=torch.float64)
    )

  def test_scalar_tensor_re_read_on_non_subscript_access(self):
    scalar = torch_wrapper.ScalarTensor(self.parent, "time", add_batch_dim=True)
    self.assertEqual(scalar.item(), 0.002)

    # Mutate parent time (e.g. mj_step advancing time in C++)
    self.parent.time = 0.004

    # Verify non-subscript operations see the updated parent value
    self.assertEqual(scalar.item(), 0.004)
    cloned = scalar.clone()
    self.assertEqual(cloned.item(), 0.004)
    torch.testing.assert_close(
        scalar + 1, torch.tensor([1.004], dtype=torch.float64)
    )
    self.assertIn("0.004", repr(scalar))


class TorchWrapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.nworld = 4
    wp.init()
    wp.set_device(_DEVICE)
    self.device = _DEVICE
    m = mujoco.MjModel.from_xml_string(XML)
    self.mw = mjw.put_model(m)
    with wp.ScopedDevice(self.device):
      self.dw = mjw.make_data(m, nworld=self.nworld)

  def test_qpos_type_and_shape(self):
    dw = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)
    self.assertIsInstance(dw.qpos, torch_wrapper.TorchTensor)
    self.assertIsInstance(dw.qpos[:], torch.Tensor)
    self.assertIsInstance(dw.qpos._array, wp.array)
    self.assertEqual(dw.qpos.shape, (self.nworld, self.mw.nq))
    self.assertEqual(dw.qpos.shape, dw.qpos._array.shape)

  def test_qpos_set_and_get(self):
    dw = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)

    ptr_before = dw.qpos._array.ptr
    qpos_val = torch.randn(
        self.nworld,
        self.mw.nq,
        device=dw.qpos.device,
        dtype=dw.qpos.dtype,
    )
    dw.qpos[:] = qpos_val
    torch.testing.assert_close(dw.qpos[:], qpos_val)
    np.testing.assert_array_equal(
        dw.qpos._array.numpy(), qpos_val.cpu().numpy()
    )
    self.assertEqual(dw.qpos._array.ptr, ptr_before)

  def test_qpos_set_item(self):
    dw = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)

    ptr_before = dw.qpos._array.ptr
    qpos_val = torch.randn(
        self.mw.nq, device=dw.qpos.device, dtype=dw.qpos.dtype
    )
    dw.qpos[0] = qpos_val
    torch.testing.assert_close(dw.qpos[0], qpos_val)
    np.testing.assert_array_equal(
        dw.qpos._array.numpy()[0], qpos_val.cpu().numpy()
    )
    self.assertEqual(dw.qpos._array.ptr, ptr_before)

  def test_qpos_set_raises(self):
    dw = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)
    with self.assertRaises(AttributeError):
      dw.qpos = torch.randn(self.nworld, self.mw.nq)

  def test_geom_pos_tile_on_access(self):
    m = mujoco.MjModel.from_xml_string(XML)
    mw = mjw.put_model(m)
    self.assertTrue(mw.geom_pos._is_batched)
    self.assertEqual(mw.geom_pos.shape[0], 1)

    mwt = torch_wrapper.TorchWrapper(mw, nworld=self.nworld)
    self.assertEqual(mwt.geom_pos.shape, (self.nworld, 2, 3))
    self.assertEqual(mwt.obj.geom_pos.shape, mw.geom_pos.shape)
    self.assertEqual(mwt.obj.geom_pos.dtype, wp.vec3)

  def test_register_graph_capture_callback(self):
    m = mujoco.MjModel.from_xml_string(XML)
    mw = mjw.put_model(m)
    mwt = torch_wrapper.TorchWrapper(mw, nworld=self.nworld)

    trigger_count = [0]

    def cb():
      trigger_count[0] += 1

    mwt.register_graph_capture_callback(cb)

    # nworld 1 field (geom_pos) has not been tiled yet.
    self.assertEqual(mwt.obj.geom_pos.shape[0], 1)
    self.assertEqual(trigger_count[0], 0)
    # Accessing geom_pos triggers tiling and graph capture callback.
    _ = mwt.geom_pos
    self.assertEqual(mwt.obj.geom_pos.shape[0], self.nworld)
    self.assertEqual(trigger_count[0], 1)

    # Accessing geom_pos again should NOT trigger callback again.
    _ = mwt.geom_pos
    self.assertEqual(trigger_count[0], 1)

  def test_unsupported_field(self):
    dw = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)
    with self.assertRaises(AttributeError):
      _ = dw.contact
    with self.assertRaises(AttributeError):
      _ = dw.efc
    with self.assertRaises(AttributeError):
      _ = dw.state

  def test_opt_field_access(self):
    m = mujoco.MjModel.from_xml_string(XML)
    mw = torch_wrapper.TorchWrapper(mjw.put_model(m), nworld=self.nworld)
    opt = mw.opt
    self.assertIsInstance(opt, torch_wrapper.TorchWrapper)
    self.assertIsInstance(opt.gravity, torch_wrapper.TorchTensor)
    np.testing.assert_array_almost_equal(
        opt.gravity[:].cpu().numpy()[0], m.opt.gravity
    )
    with self.assertRaises(AttributeError):
      _ = opt.invalid_field


class TorchWrapperCPUTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.set_device(_DEVICE)
    self.device = "cpu"
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    self.m, self.d = m, d

  def test_qpos_type_and_shape(self):
    dw = torch_wrapper.TorchWrapper(self.d, nworld=1)
    self.assertIsInstance(dw.qpos, torch_wrapper.TorchTensor)
    self.assertIsInstance(dw.qpos[:], torch.Tensor)
    self.assertIsInstance(dw.qpos._array, np.ndarray)
    self.assertEqual(dw.qpos.shape, (1, self.m.nq))
    self.assertEqual(dw.qpos._array.shape, (self.m.nq,))

  def test_qpos_set_and_get(self):
    dw = torch_wrapper.TorchWrapper(self.d, nworld=1)

    qpos_val = torch.randn(
        1,
        self.m.nq,
        device=dw.qpos.device,
        dtype=dw.qpos.dtype,
    )
    dw.qpos[:] = qpos_val
    torch.testing.assert_close(dw.qpos[:], qpos_val)
    np.testing.assert_array_equal(dw.qpos._array, qpos_val.cpu().numpy()[0])

  def test_qpos_set_item(self):
    dw = torch_wrapper.TorchWrapper(self.d, nworld=1)

    qpos_val = torch.randn(
        self.m.nq, device=dw.qpos.device, dtype=dw.qpos.dtype
    )
    dw.qpos[0] = qpos_val
    torch.testing.assert_close(dw.qpos[0], qpos_val)
    np.testing.assert_array_equal(dw.qpos._array, qpos_val.cpu().numpy())

  def test_qpos_set_raises(self):
    dw = torch_wrapper.TorchWrapper(self.d, nworld=1)
    with self.assertRaises(AttributeError):
      dw.qpos = torch.randn(1, self.m.nq)

  def test_model_field_batch_dim(self):
    mw = torch_wrapper.TorchWrapper(self.m, nworld=1)
    # Batched fields having leading batch dim.
    self.assertEqual(mw.geom_pos.shape, (1, self.m.ngeom, 3))
    self.assertEqual(mw.geom_pos._array.shape, (self.m.ngeom, 3))
    # Non-batched fields do NOT have leading batch dim.
    self.assertEqual(mw.body_parentid.shape, (self.m.nbody,))
    self.assertEqual(mw.body_parentid._array.shape, (self.m.nbody,))

  def test_model_values_match(self):
    mw = torch_wrapper.TorchWrapper(self.m)
    np.testing.assert_array_equal(mw.geom_pos[0].numpy(), self.m.geom_pos)
    np.testing.assert_array_equal(
        mw.body_parentid[:].numpy(), self.m.body_parentid
    )

  def test_batched_opt_field(self):
    mw = torch_wrapper.TorchWrapper(self.m)
    self.assertIsInstance(mw.opt.gravity, torch_wrapper.TorchTensor)
    self.assertEqual(mw.opt.gravity.shape, (1, 3))
    np.testing.assert_array_almost_equal(
        mw.opt.gravity.numpy()[0], self.m.opt.gravity
    )

  def test_scalar_field(self):
    mujoco.mj_step(self.m, self.d)
    dw = torch_wrapper.TorchWrapper(self.d)
    self.assertIsInstance(dw.time, torch_wrapper.TorchTensor)
    self.assertEqual(dw.time.shape, (1,))
    self.assertEqual(dw.time[0].item(), self.d.time)

    # set scalar field
    original_time = self.d.time
    dw.time[0] = 99.0
    self.assertEqual(self.d.time, 99.0)
    self.assertNotEqual(self.d.time, original_time)

    # test scalar field reflects changes after mj_step
    initial_time = dw.time[0].clone()  # pyrefly: ignore[missing-attribute]
    mujoco.mj_step(self.m, self.d)
    new_time = dw.time[0]
    expected_time = initial_time + self.m.opt.timestep
    torch.testing.assert_close(
        new_time.to(torch.float32), expected_time.to(torch.float32)  # pyrefly: ignore[missing-attribute]
    )

  def test_values_sync_to_mjdata(self):
    dw = torch_wrapper.TorchWrapper(self.d)
    dw.qpos[0] = torch.arange(self.m.nq, dtype=dw.qpos.dtype)
    np.testing.assert_array_equal(self.d.qpos, np.arange(self.m.nq))

  def test_unsupported_model_field(self):
    mw = torch_wrapper.TorchWrapper(self.m)
    with self.assertRaises(AttributeError):
      _ = mw.stat
    with self.assertRaises(AttributeError):
      _ = mw.invalid_field


def wrap_model(
    backend: types.Impl, m: mujoco.MjModel, nworld: int = 1
) -> torch_wrapper.TorchWrapper:
  if backend == types.Impl.WARP:
    return torch_wrapper.TorchWrapper(mjw.put_model(m), nworld=nworld)
  return torch_wrapper.TorchWrapper(m, nworld=nworld)


def wrap_data(
    backend: types.Impl, m: mujoco.MjModel, d: mujoco.MjData, nworld: int = 1
) -> torch_wrapper.TorchWrapper:
  if backend == types.Impl.WARP:
    return torch_wrapper.TorchWrapper(mjw.put_data(m, d, nworld=nworld))
  return torch_wrapper.TorchWrapper(d, nworld=nworld)


class BindTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.set_device(_DEVICE)
    wp.config.kernel_cache_dir = "/tmp/kernel_cache"

  XML = """
  <mujoco model="test_bind_model">
      <worldbody>
        <body pos="10 20 30" name="body1">
          <joint axis="1 0 0" type="ball" name="joint1"/>
          <geom size="1 2 3" type="box" name="geom1" pos="0 1 0"/>
        </body>
        <body pos="40 50 60" name="body2">
          <joint axis="0 1 0" type="slide" name="joint2"/>
          <geom size="4 5 6" type="box" name="geom2"/>
        </body>
        <body pos="70 80 90" name="body3">
          <joint axis="0 0 1" type="slide" name="joint3"/>
          <geom size="7 8 9" type="box" name="geom3"/>
        </body>
        <body pos="100 110 120" name="body4" mocap="true"/>
        <body pos="130 140 150" name="body5" mocap="true"/>
      </worldbody>

      <actuator>
        <motor joint="joint1"/>
        <motor joint="joint2"/>
        <motor joint="joint3"/>
      </actuator>

      <sensor>
        <framepos objtype="body" objname="body1"/>
        <framepos objtype="body" objname="body2"/>
        <framepos objtype="body" objname="body3"/>
      </sensor>
  </mujoco>
  """

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_bind(self, backend: types.Impl):
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    d = mujoco.MjData(m)
    mw = wrap_model(backend, m, nworld=1)
    dw = wrap_data(backend, m, d)
    mujoco.mj_step(m, d)
    if backend == types.Impl.WARP:
      mjw.step(mw.obj, dw.obj)

    # Test binding data.
    qpos = dw.bind(mw, s.joints).qpos
    np.testing.assert_array_almost_equal(
        qpos.cpu().numpy()[0], d.bind(s.joints).qpos
    )
    # Test that the bound accessor is cached.
    bound1 = dw.bind(mw, s.joints)
    bound2 = dw.bind(mw, s.joints)
    self.assertEqual(bound1, bound2)
    self.assertLen(bound1._index_cache, 1)
    # Test that the accessor indices are unique on device.
    self.assertEqual(
        bound1._index_cache["qpos"].data_ptr(),
        bound2._index_cache["qpos"].data_ptr(),
    )

    # Test setting on a bound data accessor.
    dw.bind(mw, s.joint("joint3")).set("qpos", torch.tensor([[42.0]]))
    out = dw.qpos[:, mw.jnt_qposadr[s.joint("joint3").id]]
    self.assertEqual(out[0], 42.0)
    # Check the underlying array directly.
    if backend == types.Impl.WARP:
      out = dw.obj.qpos.numpy()[:, mw.jnt_qposadr[s.joint("joint3").id]][0]
    elif backend == types.Impl.CPP:
      out = dw.obj.qpos[mw.jnt_qposadr[s.joint("joint3").id]]
    self.assertEqual(out, 42.0)

    # Test binding model.
    geom_pos = mw.bind(s.geoms).pos
    np.testing.assert_array_equal(geom_pos.cpu().numpy()[0], m.geom_pos)
    # Test that the bound accessor is cached.
    bound1 = mw.bind(s.geoms)
    bound2 = mw.bind(s.geoms)
    self.assertEqual(bound1, bound2)
    self.assertLen(mw._bound_cache, 1)

    # Test setting on a bound model accessor.
    mw.bind(s.geom("geom1")).set("pos", torch.tensor([42.0, 1.0, 2.1]))
    out = mw.geom_pos[:, s.geom("geom1").id].cpu().numpy()
    np.testing.assert_array_equal(
        out[0], torch.tensor([42.0, 1.0, 2.1], dtype=torch.float32)
    )
    # Check the underlying array directly.
    if backend == types.Impl.WARP:
      out_geom = mw.obj.geom_pos.numpy()[:, s.geom("geom1").id][0]
    elif backend == types.Impl.CPP:
      out_geom = mw.obj.geom_pos[s.geom("geom1").id]
    np.testing.assert_array_almost_equal(
        out_geom, np.array([42.0, 1.0, 2.1], dtype=float)  # pyrefly: ignore[unbound-name]
    )

    # Test setting explicit dof_ fields on bound joint accessors.
    mw.bind(s.joint("joint2")).set("dof_damping", torch.tensor([[13.37]]))
    out_damp = mw.dof_damping[:, 3].cpu().numpy()
    self.assertAlmostEqual(out_damp[0], 13.37, places=5)
    if backend == types.Impl.WARP:
      out_damp_raw = mw.obj.dof_damping.numpy()[:, 3][0]
    elif backend == types.Impl.CPP:
      out_damp_raw = mw.obj.dof_damping[3]
    self.assertAlmostEqual(out_damp_raw, 13.37, places=5)  # pyrefly: ignore[unbound-name]

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_bind_get(self, backend: types.Impl):
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    d = mujoco.MjData(m)
    mw = wrap_model(backend, m, nworld=1)
    dw = wrap_data(backend, m, d)
    mujoco.mj_step(m, d)
    if backend == types.Impl.WARP:
      mjw.step(mw.obj, dw.obj)

    # Test getting.
    np.testing.assert_array_equal(
        mw.bind(s.bodies).pos.cpu().numpy()[0], m.body_pos
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies).xpos.cpu().numpy()[0], d.xpos
    )
    np.testing.assert_array_equal(m.bind(s.bodies[0]).mass, m.body_mass[0])
    np.testing.assert_array_equal(m.bind(s.bodies[0:1]).mass, [m.body_mass[0]])
    np.testing.assert_array_equal(
        mw.bind(s.bodies[0]).mass.cpu().numpy()[0], m.body_mass[0]
    )
    np.testing.assert_array_equal(
        mw.bind(s.bodies[0:1]).mass.cpu().numpy()[0], [m.body_mass[0]]
    )
    for i in range(m.nbody):
      np.testing.assert_array_equal(m.bind(s.bodies[i]).pos, m.body_pos[i, :])
      np.testing.assert_array_equal(
          mw.bind(s.bodies[i]).pos.cpu().numpy()[0], m.body_pos[i, :]
      )
      np.testing.assert_array_equal(d.bind(s.bodies[i]).xpos, d.xpos[i, :])
      np.testing.assert_array_equal(
          dw.bind(mw, s.bodies[i]).xpos.cpu().numpy()[0], d.xpos[i, :]
      )
      np.testing.assert_array_equal(
          dw.bind(mw, s.bodies[i]).xfrc_applied.cpu().numpy()[0],
          d.xfrc_applied[i, :],
      )

    np.testing.assert_array_equal(
        mw.bind(s.geoms).size.cpu().numpy()[0], m.geom_size
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.geoms).xpos.cpu().numpy()[0], d.geom_xpos
    )
    for i in range(m.ngeom):
      np.testing.assert_array_equal(m.bind(s.geoms[i]).size, m.geom_size[i, :])
      np.testing.assert_array_equal(
          mw.bind(s.geoms[i]).size.cpu().numpy()[0], m.geom_size[i, :]
      )
      np.testing.assert_array_equal(d.bind(s.geoms[i]).xpos, d.geom_xpos[i, :])
      np.testing.assert_array_equal(
          dw.bind(mw, s.geoms[i]).xpos.cpu().numpy()[0], d.geom_xpos[i, :]
      )

    dw = wrap_data(backend, m, d)
    np.testing.assert_array_equal(
        mw.bind(s.joints).axis.cpu().numpy()[0], m.jnt_axis
    )
    np.testing.assert_array_equal(
        mw.bind(s.joints).qposadr.cpu().numpy(), m.jnt_qposadr
    )
    np.testing.assert_array_equal(
        mw.bind(s.joints).dofadr.cpu().numpy(), m.jnt_dofadr
    )
    np.testing.assert_array_equal(dw.bind(mw, s.joints[1]).id.cpu().numpy(), 1)
    np.testing.assert_array_equal(
        dw.bind(mw, s.joints[1:2]).id.cpu().numpy(), 1
    )
    qposnum = [4, 1, 1]  # one ball joint (4) and two slide joints (1)
    dofnum = [3, 1, 1]  # one ball joint (3) and two slide joints (1)
    for i in range(m.njnt):
      np.testing.assert_array_equal(m.bind(s.joints[i]).axis, m.jnt_axis[i, :])
      np.testing.assert_array_equal(
          mw.bind(s.joints[i]).axis.cpu().numpy()[0], m.jnt_axis[i, :]
      )
      np.testing.assert_array_almost_equal(
          dw.bind(mw, s.joints[i]).qpos.cpu().numpy()[0],
          d.qpos[m.jnt_qposadr[i] : m.jnt_qposadr[i] + qposnum[i]],
          decimal=6,
      )
      np.testing.assert_array_almost_equal(
          dw.bind(mw, s.joints[i]).qvel.cpu().numpy()[0],
          d.qvel[m.jnt_dofadr[i] : m.jnt_dofadr[i] + dofnum[i]],
          decimal=6,
      )
      np.testing.assert_array_almost_equal(
          dw.bind(mw, s.joints[i]).qacc.cpu().numpy()[0],
          d.qacc[m.jnt_dofadr[i] : m.jnt_dofadr[i] + dofnum[i]],
          decimal=6,
      )
      np.testing.assert_array_almost_equal(
          dw.bind(mw, s.joints[i]).qfrc_actuator.cpu().numpy()[0],
          d.qfrc_actuator[m.jnt_dofadr[i] : m.jnt_dofadr[i] + dofnum[i]],
          decimal=6,
      )

    # Test explicit dof_ field mapping on joint bindings (preserves order, resolves multi-DOF joints)
    np.testing.assert_array_equal(
        mw.bind(s.joints).dof_damping.cpu().numpy()[0], m.dof_damping
    )
    np.testing.assert_array_equal(
        mw.bind(s.joint("joint2")).dof_damping.cpu().numpy()[0],
        m.dof_damping[3:4],
    )
    # Joint 2 is index 3 (1 DOF), Joint 1 is ball (3 DOFs: indices 0, 1, 2)
    np.testing.assert_array_equal(
        mw.bind([s.joint("joint2"), s.joint("joint1")])
        .dof_damping.cpu()
        .numpy()[0],
        np.concatenate([m.dof_damping[3:4], m.dof_damping[0:3]]),
    )

    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], d.ctrl
    )
    for i in range(m.nu):
      np.testing.assert_array_equal(d.bind(s.actuators[i]).ctrl, d.ctrl[i])
      np.testing.assert_array_equal(
          dw.bind(mw, s.actuators[i]).ctrl.cpu().numpy()[0], d.ctrl[i]
      )

    np.testing.assert_array_equal(
        dw.bind(mw, s.sensors).sensordata.cpu().numpy()[0], d.sensordata
    )
    for i in range(m.nsensor):
      np.testing.assert_array_equal(
          dw.bind(mw, s.sensors[i]).sensordata.cpu().numpy()[0],
          d.sensordata[m.sensor_adr[i] : m.sensor_adr[i] + m.sensor_dim[i]],
      )

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_bind_set(self, backend: types.Impl):
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    d = mujoco.MjData(m)
    mw = wrap_model(backend, m, nworld=1)
    dw = wrap_data(backend, m, d)
    mujoco.mj_step(m, d)
    if backend == types.Impl.WARP:
      mjw.step(mw.obj, dw.obj)

    # test setting
    dw = wrap_data(backend, m, d)
    np.testing.assert_array_equal(d.ctrl, [0, 0, 0])
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], d.ctrl
    )
    dw.bind(mw, s.actuators).set(
        "ctrl",
        torch.tensor([1, 2, 3], dtype=dw.ctrl.dtype, device=dw.qpos.device),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], [1, 2, 3]
    )
    dw.bind(mw, s.actuators[1:]).set(
        "ctrl", torch.tensor([4, 5], dtype=dw.ctrl.dtype, device=dw.qpos.device)
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], [1, 4, 5]
    )
    dw.bind(mw, s.actuators[1]).set(
        "ctrl", torch.tensor([6], dtype=dw.ctrl.dtype, device=dw.qpos.device)
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], [1, 6, 5]
    )
    dw.bind(mw, s.actuators[1]).set(
        "ctrl", torch.tensor(7, dtype=dw.ctrl.dtype, device=dw.qpos.device)
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], [1, 7, 5]
    )

    qpos_1step = [1.00000e00, -3.67875e-06, 0, 0, 0, -3.924e-05]
    qpos_desired = [1, 0, 0, 0, 0, 8]
    np.testing.assert_array_almost_equal(d.qpos, qpos_1step)
    np.testing.assert_array_almost_equal(
        dw.bind(mw, s.joints).qpos.cpu().numpy()[0], d.qpos
    )
    dw.bind(mw, s.joints[::2]).set(
        "qpos",
        torch.tensor(
            [1, 0, 0, 0, 8], dtype=dw.qpos.dtype, device=dw.qpos.device
        ),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.joints).qpos.cpu().numpy()[0], qpos_desired
    )
    dw.bind(mw, s.joints[0]).set(
        "qpos",
        torch.tensor(
            qpos_desired[:4], dtype=dw.qpos.dtype, device=dw.qpos.device
        ),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.joints[0]).qpos.cpu().numpy()[0], qpos_desired[:4]
    )
    dw.bind(mw, s.joints[::2]).set(
        "qvel",
        torch.tensor(
            [2.0, -1.2, 0.5, 0.3], dtype=dw.qvel.dtype, device=dw.qpos.device
        ),
    )
    np.testing.assert_array_almost_equal(
        dw.bind(mw, s.joints).qvel.cpu().numpy()[0],
        [2.0, -1.2, 0.5, 0.0, 0.3],
        decimal=6,
    )
    dw.bind(mw, s.joints[::2]).set(
        "qacc",
        torch.tensor(
            [3.0, -2.1, 0.6, 0.4], dtype=dw.qacc.dtype, device=dw.qpos.device
        ),
    )
    np.testing.assert_array_almost_equal(
        dw.bind(mw, s.joints).qacc.cpu().numpy()[0],
        [3.0, -2.1, 0.6, 0.0, 0.4],
        decimal=6,
    )

    dw.bind(mw, s.bodies[1]).set(
        "xfrc_applied",
        torch.tensor(
            [1, 2, 3, 4, 5, 6],
            dtype=dw.xfrc_applied.dtype,
            device=dw.qpos.device,
        ),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies[1]).xfrc_applied.cpu().numpy()[0],
        [1, 2, 3, 4, 5, 6],
    )
    for body in s.bodies[:1] + s.bodies[2:]:
      np.testing.assert_array_equal(
          dw.bind(mw, body).xfrc_applied.cpu().numpy()[0], [0, 0, 0, 0, 0, 0]
      )
    dw.bind(mw, s.bodies[0:2]).set(
        "xfrc_applied",
        torch.tensor(
            np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]),
            dtype=dw.xfrc_applied.dtype,
            device=dw.qpos.device,
        ),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies[0]).xfrc_applied.cpu().numpy()[0],
        [1, 1, 1, 1, 1, 1],
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies[1]).xfrc_applied.cpu().numpy()[0],
        [2, 2, 2, 2, 2, 2],
    )
    dw = wrap_data(backend, m, d)
    dw.bind(mw, s.bodies[-1]).set(
        "mocap_pos",
        torch.tensor(
            [1, 2, 3], dtype=dw.mocap_pos.dtype, device=dw.mocap_pos.device
        ),
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies[-2]).mocap_pos.cpu().numpy()[0], [100, 110, 120]
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.bodies[-1]).mocap_pos.cpu().numpy()[0], [1, 2, 3]
    )

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_assertions(self, backend: types.Impl):
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    d = mujoco.MjData(m)
    mw = wrap_model(backend, m, nworld=1)
    dw = wrap_data(backend, m, d)
    mujoco.mj_step(m, d)
    if backend == types.Impl.WARP:
      mjw.step(mw.obj, dw.obj)

    # test bind() argument validation
    with self.assertRaisesRegex(TypeError, "takes 2 positional arguments"):
      dw.bind()  # pyrefly: ignore[no-matching-overload]
    with self.assertRaisesRegex(TypeError, "takes 2 positional arguments"):
      dw.bind(s.geoms)
    with self.assertRaisesRegex(
        TypeError, "requires a model wrapper as the first argument"
    ):
      dw.bind(s.geoms, mw)
    with self.assertRaisesRegex(ValueError, "specs must not be empty"):
      dw.bind(mw, [])
    with self.assertRaisesRegex(TypeError, "takes 1 positional argument"):
      mw.bind()  # pyrefly: ignore[no-matching-overload]
    with self.assertRaisesRegex(TypeError, "takes 1 positional argument"):
      mw.bind(mw, s.geoms)
    with self.assertRaisesRegex(ValueError, "specs must not be empty"):
      mw.bind([])

    # test attribute and type mismatches
    with self.assertRaisesRegex(
        AttributeError, "ctrl is not available for this type"
    ):
      print(dw.bind(mw, s.geoms).ctrl)
    with self.assertRaises(AttributeError):
      print(dw.bind(mw, s.actuators).actuator_ctrl)
    with self.assertRaises(AttributeError):
      print(
          dw.bind(mw, s.actuators).set(
              "actuator_ctrl", [1, 2, 3]  # pyrefly: ignore[bad-argument-type]
          )
      )
    with self.assertRaisesRegex(
        AttributeError, "qpos, qvel, qacc, qfrc are not available for this type"
    ):
      print(dw.bind(mw, s.geoms).qpos)

    # test that modified names do not raise an error
    s.actuators[0].name = "modified_actuator_name"
    np.testing.assert_array_equal(
        dw.bind(mw, s.actuators).ctrl.cpu().numpy()[0], d.ctrl
    )
    s.geoms[0].name = "modified_geom_name"
    np.testing.assert_array_equal(
        mw.bind(s.geoms[0]).pos.cpu().numpy()[0], m.geom_pos[0, :]
    )

    # what happens when we bind to an actuator that was removed?
    bygone_actuators = []
    for act in s.actuators:
      bygone_actuators.append(act)
      s.delete(act)
    m = s.compile()
    d = mujoco.MjData(m)
    mw = wrap_model(backend, m, nworld=1)
    dw = wrap_data(backend, m, d)
    with self.assertRaisesRegex(ValueError, "Invalid spec ID: -1"):
      dw.bind(mw, bygone_actuators)

    # test that adding a body requires recompilation
    s.worldbody.add_body()
    with self.assertRaises(ValueError):
      mw.bind(s.bodies)

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_batched(self, backend: types.Impl):
    if backend != types.Impl.WARP:
      self.skipTest("Batching only supported for mujoco_warp backend.")
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    d = mujoco.MjData(m)

    # test batched data
    batch_size = 16
    mw = wrap_model(backend, m, nworld=batch_size)
    dw = wrap_data(backend, m, d, nworld=batch_size)
    device = dw.qpos.device
    for i in range(m.nbody):
      np.testing.assert_array_equal(
          dw.bind(mw, s.bodies[i]).xpos.cpu().numpy(),
          [d.xpos[i, :]] * batch_size,
      )

    # set batched data value with env_mask
    env_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    env_mask[2:4] = True
    dw.bind(mw, s.joints[1:3]).set(
        "qpos",
        torch.tensor([1, 2], dtype=dw.qpos.dtype, device=device),
        env_mask,
    )
    np.testing.assert_array_equal(
        dw.bind(mw, s.joints[1:3]).qpos.cpu().numpy()[2:4],
        np.array([[1, 2]] * 2),
    )

    self.assertEqual(mw.bind(s.bodies).pos.shape, (batch_size, m.nbody, 3))
    body_pos = mw.bind(s.bodies).pos
    mw.bind(s.bodies).pos[:, 0, 0] = 1.0
    np.testing.assert_array_equal(
        mw.bind(s.bodies).pos.cpu().numpy(),
        body_pos.cpu().numpy(),
    )

    # set batched model value with env_mask
    mw.bind(s.bodies[0:2]).set(
        "pos",
        torch.tensor([1, 2, 3], dtype=mw.geom_pos.dtype, device=device),
        env_mask,
    )
    np.testing.assert_array_equal(
        mw.bind(s.bodies[0:2]).pos.cpu().numpy()[2:4],
        np.array([[[1, 2, 3]] * 2] * 2),
    )

  @parameterized.parameters(types.Impl.CPP, types.Impl.WARP)
  def test_bind_pattern(self, backend: types.Impl):
    s = mujoco.MjSpec.from_string(self.XML)
    m = s.compile()
    mw = wrap_model(backend, m, nworld=1)

    bound = mw.bind_pattern(s, mujoco.mjtObj.mjOBJ_GEOM, "geom.*")
    self.assertEqual(bound.id.shape, (3,))
    np.testing.assert_array_equal(bound.pos.cpu().numpy()[0], m.geom_pos)

    bound_single = mw.bind_pattern(s, mujoco.mjtObj.mjOBJ_GEOM, "geom1")
    self.assertIsNot(bound, bound_single)
    np.testing.assert_array_equal(
        bound_single.pos.cpu().numpy()[0], m.geom_pos[0]
    )


def _dtype_matches(mj_dtype: torch.dtype, mjw_dtype: torch.dtype) -> bool:
  """Checks dtypes match, allowing 64-bit CPU vs 32-bit GPU."""
  if mj_dtype == mjw_dtype:
    return True
  pairs = {
      (torch.float64, torch.float32),
      (torch.int64, torch.int32),
      (torch.uint8, torch.int32),
      (torch.uint8, torch.bool),
      (torch.bool, torch.int32),
  }
  return (mj_dtype, mjw_dtype) in pairs


def _check_fields_recursive(
    test_obj,
    mj_wrapper: torch_wrapper.TorchWrapper,
    mjw_wrapper: torch_wrapper.TorchWrapper,
    obj_type: str,
    prefix: str = "",
):
  """Checks dims and dtype between mj and mj_warp wrappers."""
  for name in torch_wrapper._union_mj_fields().get(obj_type, {}):
    full_name = prefix + name
    try:
      mj_val = getattr(mj_wrapper, name)
      mjw_val = getattr(mjw_wrapper, name)
    except Exception as e:
      raise AssertionError(f"Failed to access field '{full_name}': {e}") from e

    if isinstance(mj_val, torch_wrapper.TorchWrapper):
      _check_fields_recursive(
          test_obj, mj_val, mjw_val, name.capitalize(), f"{full_name}."
      )
      continue

    test_obj.assertEqual(
        len(mj_val.shape),
        len(mjw_val.shape),
        f"{full_name} ndim mismatch: mj={len(mj_val.shape)},"
        f" mjw={len(mjw_val.shape)}",
    )
    test_obj.assertTrue(
        _dtype_matches(mj_val.dtype, mjw_val.dtype),
        f"{full_name} dtype mismatch: mj={mj_val.dtype}, mjw={mjw_val.dtype}",
    )


class TorchWrapperXPU(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.set_device(_DEVICE)

    self.nworld = 1
    self.m = mujoco.MjModel.from_xml_string(XML)
    self.d = mujoco.MjData(self.m)

    self.mw = mjw.put_model(self.m)
    with wp.ScopedDevice(_DEVICE):
      self.dw = mjw.make_data(self.m, nworld=self.nworld)

    self.mw_torch = torch_wrapper.TorchWrapper(self.mw, nworld=self.nworld)
    self.dw_torch = torch_wrapper.TorchWrapper(self.dw, nworld=self.nworld)

    self.m_torch = torch_wrapper.TorchWrapper(self.m)
    self.d_torch = torch_wrapper.TorchWrapper(self.d)

  def test_fields(self):
    _check_fields_recursive(self, self.m_torch, self.mw_torch, "Model")
    _check_fields_recursive(self, self.d_torch, self.dw_torch, "Data")

  def test_fields_in_union(self):
    def get_field_names(wrapper):
      names = set()
      for n in dir(wrapper):
        if n.startswith("_"):
          continue
        val = getattr(wrapper, n, None)
        if isinstance(
            val, (torch_wrapper.TorchTensor, torch_wrapper.TorchWrapper)
        ):
          names.add(n)
      return names

    cases = [
        (self.m_torch, self.mw_torch, "Model"),
        (self.m_torch.opt, self.mw_torch.opt, "Option"),
        (self.d_torch, self.dw_torch, "Data"),
    ]
    for mt, mwt, obj_type in cases:
      with self.subTest(obj_type=obj_type):
        union_names = set(torch_wrapper._union_mj_fields()[obj_type].keys())
        mj_names = get_field_names(mt)
        mjw_names = get_field_names(mwt)
        self.assertTrue(
            mj_names.issubset(union_names),
            f"{obj_type} mj extra: {mj_names - union_names}",
        )
        self.assertTrue(
            mjw_names.issubset(union_names),
            f"{obj_type} mjw extra: {mjw_names - union_names}",
        )

  def test_essential_fields_in_union(self):
    """Ensures critical fields are not accidentally filtered out."""
    union = torch_wrapper._union_mj_fields()

    essential_data = {
        "qpos", "qvel", "qacc", "ctrl", "time", "xpos", "xquat", "sensordata",
        "act", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "eq_active",
        "mocap_pos", "mocap_quat", "act_dot", "xmat", "xipos", "geom_xpos",
        "geom_xmat",
    }  # fmt: skip
    essential_model = {
        "nq", "nv", "nu", "na", "nbody", "njnt", "ngeom",
        "body_pos", "geom_pos", "jnt_qposadr", "jnt_dofadr",
        "sensor_adr", "sensor_dim", "jnt_type", "jnt_type",
        "body_mocapid",
    }  # fmt: skip
    essential_option = {"gravity", "timestep", "wind", "integrator"}

    for name in essential_data:
      self.assertIn(
          name, union["Data"], f"Essential field 'Data.{name}' missing"
      )
    for name in essential_model:
      self.assertIn(
          name, union["Model"], f"Essential field 'Model.{name}' missing"
      )
    for name in essential_option:
      self.assertIn(
          name, union["Option"], f"Essential field 'Option.{name}' missing"
      )


class TorchWrapperSetConstFieldsTest(absltest.TestCase):
  """Tests that global SET_CONST* fields in torch_wrapper.py are correct.

  These fields are output fields of mjw.set_const* calls. The test
  dynamically discovers the output fields of mjw.set_const* using wp.launch
  to avoid manually enumerating them in torch_wrapper.py.
  """

  def test_setconst_fields_match_mjwarp(self):
    """Dynamically discover set_const output fields using wp.launch."""
    rich_xml = """
    <mujoco model="recompile_test">
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <site name="site1" pos="0 0 .7"/>
        <site name="site2"/>
        <body name="floor">
          <geom name="floor" type="plane" size="1 1 0.1"/>
        </body>
        <body name="head" pos="0 0 .7">
          <geom type="sphere" size=".1" density="200"/>
          <freejoint/>
          <body pos=".2 0 -.2">
            <joint name="joint_arm" type="hinge" axis="0 1 0"/>
            <geom type="capsule" size=".05 0.2" fromto="0 0 0 .3 .1 -.05"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="spatial" limited="true" range="0 .7">
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
      </tendon>
      <actuator>
        <motor tendon="spatial" gear="10"/>
        <motor joint="joint_arm" gear="100"/>
      </actuator>
    </mujoco>
    """

    mj_model = mujoco.MjModel.from_xml_string(rich_xml)
    mj_data = mujoco.MjData(mj_model)

    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(mj_model, mj_data)

    # Intercept wp.launch to record outputs.
    launched_outputs = []
    original_launch = wp.launch

    def wp_launch_tracker(kernel, dim, *args, **kwargs):
      # We assume mjw.set_const* calls properly tag their outputs.
      outputs = kwargs.get("outputs", [])
      for out in outputs:
        if isinstance(out, wp.array) and out not in launched_outputs:
          launched_outputs.append(out)
      return original_launch(kernel, dim, *args, **kwargs)

    def discover_fields_from_call(func):
      launched_outputs.clear()
      wp.launch = wp_launch_tracker
      try:
        func(mjw_model, mjw_data)
      finally:
        wp.launch = original_launch

      discovered = set()
      for f in dataclasses.fields(type(mjw_model)):
        val = getattr(mjw_model, f.name, None)
        if isinstance(val, wp.array) and any(
            val is out for out in launched_outputs
        ):
          discovered.add(f.name)
      return discovered

    set_const_cases = [
        (
            lambda m, d: mjw.set_const(
                m, d, groups=mjw.SetConstGroup.FIXED, restore=False
            ),
            torch_wrapper.SET_CONST_FIXED_OUTPUT_FIELDS,
            "set_const_fixed",
        ),
        (
            lambda m, d: mjw.set_const(
                m, d, groups=mjw.SetConstGroup.KIN0, restore=False
            ),
            torch_wrapper.SET_CONST_KIN0_OUTPUT_FIELDS,
            "set_const_kin0",
        ),
        (
            lambda m, d: mjw.set_const(
                m, d, groups=mjw.SetConstGroup.INERT0, restore=False
            ),
            torch_wrapper.SET_CONST_INERT0_OUTPUT_FIELDS,
            "set_const_inert0",
        ),
        (
            lambda m, d: mjw.set_const(m, d, restore=False),
            torch_wrapper.SET_CONST_OUTPUT_FIELDS,
            "set_const",
        ),
    ]

    for func, expected_fields, name in set_const_cases:
      discovered = discover_fields_from_call(func)
      self.assertEqual(
          discovered,
          expected_fields,
          (
              f"Mismatch between {name} outputs and"
              f" {name.upper()}_OUTPUT_FIELDS.\nMissing from"
              f" {name.upper()}_OUTPUT_FIELDS:"
              f" {discovered - expected_fields}\nExtra in"
              f" {name.upper()}_OUTPUT_FIELDS: {expected_fields - discovered}"
          ),
      )

  def test_set_const_fields_are_tiled(self):
    """TorchWrapper.__init__ tiles SET_CONST_OUTPUT_FIELDS to nworld."""
    m = mujoco.MjModel.from_xml_string(XML)
    nworld = 4
    mjw_model = mjw.put_model(m)

    # Before wrapping, batched MjModel fields should be shape (1, ...).
    for name in torch_wrapper.SET_CONST_OUTPUT_FIELDS:
      val = getattr(mjw_model, name, None)
      self.assertEqual(
          val.shape[0], 1, f"{name} should be shape (1, ...) before wrapping"  # pyrefly: ignore[missing-attribute]
      )

    mw = torch_wrapper.TorchWrapper(mjw_model, nworld=nworld)

    # After wrapping, all derived constant fields should be tiled.
    for name in torch_wrapper.SET_CONST_OUTPUT_FIELDS:
      val = getattr(mw.obj, name, None)
      self.assertEqual(
          val.shape[0],  # pyrefly: ignore[missing-attribute]
          nworld,
          f"{name} leading dim should be {nworld} after tiling",
      )


class TorchWrapperSetConstTrackingTest(parameterized.TestCase):
  """Tests that model writes through the wrapper request set_const levels."""

  def setUp(self):
    super().setUp()
    wp.init()
    wp.set_device(_DEVICE)
    self.spec = mujoco.MjSpec.from_string(XML)
    self.m = self.spec.compile()
    self.d = mujoco.MjData(self.m)

  def _model(self, backend=types.Impl.WARP) -> torch_wrapper.TorchWrapper:
    nworld = 4 if backend == types.Impl.WARP else 1
    return wrap_model(backend, self.m, nworld=nworld)

  def test_initial_level_is_none(self):
    model = self._model()
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_setitem_requests_level(self, backend):
    model = self._model(backend)
    model.body_mass[:, 1] = 2.0
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.FULL)

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_bound_set_requests_level(self, backend):
    model = self._model(backend)
    bound = model.bind_pattern(self.spec, mujoco.mjtObj.mjOBJ_BODY, "sphere")
    bound.set("ipos", bound.ipos + 0.01)
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.FULL)

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_geom_size_level_depends_on_tendon_wrap(self, backend):
    # Without geom tendon wrapping, geom_size only requires FIXED.
    model = self._model(backend)
    model.geom_size[:, 1] = 0.2
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.FIXED)

    # With sphere/cylinder geom tendon wrapping, geom_size requires FULL.
    wrap_xml = """
    <mujoco>
      <worldbody>
        <site name="s1" pos="-1 0 1"/>
        <site name="s2" pos="1 0 1"/>
        <body name="sphere" pos="0 0 1">
          <freejoint/>
          <geom name="sphere" type="sphere" size="0.2"/>
        </body>
      </worldbody>
      <tendon>
        <spatial>
          <site site="s1"/>
          <geom geom="sphere"/>
          <site site="s2"/>
        </spatial>
      </tendon>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(wrap_xml)
    nworld = 4 if backend == types.Impl.WARP else 1
    wrap_mw = wrap_model(backend, m, nworld=nworld)
    wrap_mw.geom_size[:, 0] = 0.3
    self.assertEqual(
        wrap_mw.pending_set_const, torch_wrapper.SetConstLevel.FULL
    )

  def test_full_slice_assignment_requests_level(self):
    model = self._model()
    other = torch.zeros_like(model.qpos0._tensor)
    model.qpos0[...] = other
    self.assertEqual(
        model.pending_set_const,
        torch_wrapper.SetConstLevel.KIN0 | torch_wrapper.SetConstLevel.INERT0,
    )

  def test_in_place_copy_does_not_request(self):
    model = self._model()
    other = torch.zeros_like(model.qpos0._tensor)
    model.qpos0.copy_(other)
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  def test_none_level_field_does_not_request(self):
    model = self._model()
    model.geom_friction[:, 0, 0] = 0.3
    model.geom_friction.mul_(1.1)
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  def test_reads_do_not_request(self):
    model = self._model()
    _ = model.body_mass[:, 1].clone()
    _ = model.body_mass + 1.0
    _ = torch.sum(model.body_mass)
    _ = model.body_mass.add(1.0)  # Out-of-place.
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  def test_tensor_writes_are_not_tracked(self):
    model = self._model()
    model.body_mass._tensor[:, 1] = 2.0
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  def test_augmented_assignment_raises(self):
    model = self._model()
    mass = model.body_mass
    with self.assertRaises(TypeError):
      mass += 1.0
    with self.assertRaises(TypeError):
      model.body_mass *= 2.0
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  def test_multiple_writes_or_levels(self):
    model = self._model()
    model.dof_armature[:] = 0.1
    self.assertEqual(
        model.pending_set_const, torch_wrapper.SetConstLevel.INERT0
    )
    model.cam_pos[:] = 0.2
    self.assertEqual(
        model.pending_set_const,
        torch_wrapper.SetConstLevel.KIN0 | torch_wrapper.SetConstLevel.INERT0,
    )
    model.body_mass[:, 1] = 2.0
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.FULL)

  def test_flush_returns_and_clears(self):
    model = self._model()
    model.qpos0[:] = 0.0
    self.assertEqual(
        model.flush_set_const(),
        torch_wrapper.SetConstLevel.KIN0 | torch_wrapper.SetConstLevel.INERT0,
    )
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)
    self.assertEqual(model.flush_set_const(), torch_wrapper.SetConstLevel.NONE)
    # Writes after a flush are tracked again.
    model.body_mass[:, 1] = 2.0
    self.assertEqual(model.flush_set_const(), torch_wrapper.SetConstLevel.FULL)

  def test_set_const_kernels_do_not_request(self):
    model = self._model()
    dw = mjw.put_data(self.m, self.d, nworld=model.nworld)
    model.body_mass[:, 1] = 2.0
    model.flush_set_const()
    mjw.set_const(model.obj, dw)
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_data_wrapper_raises(self, backend):
    nworld = 4 if backend == types.Impl.WARP else 1
    data = wrap_data(backend, self.m, self.d, nworld=nworld)
    data.qpos[:] = 0.0  # Data writes are never tracked.
    self.assertEqual(data.pending_set_const, torch_wrapper.SetConstLevel.NONE)
    with self.assertRaises(TypeError):
      data.flush_set_const()
    with self.assertRaises(TypeError):
      data.clear_defaults()
    with self.assertRaises(TypeError):
      data.restore_defaults(torch.ones(nworld, dtype=torch.bool))

  def test_option_wrapper_is_not_tracked(self):
    model = self._model()
    model.opt.gravity[:] = 0.0
    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.NONE)

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_default_captured_once_on_first_write(self, backend):
    model = self._model(backend)
    self.assertEmpty(model._defaults)
    initial_mass = model.body_mass[:, 1].clone()
    initial_friction = model.geom_friction[:, 0, 0].clone()

    # First write captures the pre-write default.
    model.body_mass[:, 1] = 2.0
    model.geom_friction[:, 0, 0] = 0.3
    self.assertCountEqual(
        model._defaults.keys(), ["body_mass", "geom_friction"]
    )
    saved_mass_ptr = model._defaults["body_mass"].data_ptr()
    torch.testing.assert_close(model._defaults["body_mass"][:, 1], initial_mass)
    torch.testing.assert_close(
        model._defaults["geom_friction"][:, 0, 0], initial_friction
    )

    # Subsequent writes do not re-clone or overwrite the saved default.
    model.body_mass[:, 1] = 5.0
    self.assertEqual(model._defaults["body_mass"].data_ptr(), saved_mass_ptr)
    torch.testing.assert_close(model._defaults["body_mass"][:, 1], initial_mass)

  def test_clear_defaults(self):
    model = self._model()
    model.body_mass[:, 1] = 2.0
    self.assertIn("body_mass", model._defaults)

    # Clearing defaults drops previous defaults; next write captures 2.0.
    model.clear_defaults()
    self.assertEmpty(model._defaults)

    model.body_mass[:, 1] = 5.0
    torch.testing.assert_close(
        model._defaults["body_mass"][:, 1],
        torch.full_like(model.body_mass[:, 1], 2.0),
    )

  def test_restore_defaults_in_place_and_masked(self):
    model = self._model()
    initial_mass = model.body_mass[:, 1].clone()
    tensor_ptr = model.body_mass._tensor.data_ptr()

    model.body_mass[:, 1] = 10.0
    model.flush_set_const()
    saved_ptr = model._defaults["body_mass"].data_ptr()

    # Partial restore on worlds 0 and 2 marks pending_set_const.
    mask = torch.tensor(
        [True, False, True, False],
        dtype=torch.bool,
        device=model.body_mass.device,
    )
    model.restore_defaults(mask)

    self.assertEqual(model.pending_set_const, torch_wrapper.SetConstLevel.FULL)
    self.assertEqual(model.body_mass._tensor.data_ptr(), tensor_ptr)
    self.assertEqual(model._defaults["body_mass"].data_ptr(), saved_ptr)
    torch.testing.assert_close(model.body_mass[0, 1], initial_mass[0])
    torch.testing.assert_close(model.body_mass[2, 1], initial_mass[2])
    torch.testing.assert_close(
        model.body_mass[1, 1], torch.tensor(10.0, device=model.body_mass.device)
    )
    torch.testing.assert_close(
        model.body_mass[3, 1], torch.tensor(10.0, device=model.body_mass.device)
    )

  def test_restore_defaults_non_batch_field(self):
    model = self._model()
    initial = model.geom_conaffinity[0].clone()
    model.geom_conaffinity[0] = initial + 1

    # Non-batched fields are shared by all worlds, so any mask restores them.
    mask = torch.tensor(
        [True, False, False, False], device=model.body_mass.device
    )
    model.restore_defaults(mask)
    torch.testing.assert_close(model.geom_conaffinity[0], initial)

  def test_restore_defaults_cpu_does_not_use_cuda_stream(self):
    model = self._model(types.Impl.CPP)
    initial_mass = model.body_mass[:, 1].clone()
    model.body_mass[:, 1] = 10.0

    with mock.patch.object(torch.cuda, "stream") as mock_stream:
      model.restore_defaults(torch.tensor([True]))
      mock_stream.assert_not_called()

    torch.testing.assert_close(model.body_mass[:, 1], initial_mass)


if __name__ == "__main__":
  absltest.main()
