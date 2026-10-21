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
"""Tests for the Simulator class."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from reaf.simulation.torch._src.core import simulator
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch

_TEST_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="body1" pos="0 0 1">
      <joint name="joint1" type="slide" axis="0 0 1"/>
      <geom name="geom1" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SimulatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, "cpu", 1),
      ("warp_cpu", types.Impl.WARP, "cpu", 2),
      ("warp_gpu", types.Impl.WARP, "cuda:0", 2),
  )
  def test_simulator_step_and_forward(self, impl, xpu_device, nworld):
    if not torch.cuda.is_available() and xpu_device.startswith("cuda"):
      self.skipTest("CUDA not available")
    spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator.Simulator(
        mj_spec=spec,
        nworld=nworld,
        xpu_device=xpu_device,
        impl=impl,
    )

    self.assertIsNotNone(sim.model)
    self.assertIsNotNone(sim.data)
    self.assertEqual(sim.impl, impl)

    # Initial forward kinematics
    sim.forward()

    # Step physics
    initial_qpos = sim.data.qpos.clone()
    sim.step()
    # Object falling under gravity
    self.assertFalse(torch.allclose(sim.data.qpos, initial_qpos))

    # Reset
    sim.reset()
    self.assertTrue(
        torch.allclose(sim.data.qpos, torch.zeros_like(sim.data.qpos))
    )

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, "cpu", 1),
      ("warp_cpu", types.Impl.WARP, "cpu", 2),
      ("warp_gpu", types.Impl.WARP, "cuda:0", 2),
  )
  def test_simulator_partial_reset(self, impl, xpu_device, nworld):
    """Tests partial reset for a subset of environments."""
    if not torch.cuda.is_available() and xpu_device.startswith("cuda"):
      self.skipTest("CUDA not available")
    spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator.Simulator(
        mj_spec=spec,
        nworld=nworld,
        xpu_device=xpu_device,
        impl=impl,
    )

    # Initial forward kinematics
    sim.forward()
    initial_qpos = sim.data.qpos.clone()

    # Step physics
    sim.step()
    stepped_qpos = sim.data.qpos.clone()
    self.assertFalse(torch.allclose(stepped_qpos, initial_qpos))

    # Reset a subset of environments (env 0)
    reset_env_mask = torch.zeros(nworld, dtype=torch.bool, device=xpu_device)
    reset_env_mask[0] = True
    sim.reset(env_mask=reset_env_mask)

    # Check that reset environment has its qpos reset
    self.assertTrue(
        torch.allclose(
            sim.data.qpos[reset_env_mask], initial_qpos[reset_env_mask]
        )
    )

    if nworld > 1:
      # Check that non-reset environments are not reset
      non_reset_env_mask = ~reset_env_mask
      self.assertTrue(
          torch.allclose(
              sim.data.qpos[non_reset_env_mask],
              stepped_qpos[non_reset_env_mask],
          )
      )
      self.assertFalse(
          torch.allclose(
              sim.data.qpos[non_reset_env_mask],
              initial_qpos[non_reset_env_mask],
          )
      )

      # Reset the remaining environments and verify all are reset
      sim.reset(env_mask=non_reset_env_mask)
      self.assertTrue(torch.allclose(sim.data.qpos, initial_qpos))

  def test_nvmax_non_warp_raises_error(self):
    spec = mujoco.MjSpec.from_string(_TEST_XML)
    with self.assertRaises(ValueError):
      simulator.Simulator(
          mj_spec=spec,
          nworld=1,
          xpu_device="cpu",
          impl=types.Impl.CPP,
          nvmax=10,
      )

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, "cpu", 1),
      ("warp_cpu", types.Impl.WARP, "cpu", 2),
      ("warp_gpu", types.Impl.WARP, "cuda:0", 2),
  )
  def test_set_const_runs_tracked_level(self, impl, xpu_device, nworld):
    """set_const() runs the level tracked by model writes and flushes it."""
    if not torch.cuda.is_available() and xpu_device.startswith("cuda"):
      self.skipTest("CUDA not available")
    spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator.Simulator(
        mj_spec=spec, nworld=nworld, xpu_device=xpu_device, impl=impl
    )
    body_id = sim.mj_model.body("body1").id

    # Nothing written: no-op.
    self.assertEqual(sim.set_const(), torch_wrapper.SetConstLevel.NONE)

    # body_mass requires FULL; body_subtreemass is derived from it.
    sim.model.body_mass[:, body_id] = 5.0
    self.assertEqual(sim.set_const(), torch_wrapper.SetConstLevel.FULL)
    torch.testing.assert_close(
        sim.model.body_subtreemass[:, body_id],
        torch.full_like(sim.model.body_subtreemass[:, body_id], 5.0),
    )
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )
    self.assertEqual(sim.set_const(), torch_wrapper.SetConstLevel.NONE)

    # Partial levels dispatch to the matching kernel with restore=False.
    cases = [
        (
            "dof_armature",
            torch_wrapper.SetConstLevel.STAGE_0,
            "set_const_0",
            {"restore": False},
        ),
        (
            None,
            torch_wrapper.SetConstLevel.FIXED,
            "set_const_fixed",
            {},
        ),
        (
            "body_mass",
            torch_wrapper.SetConstLevel.FULL,
            "set_const",
            {"restore": False},
        ),
    ]
    for field, level, mjw_fn, kwargs in cases:
      with self.subTest(level=level):
        if field is not None:
          getattr(sim.model, field)[:] = 0.25
        else:
          object.__setattr__(sim.model, "_pending_set_const", level)
        if impl == types.Impl.CPP:
          with mock.patch.object(simulator.mujoco, "mj_setConst") as fn:
            self.assertEqual(sim.set_const(), level)
          fn.assert_called_once_with(sim.mj_model, sim.mj_data)
        else:
          with mock.patch.object(simulator.mjw, mjw_fn) as fn:
            self.assertEqual(sim.set_const(), level)
          fn.assert_called_once_with(sim.warp_model, sim.warp_data, **kwargs)

  @parameterized.named_parameters(
      ("cpp", types.Impl.CPP, "cpu", 1),
      ("warp_cpu", types.Impl.WARP, "cpu", 2),
      ("warp_gpu", types.Impl.WARP, "cuda:0", 2),
  )
  def test_set_const_preserves_state(self, impl, xpu_device, nworld):
    """set_const() preserves integration state across backends."""
    if not torch.cuda.is_available() and xpu_device.startswith("cuda"):
      self.skipTest("CUDA not available")
    spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator.Simulator(
        mj_spec=spec, nworld=nworld, xpu_device=xpu_device, impl=impl
    )
    body_id = sim.mj_model.body("body1").id
    sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    size = mujoco.mj_stateSize(sim.mj_model, sig)

    def get_state():
      if impl == types.Impl.CPP:
        state = torch.zeros(size, dtype=torch.float64)
        mujoco.mj_getState(sim.mj_model, sim.mj_data, state.numpy(), sig)
        return state
      with simulator.wp.ScopedDevice(xpu_device):
        state = simulator.wp.zeros((nworld, size), dtype=float)
        simulator.mjw.get_state(sim.warp_model, sim.warp_data, state, sig)
      return simulator.wp.to_torch(state)

    sim.data.qpos[:] = 1.5
    expected_state = get_state()

    sim.model.body_mass[:, body_id] = 5.0
    sim.set_const()

    torch.testing.assert_close(get_state(), expected_state)


if __name__ == "__main__":
  absltest.main()
