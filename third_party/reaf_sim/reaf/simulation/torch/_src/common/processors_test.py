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

"""Tests for command and action processors."""

import logging
from unittest import mock

from absl.testing import absltest
from reaf.simulation.torch._src.common import math
from reaf.simulation.torch._src.common import noise
from reaf.simulation.torch._src.common import processors
import torch

_M = mock.MagicMock()
_SIM_DT = 0.005  # 5ms physics timestep


class StochasticSubstepCommandsTest(absltest.TestCase):

  def test_noise_only(self):
    proc = processors.StochasticSubstepCommands(
        keys="ctrl",
        noise=noise.Noise(dist=noise.Uniform(low=0.5, high=1.0), op="add"),
        nworld=64,
        xpu_device="cpu",
    )
    cmd = {"ctrl": torch.zeros(64, 3)}
    proc.process_substep_commands(_M, cmd)
    self.assertTrue(torch.all(cmd["ctrl"] > 0).item())

  def test_delay_only(self):
    proc = processors.StochasticSubstepCommands(
        keys="ctrl",
        delay=(0.005, 0.005),
        nworld=1,
        xpu_device="cpu",
        simulation_timestep=_SIM_DT,
    )
    cmd1 = {"ctrl": torch.ones(1, 2)}
    proc.process_substep_commands(_M, cmd1)
    cmd2 = {"ctrl": torch.ones(1, 2) * 5}
    proc.process_substep_commands(_M, cmd2)
    torch.testing.assert_close(cmd2["ctrl"], torch.ones(1, 2))

  def test_reset(self):
    proc = processors.StochasticSubstepCommands(
        keys="ctrl",
        delay=(0.005, 0.005),
        nworld=2,
        xpu_device="cpu",
        simulation_timestep=_SIM_DT,
    )
    cmd1 = {"ctrl": torch.ones(2, 2)}
    proc.process_substep_commands(_M, cmd1)
    proc.reset()
    cmd2 = {"ctrl": torch.ones(2, 2) * 9}
    proc.process_substep_commands(_M, cmd2)
    torch.testing.assert_close(cmd2["ctrl"], torch.ones(2, 2) * 9)

  def test_passthrough(self):
    """Empty modifier config = pure passthrough."""
    proc = processors.StochasticSubstepCommands(
        keys=("arm", "grip"),
        nworld=2,
        xpu_device="cpu",
    )
    arm = torch.randn(2, 6)
    grip = torch.randn(2, 1)
    cmd = {"arm": arm, "grip": grip}
    proc.process_substep_commands(_M, cmd)
    torch.testing.assert_close(cmd["arm"], arm)
    torch.testing.assert_close(cmd["grip"], grip)

  def test_zero_delay_warning(self):
    """Delay smaller than simulation_timestep should log a warning."""
    with self.assertLogs(level=logging.WARNING) as cm:
      processors.StochasticSubstepCommands(
          keys="ctrl",
          delay=(0.001, 0.001),
          nworld=1,
          xpu_device="cpu",
          simulation_timestep=_SIM_DT,
      )
    self.assertTrue(any("no-op" in msg for msg in cm.output))

  def test_bias_and_reset(self):
    torch.manual_seed(0)
    nworld = 4
    procs = [
        processors.StochasticSubstepCommands(
            keys="ctrl",
            bias=noise.uniform(0.1, 0.9, op="add"),
            nworld=nworld,
            xpu_device="cpu",
        ),
        processors.StochasticSubstepCommands(
            keys="quat",
            bias=noise.Noise(dist=noise.SO3Uniform(low=-0.2, high=0.2)),
            nworld=nworld,
            xpu_device="cpu",
        ),
        processors.StochasticSubstepCommands(
            keys="rot6d",
            bias=noise.Noise(dist=noise.SO3Gaussian(mean=0.0, std=0.1)),
            nworld=nworld,
            xpu_device="cpu",
        ),
    ]
    q_in = noise.sample_so3_uniform(
        (nworld, 4), torch.device("cpu"), low=-1.0, high=1.0
    )
    r6_in = math.rot_matrix_to_6d(math.quat_to_matrix(q_in))
    make_cmd = lambda: {
        "ctrl": torch.zeros(nworld, 3),
        "quat": q_in.clone(),
        "rot6d": r6_in.clone(),
    }

    cmd0, cmd1 = make_cmd(), make_cmd()
    for p in procs:
      p.process_substep_commands(_M, cmd0)
      p.process_substep_commands(_M, cmd1)
    for k in ("ctrl", "quat", "rot6d"):
      torch.testing.assert_close(cmd0[k], cmd1[k])
    self.assertTrue((cmd0["ctrl"] >= 0.1).all() and (cmd0["ctrl"] <= 0.9).all())
    torch.testing.assert_close(
        torch.linalg.norm(cmd0["quat"], dim=-1), torch.ones(nworld)
    )

    mask = torch.tensor([True, False, True, False])
    for p in procs:
      p.reset(mask)
    cmd2 = make_cmd()
    for p in procs:
      p.process_substep_commands(_M, cmd2)
    for k in ("ctrl", "quat", "rot6d"):
      torch.testing.assert_close(cmd2[k][1], cmd0[k][1])
      self.assertFalse(torch.allclose(cmd2[k][0], cmd0[k][0]))

    for _ in range(50):
      for p in procs:
        p.reset()
    cmd50 = make_cmd()
    for p in procs:
      p.process_substep_commands(_M, cmd50)
    self.assertTrue(
        (cmd50["ctrl"] >= 0.1).all() and (cmd50["ctrl"] <= 0.9).all()
    )


class StochasticSubstepMeasurementsTest(absltest.TestCase):

  def test_noise_only(self):
    proc = processors.StochasticSubstepMeasurements(
        keys="pos",
        noise=noise.Noise(dist=noise.Uniform(low=0.5, high=1.0), op="add"),
        nworld=64,
        xpu_device="cpu",
    )
    meas = {"pos": torch.zeros(64, 3)}
    proc.process_substep_measurements(_M, meas)
    self.assertTrue(torch.all(meas["pos"] > 0).item())

  def test_delay_only(self):
    proc = processors.StochasticSubstepMeasurements(
        keys="pos",
        delay=(0.005, 0.005),
        nworld=1,
        xpu_device="cpu",
        simulation_timestep=_SIM_DT,
    )
    meas1 = {"pos": torch.ones(1, 2)}
    proc.process_substep_measurements(_M, meas1)
    meas2 = {"pos": torch.ones(1, 2) * 5}
    proc.process_substep_measurements(_M, meas2)
    torch.testing.assert_close(meas2["pos"], torch.ones(1, 2))

  def test_reset(self):
    proc = processors.StochasticSubstepMeasurements(
        keys="pos",
        delay=(0.005, 0.005),
        nworld=2,
        xpu_device="cpu",
        simulation_timestep=_SIM_DT,
    )
    meas1 = {"pos": torch.ones(2, 2)}
    proc.process_substep_measurements(_M, meas1)
    proc.reset()
    meas2 = {"pos": torch.ones(2, 2) * 9}
    proc.process_substep_measurements(_M, meas2)
    torch.testing.assert_close(meas2["pos"], torch.ones(2, 2) * 9)

  def test_passthrough(self):
    """Empty modifier config = pure passthrough."""
    proc = processors.StochasticSubstepMeasurements(
        keys=("pos", "vel"),
        nworld=2,
        xpu_device="cpu",
    )
    pos = torch.randn(2, 6)
    vel = torch.randn(2, 6)
    meas = {"pos": pos, "vel": vel}
    proc.process_substep_measurements(_M, meas)
    torch.testing.assert_close(meas["pos"], pos)
    torch.testing.assert_close(meas["vel"], vel)

  def test_zero_delay_warning(self):
    """Delay smaller than simulation_timestep should log a warning."""
    with self.assertLogs(level=logging.WARNING) as cm:
      processors.StochasticSubstepMeasurements(
          keys="pos",
          delay=(0.001, 0.001),
          nworld=1,
          xpu_device="cpu",
          simulation_timestep=_SIM_DT,
      )
    self.assertTrue(any("no-op" in msg for msg in cm.output))

  def test_bias_and_reset(self):
    torch.manual_seed(0)
    nworld = 4
    procs = [
        processors.StochasticSubstepMeasurements(
            keys="pos",
            bias=noise.uniform(0.1, 0.9, op="add"),
            nworld=nworld,
            xpu_device="cpu",
        ),
        processors.StochasticSubstepMeasurements(
            keys="quat",
            bias=noise.Noise(dist=noise.SO3Uniform(low=-0.2, high=0.2)),
            nworld=nworld,
            xpu_device="cpu",
        ),
        processors.StochasticSubstepMeasurements(
            keys="rot6d",
            bias=noise.Noise(dist=noise.SO3Gaussian(mean=0.0, std=0.1)),
            nworld=nworld,
            xpu_device="cpu",
        ),
    ]
    q_in = noise.sample_so3_uniform(
        (nworld, 4), torch.device("cpu"), low=-1.0, high=1.0
    )
    r6_in = math.rot_matrix_to_6d(math.quat_to_matrix(q_in))
    make_meas = lambda: {
        "pos": torch.zeros(nworld, 3),
        "quat": q_in.clone(),
        "rot6d": r6_in.clone(),
    }

    meas0, meas1 = make_meas(), make_meas()
    for p in procs:
      p.process_substep_measurements(_M, meas0)
      p.process_substep_measurements(_M, meas1)
    for k in ("pos", "quat", "rot6d"):
      torch.testing.assert_close(meas0[k], meas1[k])
    self.assertTrue((meas0["pos"] >= 0.1).all() and (meas0["pos"] <= 0.9).all())
    torch.testing.assert_close(
        torch.linalg.norm(meas0["quat"], dim=-1), torch.ones(nworld)
    )

    mask = torch.tensor([True, False, True, False])
    for p in procs:
      p.reset(mask)
    meas2 = make_meas()
    for p in procs:
      p.process_substep_measurements(_M, meas2)
    for k in ("pos", "quat", "rot6d"):
      torch.testing.assert_close(meas2[k][1], meas0[k][1])
      self.assertFalse(torch.allclose(meas2[k][0], meas0[k][0]))

    for _ in range(50):
      for p in procs:
        p.reset()
    meas50 = make_meas()
    for p in procs:
      p.process_substep_measurements(_M, meas50)
    self.assertTrue(
        (meas50["pos"] >= 0.1).all() and (meas50["pos"] <= 0.9).all()
    )


class TransformActionsTest(absltest.TestCase):

  def _make(
      self,
      input_key="actions",
      output_key="robot/ctrl",
      scale=(1.0, 1.0, 1.0, 1.0),
      offset=None,
      clip=None,
      name="TransformActions",
  ):
    return processors.TransformActions(
        name=name,
        input_key=input_key,
        output_key=output_key,
        scale=scale,
        offset=offset,
        clip=clip,
        xpu_device="cpu",
    )

  def test_scale_scalar(self):
    mod = self._make(scale=0.5)
    result = mod.process_commands({"actions": torch.ones(2, 1)})
    torch.testing.assert_close(result["robot/ctrl"], torch.full((2, 1), 0.5))

  def test_scale_tuple(self):
    mod = self._make(scale=(1.0, 2.0, 3.0, 4.0))
    result = mod.process_commands({"actions": torch.ones(2, 4)})
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_offset_scalar(self):
    mod = self._make(offset=1.0)
    result = mod.process_commands({"actions": torch.zeros(2, 4)})
    torch.testing.assert_close(result["robot/ctrl"], torch.ones(2, 4))

  def test_offset_tuple(self):
    mod = self._make(offset=(1.0, 2.0, 3.0, 4.0))
    result = mod.process_commands({"actions": torch.zeros(2, 4)})
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_no_offset(self):
    mod = self._make()
    action = torch.ones(2, 4) * 3.0
    result = mod.process_commands({"actions": action})
    torch.testing.assert_close(result["robot/ctrl"], action)

  def test_clip(self):
    mod = self._make(clip=(-0.5, 0.5))
    action = torch.tensor([[1.0, -1.0, 0.3, -0.3]] * 2)
    result = mod.process_commands({"actions": action})
    expected = torch.tensor([[0.5, -0.5, 0.3, -0.3]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_clip_vectorized(self):
    """Per-dimension clip bounds."""
    mod = self._make(clip=([-1.0, -2.0, -3.0, -4.0], [1.0, 2.0, 3.0, 4.0]))
    action = torch.tensor([[5.0, -5.0, 2.0, -2.0]] * 2)
    result = mod.process_commands({"actions": action})
    expected = torch.tensor([[1.0, -2.0, 2.0, -2.0]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_clip_min_only(self):
    """One-sided: lower bound only, no upper."""
    mod = self._make(clip=(0.0, None))
    action = torch.tensor([[1.0, -1.0, 0.3, -0.3]] * 2)
    result = mod.process_commands({"actions": action})
    expected = torch.tensor([[1.0, 0.0, 0.3, 0.0]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_clip_max_only(self):
    """One-sided: upper bound only, no lower."""
    mod = self._make(clip=(None, 0.5))
    action = torch.tensor([[1.0, -1.0, 0.3, -0.3]] * 2)
    result = mod.process_commands({"actions": action})
    expected = torch.tensor([[0.5, -1.0, 0.3, -0.3]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_no_clip(self):
    mod = self._make()
    action = torch.tensor([[100.0, -100.0, 0.0, 0.0]] * 2)
    result = mod.process_commands({"actions": action})
    torch.testing.assert_close(result["robot/ctrl"], action)

  def test_scale_then_clip(self):
    """Scale=2.0, clip=(-1, 1): value 0.8 -> scaled 1.6 -> clipped 1.0."""
    mod = self._make(scale=2.0, clip=(-1.0, 1.0))
    action = torch.tensor([[0.8, -0.8, 0.3, -0.3]] * 2)
    result = mod.process_commands({"actions": action})
    expected = torch.tensor([[1.0, -1.0, 0.6, -0.6]] * 2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_scale_offset_clip(self):
    """Scale=0.1, offset=1.0, clip=(0, 1.2): value 5.0 -> 0.5 -> 1.5 -> 1.2."""
    mod = self._make(scale=0.1, offset=1.0, clip=(0.0, 1.2))
    result = mod.process_commands({"actions": torch.full((2, 1), 5.0)})
    expected = torch.full((2, 1), 1.2)
    torch.testing.assert_close(result["robot/ctrl"], expected)

  def test_name(self):
    mod = self._make(name="my_modifier")
    self.assertEqual(mod.name, "my_modifier")


class DeltaActionsTest(absltest.TestCase):

  def _make(self, name="DeltaActions", **action_cfg):
    return processors.DeltaActions(
        name=name,
        output_key="robot/ctrl",
        nworld=2,
        xpu_device="cpu",
        **action_cfg,
    )

  def test_integration_and_reset(self):
    proc = self._make(
        scale=[0.1, 0.2],
        offset=[1.0, 2.0],
        clip=((0.0, 0.0), (2.0, 3.0)),
    )
    # Step 1: delta action [1.0, 2.0] -> scaled [0.1, 0.4] -> pos [1.1, 2.4]
    action1 = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    res1 = proc.process_commands({"actions": action1})
    torch.testing.assert_close(
        res1["robot/ctrl"], torch.tensor([[1.1, 2.4], [1.1, 2.4]])
    )

    # Step 2: delta action [10.0, 10.0] -> clipped to max [2.0, 3.0]
    action2 = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    res2 = proc.process_commands({"actions": action2})
    torch.testing.assert_close(
        res2["robot/ctrl"], torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    )

    # Reset env 0 only (using boolean mask or index tensor)
    proc.reset(env_mask=torch.tensor([True, False]))
    self.assertEqual(proc.name, "DeltaActions")
    # Env 0 back to initial [1.0, 2.0], Env 1 stays at [2.0, 3.0]
    action3 = torch.zeros(2, 2)
    res3 = proc.process_commands({"actions": action3})
    torch.testing.assert_close(
        res3["robot/ctrl"], torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    )

    # Full reset
    proc.reset()
    res4 = proc.process_commands({"actions": action3})
    torch.testing.assert_close(
        res4["robot/ctrl"], torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    )

  def test_action_clipping(self):
    proc = self._make(
        scale=1.0,
        offset=[0.0],
        action_clip=(-0.5, 0.5),
    )
    # Incoming delta 2.0 clamped to 0.5 -> pos 0.5
    res = proc.process_commands({"actions": torch.tensor([[2.0], [-2.0]])})
    torch.testing.assert_close(res["robot/ctrl"], torch.tensor([[0.5], [-0.5]]))

  def test_action_smoothing(self):
    proc = self._make(
        scale=1.0,
        offset=[0.0],
        smoothing=0.5,
    )
    # Step 1: delta 1.0, prev 0.0 -> (0.5)*1.0 + 0 = 0.5 -> pos 0.5
    res1 = proc.process_commands({"actions": torch.tensor([[1.0], [1.0]])})
    torch.testing.assert_close(res1["robot/ctrl"], torch.tensor([[0.5], [0.5]]))

    # Step 2: delta 1.0, prev 0.5 -> (0.5)*1.0 + (0.5)*0.5 = 0.75 -> pos 1.25
    res2 = proc.process_commands({"actions": torch.tensor([[1.0], [1.0]])})
    torch.testing.assert_close(
        res2["robot/ctrl"], torch.tensor([[1.25], [1.25]])
    )


if __name__ == "__main__":
  absltest.main()
