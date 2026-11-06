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

"""Tests for MeasurementDevice."""

from absl.testing import absltest
import mujoco
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import functional_device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


def _make_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="arm", pos=(1.0, 2.0, 3.0))
  body.add_joint(name="j0", type=mujoco.mjtJoint.mjJNT_HINGE)
  body.add_geom(
      name="geom0",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.1, 0.0, 0.0),
  )
  return spec


def _scaled_xpos(
    model: types.Model,
    data: types.Data,
    *,
    body_id: int,
    scale: torch.Tensor,
) -> torch.Tensor:
  del model
  return data.xpos[:, body_id] * scale


class MeasurementDeviceTest(absltest.TestCase):

  def test_callable_and_term_params(self):
    sim = simulator_module.Simulator(
        mj_spec=_make_spec(),
        nworld=1,
        impl=types.Impl.CPP,
    )
    sim.forward()

    dev = functional_device.MeasurementDevice(
        name="state",
        measurements={
            "xpos": lambda model, data: data.xpos,
            "scaled_arm": config.Term(
                _scaled_xpos,
                body_id=1,
                scale=torch.tensor([2.0, 3.0, 4.0]),
            ),
            "qpos": config.Term(
                lambda model, data: data.qpos,
                output_shape=(1,),
            ),
        },
    )
    self.assertEqual(dev.substep_measurements_spec()["state/qpos"].shape, (1,))

    dev.initialize(sim)
    meas_spec = dev.substep_measurements_spec()
    self.assertEqual(meas_spec["state/xpos"].shape, (2, 3))
    self.assertEqual(meas_spec["state/scaled_arm"].shape, (3,))
    self.assertEqual(meas_spec["state/qpos"].shape, (1,))

    measurements = dev.get_substep_measurements(sim)
    self.assertEqual(measurements["state/xpos"].shape, (1, 2, 3))
    torch.testing.assert_close(
        measurements["state/scaled_arm"],
        torch.tensor([[2.0, 6.0, 12.0]]),
        check_dtype=False,
    )


if __name__ == "__main__":
  absltest.main()
