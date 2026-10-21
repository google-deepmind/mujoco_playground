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

"""Tests for ContactSensorDevice."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.common import contact_sensor_device
from reaf.simulation.torch._src.common.contact_sensor_device import ContactField
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_contact_scene_spec(
    left_pos=(-0.2, 0, 0.5),
    right_pos=(0.2, 0, 0.5),
) -> mujoco.MjSpec:
  """Create a spec with multiple bodies that can contact."""
  spec = mujoco.MjSpec()

  # Floor
  spec.worldbody.add_geom(
      name="floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(1, 1, 0.1),
      pos=(0, 0, 0),
  )

  # Left box
  left_body = spec.worldbody.add_body(name="left_box", pos=left_pos)
  left_body.add_freejoint(name="left_joint")
  left_body.add_geom(
      name="left_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.1, 0.1, 0.1),
      mass=1.0,
  )

  # Right box
  right_body = spec.worldbody.add_body(name="right_box", pos=right_pos)
  right_body.add_freejoint(name="right_joint")
  right_body.add_geom(
      name="right_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.1, 0.1, 0.1),
      mass=1.0,
  )

  return spec


class ContactSensorDeviceTest(parameterized.TestCase):

  def test_configure_spec_single_body(self):
    spec = _make_contact_scene_spec()

    dev = contact_sensor_device.ContactSensorDevice(
        name="contact", fields=("found",)
    )
    dev.configure_spec(
        spec,
        objtype="body",
        objname="left_box",
        reftype="geom",
        refname="floor",
    )

    mj_model = spec.compile()
    self.assertEqual(mj_model.nsensor, 1)

  def test_configure_spec_pattern_matching(self):
    spec = _make_contact_scene_spec()

    dev = contact_sensor_device.ContactSensorDevice(
        name="foot", fields=("found", "force")
    )
    dev.configure_spec(
        spec,
        objtype="body",
        objname=("left_box", "right_box"),
        reftype="geom",
        refname="floor",
    )

    mj_model = spec.compile()
    self.assertEqual(mj_model.nsensor, 2)

  def test_configure_spec_regex_pattern(self):
    spec = _make_contact_scene_spec()

    dev = contact_sensor_device.ContactSensorDevice(
        name="box", fields=("found",)
    )
    dev.configure_spec(
        spec,
        objtype="body",
        objname=".*_box",  # Regex matching both boxes
    )

    mj_model = spec.compile()
    self.assertEqual(mj_model.nsensor, 2)

  @parameterized.parameters(
      dict(
          left_pos=(-0.2, 0, 0.5),
          right_pos=(0.2, 0, 0.5),
          fields=("found", "force"),
          expected_dims=4,
          expected_left_found=False,
          expected_right_found=False,
      ),
      dict(
          left_pos=(-0.2, 0, 0.1),
          right_pos=(0.2, 0, 5.0),
          fields=("found",),
          expected_dims=1,
          expected_left_found=True,
          expected_right_found=False,
      ),
  )
  def test_contact_measurements(
      self,
      left_pos: tuple[float, float, float],
      right_pos: tuple[float, float, float],
      fields: tuple[ContactField, ...],
      expected_dims: int,
      expected_left_found: bool,
      expected_right_found: bool,
  ):
    spec = _make_contact_scene_spec(left_pos=left_pos, right_pos=right_pos)

    dev = contact_sensor_device.ContactSensorDevice(
        name="contact",
        fields=fields,
    )
    dev.configure_spec(
        spec,
        objtype="body",
        objname=("left_box", "right_box"),
        reftype="geom",
        refname="floor",
    )

    if not torch.cuda.is_available():
      self.skipTest("Warp put_model/put_data requires CUDA.")
    nworld = 2

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=nworld,
        impl=types.Impl.WARP,
    )

    # Already instantiated above.
    dev.initialize(sim)

    # Check measurements spec.
    meas_spec = dev.substep_measurements_spec()
    self.assertIn("contact/data", meas_spec)
    self.assertEqual(meas_spec["contact/data"].shape, (2, expected_dims))

    # Check measurement values.
    measurements = dev.get_substep_measurements(sim)
    contact_data = measurements["contact/data"]
    self.assertEqual(contact_data.shape, (nworld, 2, expected_dims))

    left_found = torch.all(contact_data[:, 0, 0] > 0.0).item()
    right_found = torch.all(contact_data[:, 1, 0] > 0.0).item()
    self.assertEqual(left_found, expected_left_found)
    self.assertEqual(right_found, expected_right_found)

  def test_negative_history_length_raises(self):
    with self.assertRaisesRegex(ValueError, "history_length must be >= 0"):
      contact_sensor_device.ContactSensorDevice(
          name="contact", fields=("found",), history_length=-1
      )

  def _make_sim_and_device(
      self, fields: tuple[ContactField, ...], history_length: int
  ) -> tuple[
      simulator_module.Simulator, contact_sensor_device.ContactSensorDevice
  ]:
    spec = _make_contact_scene_spec(left_pos=(-0.2, 0, 0.1))
    dev = contact_sensor_device.ContactSensorDevice(
        name="contact", fields=fields, history_length=history_length
    )
    dev.configure_spec(
        spec,
        objtype="body",
        objname=("left_box", "right_box"),
        reftype="geom",
        refname="floor",
    )
    if not torch.cuda.is_available():
      self.skipTest("Warp put_model/put_data requires CUDA.")
    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=2,
        impl=types.Impl.WARP,
    )
    dev.initialize(sim)
    return sim, dev

  def test_per_field_measurements(self):
    # Out of MuJoCo bit order on purpose: data is stacked as (found, force).
    sim, dev = self._make_sim_and_device(("force", "found"), history_length=0)

    meas_spec = dev.substep_measurements_spec()
    self.assertEqual(meas_spec["contact/data"].shape, (2, 4))
    self.assertEqual(meas_spec["contact/found"].shape, (2, 1))
    self.assertEqual(meas_spec["contact/force"].shape, (2, 3))
    self.assertNotIn("contact/found_history", meas_spec)

    m = dev.get_substep_measurements(sim)
    self.assertEqual(set(m), set(meas_spec))
    torch.testing.assert_close(m["contact/found"], m["contact/data"][..., :1])
    torch.testing.assert_close(m["contact/force"], m["contact/data"][..., 1:])
    # Left box rests on the floor, right box does not.
    self.assertTrue(torch.all(m["contact/found"][:, 0] > 0).item())
    self.assertTrue(torch.all(m["contact/found"][:, 1] == 0).item())

  def test_per_field_history(self):
    sim, dev = self._make_sim_and_device(("found", "force"), history_length=3)

    meas_spec = dev.substep_measurements_spec()
    self.assertEqual(meas_spec["contact/found_history"].shape, (2, 3, 1))
    self.assertEqual(meas_spec["contact/force_history"].shape, (2, 3, 3))

    m = dev.get_substep_measurements(sim)
    first = {k: m[f"contact/{k}"].clone() for k in ("found", "force")}
    for k in ("found", "force"):
      history = m[f"contact/{k}_history"]
      self.assertEqual(
          history.shape, (2, *meas_spec[f"contact/{k}_history"].shape)
      )
      torch.testing.assert_close(history[:, :, 0], first[k])
      torch.testing.assert_close(
          history[:, :, 1:], torch.zeros_like(history[:, :, 1:])
      )

    sim.step()
    m = dev.get_substep_measurements(sim)
    for k in ("found", "force"):
      history = m[f"contact/{k}_history"]
      torch.testing.assert_close(history[:, :, 0], m[f"contact/{k}"])
      torch.testing.assert_close(history[:, :, 1], first[k])

    force_history = m["contact/force_history"]
    dev.reset(sim, env_mask=torch.tensor([True, False], device=sim.xpu_device))
    torch.testing.assert_close(
        force_history[0], torch.zeros_like(force_history[0])
    )
    torch.testing.assert_close(force_history[1, :, 1], first["force"][1])


if __name__ == "__main__":
  absltest.main()
