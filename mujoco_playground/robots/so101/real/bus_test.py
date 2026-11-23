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

"""Tests for the rustypot-backed SO101 FeetechBus."""

import sys
import types
from unittest import mock

from absl.testing import absltest
from mujoco_playground.robots.so101.real import bus as bus_mod
from mujoco_playground.robots.so101.real import constant


class _FakeRustypotBus:
  """In-memory stand-in for `rustypot.Bus`."""

  def __init__(self, port, baudrate, timeout, motors, missing_ids=()):
    self.port = port
    self.baudrate = baudrate
    self.timeout = timeout
    self.motors = dict(motors)
    self.missing_ids = set(missing_ids)
    self.registers: dict[tuple[int, str], int] = {}
    self.torque: dict[int, bool] = {}
    self.last_torque_retries: int | None = None
    self.closed = False

  def scan(self, definition, ids=None):
    del definition
    target = self.motors if ids is None else ids
    return {mid: 777 for mid in target if mid not in self.missing_ids}

  def set_torque(self, ids, enabled, retries=0):
    self.last_torque_retries = retries
    for motor_id in ids:
      self.torque[motor_id] = enabled
    return {}

  def write_register(self, motor_id, name, value, retries=0):
    del retries
    self.registers[(motor_id, name)] = value

  def sync_read_register(self, ids, name, retries=0):
    del retries
    return [self.registers[(motor_id, name)] for motor_id in ids]

  def sync_write_register(self, ids, name, values, retries=0):
    del retries
    for motor_id, val in zip(ids, values):
      self.registers[(motor_id, name)] = val

  def close(self):
    self.closed = True


def _connect_fake_bus() -> tuple[bus_mod.FeetechBus, _FakeRustypotBus]:
  created: list[_FakeRustypotBus] = []

  def _make_bus(port, baudrate, timeout, motors):
    fb = _FakeRustypotBus(port, baudrate, timeout, motors)
    created.append(fb)
    return fb

  fake_rustypot = types.SimpleNamespace(
      Bus=_make_bus,
      Sts3215PyController=types.SimpleNamespace(definition=lambda: "sts3215"),
  )
  b = bus_mod.FeetechBus("/dev/ttyFake")
  with mock.patch.dict(sys.modules, {"rustypot": fake_rustypot}):
    b.connect()
  return b, created[0]


class FeetechBusTest(absltest.TestCase):

  def test_sync_read_and_write_round_trip(self):
    b, fake = _connect_fake_bus()
    self.assertEqual(fake.baudrate, bus_mod.BAUDRATE)
    self.assertEqual(set(fake.motors), set(constant.MOTOR_IDS.values()))

    goals = {name: 1000 + i for i, name in enumerate(constant.JOINTS)}
    b.sync_write("Goal_Position", goals, normalize=False)
    read_back = b.sync_read("Goal_Position", normalize=False)
    self.assertEqual(read_back, goals)

    b.disconnect()
    self.assertTrue(fake.closed)
    self.assertEqual(fake.last_torque_retries, 5)
    self.assertTrue(all(not v for v in fake.torque.values()))

  def test_connect_closes_port_when_motor_missing(self):
    created: list[_FakeRustypotBus] = []

    def _make_bus(port, baudrate, timeout, motors):
      fb = _FakeRustypotBus(port, baudrate, timeout, motors, missing_ids=(6,))
      created.append(fb)
      return fb

    fake_rustypot = types.SimpleNamespace(
        Bus=_make_bus,
        Sts3215PyController=types.SimpleNamespace(definition=lambda: "sts3215"),
    )
    b = bus_mod.FeetechBus("/dev/ttyFake")
    with mock.patch.dict(sys.modules, {"rustypot": fake_rustypot}):
      with self.assertRaisesRegex(ConnectionError, "gripper"):
        b.connect()
    self.assertTrue(created[0].closed)

  def test_configure_follower_writes_pid_and_gripper_limits(self):
    b, fake = _connect_fake_bus()
    b.configure_follower()

    self.assertTrue(all(fake.torque.values()))
    pan_id = constant.MOTOR_IDS["shoulder_pan"]
    gripper_id = constant.MOTOR_IDS[constant.GRIPPER_JOINT]
    self.assertEqual(fake.registers[(pan_id, "p_coefficient")], 16)
    self.assertEqual(fake.registers[(pan_id, "d_coefficient")], 32)
    self.assertNotIn((pan_id, "max_torque_limit"), fake.registers)
    self.assertEqual(fake.registers[(gripper_id, "max_torque_limit")], 500)
    self.assertEqual(fake.registers[(gripper_id, "protection_current")], 250)
    self.assertEqual(fake.registers[(gripper_id, "overload_torque")], 25)

  def test_set_half_turn_homings_clamps_to_sign_magnitude_limit(self):
    b, fake = _connect_fake_bus()
    fake.registers[(constant.MOTOR_IDS["shoulder_pan"], "present_position")] = (
        2500
    )
    fake.registers[(constant.MOTOR_IDS["wrist_roll"], "present_position")] = (
        4095
    )

    offsets = b.set_half_turn_homings(["shoulder_pan", "wrist_roll"])
    self.assertEqual(offsets, {"shoulder_pan": 453, "wrist_roll": 2047})
    self.assertEqual(
        fake.registers[(constant.MOTOR_IDS["shoulder_pan"], "homing_offset")],
        453,
    )
    self.assertEqual(
        fake.registers[(constant.MOTOR_IDS["wrist_roll"], "homing_offset")],
        2047,
    )


if __name__ == "__main__":
  absltest.main()
