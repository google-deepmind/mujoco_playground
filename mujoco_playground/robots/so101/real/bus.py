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

"""Minimal Feetech STS3215 serial bus for the SO101 arm, backed by rustypot."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from mujoco_playground.robots.so101.real import constant

BAUDRATE = 1_000_000  # Feetech STS3215 default bus rate (1 Mbps).
TIMEOUT_SECS = 1.0  # Serial read timeout in seconds.
OPERATING_MODE_POSITION = (
    0  # STS3215 Operating_Mode register value for position control.
)

# Low-latency bus timing and max acceleration registers (units of 2 us / 100 step/s^2).
_RETURN_DELAY_TIME = 0
_MAXIMUM_ACCELERATION = 254
_ACCELERATION = 254
# Follower position-mode PID gains (STS3215 defaults for SO-101).
_POSITION_P_COEFFICIENT = 16
_POSITION_I_COEFFICIENT = 0
_POSITION_D_COEFFICIENT = 32
# Gripper torque/current protection limits (50% max torque, 50% current, 25% overload).
_GRIPPER_MAX_TORQUE_LIMIT = 500
_GRIPPER_PROTECTION_CURRENT = 250
_GRIPPER_OVERLOAD_TORQUE = 25
# STS3215 Homing_Offset is 11-bit sign-magnitude: [-2047, 2047].
_HOMING_OFFSET_LIMIT = (1 << 11) - 1


class FeetechBus:
  """Raw register bus for SO101 Feetech STS3215 servos via `rustypot.Bus`."""

  def __init__(
      self,
      port: str,
      *,
      motor_ids: Mapping[str, int] = constant.MOTOR_IDS,
  ):
    self._port = port
    self._motor_ids = dict(motor_ids)
    self._bus: Any = None

  @property
  def bus(self) -> "FeetechBus":
    """Self-reference so callers using `robot.bus` work uniformly."""
    return self

  def _require_bus(self) -> Any:
    if self._bus is None:
      raise RuntimeError(f"FeetechBus on {self._port!r} is not connected.")
    return self._bus

  def _resolve_motors(self, motors: Sequence[str] | None) -> list[str]:
    return list(self._motor_ids) if motors is None else list(motors)

  def connect(self) -> None:
    """Opens the serial port and verifies all configured STS3215 motors respond."""
    import rustypot  # pyrefly: ignore[missing-import]

    defn = rustypot.Sts3215PyController.definition()
    ids = list(self._motor_ids.values())
    bus = rustypot.Bus(
        self._port, BAUDRATE, TIMEOUT_SECS, {mid: defn for mid in ids}
    )
    found = bus.scan(defn, ids)
    missing = [n for n, mid in self._motor_ids.items() if mid not in found]
    if missing:
      bus.close()
      raise ConnectionError(
          f"Motors not responding on {self._port!r}: {missing}"
      )
    self._bus = bus

  def disconnect(self, *, disable_torque: bool = True) -> None:
    """Optionally disables torque and closes the serial port."""
    if self._bus is None:
      return
    try:
      if disable_torque:
        self.disable_torque(num_retry=5)
    finally:
      self._bus.close()
      self._bus = None

  def disable_torque(
      self, motors: Sequence[str] | None = None, *, num_retry: int = 0
  ) -> None:
    """Disables torque (and unlocks EEPROM) on selected motors."""
    names = self._resolve_motors(motors)
    ids = [self._motor_ids[n] for n in names]
    errors = self._require_bus().set_torque(ids, False, retries=num_retry)
    if errors:
      raise RuntimeError(f"Failed to disable torque on motors: {errors}")

  def enable_torque(
      self, motors: Sequence[str] | None = None, *, num_retry: int = 0
  ) -> None:
    """Enables torque (and locks EEPROM) on selected motors."""
    names = self._resolve_motors(motors)
    ids = [self._motor_ids[n] for n in names]
    errors = self._require_bus().set_torque(ids, True, retries=num_retry)
    if errors:
      raise RuntimeError(f"Failed to enable torque on motors: {errors}")

  def write(
      self, register: str, motor: str, value: int, *, num_retry: int = 0
  ) -> None:
    """Writes an integer value to a single motor register."""
    self._require_bus().write_register(
        self._motor_ids[motor],
        register.lower(),
        int(value),
        retries=num_retry,
    )

  def sync_read(
      self,
      register: str,
      motors: Sequence[str] | None = None,
      *,
      normalize: bool = False,
      num_retry: int = 2,
  ) -> dict[str, int]:
    """Sync-reads a raw integer register across the selected motors."""
    if normalize:
      raise ValueError("FeetechBus only supports raw (normalize=False) reads.")
    names = self._resolve_motors(motors)
    ids = [self._motor_ids[n] for n in names]
    values = self._require_bus().sync_read_register(
        ids, register.lower(), retries=num_retry
    )
    return dict(zip(names, values))

  def sync_write(
      self,
      register: str,
      values: Mapping[str, int],
      *,
      normalize: bool = False,
      num_retry: int = 0,
  ) -> None:
    """Sync-writes raw integer register values across the given motors."""
    if normalize:
      raise ValueError("FeetechBus only supports raw (normalize=False) writes.")
    ids = [self._motor_ids[n] for n in values]
    vals = [int(v) for v in values.values()]
    self._require_bus().sync_write_register(
        ids, register.lower(), vals, retries=num_retry
    )

  def configure_motors(self) -> None:
    """Writes low-latency return delay and acceleration limits to all motors."""
    for name in self._motor_ids:
      self.write("return_delay_time", name, _RETURN_DELAY_TIME)
      self.write("maximum_acceleration", name, _MAXIMUM_ACCELERATION)
      self.write("acceleration", name, _ACCELERATION)

  def configure_leader(self) -> None:
    """Configures a torque-free leader arm in position mode."""
    self.disable_torque()
    self.configure_motors()
    for name in self._motor_ids:
      self.write("operating_mode", name, OPERATING_MODE_POSITION)

  def configure_follower(self) -> None:
    """Configures follower PID gains, gripper protection, and enables torque."""
    self.disable_torque()
    try:
      self.configure_motors()
      for name in self._motor_ids:
        self.write("operating_mode", name, OPERATING_MODE_POSITION)
        self.write("p_coefficient", name, _POSITION_P_COEFFICIENT)
        self.write("i_coefficient", name, _POSITION_I_COEFFICIENT)
        self.write("d_coefficient", name, _POSITION_D_COEFFICIENT)
        if name == constant.GRIPPER_JOINT:
          self.write("max_torque_limit", name, _GRIPPER_MAX_TORQUE_LIMIT)
          self.write("protection_current", name, _GRIPPER_PROTECTION_CURRENT)
          self.write("overload_torque", name, _GRIPPER_OVERLOAD_TORQUE)
    finally:
      self.enable_torque()

  def set_half_turn_homings(
      self, motors: Sequence[str] | None = None
  ) -> dict[str, int]:
    """Sets each motor's EEPROM Homing_Offset so its current pose reads 2047."""
    names = self._resolve_motors(motors)
    for name in names:
      self.write("homing_offset", name, 0)
      self.write("min_position_limit", name, 0)
      self.write("max_position_limit", name, constant.ENCODER_MAX)

    positions = self.sync_read("present_position", names, normalize=False)
    half_turn = constant.ENCODER_MAX // 2
    offsets: dict[str, int] = {}
    for name, pos in positions.items():
      # On Feetech STS3215, present = actual - offset; clamp to [-2047, 2047].
      offset = max(
          -_HOMING_OFFSET_LIMIT,
          min(_HOMING_OFFSET_LIMIT, int(pos) - half_turn),
      )
      self.write("homing_offset", name, offset)
      offsets[name] = offset
    return offsets
