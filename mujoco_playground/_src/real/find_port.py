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

"""Identification of USB serial devices."""

import dataclasses
import json
import pathlib
from typing import Mapping
from typing import Sequence

from mujoco_playground._src import paths


@dataclasses.dataclass(frozen=True)
class SerialPort:
  """A USB serial device."""

  device: str  # e.g. /dev/ttyACM0 or /dev/tty.usbmodem588A0217751.
  serial_number: str | None  # None if the adapter reports none.


class PortResolutionError(RuntimeError):
  """Raised when a role cannot be mapped to exactly one device."""


def list_ports() -> list[SerialPort]:
  """Returns every USB serial device currently attached, sorted by path."""
  from serial.tools import list_ports as list_ports_module  # pyrefly: ignore[missing-import, missing-source-for-stubs]

  ports = [
      SerialPort(device=port.device, serial_number=port.serial_number)
      for port in list_ports_module.comports()
  ]
  return sorted(ports, key=lambda port: port.device)


def format_ports(ports: Sequence[SerialPort]) -> str:
  """Renders ports as a table for CLI output and error messages."""
  if not ports:
    return "  (no USB serial devices found)"
  lines = []
  for port in ports:
    serial_number = port.serial_number or "<none reported>"
    lines.append(f"  {port.device}  serial={serial_number}")
  return "\n".join(lines)


def _resolve(
    roles: Mapping[str, str], ports: Sequence[SerialPort]
) -> dict[str, str]:
  """Maps each role to the path of the device with its USB serial number."""
  by_serial: dict[str, list[SerialPort]] = {}
  for port in ports:
    if port.serial_number is not None:
      by_serial.setdefault(port.serial_number, []).append(port)

  resolved = {}
  for role, serial_number in roles.items():
    if not serial_number:
      raise PortResolutionError(
          f"No USB serial number configured for role '{role}'. Re-run the"
          " robot's port scan (robots/<robot>/real/ports.py --identify) to"
          " generate its config."
      )

    matches = by_serial.get(serial_number, [])
    if not matches:
      raise PortResolutionError(
          f"No attached device has USB serial number '{serial_number}'"
          f" (configured for role '{role}'). Attached"
          f" devices:\n{format_ports(ports)}\n\nCheck the device is powered and"
          " plugged in, or re-run the port scan if the hardware changed."
      )
    if len(matches) > 1:
      device_paths = ", ".join(port.device for port in matches)
      raise PortResolutionError(
          f"USB serial number '{serial_number}' (role '{role}') matches"
          f" multiple devices: {device_paths}. Serial numbers are expected to"
          " be unique per device; this usually means a clone adapter that"
          " reports"
          " a shared serial number, which cannot be identified by serial alone."
      )
    resolved[role] = matches[0].device

  return resolved


def resolve_current(roles: Mapping[str, str]) -> dict[str, str]:
  """Resolves `roles` against the devices attached to this device right now."""
  return _resolve(roles, list_ports())


def config_path(robot: str) -> pathlib.Path:
  """Returns where a robot's machine-local port config lives."""
  return pathlib.Path(paths.local_dir() / f"{robot}_ports.json")


def write_port_config(robot: str, roles: Mapping[str, str]) -> pathlib.Path:
  """Writes a robot's port config and returns the path written."""
  path = config_path(robot)
  path.write_text(json.dumps(dict(roles), indent=2, sort_keys=True) + "\n")
  return path


def load_port_config(
    robot: str, roles: Sequence[str], scan_command: str
) -> Mapping[str, str]:
  """Returns the role-to-serial-number map from a robot's port config."""
  path = config_path(robot)
  if not path.is_file():
    raise PortResolutionError(
        f"No port config for '{robot}' at {path}\n\nGenerate it with:\n\n   "
        f" {scan_command}\n"
    )

  try:
    serial_numbers = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise PortResolutionError(
        f"{path} is not valid JSON ({exc}). Regenerate it with:\n\n   "
        f" {scan_command}\n"
    ) from exc

  missing = [role for role in roles if not serial_numbers.get(role)]
  if missing:
    raise PortResolutionError(
        f"{path} has no USB serial number for: {', '.join(missing)}\n\n"
        f"Regenerate it with:\n\n    {scan_command}\n"
    )
  return serial_numbers
