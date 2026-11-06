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

"""SO101 calibration with interactive CLI.

Calibration makes joint angles physically meaningful and consistent across
arms: the same reading means the same pose on the leader, the follower, and
the the sim. For each arm run:

  uv run --extra so101 python -m
  mujoco_playground.robots.so101.real.get_calibration
  uv run --extra so101 python -m
  mujoco_playground.robots.so101.real.get_calibration --roles=leader

Which obtains two measurements things:

1. **Middle pose.** The position shown in `assets/middle_pose.png` becomes
   the reference middle pose; `set_half_turn_homings` writes the homing offset
   to EEPROM so the middle pose is centered at 2047 (the half-turn).
2. **Range.** The recorded raw `range_min`/`range_max` are used to clamp goals
   for safety; for the gripper they also define its 0-100% scale, so sweep it
   fully.

Rerun the calibration only when the mechanics change.
"""

import json
import pathlib
from typing import Any
from typing import Mapping
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from mujoco_playground import paths
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import get_ports

# Per-motor fields, all raw encoder ticks (0..ENCODER_MAX).
FIELDS = ("id", "zero", "range_min", "range_max")

CALIBRATE_COMMAND = (
    "uv run --extra so101 python -m"
    f" {__package__}.{pathlib.Path(__file__).stem}"
)


class CalibrationError(RuntimeError):
  """Raised when an arm's calibration is missing or malformed."""


def config_path() -> pathlib.Path:
  """Returns where this machine's SO101 calibration lives."""
  return pathlib.Path(paths.local_dir()) / f"{constant.ROBOT}_calibration.json"


def write(
    role: str,
    motors: Mapping[str, Mapping[str, int]],
    serial_number: str | None = None,
) -> pathlib.Path:
  """Writes one arm's calibration, preserving the other arm in the file."""
  path = config_path()
  config: dict[str, Any] = {}
  if path.is_file():
    try:
      config = json.loads(path.read_text())
    except json.JSONDecodeError:
      backup = path.with_suffix(".corrupt.json")
      path.replace(backup)
      logging.warning(
          "%s was not valid JSON, so the calibration for any other arm could"
          " not be read. It has been moved to %s (recover it by hand if"
          " needed) and a fresh file written for '%s'.",
          path,
          backup,
          role,
      )
  config[role] = {name: dict(fields) for name, fields in motors.items()}
  if serial_number is not None:
    config.setdefault("serials", {})[role] = serial_number
  path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
  return path


def load(
    role: str, expected_serial: str | None = None
) -> Mapping[str, Mapping[str, int]]:
  """Returns one arm's per-motor calibration as plain dicts."""
  path = config_path()
  if not path.is_file():
    raise CalibrationError(
        f"No calibration for '{constant.ROBOT}' at {path}\n\nGenerate it"
        f" with:\n\n    {CALIBRATE_COMMAND}\n"
    )

  try:
    config = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise CalibrationError(
        f"{path} is not valid JSON ({exc}). Regenerate it with:\n\n   "
        f" {CALIBRATE_COMMAND}\n"
    ) from exc

  motors = config.get(role)
  if motors is None:
    raise CalibrationError(
        f"{path} has no calibration for role '{role}'. Generate it"
        f" with:\n\n    {CALIBRATE_COMMAND}\n"
    )

  _validate(motors, constant.JOINTS, role, path)

  stored_serial = config.get("serials", {}).get(role)
  if (
      expected_serial is not None
      and stored_serial is not None
      and stored_serial != expected_serial
  ):
    raise CalibrationError(
        f"The stored calibration for '{role}' was made for the arm with USB"
        f" serial '{stored_serial}', but the attached {role} has serial"
        f" '{expected_serial}' — a different physical arm. Calibrate"
        f" it with:\n\n    {CALIBRATE_COMMAND}\n"
    )
  return motors


def _validate(
    motors: Mapping[str, Any],
    motor_names: Sequence[str],
    role: str,
    path: pathlib.Path,
) -> None:
  missing = [name for name in motor_names if name not in motors]
  incomplete = [
      name
      for name in motor_names
      if name in motors and any(field not in motors[name] for field in FIELDS)
  ]
  if missing or incomplete:
    problems = []
    if missing:
      problems.append(f"missing motors: {', '.join(missing)}")
    if incomplete:
      problems.append(f"incomplete motors: {', '.join(incomplete)}")
    raise CalibrationError(
        f"Calibration for '{role}' in {path} is invalid"
        f" ({'; '.join(problems)}). Regenerate it with:\n\n   "
        f" {CALIBRATE_COMMAND}\n"
    )


# CLI ------------------------------------------------------------------------

_ROLES = flags.DEFINE_list(
    "roles",
    list(constant.ROLES),
    "Arms to calibrate, in order.",
)


def _make_bus(port: str):
  """Builds a FeetechMotorsBus with the SO101 motor table."""
  from lerobot import motors as motors_mod  # pyrefly: ignore[missing-import]
  from lerobot.motors import feetech  # pyrefly: ignore[missing-import]

  norm = {
      name: (
          motors_mod.MotorNormMode.RANGE_0_100
          if name == "gripper"
          else motors_mod.MotorNormMode.DEGREES
      )
      for name in constant.JOINTS
  }
  return feetech.FeetechMotorsBus(
      port=port,
      motors={
          name: motors_mod.Motor(
              constant.MOTOR_IDS[name], constant.MOTOR_MODEL, norm[name]
          )
          for name in constant.JOINTS
      },
  )


def _make_passive(bus, role: str) -> None:
  """Disables torque on every motor and verifies by reading it back."""
  bus.disable_torque(num_retry=3)
  torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
  still_on = sorted(name for name, value in torque.items() if value != 0)
  if still_on:
    raise RuntimeError(
        f"Torque is still enabled on the {role} for:"
        f" {', '.join(still_on)}. Power-cycle the arm and re-run; if it"
        " persists, check the servo bus wiring."
    )
  print(f"\nTorque verified OFF on all {role} motors — the arm is passive.")


def _capture_middle(bus) -> Mapping[str, int]:
  """Sets the homing offset in EEPROM so the middle pose is centered at 2047."""
  input("Hold the pose and press ENTER to capture...")
  bus.set_half_turn_homings()
  middle = bus.sync_read("Present_Position", normalize=False)
  print("  captured: " + str(dict(middle)))
  return middle


def _record_ranges(
    bus, motor_names: Sequence[str]
) -> tuple[dict[str, int], dict[str, int]]:
  """Interactively records each motor's range of motion until ENTER."""
  from lerobot.utils import utils as lerobot_utils  # pyrefly: ignore[missing-import]

  positions = bus.sync_read("Present_Position", motor_names, normalize=False)
  mins = {name: int(positions[name]) for name in motor_names}
  maxes = {name: int(positions[name]) for name in motor_names}
  while True:
    positions = bus.sync_read("Present_Position", motor_names, normalize=False)
    print("\n-------------------------------------------")
    print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
    for name in motor_names:
      pos = int(positions[name])
      mins[name] = min(pos, mins[name])
      maxes[name] = max(pos, maxes[name])
      print(f"{name:<15} | {mins[name]:>6} | {pos:>6} | {maxes[name]:>6}")
    if lerobot_utils.enter_pressed():
      break
    # Move the cursor up to overwrite the previous table.
    lerobot_utils.move_cursor_up(len(motor_names) + 3)

  unmoved = sorted(n for n in motor_names if mins[n] == maxes[n])
  if unmoved:
    raise ValueError(
        f"These motors never moved during the sweep: {', '.join(unmoved)}."
    )
  return mins, maxes


def _calibrate_arm(role: str, port: str, serial_number: str) -> None:
  """Runs the interactive calibration for one arm and saves the result."""
  from lerobot.motors import feetech  # pyrefly: ignore[missing-import]

  bus = _make_bus(port)
  bus.connect()
  try:
    _make_passive(bus, role)
    for name in constant.JOINTS:
      bus.write("Operating_Mode", name, feetech.OperatingMode.POSITION.value)

    print(f"Support the {role} by hand — it may fall under gravity.")
    print(
        f"Hold the {role} in the MIDDLE pose (see assets/middle_pose.png next"
        " to this module): pan centered, upper arm vertical, elbow bent 90"
        " degrees so the forearm is level, wrist straight, grippers should be"
        " vertical and the camera is on the left side. This pose defines the"
        " reference for the leader, the follower, and the sim, so pose it as"
        " consistently as you can."
    )
    middle = _capture_middle(bus)

    sweep = [j for j in constant.JOINTS if j != "wrist_roll"]
    print(
        f"\nMove every joint of the {role} except wrist_roll through its FULL"
        " range of motion, one at a time. (wrist_roll rotates continuously"
        " and is assigned the full encoder range automatically)."
        " Live values will stream below; press ENTER when done."
    )
    range_mins, range_maxes = _record_ranges(bus, sweep)

    motors = {
        name: {
            "id": constant.MOTOR_IDS[name],
            "zero": int(middle[name]),
            "range_min": int(range_mins[name]) if name != "wrist_roll" else 0,
            "range_max": (
                int(range_maxes[name])
                if name != "wrist_roll"
                else constant.ENCODER_MAX
            ),
        }
        for name in constant.JOINTS
    }

    # The saved file is the single source of truth: runtime code loads it on
    # every connect and does its own tick->radian conversion. The USB serial
    # ties it to this physical arm.
    path = write(role, motors, serial_number)
    print(f"\nCalibrated the {role}. Saved to:\n\n  {path}\n")
  finally:
    bus.disconnect()


def main(argv):
  del argv  # Unused.
  for role in _ROLES.value:
    if role not in constant.ROLES:
      raise app.UsageError(
          f"Unknown role '{role}'; expected one of {sorted(constant.ROLES)}."
      )

  # Resolved up front so an unplugged arm fails before anything is calibrated.
  ports = {role: get_ports.resolve(role) for role in _ROLES.value}

  for role, device_path in ports.items():
    _calibrate_arm(role, device_path, get_ports.serial_number(role))

  print("Done. Verify with:")
  print(
      '\n  uv run --extra so101 python -c "from'
      " mujoco_playground.robots.so101.real import"
      f" {pathlib.Path(__file__).stem};"
      f" print({pathlib.Path(__file__).stem}.load('{_ROLES.value[0]}'))\""
  )


if __name__ == "__main__":
  app.run(main)
