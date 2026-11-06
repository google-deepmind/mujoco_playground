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

"""Maps USB devices to SO101 arms.

Serial numbers are per-device, so the map itself is generated per machine into a
gitignored config. Generate the config with:

  uv run --extra so101 python -m mujoco_playground.robots.so101.real.ports --identify
"""

import pathlib

from mujoco_playground._src.real import find_port
from mujoco_playground.robots.so101.real import constant

SCAN_COMMAND = (
    "uv run --extra so101 python -m"
    f" {__package__}.{pathlib.Path(__file__).stem} --identify"
)


def serial_number(role: str) -> str:
  """Returns the configured USB serial number for one SO101 role."""
  return find_port.load_port_config(constant.ROBOT, [role], SCAN_COMMAND)[role]


def resolve(role: str) -> str:
  """Returns the device path for one SO101 role."""
  return find_port.resolve_current({role: serial_number(role)})[role]


if __name__ == "__main__":
  from absl import app
  from absl import flags
  from mujoco_playground._src.real import find_hardware

  _IDENTIFY = flags.DEFINE_bool(
      "identify",
      False,
      "Interactively attribute each arm by unplugging it, then write this"
      " machine's port config.",
  )
  _SETTLE_SECONDS = flags.DEFINE_float(
      "settle_seconds",
      1.0,
      "Delay after unplugging before re-enumerating, to let the OS catch up.",
  )

  def main(argv):
    del argv  # Unused.
    find_hardware.run_ports(
        constant.ROBOT,
        constant.ROLES,
        identify=_IDENTIFY.value,
        settle_seconds=_SETTLE_SECONDS.value,
    )

  app.run(main)
