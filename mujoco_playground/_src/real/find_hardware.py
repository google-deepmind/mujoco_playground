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

"""Interactive tools for identifying a robot's hardware."""

import time
from typing import Sequence

from absl import app
from mujoco_playground._src.real import find_opencv_camera
from mujoco_playground._src.real import find_port


def identify_serial_role(role: str, settle_seconds: float) -> str:
  """Returns the USB serial number of the device unplugged for `role`."""
  before = {port.device: port for port in find_port.list_ports()}
  input(f"\nUnplug the {role} device, then press ENTER...")
  time.sleep(settle_seconds)
  after = {port.device for port in find_port.list_ports()}

  removed = [port for path, port in before.items() if path not in after]
  if not removed:
    raise app.UsageError(
        f"No device disappeared when unplugging the {role}. Either it was"
        " already unplugged, or the OS had not re-enumerated yet -- retry with"
        " a larger --settle_seconds."
    )
  if len(removed) > 1:
    paths = ", ".join(port.device for port in removed)
    raise app.UsageError(
        f"More than one device disappeared ({paths}). Unplug exactly one device"
        " at a time."
    )

  port = removed[0]
  if port.serial_number is None:
    raise app.UsageError(
        f"{port.device} does not report a USB serial number, so it cannot be"
        " identified this way. It needs a udev rule or a different"
        " adapter."
    )

  print(f"  {role}: {port.device}  serial={port.serial_number}")
  input(f"Plug the {role} back in, then press ENTER...")
  time.sleep(settle_seconds)
  return port.serial_number


def identify_ports(
    robot: str, roles: Sequence[str], settle_seconds: float
) -> None:
  """Attributes each serial role by unplugging and writes the port config."""
  serial_numbers = {
      role: identify_serial_role(role, settle_seconds) for role in roles
  }
  path = find_port.write_port_config(robot, serial_numbers)

  print(f"\nWrote {robot} port config to:\n\n  {path}\n")
  for role, serial_number in serial_numbers.items():
    print(f"  {role}: {serial_number}")


def identify_cameras(robot: str, camera_names: Sequence[str]) -> None:
  """Attributes each camera by preview image and writes the camera config."""
  found = find_opencv_camera.list_cameras()
  if not found:
    raise app.UsageError(
        "No cameras detected. Check the camera is plugged in (and that this"
        " terminal has camera permission on macOS)."
    )

  print("\nDetected cameras; saving one preview frame each:")
  previews: dict[str, find_opencv_camera.CameraId] = {}
  for camera_id in found:
    preview = find_opencv_camera.save_preview(camera_id)
    label = "(no frame)" if preview is None else str(preview)
    print(f"  {camera_id}: {label}")
    previews[str(camera_id)] = camera_id

  camera_map: dict[str, find_opencv_camera.CameraId] = {}
  for name in camera_names:
    while True:
      entry = input(
          f"\nWhich camera is '{name}'? Enter one of"
          f" {sorted(previews)} after looking at the previews: "
      ).strip()
      if entry in previews:
        camera_map[name] = find_opencv_camera.stable_id(previews[entry])
        break
      print(f"  '{entry}' is not one of the detected cameras.")

  path = find_opencv_camera.write_camera_config(robot, camera_map)
  print(f"\nWrote {robot} camera config to:\n\n  {path}\n")
  for name, camera_id in camera_map.items():
    print(f"  {name}: {camera_id}")


def run_ports(
    robot: str,
    roles: Sequence[str],
    identify: bool,
    settle_seconds: float = 1.0,
) -> None:
  """Lists serial devices and writes the port config."""
  attached = find_port.list_ports()
  print("Attached USB serial devices:")
  print(find_port.format_ports(attached))

  if not identify:
    print("\nRe-run with --identify to attribute each device to a role.")
    return
  identify_ports(robot, roles, settle_seconds)


def run_cameras(
    robot: str, camera_names: Sequence[str], identify: bool
) -> None:
  """Lists cameras and writes the camera config."""
  if not identify:
    found = find_opencv_camera.list_cameras()
    print("Detected cameras:")
    for camera_id in found:
      print(f"  {camera_id}")
    if not found:
      print("  (none)")
    print("\nRe-run with --identify to attribute each camera by preview.")
    return
  identify_cameras(robot, camera_names)
