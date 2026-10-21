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

"""Tests for serial port resolution."""

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from mujoco_playground._src.real import find_port

_SCAN = "uv run python -m robots.so101.real.scan_ports --identify"


def _port(device: str, serial_number: str | None) -> find_port.SerialPort:
  return find_port.SerialPort(device=device, serial_number=serial_number)


class ResolveTest(absltest.TestCase):

  def test_maps_roles_to_device_paths(self):
    attached = [_port("/dev/ttyACM0", "AAA"), _port("/dev/ttyACM1", "BBB")]

    resolved = find_port._resolve(
        {"follower": "BBB", "leader": "AAA"}, attached
    )

    self.assertEqual(
        resolved, {"follower": "/dev/ttyACM1", "leader": "/dev/ttyACM0"}
    )

  def test_resolution_is_independent_of_enumeration_order(self):
    roles = {"follower": "AAA"}
    first_boot = [_port("/dev/ttyACM0", "AAA"), _port("/dev/ttyACM1", "BBB")]
    second_boot = [_port("/dev/ttyACM0", "BBB"), _port("/dev/ttyACM1", "AAA")]

    self.assertEqual(
        find_port._resolve(roles, first_boot), {"follower": "/dev/ttyACM0"}
    )
    self.assertEqual(
        find_port._resolve(roles, second_boot), {"follower": "/dev/ttyACM1"}
    )

  def test_unconfigured_role_raises(self):
    with self.assertRaisesRegex(find_port.PortResolutionError, "follower"):
      find_port._resolve({"follower": ""}, [_port("/dev/ttyACM0", "AAA")])

  def test_unattached_serial_number_lists_what_is_attached(self):
    attached = [_port("/dev/ttyACM0", "AAA")]

    with self.assertRaisesRegex(find_port.PortResolutionError, "/dev/ttyACM0"):
      find_port._resolve({"follower": "MISSING"}, attached)

  def test_duplicate_serial_numbers_raise(self):
    attached = [_port("/dev/ttyACM0", "AAA"), _port("/dev/ttyACM1", "AAA")]

    with self.assertRaisesRegex(
        find_port.PortResolutionError, "multiple devices"
    ):
      find_port._resolve({"follower": "AAA"}, attached)

  def test_ports_without_serial_numbers_are_ignored(self):
    attached = [_port("/dev/ttyACM0", None), _port("/dev/ttyACM1", "AAA")]

    resolved = find_port._resolve({"follower": "AAA"}, attached)

    self.assertEqual(resolved, {"follower": "/dev/ttyACM1"})


class ResolveAttachedTest(absltest.TestCase):

  def test_resolves_against_currently_attached_devices(self):
    attached = [_port("/dev/ttyACM0", "AAA"), _port("/dev/ttyACM1", "BBB")]

    with mock.patch.object(find_port, "list_ports", return_value=attached):
      resolved = find_port.resolve_current({"follower": "BBB"})

    self.assertEqual(resolved, {"follower": "/dev/ttyACM1"})


class PortConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self._local_dir = pathlib.Path(tmp.name)
    patcher = mock.patch.object(
        find_port.paths, "local_dir", return_value=self._local_dir
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_round_trips_through_a_written_config(self):
    find_port.write_port_config("so101", {"follower": "AAA", "leader": "BBB"})

    loaded = find_port.load_port_config("so101", ["follower", "leader"], _SCAN)

    self.assertEqual(loaded, {"follower": "AAA", "leader": "BBB"})

  def test_missing_config_names_the_scan_command(self):
    with self.assertRaisesRegex(find_port.PortResolutionError, "scan_ports"):
      find_port.load_port_config("so101", ["follower"], _SCAN)

  def test_missing_role_names_the_role(self):
    find_port.write_port_config("so101", {"follower": "AAA"})

    with self.assertRaisesRegex(find_port.PortResolutionError, "leader"):
      find_port.load_port_config("so101", ["follower", "leader"], _SCAN)

  def test_config_path_is_under_the_local_dir(self):
    self.assertEqual(
        find_port.config_path("so101"), self._local_dir / "so101_ports.json"
    )

  def test_unreadable_config_names_the_scan_command(self):
    find_port.config_path("so101").write_text("not json")

    with self.assertRaisesRegex(find_port.PortResolutionError, "scan_ports"):
      find_port.load_port_config("so101", ["follower"], _SCAN)


class FormatPortsTest(absltest.TestCase):

  def test_empty_is_reported_explicitly(self):
    self.assertIn("no USB serial devices", find_port.format_ports([]))

  def test_missing_serial_number_is_labelled(self):
    self.assertIn(
        "<none reported>", find_port.format_ports([_port("/dev/ttyACM0", None)])
    )


if __name__ == "__main__":
  absltest.main()
