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

"""Tests for camera identification."""

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from mujoco_playground._src.real import find_opencv_camera

_SCAN = "uv run python -m robots.so101.real.scan_hardware --identify"


class CameraConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self._local_dir = pathlib.Path(tmp.name)
    patcher = mock.patch.object(
        find_opencv_camera.paths, "local_dir", return_value=self._local_dir
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_round_trips_indices_and_paths(self):
    device = self._local_dir / "video0"
    device.touch()
    find_opencv_camera.write_camera_config(
        "so101", {"wrist": 1, "overhead": str(device)}
    )

    loaded = find_opencv_camera.load_camera_config(
        "so101", ["wrist", "overhead"], _SCAN
    )

    self.assertEqual(loaded, {"wrist": 1, "overhead": str(device)})

  def test_missing_config_names_the_scan_command(self):
    with self.assertRaisesRegex(
        find_opencv_camera.CameraResolutionError, "scan_hardware"
    ):
      find_opencv_camera.load_camera_config("so101", ["wrist"], _SCAN)

  def test_missing_name_is_reported(self):
    find_opencv_camera.write_camera_config("so101", {"wrist": 0})

    with self.assertRaisesRegex(
        find_opencv_camera.CameraResolutionError, "overhead"
    ):
      find_opencv_camera.load_camera_config(
          "so101", ["wrist", "overhead"], _SCAN
      )

  def test_dangling_device_path_advises_rescan(self):
    find_opencv_camera.write_camera_config(
        "so101", {"wrist": "/dev/no/such/camera"}
    )

    with self.assertRaisesRegex(
        find_opencv_camera.CameraResolutionError, "scan_hardware"
    ):
      find_opencv_camera.load_camera_config("so101", ["wrist"], _SCAN)

  def test_index_zero_is_a_valid_camera_id(self):
    find_opencv_camera.write_camera_config("so101", {"wrist": 0})

    loaded = find_opencv_camera.load_camera_config("so101", ["wrist"], _SCAN)

    self.assertEqual(loaded, {"wrist": 0})


class StableIdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self._tmp = pathlib.Path(tmp.name)

  def test_maps_device_node_to_by_id_symlink(self):
    node = self._tmp / "video2"
    node.touch()
    by_id = self._tmp / "by-id"
    by_id.mkdir()
    link = by_id / "usb-Wrist_Cam_123-video-index0"
    link.symlink_to(node)

    self.assertEqual(find_opencv_camera.stable_id(str(node), by_id), str(link))

  def test_indices_pass_through(self):
    self.assertEqual(find_opencv_camera.stable_id(1), 1)

  def test_unlinked_node_passes_through(self):
    node = self._tmp / "video2"
    node.touch()
    by_id = self._tmp / "by-id"
    by_id.mkdir()

    self.assertEqual(find_opencv_camera.stable_id(str(node), by_id), str(node))


if __name__ == "__main__":
  absltest.main()
