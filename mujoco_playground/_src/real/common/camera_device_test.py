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

"""Tests for the shared OpenCV camera reader and REAF device."""

import threading
import types
from unittest import mock

from absl.testing import absltest
from mujoco_playground._src.real.common import camera_device
import numpy as np

try:
  import cv2  # `identify` extra (`opencv-python-headless`).  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  _HAS_OPENCV = True
except ImportError:
  _HAS_OPENCV = False


class _FakeCamera:
  """Stands in for a connected OpenCVCamera."""

  def __init__(self, width: int = 320, height: int = 240):
    self.config = types.SimpleNamespace(width=width, height=height)
    self.frame_lock = threading.Lock()
    self.latest_frame: np.ndarray | None = np.full(
        (height, width, 3), 7, dtype=np.uint8
    )
    self.connected = False
    self.primed = False
    self.disconnected = False

  def connect(self) -> None:
    self.connected = True

  def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
    del timeout_ms
    self.primed = True
    assert self.latest_frame is not None
    return self.latest_frame

  def disconnect(self) -> None:
    self.disconnected = True


class _FakeVideoCapture:
  """Stands in for `cv2.VideoCapture`."""

  def __init__(self, frame_bgr: np.ndarray, opened: bool = True):
    self._frame = frame_bgr
    self._opened = opened
    self.released = False
    self.props: dict[int, float] = {}

  def isOpened(self) -> bool:  # pylint: disable=invalid-name
    return self._opened

  def set(self, prop: int, value: float) -> bool:
    self.props[prop] = value
    return True

  def read(self) -> tuple[bool, np.ndarray | None]:
    if not self._opened or self.released:
      return False, None
    return True, self._frame.copy()

  def release(self) -> None:
    self.released = True


@absltest.skipUnless(
    _HAS_OPENCV, "OpenCV is required (requires `identify` extra)."
)
class OpenCVCameraTest(absltest.TestCase):

  def test_converts_bgr_to_rgb_and_resizes_to_configured_dimensions(self):
    # Simulate a V4L2 device (like RealSense D405) returning 424x240 BGR when
    # 320x240 is requested: blue=10, green=20, red=200 in BGR.
    bgr = np.zeros((240, 424, 3), dtype=np.uint8)
    bgr[:, :] = [10, 20, 200]
    fake_cap = _FakeVideoCapture(bgr)

    with mock.patch.object(cv2, "VideoCapture", return_value=fake_cap):
      cam = camera_device.OpenCVCamera(
          camera_device.OpenCVCameraConfig(
              index_or_path="/dev/video4", width=320, height=240, fps=30
          )
      )
      cam.connect()
      try:
        frame = cam.async_read(timeout_ms=1000)
      finally:
        cam.disconnect()

    self.assertEqual(frame.shape, (240, 320, 3))
    self.assertEqual(frame.dtype, np.uint8)
    # BGR [10, 20, 200] becomes RGB [200, 20, 10].
    np.testing.assert_array_equal(frame[0, 0], [200, 20, 10])
    self.assertTrue(fake_cap.released)

  def test_unopened_capture_raises_connection_error(self):
    fake_cap = _FakeVideoCapture(np.zeros((10, 10, 3), np.uint8), opened=False)
    with mock.patch.object(cv2, "VideoCapture", return_value=fake_cap):
      cam = camera_device.OpenCVCamera(
          camera_device.OpenCVCameraConfig(index_or_path="/dev/video99")
      )
      with self.assertRaisesRegex(ConnectionError, "video99"):
        cam.connect()
    self.assertTrue(fake_cap.released)

  def test_create_camera_device_initializes_and_streams(self):
    bgr = np.full((240, 320, 3), 42, dtype=np.uint8)
    fake_cap = _FakeVideoCapture(bgr)
    with mock.patch.object(cv2, "VideoCapture", return_value=fake_cap):
      dev = camera_device.create_camera_device(
          index_or_path=0,
          name="wrist_camera",
          config=camera_device.CameraConfig(width=320, height=240, fps=30),
      )
      try:
        meas = dev.get_measurements()
      finally:
        dev.close()

    self.assertEqual(meas["wrist_camera"].shape, (240, 320, 3))
    self.assertTrue(fake_cap.released)


class OpenCVCameraDeviceTest(absltest.TestCase):

  def test_measurement_spec_matches_camera_config(self):
    device = camera_device.OpenCVCameraDevice(
        _FakeCamera(width=320, height=240), name="wrist_camera"
    )

    spec = device.measurements_spec()["wrist_camera"]

    self.assertEqual(spec.shape, (240, 320, 3))
    self.assertEqual(spec.dtype, np.uint8)

  def test_is_measurement_only(self):
    device = camera_device.OpenCVCameraDevice(_FakeCamera())

    self.assertEmpty(device.commands_spec())

  def test_initialize_connects_and_primes_camera(self):
    camera = _FakeCamera()
    device = camera_device.OpenCVCameraDevice(camera, name="wrist")

    device.initialize()

    self.assertTrue(camera.connected)
    self.assertTrue(camera.primed)

  def test_reads_the_latest_frame(self):
    camera = _FakeCamera()
    device = camera_device.OpenCVCameraDevice(camera, name="wrist")

    frame = device.get_measurements()["wrist"]

    np.testing.assert_array_equal(frame, camera.latest_frame)

  def test_missing_frame_falls_back_to_the_previous_one(self):
    camera = _FakeCamera()
    device = camera_device.OpenCVCameraDevice(camera, name="wrist")
    good = device.get_measurements()["wrist"].copy()

    camera.latest_frame = None
    frame = device.get_measurements()["wrist"]

    np.testing.assert_array_equal(frame, good)

  def test_close_disconnects(self):
    camera = _FakeCamera()
    device = camera_device.OpenCVCameraDevice(camera)

    device.close()

    self.assertTrue(camera.disconnected)


if __name__ == "__main__":
  absltest.main()
