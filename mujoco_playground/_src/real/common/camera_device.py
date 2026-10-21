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

"""Shared OpenCV camera reader and REAF measurement device."""

from collections.abc import Mapping
import dataclasses
import threading
import time
from typing import Any
from typing import Dict

from absl import logging
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import device as reaf_device
from typing_extensions import override

_CONNECT_RETRIES = 3


@dataclasses.dataclass(frozen=True)
class OpenCVCameraConfig:
  """Configuration for an OpenCV camera."""

  width: int = 320
  height: int = 240
  fps: int = 30
  index_or_path: int | str | None = None


# Alias for backward compatibility.
CameraConfig = OpenCVCameraConfig


class OpenCVCamera:
  """Streams RGB frames from a local V4L2/OpenCV camera on a background thread."""

  def __init__(self, config: OpenCVCameraConfig):
    self.config = config
    self.frame_lock = threading.Lock()
    self.latest_frame: np.ndarray | None = None
    self._capture: Any = None
    self._thread: threading.Thread | None = None
    self._stop_event = threading.Event()
    self._first_frame_event = threading.Event()

  def connect(self) -> None:
    """Opens the video capture device and starts the reader thread."""
    try:
      import cv2  # `identify` extra (`opencv-python-headless`).  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
      raise ImportError(
          "OpenCVCamera requires opencv-python-headless. Install with:\n\n"
          "    uv sync --extra identify\n"
      ) from exc

    self.disconnect()
    capture = cv2.VideoCapture(self.config.index_or_path)
    if not capture.isOpened():
      capture.release()
      raise ConnectionError(
          f"Could not open camera at {self.config.index_or_path!r}."
      )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
    capture.set(cv2.CAP_PROP_FPS, self.config.fps)

    self._capture = capture
    self._stop_event.clear()
    self._first_frame_event.clear()
    self._thread = threading.Thread(
        target=self._read_loop,
        name=f"opencv_camera_{self.config.index_or_path}",
        daemon=True,
    )
    self._thread.start()

  def _read_loop(self) -> None:
    """Continuously reads frames and converts them to `(H, W, 3)` uint8 RGB."""
    import cv2  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    while not self._stop_event.is_set():
      capture = self._capture
      if capture is None:
        break
      ok, frame = capture.read()
      if not ok or frame is None:
        time.sleep(0.005)
        continue

      if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
      elif frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
      else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      if (rgb.shape[0], rgb.shape[1]) != (
          self.config.height,
          self.config.width,
      ):
        rgb = cv2.resize(
            rgb,
            (self.config.width, self.config.height),
            interpolation=cv2.INTER_AREA,
        )

      rgb = np.asarray(rgb, dtype=np.uint8)
      with self.frame_lock:
        self.latest_frame = rgb
      self._first_frame_event.set()

  def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
    """Waits for at least one frame to arrive and returns the latest frame."""
    self._first_frame_event.wait(timeout=timeout_ms / 1000.0)
    with self.frame_lock:
      if self.latest_frame is None:
        raise TimeoutError(
            f"Timed out after {timeout_ms}ms waiting for a frame from"
            f" {self.config.index_or_path!r}."
        )
      return self.latest_frame

  def disconnect(self) -> None:
    """Stops the background reader thread and releases the capture device."""
    self._stop_event.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join(timeout=1.0)
    self._thread = None
    if self._capture is not None:
      self._capture.release()
      self._capture = None


class OpenCVCameraDevice(reaf_device.Device):
  """Exposes an `OpenCVCamera` as a REAF measurement-only device."""

  def __init__(self, camera: Any, name: str = "camera"):
    self._camera = camera
    self._name = name
    config = camera.config
    self._shape: tuple[int, int, int] = (config.height, config.width, 3)
    self._last_frame: np.ndarray = np.zeros(self._shape, dtype=np.uint8)

  @property
  @override
  def name(self) -> str:
    return self._name

  def initialize(self) -> None:
    """Connects the camera and primes its background reader thread."""
    for attempt in range(_CONNECT_RETRIES):
      try:
        self._camera.connect()
        self._camera.async_read(timeout_ms=1000)
        return
      except (ConnectionError, TimeoutError) as exc:
        logging.warning(
            "Failed to connect camera '%s' (attempt %d): %s",
            self._name,
            attempt + 1,
            exc,
        )
        try:
          self._camera.disconnect()
        except Exception:  # pylint: disable=broad-except
          pass
    raise ConnectionError(
        f"Could not connect camera '{self._name}' after"
        f" {_CONNECT_RETRIES} attempts."
    )

  @override
  def measurements_spec(self) -> Dict[str, specs.Array]:
    return {self._name: specs.Array(shape=self._shape, dtype=np.uint8)}

  @override
  def commands_spec(self) -> Dict[str, gdmr_types.AnyArraySpec]:
    return {}  # A sensor: no commands.

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    with self._camera.frame_lock:
      frame = self._camera.latest_frame
      if frame is not None:
        self._last_frame = np.array(frame, dtype=np.uint8)
    return {self._name: self._last_frame}

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    del commands  # A sensor: no commands.

  def close(self) -> None:
    try:
      self._camera.disconnect()
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning("Error disconnecting camera '%s': %s", self._name, exc)


def create_opencv_camera_device(
    index_or_path: int | str,
    name: str = "camera",
    config: OpenCVCameraConfig | None = None,
) -> OpenCVCameraDevice:
  """Connects an OpenCV camera and returns its initialized REAF device."""
  if config is None:
    config = OpenCVCameraConfig()
  camera = OpenCVCamera(
      OpenCVCameraConfig(
          index_or_path=index_or_path,
          width=config.width,
          height=config.height,
          fps=config.fps,
      )
  )
  device = OpenCVCameraDevice(camera, name=name)
  device.initialize()
  return device


# Alias for backward compatibility.
create_camera_device = create_opencv_camera_device
