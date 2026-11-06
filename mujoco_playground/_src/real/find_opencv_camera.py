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

"""Identification of cameras connected to this device using opencv.

Only cameras the OS exposes as local capture hardware are visible here: opencv
enumerates `/dev/video*` on Linux and integer indices elsewhere, so a camera
behind a robot API or a network stream cannot be discovered this way.
"""

import json
import pathlib
import platform
from typing import Mapping
from typing import Sequence
from typing import Union

from mujoco_playground._src import paths

CameraId = Union[str, int]
_BY_ID_DIR = pathlib.Path("/dev/v4l/by-id")
_MAX_OPENCV_INDEX = 10


class CameraResolutionError(RuntimeError):
  """Raised when a camera name cannot be mapped to a usable device."""


def config_path(robot: str) -> pathlib.Path:
  """Returns where a robot's machine-local camera config lives.

  The values it holds are opencv ids (`index_or_path`), not a device-neutral
  identity.
  """
  return pathlib.Path(paths.local_dir() / f"{robot}_cameras.json")


def write_camera_config(
    robot: str, cameras: Mapping[str, CameraId]
) -> pathlib.Path:
  """Writes a robot's camera config and returns the path written."""
  path = config_path(robot)
  path.write_text(json.dumps(dict(cameras), indent=2, sort_keys=True) + "\n")
  return path


def load_camera_config(
    robot: str, names: Sequence[str], scan_command: str
) -> Mapping[str, CameraId]:
  """Returns the name-to-camera-id map from a robot's camera config."""
  path = config_path(robot)
  if not path.is_file():
    raise CameraResolutionError(
        f"No camera config for '{robot}' at {path}\n\nGenerate it with:\n\n   "
        f" {scan_command}\n"
    )

  try:
    cameras = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise CameraResolutionError(
        f"{path} is not valid JSON ({exc}). Regenerate it with:\n\n   "
        f" {scan_command}\n"
    ) from exc

  missing = [name for name in names if cameras.get(name) is None]
  if missing:
    raise CameraResolutionError(
        f"{path} has no camera id for: {', '.join(missing)}\n\nRegenerate it"
        f" with:\n\n    {scan_command}\n"
    )

  for name in names:
    camera_id = cameras[name]
    if isinstance(camera_id, str) and not pathlib.Path(camera_id).exists():
      raise CameraResolutionError(
          f"Camera '{name}' points at {camera_id}, which does not exist."
          " Check the camera is plugged in, or regenerate the config"
          f" with:\n\n    {scan_command}\n"
      )
  return cameras


def stable_id(
    camera_id: CameraId, by_id_dir: pathlib.Path = _BY_ID_DIR
) -> CameraId:
  """Returns the stable /dev/v4l/by-id path for a Linux device node, if any.

  Indices (macOS/Windows) and nodes without a by-id symlink pass through
  unchanged.
  """
  if not isinstance(camera_id, str) or not by_id_dir.is_dir():
    return camera_id
  target = pathlib.Path(camera_id).resolve()
  for link in sorted(by_id_dir.iterdir()):
    if link.resolve() == target:
      return str(link)
  return camera_id


def list_cameras() -> list[CameraId]:
  """Returns the ids of cameras that can currently be opened.

  Slow (opens every candidate device); intended for setup tools, not runtime.
  """
  import cv2  # pyrefly: ignore[missing-import]

  candidates: list[CameraId]
  if platform.system() == "Linux":
    candidates = sorted(str(p) for p in pathlib.Path("/dev").glob("video*"))
  else:
    candidates = list(range(_MAX_OPENCV_INDEX))

  found = []
  for candidate in candidates:
    capture = cv2.VideoCapture(candidate)
    if capture.isOpened():
      found.append(candidate)
    capture.release()
  return found


def save_preview(camera_id: CameraId) -> pathlib.Path | None:
  """Captures one frame and writes it as a PNG; returns None on read failure."""
  import cv2  # pyrefly: ignore[missing-import]

  capture = cv2.VideoCapture(camera_id)
  ok, frame = capture.read()
  capture.release()
  if not ok:
    return None

  preview_dir = pathlib.Path(paths.local_dir() / "camera_previews")
  preview_dir.mkdir(parents=True, exist_ok=True)
  safe = str(camera_id).replace("/", "_").lstrip("_")
  path = preview_dir / f"camera_{safe}.png"
  cv2.imwrite(str(path), frame)
  return path
