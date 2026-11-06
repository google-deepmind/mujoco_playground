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

"""Resolves which camera belongs to the YAM.

Camera ids are machine-local (a /dev path on Linux, an OpenCV index on macOS),
generated into a gitignored config. Generate with:

  uv run --extra yam python -m mujoco_playground.robots.yam.real.get_cameras --identify
"""

from collections.abc import Mapping
import pathlib

from mujoco_playground._src.real import find_opencv_camera
from mujoco_playground.robots.yam.real import constant

WRIST = "wrist"
CAMERAS = (WRIST,)
SCAN_COMMAND = (
    "uv run --extra yam python -m"
    f" {__package__}.{pathlib.Path(__file__).stem} --identify"
)


def resolve() -> Mapping[str, find_opencv_camera.CameraId]:
  """Returns the camera id for each YAM camera name."""
  return find_opencv_camera.load_camera_config(
      constant.ROBOT, CAMERAS, SCAN_COMMAND
  )


if __name__ == "__main__":
  from absl import app
  from absl import flags
  from mujoco_playground._src.real import find_hardware

  _IDENTIFY = flags.DEFINE_bool(
      "identify",
      False,
      "Save one preview frame per detected camera and attribute each YAM"
      " camera by picking its preview, then write this machine's camera"
      " config.",
  )

  def main(argv):
    del argv  # Unused.
    find_hardware.run_cameras(constant.ROBOT, CAMERAS, identify=_IDENTIFY.value)

  app.run(main)
