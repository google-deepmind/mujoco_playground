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

"""Real-hardware building blocks shared across robots."""

from mujoco_playground._src.real.common.camera_device import create_opencv_camera_device as create_opencv_camera_device
from mujoco_playground._src.real.common.camera_device import OpenCVCamera as OpenCVCamera
from mujoco_playground._src.real.common.camera_device import OpenCVCameraConfig as OpenCVCameraConfig
from mujoco_playground._src.real.common.camera_device import OpenCVCameraDevice as OpenCVCameraDevice
from mujoco_playground._src.real.common.clock_device import ClockDevice as ClockDevice
from mujoco_playground._src.real.common.log_reader import iter_features as iter_features
from mujoco_playground._src.real.common.logger import create_logger as create_logger
from mujoco_playground._src.real.common.motion import cosine_interpolate as cosine_interpolate
