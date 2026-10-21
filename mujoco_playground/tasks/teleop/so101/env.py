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

"""The teleop environment.

The environment contains only the follower (the controllable system); the
leader arm, a gamepad, or a learned policy live outside it. The device layer
is either the real robot or a simulated one. Both use the same commands
and measurements.
"""

import datetime
from typing import Any
from typing import Literal

from etils import epath
from mujoco_playground import paths
from mujoco_playground import wrapper as wrapper_lib
from mujoco_playground._src.objects import primitive
from mujoco_playground._src.real.common import motion
from mujoco_playground.real import instruction_feature
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as so101_device
from mujoco_playground.robots.so101.sim import device as so101_sim_device
from mujoco_playground.robots.so101.sim import scene as so101_scene
import numpy as np
from reaf.common import environment_reset_from_callable
from reaf.common import maximum_steps_termination_checker
from reaf.common import timestamp_features_producer
from reaf.core import environment as reaf_environment
from reaf.core import task_layer as tl_module
from reaf.core import termination_checker as tc_module
from reaf.simulation.torch import flat_scene as build

# Teleop's control rate, matching the real arm: both device layers are built
# to step at this period, and the logger records it as the episode timestep.
CONTROL_PERIOD_SECONDS = 0.02  # 50 Hz.
EPISODE_PERIOD = datetime.timedelta(seconds=CONTROL_PERIOD_SECONDS)


def default_log_dir() -> epath.Path:
  """Returns the default directory for teleop logs."""
  return epath.Path(paths.local_dir()) / f"{constant.ROBOT}_teleop_logs"


_HOME_SETTLE_SECONDS = 1.5


def drive_home(
    device_layer: wrapper_lib.DeviceLayer,
    real_time_homing: bool = True,
) -> None:
  """Smoothly drives the follower to the home pose using cosine interpolation."""
  meas = device_layer.begin_stepping()
  meas_keys = list(constant.JOINT_POSITION_KEYS.values())
  cmd_keys = list(constant.GOAL_POSITION_KEYS.values())
  start_arr = np.array([meas[k] for k in meas_keys], dtype=np.float32)

  def _step(pose: np.ndarray) -> None:
    device_layer.step(
        {k: float(v) for k, v in zip(cmd_keys, pose, strict=True)}
    )

  motion.cosine_interpolate(
      _step,
      start_pose=start_arr,
      target_pose=constant.FOLLOWER_HOME,
      duration_secs=_HOME_SETTLE_SECONDS,
      control_timestep=CONTROL_PERIOD_SECONDS,
      real_time=real_time_homing,
  )


class RealTeleopEnvironmentCloser(reaf_environment.EnvironmentCloser):
  """Parks the real follower and closes hardware devices."""

  def __init__(
      self,
      device_layer: wrapper_lib.DeviceLayer,
      closeables: tuple[Any, ...],
      real_time_homing: bool = True,
  ):
    self._device_layer = device_layer
    self._closeables = closeables
    self._real_time_homing = real_time_homing

  def close(self) -> None:
    print("Returning the follower home before releasing it...")
    drive_home(self._device_layer, real_time_homing=self._real_time_homing)
    for closeable in self._closeables:
      closeable.close()


def _build_teleop_environment(
    device_layer: wrapper_lib.DeviceLayer,
    episode_length_in_seconds: float = 100.0,
    termination_checker: tc_module.TerminationChecker | None = None,
    instruction: str = "",
    real_time_homing: bool = True,
    environment_closer: reaf_environment.EnvironmentCloser | None = None,
) -> reaf_environment.Environment:
  """Builds the common teleop Environment over any follower DeviceLayer."""

  def _drive_follower_home(_) -> None:
    print("Driving follower to the home pose...")
    drive_home(device_layer, real_time_homing=real_time_homing)

  termination_checkers: list[tc_module.TerminationChecker] = [
      maximum_steps_termination_checker.MaximumStepsTerminationChecker(
          max_steps=int(
              episode_length_in_seconds / EPISODE_PERIOD.total_seconds()
          )
      )
  ]
  if termination_checker is not None:
    termination_checkers.append(termination_checker)
  task_layer = tl_module.TaskLayer(
      commands_processors=(),
      features_producers=(
          timestamp_features_producer.TimestampFeaturesProducer(
              timestamp_key=constant.TIMESTAMP
          ),
          instruction_feature.InstructionFeaturesProducer(instruction),
      ),
      termination_checkers=termination_checkers,
      reward_provider=None,
      discount_provider=None,
  )
  return reaf_environment.Environment(
      device_layer=device_layer,  # pyrefly: ignore[bad-argument-type]
      task_layer=task_layer,
      environment_reset=environment_reset_from_callable.EnvironmentResetFromCallable(
          _drive_follower_home
      ),
      environment_closer=environment_closer,
  )


def create_sim_env(
    sim_impl: Literal["cpp", "warp"] = "cpp",
    episode_length_in_seconds: float = 100.0,
    termination_checker: tc_module.TerminationChecker | None = None,
    instruction: str = "",
    real_time_homing: bool = False,
) -> reaf_environment.Environment:
  """Creates the simulated SO101 teleop environment."""
  workspace = so101_scene.so101_aloha_table()
  cube_frame = workspace.worldbody.add_frame(
      pos=np.array([-0.02, 0.0, so101_scene.TABLE_HEIGHT + 0.015])
  )
  cube_frame.attach_body(primitive.cube("cube").worldbody.first_body(), "", "")
  cfg = so101_sim_device.follower_device_config(
      workspace, CONTROL_PERIOD_SECONDS
  )
  cfg.sim.impl = sim_impl
  layer = wrapper_lib.NumpyDeviceLayerWrapper(
      wrapper_lib.SingleDeviceLayerWrapper(build.build_device_layer(cfg))
  )
  return _build_teleop_environment(
      device_layer=layer,
      episode_length_in_seconds=episode_length_in_seconds,
      termination_checker=termination_checker,
      instruction=instruction,
      real_time_homing=real_time_homing,
  )


def create_real_env(
    episode_length_in_seconds: float = 100.0,
    termination_checker: tc_module.TerminationChecker | None = None,
    instruction: str = "",
    real_time_homing: bool = True,
    camera_config: so101_device.CameraConfig | None = None,
) -> reaf_environment.Environment:
  """Creates the real hardware SO101 teleop environment."""
  follower = so101_device.create_follower_device()
  camera = so101_device.create_wrist_camera_device(config=camera_config)
  layer = so101_device.create_device_layer(
      follower,
      camera,
      control_period_seconds=CONTROL_PERIOD_SECONDS,
  )
  closer = RealTeleopEnvironmentCloser(
      device_layer=layer,
      closeables=(follower, camera),
      real_time_homing=real_time_homing,
  )
  return _build_teleop_environment(
      device_layer=layer,
      episode_length_in_seconds=episode_length_in_seconds,
      termination_checker=termination_checker,
      instruction=instruction,
      real_time_homing=real_time_homing,
      environment_closer=closer,
  )
