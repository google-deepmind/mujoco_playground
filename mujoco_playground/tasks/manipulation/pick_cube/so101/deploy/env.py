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

"""The SO101 pick-cube deployment environment, in simulation or on hardware."""

from collections.abc import Mapping
from collections.abc import Sequence
import math
import time
from typing import Any
from typing import Literal

from dm_env import specs
from mujoco_playground import wrapper
from mujoco_playground._src.real.common import clock_device
from mujoco_playground._src.real.common import motion
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as real_device
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from mujoco_playground.tasks.manipulation.pick_cube.so101 import task as so101_task
import numpy as np
from reaf.core import device as reaf_device
from reaf.core import environment as reaf_environment
from reaf.simulation.torch import environment as torch_environment
from reaf.simulation.torch import flat_scene as build
from reaf.simulation.torch import functional_observation
from reaf.simulation.torch import protocols
from typing_extensions import override

# The observation groups the trained policy consumes.
OBSERVATION_GROUPS = ("policy", "camera")

# The pose every episode starts from, which the policy was trained against.
_HOVER_POSE = np.array(constant.HOVER_POSE_OPEN_GRIPPER, dtype=np.float32)

# Statically stable pose the arm parks at when the environment closes.
_REST_POSE = np.array(
    list(constant.FOLLOWER_HOME[:5])
    + [constant.gripper_pct_to_rad(constant.FOLLOWER_HOME[-1])],
    dtype=np.float32,
)


# TODO(b/513593452, btaba): Replace TargetDevice with first-class TaskState
# baked into REAF Environment (get_state/set_state public API) once available.
class TargetDevice(reaf_device.Device):
  """Where the cube should end up, mirroring the simulated mocap target."""

  def __init__(self, target_pos: Sequence[float]):
    self._key = pick_task.TARGET_MOCAP_POS
    self._target_pos = np.asarray(target_pos, dtype=np.float32)

  @property
  @override
  def name(self) -> str:
    return pick_task.TARGET

  @property
  def target_pos(self) -> np.ndarray:
    return self._target_pos

  @target_pos.setter
  def target_pos(self, pos: Sequence[float] | np.ndarray) -> None:
    self._target_pos = np.asarray(pos, dtype=np.float32)

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    return {self._key: specs.Array(shape=(3,), dtype=np.float32)}

  @override
  def commands_spec(self) -> Mapping[str, Any]:
    return {}

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    return {self._key: self._target_pos.copy()}

  @override
  def set_commands(self, commands: Mapping[str, Any]) -> None:
    del commands


class HomingReset(torch_environment.EnvironmentReset):
  """Starts an episode by driving the real arm back to the hover pose."""

  def __init__(
      self,
      *,
      clock: clock_device.ClockDevice,
      state: real_device.SO101ArmStateDevice,
      actuator: real_device.SO101ArmActuatorDevice,
      duration_secs: float,
      control_timestep: float,
  ):
    self._clock = clock
    self._state = state
    self._actuator = actuator
    self._duration_secs = duration_secs
    self._control_timestep = control_timestep

  @override
  def on_reset(self, config: Any, env_mask: Any) -> None:
    del config, env_mask  # There is only one arm.
    motion.cosine_interpolate(
        self._actuator.write,
        start_pose=np.asarray(
            self._state.get_measurements()[
                f"{self._state.name}/{constant.JOINT_POS}"
            ]
        ),
        target_pose=_HOVER_POSE,
        duration_secs=self._duration_secs,
        control_timestep=self._control_timestep,
    )
    self._clock.reset()


class Shutdown(reaf_environment.EnvironmentCloser):
  """Parks the real arm and releases the hardware."""

  def __init__(
      self,
      *,
      robot: Any,
      camera: Any,
      state: real_device.SO101ArmStateDevice,
      actuator: real_device.SO101ArmActuatorDevice,
      duration_secs: float,
      control_timestep: float,
  ):
    self._robot = robot
    self._camera = camera
    self._state = state
    self._actuator = actuator
    self._duration_secs = duration_secs
    self._control_timestep = control_timestep

  @override
  def close(self) -> None:
    motion.cosine_interpolate(
        self._actuator.write,
        start_pose=np.asarray(
            self._state.get_measurements()[
                f"{self._state.name}/{constant.JOINT_POS}"
            ]
        ),
        target_pose=_REST_POSE,
        duration_secs=self._duration_secs,
        control_timestep=self._control_timestep,
    )
    self._camera.close()
    self._robot.disconnect()


def create_sim_env(
    *,
    impl: Literal["cpp", "warp"] = "cpp",
    episode_length_secs: float = 20.0,
    control_freq_hz: float | None = None,
) -> torch_environment.Environment:
  """The sim deployment environment."""
  cfg = so101_task.so101_pick_cube_pixels_env(impl=impl)
  cfg.episode_length_secs = episode_length_secs
  if control_freq_hz is not None:
    cfg.sim.control_timestep = 1.0 / control_freq_hz
    if "camera" in cfg.device_layer.devices:
      cam_params = cfg.device_layer.devices["camera"].runtime_params
      if "render_rate_hz" in cam_params:
        cam_params["render_rate_hz"] = control_freq_hz
  deploy_cfg = build.make_deployment_config(
      cfg, observation_groups=OBSERVATION_GROUPS
  )
  return build.build_env(deploy_cfg)


def create_real_env(
    *,
    target_pos: Sequence[float],
    episode_length_secs: float = 20.0,
    homing_duration_secs: float = 2.0,
    control_freq_hz: float | None = None,
    camera_config: real_device.CameraConfig | None = None,
) -> torch_environment.Environment:
  """The real hardware deployment environment."""
  cfg = so101_task.so101_pick_cube_pixels_env(impl="cpp")
  cfg.episode_length_secs = episode_length_secs
  cfg.sim.xpu_device = "cpu"
  if control_freq_hz is not None:
    cfg.sim.control_timestep = 1.0 / control_freq_hz
  deploy_cfg = build.make_deployment_config(
      cfg, observation_groups=OBSERVATION_GROUPS
  )
  control_timestep = deploy_cfg.sim.control_timestep

  robot, calibration = real_device.connect_follower()
  clock = clock_device.ClockDevice()
  model = build.build_scene(deploy_cfg).compile()
  state = real_device.SO101ArmStateDevice(
      robot.bus,
      calibration,
      home_qpos=_HOVER_POSE,
      control_timestep=control_timestep,
      model=model,
  )
  actuator = real_device.SO101ArmActuatorDevice(robot.bus, calibration)
  target = TargetDevice(target_pos)
  camera = real_device.create_wrist_camera_device(
      name=sim_constant.WRIST_CAM_RGB,
      config=camera_config,
  )

  device_layer = real_device.create_device_layer(
      clock,
      state,
      actuator,
      target,
      camera,
      control_period_seconds=control_timestep,
  )

  return torch_environment.Environment(
      device_layer=wrapper.TorchDeviceLayerWrapper(
          device_layer,
          sim=None,
          control_timestep=control_timestep,
          xpu_device="cpu",
      ),
      task_layer=build.build_task_layer(deploy_cfg),
      observation_space_adapter=functional_observation.FunctionalObservationAdapter(
          observations=deploy_cfg.observation_adapter,
          xpu_device="cpu",
          nworld=1,
      ),
      environment_reset=HomingReset(
          clock=clock,
          state=state,
          actuator=actuator,
          duration_secs=homing_duration_secs,
          control_timestep=control_timestep,
      ),
      environment_closer=Shutdown(
          robot=robot,
          camera=camera,
          state=state,
          actuator=actuator,
          duration_secs=homing_duration_secs,
          control_timestep=control_timestep,
      ),
  )
