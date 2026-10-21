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

"""REAF device for the real YAM follower arm.

The I2RT SDK (`yam` extra) runs its own 250 Hz loop on the CAN bus: joint-space
PD towards the latest target plus gravity compensation. This device sets
targets and reads the measured pose, in the canonical units of `constant.py`.
"""

from collections.abc import Mapping
import datetime
import time
from typing import Any
from typing import Dict
from typing import TypeAlias

from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import types as gdmr_types
from mujoco_playground import real
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.real import get_cameras
import numpy as np
from reaf.common import device_sequence_coordinator
from reaf.common import time_trigger
from reaf.core import device as reaf_device
from reaf.core import device_layer as dl_module
from typing_extensions import override

OpenCVCameraConfig = real.OpenCVCameraConfig

# Stand-in for `i2rt.robots.motor_chain_robot.MotorChainRobot`; i2rt is an
# optional dependency.
MotorChainRobot: TypeAlias = Any


def check_can_up(channel: str = constant.CAN_CHANNEL) -> None:
  """Raises with the fix if the SocketCAN interface is missing or down."""
  state = epath.Path(f"/sys/class/net/{channel}/operstate")
  if not state.exists():
    raise RuntimeError(
        f"No CAN interface '{channel}'. Is the USB-CAN adapter plugged in?"
        " `ip link show` lists interfaces."
    )
  # A CAN netdev reports "up" or, with some adapters, "unknown" once up.
  if state.read_text().strip() == "down":
    raise RuntimeError(
        f"CAN interface '{channel}' is down. Bring it up with:\n  sudo ip link"
        f" set {channel} up type can bitrate {constant.CAN_BITRATE}"
    )


def _close_robot(robot: MotorChainRobot) -> None:
  """Stops the SDK control loop and closes the CAN interface cleanly.

  `DMChainCanInterface.close()` sets `running = False` and immediately closes
  the CAN socket without joining its control thread. Stop the server thread
  first, then set `running = False` under `command_lock` so any in-flight CAN
  round-trip finishes before the socket is closed.
  """
  if hasattr(robot, "_stop_event"):
    robot._stop_event.set()
  if hasattr(robot, "_server_thread"):
    robot._server_thread.join()
  chain = getattr(robot, "motor_chain", None)
  if chain is not None and hasattr(chain, "command_lock"):
    with chain.command_lock:
      chain.running = False
    time.sleep(0.01)
  robot.close()


def read_joints(robot: MotorChainRobot) -> Dict[str, float]:
  """Reads the arm's pose in canonical units (rad for arm, [0, 1] for gripper)."""
  pos = np.asarray(robot.get_joint_pos(), dtype=float)
  return {
      joint: float(value)
      for joint, value in zip(constant.JOINTS, pos, strict=True)
  }


def write_goals(robot: MotorChainRobot, joints: Mapping[str, float]) -> None:
  """Clamps goal positions to `JOINT_BOUNDS` and sends them to the robot."""
  pose = np.empty(len(constant.JOINTS), dtype=float)
  for i, joint in enumerate(constant.JOINTS):
    low, high = constant.JOINT_BOUNDS[joint]
    pose[i] = float(np.clip(joints[joint], low, high))
  robot.command_joint_pos(pose)


class YamFollowerDevice(reaf_device.Device):
  """The real follower arm: reads joint positions, accepts goal positions."""

  def __init__(self, robot: MotorChainRobot):
    self._robot = robot
    self._last_cmd_timestamp_ns: int = 0

  @property
  @override
  def name(self) -> str:
    return constant.FOLLOWER

  @override
  def measurements_spec(self) -> Dict[str, specs.Array]:
    specs_dict = {
        key: specs.Array(shape=(), dtype=np.float32)
        for key in constant.JOINT_POSITION_KEYS.values()
    }
    specs_dict[constant.FOLLOWER_COMMAND_TIMESTAMP] = specs.Array(
        shape=(), dtype=np.int64
    )
    return specs_dict

  @override
  def commands_spec(self) -> Dict[str, Any]:
    return {
        constant.GOAL_POSITION_KEYS[joint]: specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=low, maximum=high
        )
        for joint, (low, high) in constant.JOINT_BOUNDS.items()
    }

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    joints = read_joints(self._robot)
    meas: dict[str, np.generic] = {
        constant.JOINT_POSITION_KEYS[joint]: np.float32(value)
        for joint, value in joints.items()
    }
    meas[constant.FOLLOWER_COMMAND_TIMESTAMP] = np.int64(
        self._last_cmd_timestamp_ns
    )
    return meas

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    goals = {
        joint: float(np.asarray(commands[key]))
        for joint, key in constant.GOAL_POSITION_KEYS.items()
    }
    write_goals(self._robot, goals)
    self._last_cmd_timestamp_ns = time.time_ns()

  def close(self) -> None:
    _close_robot(self._robot)


def connect_robot(channel: str = constant.CAN_CHANNEL) -> MotorChainRobot:
  """Connects to the I2RT `MotorChainRobot` on `channel`."""
  from i2rt.robots import get_robot  # `yam` extra.  # pyrefly: ignore[missing-import]
  from i2rt.robots import utils  # `yam` extra.  # pyrefly: ignore[missing-import]

  check_can_up(channel)
  return get_robot.get_yam_robot(
      channel=channel,
      arm_type=utils.ArmType(constant.ARM_TYPE),
      gripper_type=utils.GripperType(constant.GRIPPER_TYPE),
      zero_gravity_mode=False,
  )


def create_follower_device(
    channel: str = constant.CAN_CHANNEL,
) -> YamFollowerDevice:
  """Connects to the real follower arm and wraps it as a REAF device."""
  return YamFollowerDevice(connect_robot(channel))


def create_wrist_camera_device(
    name: str = constant.WRIST_CAMERA,
    config: real.OpenCVCameraConfig | None = None,
) -> real.OpenCVCameraDevice:
  """Returns a connected wrist-camera device streaming in the background."""
  if config is None:
    config = real.OpenCVCameraConfig(
        width=constant.CAMERA_WIDTH,
        height=constant.CAMERA_HEIGHT,
        fps=constant.CAMERA_FPS,
    )
  return real.create_opencv_camera_device(
      index_or_path=get_cameras.resolve()[get_cameras.WRIST],
      name=name,
      config=config,
  )


def create_device_layer(
    *devices: reaf_device.Device,
    control_period_seconds: float = constant.CONTROL_PERIOD,
) -> dl_module.DeviceLayer:
  """The real device layer: the devices sequenced in order, paced at its rate."""
  trigger = (
      time_trigger.TimeTrigger(
          period=datetime.timedelta(seconds=control_period_seconds)
      )
      if control_period_seconds > 0
      else None
  )
  return dl_module.DeviceLayer(
      device_coordinator=device_sequence_coordinator.DeviceSequenceCoordinator(
          devices=list(devices), name=constant.ROBOT
      ),
      commands_trigger=None,
      measurements_trigger=trigger,
  )
