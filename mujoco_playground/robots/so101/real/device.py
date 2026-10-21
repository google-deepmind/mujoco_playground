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

"""REAF device for the real SO101 follower arm."""

from collections.abc import Sequence
import dataclasses
import datetime
import time
from typing import Any
from typing import Dict
from typing import Mapping
from typing import TypeAlias

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from mujoco_playground import real
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import get_calibration
from mujoco_playground.robots.so101.real import get_cameras
from mujoco_playground.robots.so101.real import get_ports
from mujoco_playground.robots.so101.sim import constant as sim_constant
import numpy as np
from reaf.common import device_sequence_coordinator
from reaf.common import time_trigger
from reaf.core import device as reaf_device
from reaf.core import device_layer as dl_module
from typing_extensions import override

# Stand-ins for the lerobot classes. lerobot is an optional dep.
SO101Follower: TypeAlias = Any
SO101Leader: TypeAlias = Any
FeetechMotorsBus: TypeAlias = Any

# The arm's calibration data, as loaded by `get_calibration.load`.
Calibration: TypeAlias = Mapping[str, Mapping[str, int]]

_RADIANS_PER_TICK = 2.0 * np.pi / constant.TICKS_PER_TURN


class SO101FollowerDevice(reaf_device.Device):
  """The real follower arm: reads joint positions, accepts goal positions.

  Only the follower is a device: the environment contains the controllable
  system, while the leader arm is a `Policy`.
  """

  def __init__(self, robot: SO101Follower, motor_calibration: Calibration):
    self._robot = robot
    self._motor_calibration = motor_calibration
    self._bounds = _joint_range(motor_calibration)
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
        for joint, (low, high) in self._bounds.items()
    }

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    joints = read_joints(self._robot.bus, self._motor_calibration)
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
    # `write_goals` clamps to the calibrated range before writing to the bus.
    write_goals(self._robot.bus, self._motor_calibration, goals)
    self._last_cmd_timestamp_ns = time.time_ns()

  def close(self) -> None:
    self._robot.disconnect()


class SO101ArmStateDevice(reaf_device.Device):
  """The arm's measured state, mirroring the simulated `ArticulatedDevice`.

  Emits joint positions, velocities, home pose vectors, and end-effector site
  pose (`ee/xpos` and `ee/xmat`) via forward kinematics.
  """

  def __init__(
      self,
      bus: FeetechMotorsBus,
      motor_calibration: Calibration,
      *,
      home_qpos: Sequence[float] | np.ndarray,
      control_timestep: float,
      model: mujoco.MjModel,
      name: str = constant.ROBOT_STATE,
  ):
    self._bus = bus
    self._motor_calibration = motor_calibration
    self._control_timestep = control_timestep
    self._name = name
    self._home_qpos = np.array(home_qpos, dtype=np.float32)
    self._joint_pos: np.ndarray = self._home_qpos.copy()
    self._joint_vel: np.ndarray = np.zeros(
        len(constant.JOINTS), dtype=np.float32
    )

    self._model = model
    self._data = mujoco.MjData(model)
    self._ee_site_id = model.site(sim_constant.EE_SITE).id
    self._qpos_adr = np.array([
        model.joint(f"{sim_constant.ATTACHMENT_PREFIX}{j}").qposadr[0]
        for j in constant.JOINTS
    ])

  @property
  @override
  def name(self) -> str:
    return self._name

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    spec = specs.Array(shape=(len(constant.JOINTS),), dtype=np.float32)
    return {
        f"{self._name}/{constant.HOME_QPOS}": spec,
        f"{self._name}/{constant.JOINT_POS}": spec,
        f"{self._name}/{constant.JOINT_VEL}": spec,
        "ee/xpos": specs.Array(shape=(3,), dtype=np.float32),
        "ee/xmat": specs.Array(shape=(9,), dtype=np.float32),
    }

  @override
  def get_measurements(self) -> Mapping[str, Any]:
    """Reads joint sensors and computes forward kinematics."""
    joints = read_joints(self._bus, self._motor_calibration)
    joints[constant.GRIPPER_JOINT] = constant.gripper_pct_to_rad(
        joints[constant.GRIPPER_JOINT]
    )
    joint_pos = np.array(
        [joints[joint] for joint in constant.JOINTS], dtype=np.float32
    )
    self._joint_vel = (joint_pos - self._joint_pos) / self._control_timestep
    self._joint_pos = joint_pos
    self._data.qpos[self._qpos_adr] = joint_pos
    mujoco.mj_kinematics(self._model, self._data)
    return {
        f"{self._name}/{constant.HOME_QPOS}": self._home_qpos,
        f"{self._name}/{constant.JOINT_POS}": self._joint_pos,
        f"{self._name}/{constant.JOINT_VEL}": self._joint_vel,
        "ee/xpos": self._data.site_xpos[self._ee_site_id].astype(np.float32),
        "ee/xmat": self._data.site_xmat[self._ee_site_id].astype(np.float32),
    }

  @override
  def commands_spec(self) -> Dict[str, gdmr_types.AnyArraySpec]:
    return {}

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    del commands


class SO101ArmActuatorDevice(reaf_device.Device):
  """The arm's position actuators, mirroring the simulated `ActuatorDevice`.

  Consumes `<name>/ctrl`: one position target per joint, in radians, in
  canonical joint order.
  """

  def __init__(
      self,
      bus: FeetechMotorsBus,
      motor_calibration: Calibration,
      name: str = "robot",
  ):
    self._bus = bus
    self._motor_calibration = motor_calibration
    self._name = name
    self._ctrl_key = f"{name}/ctrl"
    bounds = _joint_range(motor_calibration)
    # The bus speaks percent for the gripper, but `ctrl` is radians throughout.
    bounds[constant.GRIPPER_JOINT] = constant.GRIPPER_RAD_RANGE
    self._minimum, self._maximum = (
        np.array(b, dtype=np.float32)
        for b in zip(*(bounds[joint] for joint in constant.JOINTS))
    )

  @property
  @override
  def name(self) -> str:
    return self._name

  def write(self, joint_pos: np.ndarray) -> None:
    """Sends position targets, in radians; `write_goals` clamps them."""
    goals = dict(zip(constant.JOINTS, (float(v) for v in joint_pos)))
    goals[constant.GRIPPER_JOINT] = constant.gripper_rad_to_pct(
        goals[constant.GRIPPER_JOINT]
    )
    write_goals(self._bus, self._motor_calibration, goals)

  @override
  def measurements_spec(self) -> Dict[str, specs.Array]:
    return {}

  @override
  def get_measurements(self) -> Mapping[str, gdmr_types.ArrayType]:
    return {}

  @override
  def commands_spec(self) -> Dict[str, Any]:
    return {
        self._ctrl_key: specs.BoundedArray(
            shape=(len(constant.JOINTS),),
            dtype=np.float32,
            minimum=self._minimum,
            maximum=self._maximum,
        )
    }

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    self.write(np.asarray(commands[self._ctrl_key], dtype=np.float32))


def create_follower_device() -> SO101FollowerDevice:
  """Connects to the real follower arm and wraps it as a REAF device."""
  robot, motor_calibration = connect_follower()
  return SO101FollowerDevice(robot, motor_calibration)


CameraConfig = real.OpenCVCameraConfig


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
    control_period_seconds: float,
) -> dl_module.DeviceLayer:
  """The real device layer: the devices sequenced in order, paced at its rate."""
  return dl_module.DeviceLayer(
      device_coordinator=device_sequence_coordinator.DeviceSequenceCoordinator(
          devices=list(devices), name="so101"
      ),
      commands_trigger=None,
      measurements_trigger=time_trigger.TimeTrigger(
          period=datetime.timedelta(seconds=control_period_seconds)
      ),
  )


def read_joints(
    bus: FeetechMotorsBus, motor_calibration: Calibration
) -> Dict[str, float]:
  """The arm's pose in our units, read raw off the bus."""
  return _raw_to_radians(
      bus.sync_read("Present_Position", normalize=False), motor_calibration
  )


def write_goals(
    bus: FeetechMotorsBus,
    motor_calibration: Calibration,
    joints: Mapping[str, float],
) -> None:
  """Sends goal positions (in our unit convention); torque must be enabled."""
  bus.sync_write(
      "Goal_Position",
      _radians_to_raw(joints, motor_calibration),
      normalize=False,
  )


def connect_leader() -> tuple[SO101Leader, Calibration]:
  """Returns a connected, torque-free leader and its calibration."""
  from lerobot.teleoperators import so_leader  # pyrefly: ignore[missing-import]

  robot = so_leader.SO101Leader(
      so_leader.SO101LeaderConfig(
          port=get_ports.resolve(constant.LEADER),
          id=f"{constant.ROBOT}_leader",
      )
  )
  robot.connect(calibrate=False)
  robot.bus.disable_torque()
  return robot, _load_calibration(constant.LEADER)


def _load_calibration(role: str) -> Calibration:
  """One arm's calibration, verified against its USB serial."""
  return get_calibration.load(role, get_ports.serial_number(role))


def connect_follower() -> tuple[SO101Follower, Calibration]:
  """Returns a connected follower and its calibration; the bus is used raw."""
  from lerobot.robots import so_follower  # pyrefly: ignore[missing-import]

  robot = so_follower.SO101Follower(
      so_follower.SO101FollowerConfig(
          port=get_ports.resolve(constant.FOLLOWER),
          id=f"{constant.ROBOT}_follower",
      )
  )
  robot.connect(calibrate=False)
  return robot, _load_calibration(constant.FOLLOWER)


def _raw_joint_range(
    joint: str, motor_calibration: Calibration
) -> tuple[float, float]:
  """Returns joint range as raw ticks from its calibrated zero position.

  The gripper's doubles as its 0-100% scale.
  """
  zero = motor_calibration[joint]["zero"]
  rmin = motor_calibration[joint]["range_min"]
  rmax = motor_calibration[joint]["range_max"]
  if rmin == 0 and rmax == constant.ENCODER_MAX:
    return (-float(constant.TICKS_CENTER), float(constant.TICKS_CENTER))
  lo = constant.relative_ticks(rmin, zero)
  hi = constant.relative_ticks(rmax, zero)
  assert lo <= hi, (
      f"Invalid joint range for '{joint}': lower bound ({lo}) > upper bound"
      f" ({hi}). Raw ticks: zero={zero}, range_min={rmin}, range_max={rmax}."
  )
  return (lo, hi)


def _joint_range(
    motor_calibration: Calibration,
) -> Dict[str, tuple[float, float]]:
  """Returns the calibrated travel per joint in our units."""
  return {  # pyrefly: ignore[bad-return]
      joint: (
          (constant.GRIPPER_MIN, constant.GRIPPER_MAX)
          if joint == constant.GRIPPER_JOINT
          else tuple(
              t * _RADIANS_PER_TICK
              for t in _raw_joint_range(joint, motor_calibration)
          )
      )
      for joint in constant.JOINTS
  }


def _raw_to_radians(
    raw: Mapping[str, int], motor_calibration: Calibration
) -> Dict[str, float]:
  """Raw bus ticks to radians relative to the calibrated zero pose."""
  lo, hi = _raw_joint_range(constant.GRIPPER_JOINT, motor_calibration)
  rel = {
      j: constant.relative_ticks(raw[j], motor_calibration[j]["zero"])
      for j in constant.JOINTS
  }
  return {
      joint: (
          (rel[joint] - lo) / (hi - lo) * constant.GRIPPER_MAX
          if joint == constant.GRIPPER_JOINT
          else rel[joint] * _RADIANS_PER_TICK
      )
      for joint in constant.JOINTS
  }


def _radians_to_raw(
    joints: Mapping[str, float], motor_calibration: Calibration
) -> Dict[str, int]:
  """Our units to raw goal ticks, clamped to the calibrated joint range."""
  glo, ghi = _raw_joint_range(constant.GRIPPER_JOINT, motor_calibration)
  out = {}
  for joint, value in joints.items():
    rel = (
        glo + value / constant.GRIPPER_MAX * (ghi - glo)
        if joint == constant.GRIPPER_JOINT
        else value / _RADIANS_PER_TICK
    )
    lo, hi = _raw_joint_range(joint, motor_calibration)
    rel = min(max(rel, lo), hi)
    out[joint] = int(motor_calibration[joint]["zero"] + round(rel))
  return out
