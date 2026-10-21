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

"""The simulated SO101 follower device.

`SO101Follower` speaks the real follower's protocol exactly — the same
measurement and command keys from `real/constant.py`, so the simulated device
can swap in for the real one.
"""

from collections.abc import Mapping
import time
from typing import Union

from dm_env import specs
import mujoco
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
import numpy as np
from reaf.simulation import torch as device
from reaf.simulation.torch import device as device_base
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import types as sim_types
import torch
from typing_extensions import override


class SO101Follower(device_base.Device):
  """The simulated follower arm, matches SO101 real device protocol."""

  def __init__(self, name: str = "", nworld: int = 1, xpu_device: str = "cpu"):
    del nworld
    self._name = name
    self._xpu_device = xpu_device
    self._qpos_ids: torch.Tensor | None = None
    self._actuator_ids: torch.Tensor | None = None
    self._joint_bounds: np.ndarray | None = None  # (njoint, 2), radians.
    self._gripper_range: tuple[float, float] | None = None
    self._last_cmd_timestamp_ns: int = 0

  @property
  @override
  def name(self) -> str:
    return self._name

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    mj_model = simulator.mj_model
    joints = [
        mj_model.joint(f"{sim_constant.ATTACHMENT_PREFIX}{joint}")
        for joint in constant.JOINTS
    ]
    self._qpos_ids = torch.tensor(
        [joint.qposadr.item() for joint in joints], device=self._xpu_device
    )
    self._actuator_ids = torch.tensor(
        [
            mj_model.actuator(f"{sim_constant.ATTACHMENT_PREFIX}{joint}").id
            for joint in constant.JOINTS
        ],
        device=self._xpu_device,
    )
    self._joint_bounds = np.array([joint.range for joint in joints])
    low, high = joints[-1].range
    self._gripper_range = (float(low), float(high))

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    # The MJCF joint range is the same frame as the real arm's calibrated
    # range; the gripper is commanded as its 0-100 percentage, as on the bus.
    assert self._joint_bounds is not None
    return {
        key: specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=(
                self._joint_bounds[i, 0]
                if joint != constant.GRIPPER_JOINT
                else constant.GRIPPER_MIN
            ),
            maximum=(
                self._joint_bounds[i, 1]
                if joint != constant.GRIPPER_JOINT
                else constant.GRIPPER_MAX
            ),
        )
        for i, (joint, key) in enumerate(constant.GOAL_POSITION_KEYS.items())
    }

  @override
  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    specs_dict = {
        key: specs.Array(shape=(), dtype=np.float32)
        for key in constant.JOINT_POSITION_KEYS.values()
    }
    specs_dict[constant.FOLLOWER_COMMAND_TIMESTAMP] = specs.Array(
        shape=(), dtype=np.int64
    )
    return specs_dict

  @override
  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    assert self._gripper_range is not None
    low, high = self._gripper_range
    goals = dict(consumed_substep_commands)
    gripper_key = constant.GOAL_POSITION_KEYS[constant.GRIPPER_JOINT]
    goals[gripper_key] = low + goals[gripper_key] / constant.GRIPPER_MAX * (
        high - low
    )
    goal = torch.stack(
        [goals[key] for key in constant.GOAL_POSITION_KEYS.values()], dim=-1
    )
    simulator.data.ctrl[:, self._actuator_ids] = goal.to(
        dtype=simulator.data.ctrl.dtype
    )
    self._last_cmd_timestamp_ns = time.time_ns()

  @override
  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    assert self._gripper_range is not None
    joints = simulator.data.qpos[:, self._qpos_ids]
    values = {
        constant.JOINT_POSITION_KEYS[joint]: joints[:, i]
        for i, joint in enumerate(constant.JOINTS)
    }
    low, high = self._gripper_range
    gripper_key = constant.JOINT_POSITION_KEYS[constant.GRIPPER_JOINT]
    values[gripper_key] = (
        (values[gripper_key] - low) / (high - low) * constant.GRIPPER_MAX
    )
    values[constant.FOLLOWER_COMMAND_TIMESTAMP] = torch.tensor(
        self._last_cmd_timestamp_ns, device=self._xpu_device, dtype=torch.int64
    ).expand(simulator.data.qpos.shape[0])
    return values


def follower_device_config(
    base_spec: mujoco.MjSpec, control_period_seconds: float
) -> config.Environment:
  """Returns config for the follower arm and wrist camera devices."""
  # the base_spec must contain both the robot and the scene.
  cfg = config.Environment(base_scene=base_spec)
  cfg.sim.nworld = 1
  cfg.sim.xpu_device = "cpu"
  cfg.sim.impl = "cpp"
  cfg.sim.control_timestep = control_period_seconds
  assert cfg.sim.control_timestep > cfg.sim.simulation_timestep

  cfg.device_layer.devices["follower"] = config.Component(
      SO101Follower,
  )
  cfg.device_layer.devices["wrist_camera"] = config.Component(
      device.MujocoCameraDevice,
      camera_name=sim_constant.WRIST_CAM_FULLNAME,
      width=constant.CAMERA_WIDTH,
      height=constant.CAMERA_HEIGHT,
      render_rate_hz=constant.CAMERA_FPS,
      image_output_name_prefix=constant.WRIST_CAMERA,
      rgb_output_name_suffix="",
  )
  return cfg
