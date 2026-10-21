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

"""The simulated YAM follower device.

`YamFollower` speaks the real follower's protocol — the same measurement and
command keys from `real/constant.py`, so the simulated device can swap in for
the real one.
"""

from collections.abc import Mapping
import time
from typing import Any

from dm_env import specs
import mujoco
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
import numpy as np
from reaf.simulation.torch import device as device_base
from reaf.simulation.torch import simulator as simulator_module
import torch
from typing_extensions import override


class YamFollower(device_base.Device):
  """The simulated follower arm, matching the YAM real device protocol."""

  def __init__(self, name: str = "", nworld: int = 1, xpu_device: str = "cpu"):
    del nworld
    self._name = name
    self._xpu_device = xpu_device
    self._qpos_ids: torch.Tensor | None = None
    self._actuator_ids: torch.Tensor | None = None
    self._joint_bounds: np.ndarray | None = None  # (njoint, 2)
    self._gripper_ctrl_range: tuple[float, float] | None = None
    self._gripper_travel: tuple[float, float] | None = None
    self._last_cmd_timestamp_ns: int = 0

  @property
  @override
  def name(self) -> str:
    return self._name

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    mj_model = simulator.mj_model
    mj_joints = (*constant.ARM_JOINTS, sim_constant.FINGER_JOINTS[0])
    joints = [
        mj_model.joint(f"{sim_constant.ATTACHMENT_PREFIX}{joint}")
        for joint in mj_joints
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
    # Commands map onto the actuator's ctrlrange, which reaches
    # `GRIPPER_SQUEEZE_M` past the closed stop so that a closed command keeps
    # squeezing; measurements are normalized against the physical travel.
    gripper_actuator = mj_model.actuator(
        f"{sim_constant.ATTACHMENT_PREFIX}{sim_constant.GRIPPER_ACTUATOR}"
    )
    low, high = gripper_actuator.ctrlrange
    self._gripper_ctrl_range = (float(low), float(high))
    self._gripper_travel = (0.0, float(joints[-1].range[1]))

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    # The MJCF joint range is the same frame as the real arm's limits; the
    # gripper is commanded on its normalized [0, 1] scale, as in the SDK.
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
    assert self._gripper_ctrl_range is not None
    low, high = self._gripper_ctrl_range
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
    assert self._gripper_travel is not None
    joints = simulator.data.qpos[:, self._qpos_ids]
    values = {
        constant.JOINT_POSITION_KEYS[joint]: joints[:, i]
        for i, joint in enumerate(constant.JOINTS)
    }
    low, high = self._gripper_travel
    gripper_key = constant.JOINT_POSITION_KEYS[constant.GRIPPER_JOINT]
    values[gripper_key] = torch.clamp(
        (values[gripper_key] - low) / (high - low) * constant.GRIPPER_MAX,
        constant.GRIPPER_MIN,
        constant.GRIPPER_MAX,
    )
    values[constant.FOLLOWER_COMMAND_TIMESTAMP] = torch.tensor(
        self._last_cmd_timestamp_ns, device=self._xpu_device, dtype=torch.int64
    ).expand(simulator.data.qpos.shape[0])
    return values
