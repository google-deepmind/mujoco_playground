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

"""A leader agent for the SO101."""

from collections.abc import Mapping
import time
from typing import Any
from typing import Tuple

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as policy_interface
from gdm_robotics.interfaces import types as gdmr_types
from mujoco_playground._src.real.common import motion
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device
import numpy as np

_HOMING_DURATION_S = 1.5
_HOMING_RATE_HZ = 50.0


class SO101LeaderAgent(policy_interface.Policy):
  """Maps leader arm motion onto follower goal positions.

  `reset` will torque-drive the leader to `LEADER_HOME`, then torque is
  released and `step` passes the leader's pose through as the follower goal.
  """

  def __init__(
      self,
      leader: Any,
      motor_calibration: device.Calibration,
      action_spec: Mapping[str, specs.BoundedArray],
      real_time_homing: bool = True,
  ):
    self._leader = leader
    self._motor_calibration = motor_calibration
    self._action_spec = dict(action_spec)
    self._home = constant.LEADER_HOME
    self._real_time_homing = real_time_homing

  def reset(self) -> None:
    """Drives the leader to the HOME pose and waits for it to arrive."""
    self._leader_go_to(self._home, real_time=self._real_time_homing)

  def initial_state(self) -> tuple[()]:
    """Makes sure the torque is disabled so the leader can be moved by hand."""
    self._leader.bus.disable_torque(num_retry=5)
    return ()

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: Any,
  ) -> Tuple[
      Tuple[dict[str, Any], dict[str, Any]],
      Any,
  ]:
    del timestep  # Pass-through: the action depends only on the leader pose.
    t0_ns = time.time_ns()
    leader = device.read_joints(self._leader.bus, self._motor_calibration)
    action = {
        # `leader` holds Python floats, so the clip has to be cast back to the
        # spec's dtype: np.clip would otherwise widen it to float64.
        key: (
            self._action_spec[key].dtype.type(
                np.clip(
                    leader[joint],
                    self._action_spec[key].minimum,
                    self._action_spec[key].maximum,
                )
            )
        )
        for joint, key in constant.GOAL_POSITION_KEYS.items()
    }
    # The raw pose rides along as policy extra so logs keep what the operator
    # actually did, not only the clipped command.
    extra: dict[str, np.generic] = {
        key: np.float32(leader[joint])
        for joint, key in constant.LEADER_RAW_KEYS.items()
    }
    extra[constant.LEADER_TIMESTAMP] = np.int64(t0_ns)
    return (action, extra), prev_state

  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> Tuple[
      Tuple[dict[str, specs.BoundedArray], dict[str, specs.Array]],
      tuple[()],
  ]:
    """Returns the spec of the action, extra output, and state."""
    del timestep_spec  # The action spec is fixed by the follower.
    extra_spec = {
        key: specs.Array(shape=(), dtype=np.float32)
        for key in tuple(constant.LEADER_RAW_KEYS.values())
    }
    extra_spec[constant.LEADER_TIMESTAMP] = specs.Array(
        shape=(), dtype=np.int64
    )
    return ((self._action_spec, extra_spec), ())

  def _leader_go_to(self, pose: np.ndarray, real_time: bool) -> None:
    """Sends the leader to the given pose (calibrated frame) smoothly."""
    print("Enabling leader torque and driving to the home pose...")
    self._leader.bus.enable_torque(num_retry=5)
    start_joints = device.read_joints(self._leader.bus, self._motor_calibration)
    start_arr = np.array(
        [start_joints[j] for j in constant.JOINTS], dtype=np.float32
    )

    def _write_leader(p: np.ndarray) -> None:
      goals = {j: float(p[i]) for i, j in enumerate(constant.JOINTS)}
      device.write_goals(self._leader.bus, self._motor_calibration, goals)

    motion.cosine_interpolate(
        _write_leader,
        start_pose=start_arr,
        target_pose=pose,
        duration_secs=_HOMING_DURATION_S,
        control_timestep=1.0 / _HOMING_RATE_HZ,
        real_time=real_time,
    )
    print("Leader arm reached home.")
