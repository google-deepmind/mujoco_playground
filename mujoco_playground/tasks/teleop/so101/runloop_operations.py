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

"""Runloop operations for SO101 teleop."""

import time
from typing import Any

import dm_env
from gdm_robotics.runtime import runloop as runloop_lib
import mujoco
from mujoco_playground.real import foot_pedal
from mujoco_playground.tasks.teleop.so101 import env as teleop_env
from mujoco_playground.tasks.teleop.so101 import leader_agent
from reaf.common import time_trigger


class LeaderResetOperation(runloop_lib.RunloopRuntimeOperations):
  """Drives the leader to the HOME pose before each episode."""

  def __init__(self, policy: leader_agent.SO101LeaderAgent):
    self._policy = policy

  def before_episode_reset(self) -> bool:
    self._policy.reset()
    return True


class FootPedalOperation(runloop_lib.RunloopRuntimeOperations):
  """Foot-pedal episode protocol, on the keyboard shim.

  LEFT ('a') starts an episode

  During an episode:
  LEFT ends it as FAILURE
  MIDDLE ('s') as INVALID
  RIGHT ('d') as SUCCESS

  The outcome becomes the episode's log label.
  """

  def __init__(self, pedal: foot_pedal.KeyboardFootPedal, termination: Any):
    self._pedal = pedal
    self._termination = termination
    self._episode = 0
    self.outcome: str | None = None

  def before_episode_reset(self) -> bool:
    self._episode += 1
    print(
        f"\n=== Episode {self._episode} ===\n"
        "Press 'a' (left pedal) to START it. Then, during the episode:"
        " 'a' = failure, 's' = invalid, 'd' = success.\n"
        "(Ctrl-C to finish the session.)",
        flush=True,
    )
    while not self._pedal.state.left_pressed:
      time.sleep(0.02)
    print(f"Episode {self._episode} started.", flush=True)
    return True

  def inspect_timestep(self, timestep: dm_env.TimeStep) -> None:
    if timestep.first():
      self.outcome = None
    state = self._pedal.state
    if state.left_pressed:
      self.outcome = "failure"
    elif state.middle_pressed:
      self.outcome = "invalid"
    elif state.right_pressed:
      self.outcome = "success"
    else:
      return
    print(
        f"Ending episode {self._episode} as {self.outcome.upper()};"
        " writing it to the log...",
        flush=True,
    )
    self._termination.do_terminate()


class RateLimiterOperation(runloop_lib.RunloopRuntimeOperations):
  """Paces the loop to the control rate — simulation only."""

  def __init__(self):
    self._trigger = time_trigger.TimeTrigger(period=teleop_env.EPISODE_PERIOD)

  def inspect_timestep(self, timestep: dm_env.TimeStep) -> None:
    del timestep
    self._trigger.wait_for_event()


class ViewerOperation(runloop_lib.RunloopRuntimeOperations):
  """Shows the simulated scene while the runloop drives it.

  Closing the window ends the session; episode control stays with the foot
  pedal, exactly as on the real robot.
  """

  def __init__(self, viewer: Any, sim: Any, termination: Any):
    self._viewer = viewer
    self._sim = sim
    self._viewer_data = viewer.d if hasattr(viewer, "d") else None
    self._termination = termination

  def before_episode_reset(self) -> bool:
    return self._viewer.is_running()

  def inspect_timestep(self, timestep: dm_env.TimeStep) -> None:
    del timestep  # The scene is read from the simulation itself.
    if not self._viewer.is_running():
      self._termination.do_terminate()
      return
    with self._viewer.lock():
      self._viewer.d.qpos[:] = self._sim.data.qpos[0].cpu().numpy()
      mujoco.mj_forward(self._viewer.m, self._viewer.d)
    self._viewer.sync()
