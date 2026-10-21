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

"""Teleoperates the SO101 follower (real or simulated) from the leader arm.

  uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop         # real
  uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop --sim   # sim

The environment drives the follower arm and reads off of the joint positions
and camera. The leader arm is a `Policy` driving the environment through the
runloop.

At every episode start, both arms drive themselves to the recorded HOME pose:
the leader agent's `reset` torque-drives the leader there and releases, and
the environment reset homes the follower. Teleop is then a direct pass-through
in the shared calibrated frame.
"""

import sys
from typing import Any

from absl import app
from absl import flags
from absl import logging
from etils import epath
from gdm_robotics.runtime import runloop as runloop_lib
from mujoco_playground import real
from mujoco_playground.real import foot_pedal
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as so101_device
from mujoco_playground.tasks.teleop.so101 import env as teleop_env
from mujoco_playground.tasks.teleop.so101 import leader_agent
from mujoco_playground.tasks.teleop.so101 import runloop_operations
from reaf.common import external_signal_termination_checker

_SIM = flags.DEFINE_bool(
    "sim", False, "Drive the simulated follower instead of the real one."
)
_IMPL = flags.DEFINE_enum(
    "impl",
    "cpp",
    ["cpp", "warp"],
    "Simulation backend to step with. Only read when --sim is set.",
)
_EPISODE_LENGTH = flags.DEFINE_float(
    "episode_length", 100.0, "Episode length in seconds."
)
_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes", None, "Number of episodes to run (default: until Ctrl-C)."
)
_INSTRUCTION = flags.DEFINE_string(
    "instruction",
    None,
    "Task instruction logged with each episode (prompted if omitted).",
)
_TASK_ID = flags.DEFINE_string(
    "task_id", "so101_teleop", "Task id for the episode logs."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Where to write episode logs (default: local <robot>_teleop_logs).",
)
_RESET_LEADER = flags.DEFINE_bool(
    "reset_leader",
    True,
    "Whether to torque-drive the leader arm home before each episode. Set "
    "this to False (--noreset_leader) for a passive (unactuated) leader arm.",
)
_CAMERA_WIDTH = flags.DEFINE_integer(
    "camera_width",
    constant.CAMERA_WIDTH,
    "Capture width in pixels for the real wrist camera.",
)
_CAMERA_HEIGHT = flags.DEFINE_integer(
    "camera_height",
    constant.CAMERA_HEIGHT,
    "Capture height in pixels for the real wrist camera.",
)


def _instruction() -> str:
  """The instruction logged with every episode; prompted for if not given."""
  if _INSTRUCTION.value is not None:
    return _INSTRUCTION.value
  return input("Enter the task instruction for this session: ")


def _create_logger(
    env: Any,
    policy: leader_agent.SO101LeaderAgent,
    pedal_operations: runloop_operations.FootPedalOperation,
) -> Any:
  (_, policy_extra_spec), _ = policy.step_spec(env.timestep_spec())
  output_dir = (
      epath.Path(_OUTPUT_DIR.value)
      if _OUTPUT_DIR.value
      else teleop_env.default_log_dir()
  )
  return real.create_logger(
      timestep_spec=env.timestep_spec(),
      action_spec=env.action_spec(),
      policy_extra_spec=policy_extra_spec,
      image_observation_keys=[constant.WRIST_CAMERA],
      timestamp_key=constant.TIMESTAMP,
      task_id=_TASK_ID.value,
      agent_id=constant.ROBOT,
      output_directory=output_dir,
      control_timestep_seconds=teleop_env.CONTROL_PERIOD_SECONDS,
      outcome=lambda: pedal_operations.outcome,
  )


def _launch_viewer(sim: Any) -> Any:
  """Opens a passive viewer onto the simulated follower's own state."""
  import mujoco.viewer

  if sys.platform == "darwin" and mujoco.viewer._MJPYTHON is None:  # pylint: disable=protected-access
    # macOS requires the viewer to own the main thread.
    raise SystemExit(
        "On macOS the sim viewer must run under mjpython:\n\n  uv run --extra"
        f" cpu --extra so101 mjpython -m {__package__}.teleop --sim\n"
    )
  mj_model = sim.mj_model
  return mujoco.viewer.launch_passive(
      mj_model, mujoco.MjData(mj_model), show_left_ui=False, show_right_ui=False
  )


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.WARNING)

  # Build the environment.
  instruction = _instruction()
  termination = (
      external_signal_termination_checker.ExternalSignalTerminationChecker(
          name="foot_pedal"
      )
  )
  if _SIM.value:
    env = teleop_env.create_sim_env(
        sim_impl=_IMPL.value,  # pyrefly: ignore[bad-argument-type]
        episode_length_in_seconds=_EPISODE_LENGTH.value,
        termination_checker=termination,
        instruction=instruction,
    )
  else:
    env = teleop_env.create_real_env(
        episode_length_in_seconds=_EPISODE_LENGTH.value,
        termination_checker=termination,
        instruction=instruction,
        camera_config=so101_device.CameraConfig(
            width=_CAMERA_WIDTH.value,
            height=_CAMERA_HEIGHT.value,
        ),
    )

  # Build the policy.
  leader, leader_calibration = so101_device.connect_leader()
  action_spec: Any = env.action_spec()
  assert isinstance(action_spec, dict)
  policy = leader_agent.SO101LeaderAgent(
      leader, leader_calibration, action_spec
  )

  # Create other runloop operations.
  pedal = foot_pedal.KeyboardFootPedal()
  pedal_operations = runloop_operations.FootPedalOperation(pedal, termination)
  episode_logger = _create_logger(env, policy, pedal_operations)
  operations: list[runloop_lib.RunloopRuntimeOperations] = [pedal_operations]
  if _RESET_LEADER.value:
    operations = [runloop_operations.LeaderResetOperation(policy)] + operations

  viewer = None
  if _SIM.value:
    viewer = _launch_viewer(env.device_layer.sim)  # pyrefly: ignore[missing-attribute]
    operations.append(runloop_operations.RateLimiterOperation())
    operations.append(
        runloop_operations.ViewerOperation(
            viewer, env.device_layer.sim, termination  # pyrefly: ignore[missing-attribute]
        )
    )

  # Start the runloop.
  pedal.start()
  try:
    runloop_lib.Runloop(
        environment=env,
        policy=policy,
        loggers=[episode_logger],
        runloop_runtime_operations=operations,
    ).run(num_episodes=_NUM_EPISODES.value)
  except KeyboardInterrupt:
    pass
  finally:
    # Clean up.
    pedal.stop()
    if viewer is not None:
      print("\nClosing the viewer...")
      viewer.close()

    env.close()

    print("Finalizing episode logs...")
    episode_logger.stop()  # Blocks until all episode shards are finalized.

    leader.disconnect()

    print(
        "Done. Episode logs:"
        f" {_OUTPUT_DIR.value or teleop_env.default_log_dir()}"
    )


if __name__ == "__main__":
  app.run(main)
