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

"""Runs an agent on the tabletop pick-and-place tasks in sim or on a real arm.

The arm is chosen with `--robot` (see `embodiment.EMBODIMENTS`); tasks, the
agent harness and the bridge protocol are robot-agnostic.

Examples:
  # Oracle practice run in simulation.
  run --task=stack3 --agent=oracle --episodes=10
  # Same task on the i2rt YAM arm.
  run --robot=yam --task=stack3 --agent=oracle
  # Hand control to an external agent over HTTP (see ctl.py), with a viewer.
  run --task=stack3 --agent=bridge --viewer
  # Same agent, real robot.
  run --mode=real --agent=bridge --task_prompt="Stack the three cubes."
"""

import os
import sys
import time
from typing import Any

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/wp_kernel_cache")

# pylint: disable=g-import-not-at-top
from absl import app
from absl import flags
from etils import epath
import mujoco
from mujoco_playground.tasks.agentic.tabletop_v0 import agent
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment
from mujoco_playground.tasks.agentic.tabletop_v0 import env as agentic_env
from mujoco_playground.tasks.agentic.tabletop_v0 import task as agentic_task
import numpy as np
from reaf.simulation.torch import flat_scene as build

# pylint: enable=g-import-not-at-top

_MODE = flags.DEFINE_enum("mode", "sim", ["sim", "real"], "Sim or real arm.")
_ROBOT = flags.DEFINE_enum(
    "robot", "so101", sorted(embodiment.EMBODIMENTS), "Arm embodiment."
)
_AGENT = flags.DEFINE_enum(
    "agent", "oracle", ["oracle", "bridge", "cli"], "Where plans come from."
)
_TASK = flags.DEFINE_enum(
    "task", "cube_in_bowl", sorted(agentic_task.TASKS), "Sim task to load."
)
_TASK_PROMPT = flags.DEFINE_string(
    "task_prompt", "", "Instruction for the agent (defaults to the task's)."
)
_EPISODES = flags.DEFINE_integer("episodes", 1, "Number of episodes.")
_MAX_TURNS = flags.DEFINE_integer(
    "max_turns",
    0,
    "Maximum model turns per episode (0 uses backend default: unlimited for"
    " bridge, 20 for cli, 3 for oracle).",
)
_EPISODE_LENGTH_SECS = flags.DEFINE_float(
    "episode_length_secs", 1800.0, "Maximum active episode duration in seconds."
)
_VIEWER = flags.DEFINE_bool("viewer", False, "Show the native MuJoCo viewer.")
_HTTP_PORT = flags.DEFINE_integer("http_port", 8081, "Bridge HTTP port.")
_BRIDGE_DIR = flags.DEFINE_string(
    "bridge_dir", "", "Bridge directory for obs.json and wrist_cam.png."
)
_CLI_CMD = flags.DEFINE_string(
    "cli_cmd", "gemini -p", "Command for --agent=cli."
)
_VIDEO = flags.DEFINE_string(
    "video", "", "Optional output path (.mp4 or .gif) to record the rollout."
)
_SPEED_SCALE = flags.DEFINE_float(
    "speed_scale",
    0.0,
    "Motion speed multiplier (0 uses mode default: 0.35 for real, 1.0 for"
    " sim).",
)

_PROMPT_FILE = epath.Path(__file__).parent / "AGENT_PROMPT.md"


class Viewer:
  """Native passive viewer mirroring the sim state or the measured real arm."""

  def __init__(self, env, is_real: bool, emb: embodiment.Embodiment):
    import mujoco.viewer  # pylint: disable=g-import-not-at-top

    self._is_real = is_real
    self._env = env
    self._emb = emb
    if is_real:
      cfg = agentic_task.env_config(robot=emb.name)
      deploy_cfg = build.make_deployment_config(
          cfg, observation_groups=tuple(cfg.observation_adapter)
      )
      self._model = build.build_scene(deploy_cfg).compile()
      self._data = mujoco.MjData(self._model)
      self._qpos_adr = np.array([
          self._model.joint(f"{embodiment.ATTACHMENT_PREFIX}{j}").qposadr[0]
          for j in emb.mj_joints
      ])
    else:
      sim = env.device_layer.sim
      self._model, self._data = sim.mj_model, sim.mj_data
    self._viewer = mujoco.viewer.launch_passive(
        self._model, self._data, show_left_ui=False, show_right_ui=False
    )
    lookat, distance, azimuth, elevation = emb.overview_camera
    self._viewer.cam.lookat[:] = lookat
    self._viewer.cam.distance = distance * 1.6
    self._viewer.cam.azimuth = azimuth
    self._viewer.cam.elevation = elevation

  @property
  def running(self) -> bool:
    return self._viewer.is_running()

  def sync(self, obs) -> None:
    if self._is_real:
      self._data.qpos[self._qpos_adr] = self._emb.follower_to_mj_qpos(
          obs["joint_pos"][0].cpu().numpy()
      )
    else:
      self._data.qpos[:] = self._env.device_layer.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)
    self._viewer.sync()

  def close(self) -> None:
    self._viewer.close()


def _reward(ts) -> float:
  if ts.reward is None:
    return 0.0
  val = ts.reward["reward"] if isinstance(ts.reward, dict) else ts.reward
  return float(np.asarray(val).reshape(-1)[0])


def main(argv):
  del argv
  is_real = _MODE.value == "real"
  task: agentic_task.Task | None = (
      None if is_real else agentic_task.TASKS[_TASK.value]
  )
  if _TASK_PROMPT.value:
    prompt = _TASK_PROMPT.value
  elif not is_real and task is not None:
    prompt = task.prompt
  else:
    prompt = "Manipulate tabletop objects as requested."
  bridge_dir = _BRIDGE_DIR.value or agent.default_bridge_dir()
  emb = embodiment.get(_ROBOT.value)

  if is_real:
    env = agentic_env.create_real_env(
        robot=emb.name, episode_length_secs=_EPISODE_LENGTH_SECS.value
    )
  else:
    assert task is not None
    env = agentic_env.create_sim_env(
        task=task.name,
        robot=emb.name,
        episode_length_secs=_EPISODE_LENGTH_SECS.value,
    )

  max_turns = _MAX_TURNS.value
  backend: agent.OracleBackend | agent.BridgeBackend | agent.CLIBackend
  if _AGENT.value == "oracle":
    if is_real:
      raise ValueError(
          "--agent=oracle needs simulator state; use bridge or cli."
      )
    assert task is not None
    backend = agent.OracleBackend(env, task, max_turns=max_turns or 3)
  elif _AGENT.value == "bridge":
    backend = agent.BridgeBackend(
        prompt, bridge_dir, _HTTP_PORT.value, emb, max_turns=max_turns
    )
    print(f"Bridge: http://127.0.0.1:{backend.http_port}  files: {bridge_dir}")
  else:
    backend = agent.CLIBackend(
        prompt,
        bridge_dir,
        _CLI_CMD.value,
        str(_PROMPT_FILE),
        emb,
        max_turns=max_turns or 20,
    )

  speed_scale = (
      _SPEED_SCALE.value
      if _SPEED_SCALE.value > 0
      else (0.35 if is_real else 1.0)
  )
  policy = agent.AgentPolicy(
      backend,
      emb,
      dt=env.device_layer.control_timestep,
      speed_scale=speed_scale,
  )
  viewer = Viewer(env, is_real, emb) if _VIEWER.value else None
  recorder = (
      agent.VideoRecorder(env, _VIDEO.value, emb, task_name=task.name)
      if _VIDEO.value and not is_real and task is not None
      else None
  )
  pace = viewer is not None and not is_real
  successes = 0
  try:
    for ep in range(1, _EPISODES.value + 1):
      ts = env.reset()
      policy.reset(ts.observation)
      success, steps, idle_steps = False, 0, 0
      if recorder is not None:
        recorder.record(ts.observation, policy, steps, success)
      is_bridge = (
          isinstance(backend, agent.BridgeBackend)
          and not isinstance(backend, agent.CLIBackend)
          and backend.http_port > 0
      )
      while not ts.last() and (viewer is None or viewer.running):
        if is_bridge:
          if backend.consume_reset_request():  # pyrefly: ignore[missing-attribute]
            break
          if policy.idle and not backend.has_pending_plan():  # pyrefly: ignore[missing-attribute]
            if backend.exhausted or (not is_bridge and success):
              break
            if not is_real:
              if viewer is not None:
                viewer.sync(ts.observation)
              time.sleep(0.02)
              continue
        t0 = time.perf_counter()
        action = policy.step(ts.observation, success)
        ts = env.step({"action": action})
        steps += 1
        success = (
            _reward(ts) >= 1.0 if is_bridge else (success or _reward(ts) >= 1.0)
        )
        idle_steps = (
            idle_steps + 1 if (policy.idle and backend.exhausted) else 0
        )
        if (
            not is_bridge and not policy.executor.busy and success
        ) or idle_steps > 100:
          break
        # Full rate while a plan executes, ~1 fps while waiting on the agent.
        stride = 2 if policy.executor.busy else 50
        if recorder is not None and (steps % stride == 0 or ts.last()):
          recorder.record(ts.observation, policy, steps, success)
        if viewer is not None:
          viewer.sync(ts.observation)
        if pace:
          time.sleep(max(0.0, policy.executor.dt - (time.perf_counter() - t0)))
      policy.finish(ts.observation, success)
      successes += success
      print(
          f"episode {ep}: success={success} turns={backend.turn}"
          f" steps={steps} ({successes}/{ep})",
          flush=True,
      )
      if viewer is not None and not viewer.running:
        break
  finally:
    if recorder is not None:
      recorder.close()
    if isinstance(backend, agent.BridgeBackend):
      backend.close()
    if viewer is not None:
      viewer.close()
    env.close()


if __name__ == "__main__":
  app.run(main)
