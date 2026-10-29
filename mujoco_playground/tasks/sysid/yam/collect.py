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
"""Plays the excitation trajectories on a real YAM and logs them at 50 Hz.

Usage:
  uv run --no-sync --extra yam python -m
  mujoco_playground.tasks.sysid.yam.collect

Every episode is first replayed in simulation (`preflight`); the run aborts
before touching the CAN bus if any fails. On the arm, the I2RT SDK runs its
250 Hz PD with gravity compensation (`get_yam_robot(...,
zero_gravity_mode=False)`), as the rest of Playground's YAM driver does.
"""

from collections.abc import Sequence
import time

from absl import app
from absl import flags
from absl import logging
from etils import epath
from mujoco_playground import paths
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.real import device
from mujoco_playground.tasks.sysid.yam import excitation
from mujoco_playground.tasks.sysid.yam import preflight
import numpy as np

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Where to write ep0*.npz (default: <local dir>/yam_sysid_logs).",
)
_EPISODES = flags.DEFINE_list(
    "episodes",
    list(excitation.EPISODES),
    f"Episodes to run, from {list(excitation.EPISODES)}.",
)
_MAX_TRACKING_ERROR = flags.DEFINE_float(
    "max_tracking_error",
    0.65,
    "Abort if an arm joint lags its command by more than this (rad).",
)
_MAX_INITIAL_OFFSET = flags.DEFINE_float(
    "max_initial_offset",
    0.35,
    "Refuse to start if the arm is further than this from REST_POSE (rad).",
)
_SLOW_SPEED = 0.2  # rad/s for moves outside the excitation.
_PERIOD_NS = int(round(excitation.DT * 1e9))


def _pace(t_next: int) -> int:
  t_next += _PERIOD_NS
  sleep = (t_next - time.perf_counter_ns()) * 1e-9
  if sleep > 0:
    time.sleep(sleep)
  return t_next


def _read(robot: device.MotorChainRobot) -> np.ndarray:
  q = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
  if q.shape != (len(constant.JOINTS),) or not np.all(np.isfinite(q)):
    raise RuntimeError(f"Bad joint reading: {q}")
  return q


def _move_slowly(robot: device.MotorChainRobot, target: np.ndarray) -> None:
  start = _read(robot)
  n = max(int(np.ceil(np.abs(target - start).max() / _SLOW_SPEED / 0.02)), 1)
  t_next = time.perf_counter_ns()
  for i in range(1, n + 26):  # Plus a 0.5 s hold at the target.
    robot.command_joint_pos(start + (target - start) * min(i / n, 1.0))
    t_next = _pace(t_next)


def _run_episode(
    robot: device.MotorChainRobot, q_cmd: np.ndarray
) -> dict[str, np.ndarray]:
  """50 Hz loop logging measurement and command-receipt timestamps."""
  n = len(q_cmd)
  log = {
      "t_obs_ns": np.zeros(n, np.int64),
      "t_cmd_post_ns": np.zeros(n, np.int64),
      "q_meas": np.zeros((n, len(constant.JOINTS))),
      "qvel_meas": np.zeros((n, len(constant.JOINTS))),
      "q_cmd": q_cmd,
  }
  arm = slice(0, constant.NUM_ARM_JOINTS)
  t_next = time.perf_counter_ns()
  for k in range(n):
    log["t_obs_ns"][k] = time.time_ns()
    q = log["q_meas"][k] = _read(robot)
    obs = robot.get_observations()
    log["qvel_meas"][k] = np.concatenate([obs["joint_vel"], obs["gripper_vel"]])
    if np.abs(q_cmd[k, arm] - q[arm]).max() > _MAX_TRACKING_ERROR.value:
      robot.command_joint_pos(q)
      raise RuntimeError(f"Tracking error over the limit at step {k}.")
    robot.command_joint_pos(q_cmd[k].copy())
    log["t_cmd_post_ns"][k] = time.time_ns()
    t_next = _pace(t_next)
  return log


def main(argv: Sequence[str]) -> None:
  del argv
  episodes = {
      e.name: e
      for e in excitation.all_episodes()
      if e.name in {excitation.EPISODES[k] for k in _EPISODES.value}
  }
  for episode in episodes.values():
    result = preflight.check_episode(episode)
    logging.info("Pre-flight %s", result)
    if not result.ok:
      raise RuntimeError(f"Pre-flight failed, not running: {result}")

  out = (
      epath.Path(_OUTPUT_DIR.value)
      if _OUTPUT_DIR.value
      else paths.local_dir() / "yam_sysid_logs"
  )
  out.mkdir(parents=True, exist_ok=True)

  robot = device.connect_robot()
  follower = device.YamFollowerDevice(robot)
  moved = False
  try:
    q_init = _read(robot)
    offset = np.abs(q_init[: constant.NUM_ARM_JOINTS]).max()
    if offset > _MAX_INITIAL_OFFSET.value:
      raise RuntimeError(
          f"The arm is {offset:.2f} rad from REST_POSE; move it closer first."
      )
    moved = True
    _move_slowly(robot, excitation.REST_POSE)
    for name, episode in episodes.items():
      logging.info("Running %s (%.1f s)", name, len(episode.q_cmd) * 0.02)
      log = _run_episode(robot, episode.q_cmd)
      with (out / f"{name}.npz").open("wb") as f:
        np.savez(
            f,
            t_obs_ns=log["t_obs_ns"],
            t_cmd_post_ns=log["t_cmd_post_ns"],
            q_meas=log["q_meas"],
            qvel_meas=log["qvel_meas"],
            q_cmd=log["q_cmd"],
        )
      delay = (log["t_cmd_post_ns"] - log["t_obs_ns"]) * 1e-6
      logging.info(
          "Saved %s: command delay %.2f +/- %.2f ms",
          name,
          delay.mean(),
          delay.std(),
      )
  finally:
    if moved:
      _move_slowly(robot, excitation.REST_POSE)
    follower.close()


if __name__ == "__main__":
  app.run(main)
