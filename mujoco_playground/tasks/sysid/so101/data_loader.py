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
"""MCAP trajectory dataset loader for SO101 system identification."""

import collections
import dataclasses
import glob
import os
from typing import Sequence

from etils import epath
import mujoco
from mujoco import sysid
from mujoco_playground._src.real.common import log_reader
from mujoco_playground.robots.so101.real import constant
import numpy as np


@dataclasses.dataclass(frozen=True)
class Trajectory:
  """A single recorded teleoperation episode trajectory."""

  path: epath.Path
  timestamps: np.ndarray  # (T,), seconds from episode start for measurements
  ctrl_timestamps: (
      np.ndarray
  )  # (T,), seconds from episode start for follower command receipt
  dt: float  # control period in seconds
  measured_qpos: np.ndarray  # (T, 6), joint angles in radians
  commanded_ctrl: np.ndarray  # (T, 6), goal joint angles in radians
  measured_qvel: np.ndarray  # (T, 6), joint velocities in rad/s
  leader_timestamps_ns: np.ndarray  # (T,) int64 t0
  follower_timestamps_ns: np.ndarray  # (T,) int64 t1


def _get_default_gripper_range() -> tuple[float, float]:
  """Returns the (min, max) range in radians for the SO101 gripper in MuJoCo."""
  from mujoco_playground.robots.so101.sim import scene

  spec = scene.so101()
  model = spec.compile()
  gripper_id = model.joint("gripper").id
  low, high = model.jnt_range[gripper_id]
  return float(low), float(high)


def load_trajectory(path: epath.Path) -> Trajectory:
  """Loads and synchronizes an episode MCAP log into a Trajectory."""
  grip_low, grip_high = _get_default_gripper_range()

  timestamp_key = f"observation/{constant.TIMESTAMP}"
  follower_cmd_ts_key = f"observation/{constant.FOLLOWER_COMMAND_TIMESTAMP}"
  leader_ts_key = f"extra/policy_extra/{constant.LEADER_TIMESTAMP}"

  steps = collections.Counter()
  timestamps_ns: dict[int, int] = {}
  leader_ts_ns: dict[int, int] = {}
  follower_ts_ns: dict[int, int] = {}
  measured: dict[int, dict[str, float]] = collections.defaultdict(dict)
  commanded: dict[int, dict[str, float]] = collections.defaultdict(dict)

  for topic, features in log_reader.iter_features(path):
    step = steps[topic]
    steps[topic] += 1

    if topic == "/timestep":
      timestamps_ns[step] = features[timestamp_key].int64_list.value[0]
      follower_ts_ns[step] = features[follower_cmd_ts_key].int64_list.value[0]

      for joint in constant.JOINTS:
        key = f"observation/{constant.JOINT_POSITION_KEYS[joint]}"
        measured[step][joint] = features[key].float_list.value[0]

    elif topic == "/action":
      for joint in constant.JOINTS:
        key = f"action/{constant.GOAL_POSITION_KEYS[joint]}"
        commanded[step][joint] = features[key].float_list.value[0]

    elif topic == "/policy_extra":
      leader_ts_ns[step] = features[leader_ts_key].int64_list.value[0]

  n_joints = len(constant.JOINTS)
  if not (len(timestamps_ns) == len(commanded) == len(leader_ts_ns)):
    raise ValueError(
        f"Mismatched message counts in {path}: {len(timestamps_ns)} timesteps,"
        f" {len(commanded)} actions, {len(leader_ts_ns)} policy extras."
    )
  if len(commanded) < 2:
    raise ValueError(f"Trajectory {path} has fewer than 2 steps.")

  # In an MDP step tuple (obs_k, action_k, extra_k), action_k is sent to the
  # robot during env.step(action_k), and its execution timestamp is recorded
  # in obs_{k+1} (step k+1).
  n_steps = len(commanded) - 1

  t0_ns = timestamps_ns[0]
  timestamps = np.array(
      [(timestamps_ns[s] - t0_ns) * 1e-9 for s in range(n_steps)],
      dtype=np.float64,
  )
  dt = float(np.mean(np.diff(timestamps)))

  ctrl_timestamps = np.array(
      [(follower_ts_ns[s + 1] - t0_ns) * 1e-9 for s in range(n_steps)],
      dtype=np.float64,
  )

  leader_ts = np.array(
      [leader_ts_ns[s] for s in range(n_steps)],
      dtype=np.int64,
  )
  follower_ts = np.array(
      [follower_ts_ns[s + 1] for s in range(n_steps)],
      dtype=np.int64,
  )

  qpos = np.zeros((n_steps, n_joints), dtype=np.float64)
  ctrl = np.zeros((n_steps, n_joints), dtype=np.float64)
  for i in range(n_steps):
    for j, joint in enumerate(constant.JOINTS):
      q_val = measured[i][joint]
      c_val = commanded[i][joint]

      if joint == constant.GRIPPER_JOINT:
        q_val = constant.gripper_pct_to_rad(q_val, grip_low, grip_high)
        c_val = constant.gripper_pct_to_rad(c_val, grip_low, grip_high)

      qpos[i, j] = q_val
      ctrl[i, j] = c_val

  qvel = np.asarray(np.gradient(qpos, timestamps, axis=0), dtype=np.float64)

  return Trajectory(
      path=path,
      timestamps=timestamps,
      ctrl_timestamps=ctrl_timestamps,
      dt=dt,
      measured_qpos=qpos,
      commanded_ctrl=ctrl,
      measured_qvel=qvel,
      leader_timestamps_ns=leader_ts,
      follower_timestamps_ns=follower_ts,
  )


def find_mcap_files(
    directory_or_pattern: str | epath.PathLike,
) -> list[epath.Path]:
  """Finds all matching MCAP log files sorted by modification time."""
  path_str = str(directory_or_pattern)
  if os.path.isdir(path_str):
    matched = glob.glob(os.path.join(path_str, "**/*.mcap"), recursive=True)
    files = [epath.Path(p) for p in matched]
  else:
    matched = glob.glob(path_str, recursive=True)
    files = [epath.Path(p) for p in matched if p.endswith(".mcap")]
  return sorted(files, key=lambda p: p.stat().mtime)


def load_trajectories(
    files: Sequence[epath.PathLike],
) -> list[Trajectory]:
  """Loads multiple MCAP trajectories."""
  return [load_trajectory(epath.Path(path)) for path in files]


def to_model_sequences(
    trajectories: Sequence[Trajectory],
    spec: mujoco.MjSpec,
    name: str = "so101",
) -> sysid.ModelSequences:
  """Converts loaded Trajectories into a mujoco.sysid.ModelSequences object."""
  if not trajectories:
    raise ValueError("Cannot create ModelSequences: no trajectories provided.")

  for joint in constant.JOINTS:
    sensor_name = f"{joint}_pos"
    if not any(s.name == sensor_name for s in spec.sensors):
      sensor = spec.add_sensor(name=sensor_name)
      sensor.type = mujoco.mjtSensor.mjSENS_JOINTPOS
      sensor.objtype = mujoco.mjtObj.mjOBJ_JOINT
      sensor.objname = joint

  model = spec.compile()
  sequence_names = []
  initial_states = []
  controls = []
  sensordata = []

  for traj in trajectories:
    sequence_names.append(epath.Path(traj.path).stem)
    init_state = sysid.create_initial_state(
        model, traj.measured_qpos[0], traj.measured_qvel[0]
    )
    ctrl_ts = sysid.TimeSeries.from_control_names(
        traj.ctrl_timestamps, traj.commanded_ctrl, model
    )
    meas_ts = sysid.TimeSeries.from_names(
        traj.timestamps, traj.measured_qpos, model
    )
    initial_states.append(init_state)
    controls.append(ctrl_ts)
    sensordata.append(meas_ts)

  return sysid.ModelSequences(
      name=name,
      spec=spec,
      sequence_name=sequence_names,
      initial_state=initial_states,
      control=controls,
      sensordata=sensordata,
  )
