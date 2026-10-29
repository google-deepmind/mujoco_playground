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
"""Loads YAM excitation logs (.npz, from `collect`) for system identification."""

import dataclasses
import glob
import os
from typing import Sequence

from etils import epath
import mujoco
from mujoco import sysid
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
import numpy as np

# The joints with position sensors, in the order of the measurements: the six
# arm joints, then the left finger (the right one follows it).
SENSOR_JOINTS = constant.ARM_JOINTS + (sim_constant.FINGER_JOINTS[0],)
# Samples within this distance of REST_POSE are dropped: there joints 2 and 3
# sit on their lower limits, which the sim and the real arm resolve
# differently.
REST_THRESHOLD = 0.05  # rad


@dataclasses.dataclass(frozen=True)
class Trajectory:
  """One recorded excitation episode, in MJCF units."""

  name: str
  timestamps: np.ndarray  # (T,) s, measurement times from episode start.
  ctrl_timestamps: np.ndarray  # (T,) s, when each command reached the SDK.
  measured_qpos: np.ndarray  # (T, 7): six arm joints [rad], left finger [m].
  commanded_ctrl: np.ndarray  # (T, 7): six arm joints [rad], gripper [m].
  measured_qvel: np.ndarray  # (T, 7)
  command_delay_ms: np.ndarray  # (T,)


def from_arrays(
    name: str,
    t_obs_ns: np.ndarray,
    t_cmd_post_ns: np.ndarray,
    q_meas: np.ndarray,
    q_cmd: np.ndarray,
    qvel_meas: np.ndarray | None = None,
    crop_rest: bool = True,
) -> Trajectory:
  """Converts arrays in the I2RT SDK's units to a Trajectory."""
  q_meas = np.asarray(q_meas, dtype=np.float64)
  q_cmd = np.asarray(q_cmd, dtype=np.float64)
  t_obs_ns = np.asarray(t_obs_ns, dtype=np.int64)
  t_cmd_post_ns = np.asarray(t_cmd_post_ns, dtype=np.int64)
  keep = slice(None)
  if crop_rest:
    away = np.flatnonzero(
        np.abs(q_cmd[:, : constant.NUM_ARM_JOINTS]).max(axis=1) > REST_THRESHOLD
    )
    if len(away) >= 2:
      keep = slice(away[0], away[-1] + 1)
  t_obs_ns, t_cmd_post_ns = t_obs_ns[keep], t_cmd_post_ns[keep]
  q_meas, q_cmd = q_meas[keep], q_cmd[keep]

  t0 = t_obs_ns[0]
  timestamps = (t_obs_ns - t0) * 1e-9
  meas = q_meas.copy()
  meas[:, -1] = [sim_constant.gripper_norm_to_m(g) for g in q_meas[:, -1]]
  ctrl = q_cmd.copy()
  ctrl[:, -1] = [sim_constant.gripper_norm_to_ctrl(g) for g in q_cmd[:, -1]]
  if qvel_meas is not None and np.all(np.isfinite(qvel_meas[keep])):
    qvel = np.asarray(qvel_meas, dtype=np.float64)[keep].copy()
    qvel[:, -1] *= sim_constant.GRIPPER_M_RANGE[1]
  else:
    qvel = np.gradient(meas, timestamps, axis=0)
  return Trajectory(
      name=name,
      timestamps=timestamps,
      ctrl_timestamps=(t_cmd_post_ns - t0) * 1e-9,
      measured_qpos=meas,
      commanded_ctrl=ctrl,
      measured_qvel=qvel,
      command_delay_ms=(t_cmd_post_ns - t_obs_ns) * 1e-6,
  )


def load_trajectory(path: epath.PathLike) -> Trajectory:
  path = epath.Path(path)
  with path.open("rb") as f:
    z = dict(np.load(f))
  return from_arrays(
      name=path.stem,
      t_obs_ns=z["t_obs_ns"],
      t_cmd_post_ns=z["t_cmd_post_ns"],
      q_meas=z["q_meas"],
      q_cmd=z["q_cmd"],
      qvel_meas=z.get("qvel_meas"),
  )


def find_episode_files(directory: epath.PathLike) -> list[epath.Path]:
  return sorted(
      epath.Path(p) for p in glob.glob(os.path.join(directory, "ep*.npz"))
  )


def add_sensors(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Adds a joint-position sensor per measured joint, in measurement order."""
  for joint in SENSOR_JOINTS:
    name = f"{joint}_pos"
    if not any(s.name == name for s in spec.sensors):
      sensor = spec.add_sensor(name=name)
      sensor.type = mujoco.mjtSensor.mjSENS_JOINTPOS
      sensor.objtype = mujoco.mjtObj.mjOBJ_JOINT
      sensor.objname = joint
  return spec


def to_model_sequences(
    trajectories: Sequence[Trajectory],
    spec: mujoco.MjSpec,
    name: str = "yam",
) -> sysid.ModelSequences:
  """Converts Trajectories into a mujoco.sysid.ModelSequences."""
  if not trajectories:
    raise ValueError("No trajectories provided.")
  spec = add_sensors(spec)
  model = spec.compile()
  initial_states, controls, sensordata = [], [], []
  for traj in trajectories:
    q, v = traj.measured_qpos[0], traj.measured_qvel[0]
    # The right finger mirrors the left one through an equality constraint.
    initial_states.append(
        sysid.create_initial_state(
            model, np.append(q, q[-1]), np.append(v, v[-1])
        )
    )
    controls.append(
        sysid.TimeSeries.from_control_names(
            traj.ctrl_timestamps, traj.commanded_ctrl, model
        )
    )
    sensordata.append(
        sysid.TimeSeries.from_names(traj.timestamps, traj.measured_qpos, model)
    )
  return sysid.ModelSequences(
      name=name,
      spec=spec,
      sequence_name=[t.name for t in trajectories],
      initial_state=initial_states,
      control=controls,
      sensordata=sensordata,
  )
