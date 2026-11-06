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

"""Smooth Cartesian IK controller with workspace safety clamping.

Robot-agnostic: the arm is described by an `embodiment.Embodiment`, and the
IK is a task-priority damped least-squares solver on the pinch site. Position
is the primary task; the tool orientation is tracked in the position
nullspace, so a 5-DoF arm (SO101) gets exact position and best-effort
orientation while a 6-DoF arm (YAM) gets the full pose.
"""

from collections.abc import Mapping
from typing import Any
from typing import Union

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_producer
import torch

# Safety cap on the Cartesian speed of the smoothed IK target (m/s). The plan
# executor in agent.py generates smooth trajectories well under this cap, so
# it only engages on discontinuous commands (e.g. a raw jump after reset).
DEFAULT_MAX_LINEAR_SPEED = 0.5
# Safety cap on wrist yaw and pitch rate (rad per control step).
MAX_ANGLE_STEP = 0.12

PITCH_DOWN = embodiment_lib.PITCH_DOWN


def clamp_workspace(
    xyz: np.ndarray, emb: embodiment_lib.Embodiment
) -> np.ndarray:
  """Clamps a 3D Cartesian point (or batch) to the embodiment's safe box."""
  return np.clip(xyz, emb.workspace_min, emb.workspace_max)


class PinchIK:
  """Task-priority DLS IK for the pinch site of a standalone robot model."""

  def __init__(
      self,
      emb: embodiment_lib.Embodiment,
      *,
      damping: float = 2e-2,
      rot_damping: float = 0.1,
      rot_gain: float = 0.5,
      posture_gain: float = 0.05,
      max_iters: int = 20,
  ):
    self.emb = emb
    self.model = emb.spec_fn().compile()
    self.data = mujoco.MjData(self.model)
    self._site = self.model.site(emb.pinch_site).id
    arm = [self.model.joint(j) for j in emb.arm_joints]
    self.qadr = np.array([j.qposadr[0] for j in arm])
    self._dofadr = np.array([j.dofadr[0] for j in arm])
    self.jnt_min = np.array([j.range[0] for j in arm])
    self.jnt_max = np.array([j.range[1] for j in arm])
    self.home = np.asarray(emb.home_qpos[: emb.n_arm], dtype=np.float64)
    self._damping = damping
    self._rot_damping = rot_damping
    self._rot_gain = rot_gain
    self._posture_gain = posture_gain
    self._max_iters = max_iters

  def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the pinch site (position, rotation matrix) at `q_arm`."""
    self.data.qpos[self.qadr] = q_arm
    mujoco.mj_kinematics(self.model, self.data)
    return (
        self.data.site_xpos[self._site].copy(),
        self.data.site_xmat[self._site].reshape(3, 3).copy(),
    )

  def _step(
      self,
      j_pos: np.ndarray,
      j_rot: np.ndarray,
      pos_err: np.ndarray,
      rot_err: np.ndarray,
      q: np.ndarray,
      free: np.ndarray,
      secondary: bool = True,
  ) -> np.ndarray:
    """One task-priority DLS step using only the `free` joints."""
    eye = np.diag(free.astype(np.float64))
    j_pos, j_rot = j_pos * free, j_rot * free
    # Primary task: pinch position.
    jp_pinv = j_pos.T @ np.linalg.inv(
        j_pos @ j_pos.T + self._damping**2 * np.eye(3)
    )
    dq = jp_pinv @ pos_err
    if not secondary:
      return dq * free
    null_pos = eye - jp_pinv @ j_pos
    # Secondary task: tool orientation within the position nullspace.
    jr_null = j_rot @ null_pos
    jr_pinv = jr_null.T @ np.linalg.inv(
        jr_null @ jr_null.T + self._rot_damping**2 * np.eye(3)
    )
    dq += jr_pinv @ (self._rot_gain * rot_err - j_rot @ dq)
    # Tertiary: drift toward the home posture in what is left.
    null_rot = null_pos - jr_pinv @ jr_null
    dq += null_rot @ (self._posture_gain * (self.home - q))
    return dq * free

  def _active_set_step(
      self, j_pos, j_rot, pos_err, rot_err, q, secondary
  ) -> np.ndarray:
    """Drops joints pinned at a limit that the step pushes further."""
    free = np.ones(len(q), dtype=bool)
    dq = np.zeros_like(q)
    for _ in range(len(q)):
      dq = self._step(j_pos, j_rot, pos_err, rot_err, q, free, secondary)
      pinned = free & (
          ((q <= self.jnt_min + 1e-6) & (dq < 0))
          | ((q >= self.jnt_max - 1e-6) & (dq > 0))
      )
      if not pinned.any():
        break
      free &= ~pinned
    # Limit maximum per-iteration joint step for numerical stability.
    dq_norm = np.linalg.norm(dq)
    if dq_norm > 0.25:
      dq *= 0.25 / dq_norm
    return dq

  def solve(
      self, q_init: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray
  ) -> np.ndarray:
    """Solves for arm joints placing the pinch site at the target pose.

    Joints pinned at a limit that the step would push further are removed
    from the Jacobians (active set), so the remaining joints still solve the
    position task exactly and only the orientation degrades. Near
    singularities or for unreachable targets, the damped nullspace is not
    exact; a step that would increase the position error falls back to a
    position-only step (then to smaller steps), so the arm never wanders away
    from the closest reachable point.
    """
    q = np.clip(q_init.astype(np.float64), self.jnt_min, self.jnt_max)
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    pos, rot = self.fk(q)
    for _ in range(self._max_iters):
      pos_err = target_pos - pos
      err = float(np.linalg.norm(pos_err))
      rot_err = 0.5 * sum(
          np.cross(rot[:, i], target_rot[:, i]) for i in range(3)
      )
      if err < 2e-4 and np.linalg.norm(rot_err) < 5e-3:
        break
      mujoco.mj_comPos(self.model, self.data)
      mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site)
      j_pos, j_rot = jacp[:, self._dofadr], jacr[:, self._dofadr]

      tol = max(1e-4, 0.05 * err)
      accepted = False
      for secondary, scale in (
          (True, 1.0),
          (False, 1.0),
          (False, 0.5),
          (False, 0.25),
      ):
        dq = self._active_set_step(j_pos, j_rot, pos_err, rot_err, q, secondary)
        q_new = np.clip(q + scale * dq, self.jnt_min, self.jnt_max)
        pos_new, rot_new = self.fk(q_new)
        if np.linalg.norm(target_pos - pos_new) <= err + tol:
          q, pos, rot, accepted = q_new, pos_new, rot_new, True
          break
        self.fk(q)  # Restore kinematics at q for the next Jacobian.
      if not accepted:
        break
    return q


class SmoothCartesianIKProcessor(
    commands_processor.CommandsProcessor,
    features_producer.FeaturesProducer,
):
  """Translates a Cartesian pinch-point target into safe joint targets.

  Consumes:
    "action": Tensor of shape (nworld, 6) or (6,) containing:
      - x, y, z: Desired pinch-center position in world coordinates (meters).
      - gripper: Normalized gripper command in [0.0 (close), 1.0 (open)].
      - yaw: World yaw of the jaw closing axis (radians, modulo pi).
      - pitch: Tilt of the approach axis below horizontal (radians);
        `PITCH_DOWN` points the fingers straight at the table.

  Produces commands:
    "follower/goal_position/<joint>": Tensor of shape (nworld,) per joint.

  Features produced:
    "robot_state/joint_pos": Measured follower joint positions (nworld, n_arm +
    1).
    "ee/xpos": Pinch-center world position from FK (nworld, 3).
    "ee/xmat": Pinch-center world rotation matrix from FK (nworld, 9).
    "target_ee_pos": Current rate-limited Cartesian target (nworld, 3).
    "raw_action": Last commanded action (nworld, 7).

  The same processor runs in simulation and on the real robot, so the plan
  executor in agent.py is the only place that shapes motion; this class only
  enforces workspace and rate safety limits.
  """

  def __init__(
      self,
      name: str = "smooth_cartesian_ik",
      *,
      robot: str = "so101",
      nworld: int = 1,
      xpu_device: str = "cpu",
      control_timestep: float = 0.02,
      max_linear_speed: float = DEFAULT_MAX_LINEAR_SPEED,
  ):
    self._name = name
    self._emb = embodiment_lib.get(robot)
    self._nworld = nworld
    self._xpu_device = torch.device(xpu_device)
    self._dt = control_timestep
    self._max_step = max_linear_speed * control_timestep
    self._ik = PinchIK(self._emb)
    self._n_arm = self._emb.n_arm
    self._qpos_key = embodiment_lib.QPOS_KEY

    home_q = np.asarray(self._emb.home_qpos, dtype=np.float64)
    self._home_qpos = home_q
    hover_pos, hover_rot = self._ik.fk(home_q[: self._n_arm])
    self._hover_pose = self._pose_of(hover_pos, hover_rot)

    # Per-world state buffers. `_pose` is the rate-limited IK target
    # [x, y, z, relative yaw, pitch, azimuth_off].
    self._current_qpos = np.tile(home_q, (nworld, 1))
    self._measured_qpos = np.tile(home_q, (nworld, 1))
    self._filtered_q_err = np.zeros((nworld, self._n_arm), dtype=np.float64)
    self._pose = np.tile(self._hover_pose, (nworld, 1))
    self._last_raw_action = np.zeros((nworld, 7))
    self._needs_sync = np.ones(nworld, dtype=bool)
    for i in range(nworld):
      self._sync_raw_action(i)

  @property
  def name(self) -> str:
    return self._name

  @property
  def embodiment(self) -> embodiment_lib.Embodiment:
    return self._emb

  def _pose_of(self, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    yaw, pitch, azimuth = self._emb.orientation(pos, rot)
    rel = self._emb.relative_yaw(yaw, pos[0], pos[1], azimuth)
    az_off = embodiment_lib.wrap_pi(azimuth - self._emb.heading(pos[0], pos[1]))
    return np.array([*pos, rel, pitch, az_off])

  def _sync_raw_action(self, i: int) -> None:
    x, y, z, rel, pitch, az_off = self._pose[i]
    azimuth = self._emb.heading(x, y) + az_off
    yaw = embodiment_lib.fold(azimuth + rel)
    self._last_raw_action[i] = [x, y, z, 1.0, yaw, pitch, az_off]

  def solve_ik_single(
      self,
      q_init: np.ndarray,
      target_pos: np.ndarray,
      target_yaw: float,
      target_pitch: float = PITCH_DOWN,
      target_azimuth: float | None = None,
  ) -> np.ndarray:
    """Solves IK for one arm pose: pinch at `target_pos`, given orientation."""
    rot = self._emb.tool_rotation(
        target_pos[0], target_pos[1], target_yaw, target_pitch, target_azimuth
    )
    return self._ik.solve(q_init[: self._n_arm], target_pos, rot)

  # --- CommandsProcessor ---

  def consumed_commands_spec(self) -> Mapping[str, Any]:
    return {"action": specs.Array(shape=(7,), dtype=np.float32)}

  def produced_commands_keys(self) -> set[str]:
    return set(self._emb.goal_position_keys)

  def process_commands(  # pyrefly: ignore[bad-override]
      self, consumed_commands: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Clamps workspace, rate-limits the target, and solves joint targets."""
    raw_np = consumed_commands["action"].detach().cpu().numpy()
    if raw_np.shape[-1] == 6:
      raw_np = np.pad(raw_np.reshape(-1, 6), ((0, 0), (0, 1)))
    else:
      raw_np = raw_np.reshape(-1, 7)
    emb, n = self._emb, self._n_arm
    ctrl_out = np.zeros((self._nworld, n + 1), dtype=np.float32)

    for i in range(self._nworld):
      cmd = raw_np[i].astype(np.float64)
      self._last_raw_action[i] = cmd

      # 1. Clamp target XYZ to the safe workspace bounding box.
      desired_xyz = clamp_workspace(cmd[:3], emb)

      # 2. Rate-limit the Cartesian target, wrist yaw, pitch, and approach
      # azimuth offset.
      curr = self._pose[i]
      delta = desired_xyz - curr[:3]
      dist = float(np.linalg.norm(delta))
      if dist > self._max_step:
        delta *= self._max_step / dist
      nxt = curr.copy()
      nxt[:3] = curr[:3] + delta
      az_goal = embodiment_lib.wrap_pi(float(cmd[6]))
      nxt[5] = embodiment_lib.wrap_pi(
          curr[5]
          + np.clip(
              embodiment_lib.wrap_pi(az_goal - curr[5]),
              -MAX_ANGLE_STEP,
              MAX_ANGLE_STEP,
          )
      )
      azimuth = emb.heading(nxt[0], nxt[1]) + nxt[5]
      rel_goal = emb.relative_yaw(float(cmd[4]), nxt[0], nxt[1], azimuth)
      nxt[3] += np.clip(rel_goal - curr[3], -MAX_ANGLE_STEP, MAX_ANGLE_STEP)
      nxt[4] += np.clip(cmd[5] - curr[4], -MAX_ANGLE_STEP, MAX_ANGLE_STEP)
      self._pose[i] = nxt

      # 3. Solve task-priority IK for the arm joints.
      rot = embodiment_lib.tool_rotation(azimuth, nxt[3], nxt[4])
      q_arm = self._ik.solve(self._current_qpos[i, :n], nxt[:3], rot)
      self._current_qpos[i, :n] = q_arm

      # 3b. Proprioceptive joint-tracking feedback: low-pass filtered so steady
      # state gravity droop is compensated without amplifying transient velocity
      # lag or serial encoder jitter during motion.
      q_cmd = q_arm
      if emb.tracking_feedback_gain > 0:
        raw_err = np.clip(q_arm - self._measured_qpos[i, :n], -0.09, 0.09)
        self._filtered_q_err[i] = (
            0.85 * self._filtered_q_err[i] + 0.15 * raw_err
        )
        q_cmd = np.clip(
            q_arm + emb.tracking_feedback_gain * self._filtered_q_err[i],
            self._ik.jnt_min,
            self._ik.jnt_max,
        )

      # 4. Map the normalized gripper command [0, 1] to the gripper joint.
      grip_norm = float(np.clip(cmd[3], 0.0, 1.0))
      lo, hi = emb.gripper_cmd_range
      grip = lo + grip_norm * (hi - lo)
      self._current_qpos[i, n] = grip

      ctrl_out[i, :n] = q_cmd.astype(np.float32)
      ctrl_out[i, n] = np.float32(grip)

    return {
        key: torch.as_tensor(
            ctrl_out[:, j], dtype=torch.float32, device=self._xpu_device
        )
        for j, key in enumerate(self._emb.goal_position_keys)
    }

  # --- FeaturesProducer ---

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {
        embodiment_lib.QPOS_KEY: specs.Array(
            shape=(len(self._emb.joints),), dtype=np.float32
        ),
        embodiment_lib.EE_XPOS_KEY: specs.Array(shape=(3,), dtype=np.float32),
        embodiment_lib.EE_XMAT_KEY: specs.Array(shape=(9,), dtype=np.float32),
        "target_ee_pos": specs.Array(shape=(3,), dtype=np.float32),
        "raw_action": specs.Array(shape=(7,), dtype=np.float32),
    }

  def required_features_keys(self) -> set[str]:
    return set(self._emb.joint_position_keys)

  def produce_features(  # pyrefly: ignore[bad-override]
      self, required_features: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Synchronizes state after reset and exposes controller features."""
    joint_keys = self._emb.joint_position_keys
    meas_qpos = None
    if all(k in required_features for k in joint_keys):
      cols = [
          required_features[k].detach().cpu().numpy().reshape(self._nworld)
          for k in joint_keys
      ]
      meas_qpos = np.stack(cols, axis=-1)
    elif self._qpos_key in required_features:
      meas_qpos = required_features[self._qpos_key].detach().cpu().numpy()

    if meas_qpos is not None:
      for i in range(self._nworld):
        self._measured_qpos[i] = meas_qpos[i]
        if self._needs_sync[i]:
          self._current_qpos[i] = meas_qpos[i]
          self._filtered_q_err[i] = 0.0
          pos, rot = self._ik.fk(meas_qpos[i, : self._n_arm])
          self._pose[i] = self._pose_of(pos, rot)
          self._sync_raw_action(i)
          self._needs_sync[i] = False

    ee_xpos = np.zeros((self._nworld, 3), dtype=np.float32)
    ee_xmat = np.zeros((self._nworld, 9), dtype=np.float32)
    for i in range(self._nworld):
      pos, rot = self._ik.fk(self._measured_qpos[i, : self._n_arm])
      ee_xpos[i] = pos
      ee_xmat[i] = rot.reshape(9)

    return {
        embodiment_lib.QPOS_KEY: torch.as_tensor(
            self._measured_qpos,
            dtype=torch.float32,
            device=self._xpu_device,
        ),
        embodiment_lib.EE_XPOS_KEY: torch.as_tensor(
            ee_xpos,
            dtype=torch.float32,
            device=self._xpu_device,
        ),
        embodiment_lib.EE_XMAT_KEY: torch.as_tensor(
            ee_xmat,
            dtype=torch.float32,
            device=self._xpu_device,
        ),
        "target_ee_pos": torch.as_tensor(
            self._pose[:, :3],
            dtype=torch.float32,
            device=self._xpu_device,
        ),
        "raw_action": torch.as_tensor(
            self._last_raw_action,
            dtype=torch.float32,
            device=self._xpu_device,
        ),
    }

  # --- Reset ---

  def reset(self, env_ids: Union[torch.Tensor, slice, None] = None) -> None:
    """Marks environments for re-synchronization with measured robot state."""
    if env_ids is None:
      ids = slice(None)
    elif isinstance(env_ids, slice):
      ids = env_ids
    else:
      ids = env_ids.detach().cpu().numpy()
    self._needs_sync[ids] = True
    self._current_qpos[ids] = self._home_qpos
    self._filtered_q_err[ids] = 0.0
    self._pose[ids] = self._hover_pose
