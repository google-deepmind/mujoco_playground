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

"""MDP observations, rewards, and terminations for whole-body tracking."""

from collections.abc import Mapping

from reaf.simulation.torch import math as pg_math
from reaf.simulation.torch import termination_checker
import torch

_TERMINATE = termination_checker.TerminationResult.TERMINATE
_DO_NOT_TERMINATE = termination_checker.TerminationResult.DO_NOT_TERMINATE


def _bool_to_termination(condition: torch.Tensor) -> torch.Tensor:
  return torch.where(condition, _TERMINATE, _DO_NOT_TERMINATE)


##
# Observations.
##


def obs_motion_command(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Concatenated reference joint positions and velocities."""
  return torch.cat(
      [f["motion/joint_pos_ref"], f["motion/joint_vel_ref"]], dim=-1
  )


def obs_anchor_pos_b(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Reference anchor position in the robot anchor frame."""
  pos_b, _ = pg_math.subtract_frame_transforms(
      f["motion/anchor_pos_robot"],
      f["motion/anchor_quat_robot"],
      f["motion/anchor_pos_ref"],
      f["motion/anchor_quat_ref"],
  )
  return pos_b.flatten(start_dim=1)


def obs_anchor_ori_b(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Reference anchor 6D orientation in the robot anchor frame."""
  _, ori_b = pg_math.subtract_frame_transforms(
      f["motion/anchor_pos_robot"],
      f["motion/anchor_quat_robot"],
      f["motion/anchor_pos_ref"],
      f["motion/anchor_quat_ref"],
  )
  mat = pg_math.quat_to_matrix(ori_b)
  return mat[..., :2].flatten(start_dim=1)


def obs_robot_body_pos_b(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Tracked robot body positions in the robot anchor frame."""
  body_pos = f["motion/body_pos_robot"]
  nb = body_pos.shape[1]
  anchor_pos = f["motion/anchor_pos_robot"][:, None, :].expand(-1, nb, -1)
  anchor_quat = f["motion/anchor_quat_robot"][:, None, :].expand(-1, nb, -1)
  pos_b, _ = pg_math.subtract_frame_transforms(
      anchor_pos, anchor_quat, body_pos, f["motion/body_quat_robot"]
  )
  return pos_b.flatten(start_dim=1)


def obs_robot_body_ori_b(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Tracked robot body 6D orientations in the robot anchor frame."""
  body_quat = f["motion/body_quat_robot"]
  nb = body_quat.shape[1]
  anchor_pos = f["motion/anchor_pos_robot"][:, None, :].expand(-1, nb, -1)
  anchor_quat = f["motion/anchor_quat_robot"][:, None, :].expand(-1, nb, -1)
  _, ori_b = pg_math.subtract_frame_transforms(
      anchor_pos, anchor_quat, f["motion/body_pos_robot"], body_quat
  )
  mat = pg_math.quat_to_matrix(ori_b)
  return mat[..., :2].flatten(start_dim=1)


def obs_joint_pos_rel(
    f: Mapping[str, torch.Tensor],
    default_joint_pos: torch.Tensor,
    qpos_key: str = "robot/joint_pos",
) -> torch.Tensor:
  """Joint positions relative to the default nominal pose."""
  qpos = f[qpos_key]
  return qpos - default_joint_pos.to(device=qpos.device, dtype=qpos.dtype)


##
# Rewards.
##


def reward_motion_global_anchor_pos(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on global anchor position error."""
  error = torch.sum(
      torch.square(f["motion/anchor_pos_ref"] - f["motion/anchor_pos_robot"]),
      dim=-1,
  )
  return torch.exp(-error / std**2)


def reward_motion_global_anchor_ori(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on global anchor orientation error."""
  error = (
      pg_math.quat_error_magnitude2(
          f["motion/anchor_quat_ref"], f["motion/anchor_quat_robot"]
      )
      ** 2
  )
  return torch.exp(-error / std**2)


def reward_motion_relative_body_pos(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on mean yaw-anchored relative body position error."""
  error = torch.sum(
      torch.square(f["motion/body_pos_relative"] - f["motion/body_pos_robot"]),
      dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def reward_motion_relative_body_ori(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on mean yaw-anchored relative body orientation error."""
  error = (
      pg_math.quat_error_magnitude2(
          f["motion/body_quat_relative"], f["motion/body_quat_robot"]
      )
      ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def reward_motion_global_body_lin_vel(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on mean world-frame body linear velocity error."""
  error = torch.sum(
      torch.square(
          f["motion/body_lin_vel_ref"] - f["motion/body_lin_vel_robot"]
      ),
      dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def reward_motion_global_body_ang_vel(
    f: Mapping[str, torch.Tensor], std: float
) -> torch.Tensor:
  """Exponential kernel on mean world-frame body angular velocity error."""
  error = torch.sum(
      torch.square(
          f["motion/body_ang_vel_ref"] - f["motion/body_ang_vel_robot"]
      ),
      dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def reward_action_rate_l2(
    f: Mapping[str, torch.Tensor],
    action_key: str = "action",
    history_key: str = "action_history",
) -> torch.Tensor:
  """Squared L2 norm of consecutive raw action differences."""
  return torch.sum(torch.square(f[action_key] - f[history_key][:, 0]), dim=1)


def reward_joint_pos_limits(
    f: Mapping[str, torch.Tensor],
    soft_joint_pos_limits: torch.Tensor,
    qpos_key: str = "robot/joint_pos",
) -> torch.Tensor:
  """L1 penalty on joint positions exceeding soft joint limits."""
  joint_pos = f[qpos_key]
  limits = soft_joint_pos_limits.to(
      device=joint_pos.device, dtype=joint_pos.dtype
  )
  out_of_limits = -(joint_pos - limits[:, 0]).clip(max=0.0)
  out_of_limits += (joint_pos - limits[:, 1]).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def reward_self_collision(
    f: Mapping[str, torch.Tensor],
    sensor_key: str = "self_collision/force_history",
    force_threshold: float = 10.0,
) -> torch.Tensor:
  """Count physics substeps within a control step with self-collision > thr."""
  force_mag = torch.norm(f[sensor_key], dim=-1)
  hit = (force_mag > force_threshold).any(dim=1)
  return hit.sum(dim=-1).float()


##
# Terminations.
##


def termination_bad_anchor_pos_z_only(
    f: Mapping[str, torch.Tensor], threshold: float
) -> torch.Tensor:
  """Terminate when anchor vertical position error exceeds `threshold`."""
  failed = (
      torch.abs(
          f["motion/anchor_pos_ref"][:, -1]
          - f["motion/anchor_pos_robot"][:, -1]
      )
      > threshold
  )
  return _bool_to_termination(failed)


def termination_bad_anchor_ori(
    f: Mapping[str, torch.Tensor], threshold: float
) -> torch.Tensor:
  """Terminate when projected-gravity z difference exceeds `threshold`."""
  anchor_quat_ref = f["motion/anchor_quat_ref"]
  anchor_quat_robot = f["motion/anchor_quat_robot"]
  gravity = torch.zeros_like(f["motion/anchor_pos_ref"])
  gravity[:, 2] = -1.0
  motion_projected = pg_math.quat_rotate_inverse(anchor_quat_ref, gravity)
  robot_projected = pg_math.quat_rotate_inverse(anchor_quat_robot, gravity)
  failed = (motion_projected[:, 2] - robot_projected[:, 2]).abs() > threshold
  return _bool_to_termination(failed)


def termination_bad_motion_body_pos_z_only(
    f: Mapping[str, torch.Tensor],
    threshold: float,
    body_indexes: list[int],
) -> torch.Tensor:
  """Terminate when any specified body's relative z error exceeds `threshold`."""
  error = torch.abs(
      f["motion/body_pos_relative"][:, body_indexes, -1]
      - f["motion/body_pos_robot"][:, body_indexes, -1]
  )
  failed = torch.any(error > threshold, dim=-1)
  return _bool_to_termination(failed)


def metric_error_anchor_pos(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/anchor_pos_ref"] - f["motion/anchor_pos_robot"], dim=-1
  )


def metric_error_anchor_rot(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return pg_math.quat_error_magnitude2(
      f["motion/anchor_quat_ref"], f["motion/anchor_quat_robot"]
  )


def metric_error_anchor_lin_vel(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/anchor_lin_vel_ref"] - f["motion/anchor_lin_vel_robot"], dim=-1
  )


def metric_error_anchor_ang_vel(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/anchor_ang_vel_ref"] - f["motion/anchor_ang_vel_robot"], dim=-1
  )


def metric_error_body_pos(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/body_pos_relative"] - f["motion/body_pos_robot"], dim=-1
  ).mean(dim=-1)


def metric_error_body_rot(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return pg_math.quat_error_magnitude2(
      f["motion/body_quat_relative"], f["motion/body_quat_robot"]
  ).mean(dim=-1)


def metric_error_body_lin_vel(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/body_lin_vel_ref"] - f["motion/body_lin_vel_robot"], dim=-1
  ).mean(dim=-1)


def metric_error_body_ang_vel(f: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.norm(
      f["motion/body_ang_vel_ref"] - f["motion/body_ang_vel_robot"], dim=-1
  ).mean(dim=-1)


def metric_error_joint_pos(
    f: Mapping[str, torch.Tensor],
    qpos_key: str = "robot/joint_pos",
) -> torch.Tensor:
  return torch.norm(f["motion/joint_pos_ref"] - f[qpos_key].float(), dim=-1)


def metric_error_joint_vel(
    f: Mapping[str, torch.Tensor],
    qvel_key: str = "robot/joint_vel",
) -> torch.Tensor:
  return torch.norm(f["motion/joint_vel_ref"] - f[qvel_key].float(), dim=-1)
