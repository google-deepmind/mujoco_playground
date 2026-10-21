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

"""MDP functions (observations and rewards) for the Pick Object task."""

import mujoco
import torch


def obs_joint_pos_rel(features, qpos_key: str, home_qpos_key: str, **kwargs):
  """Joint positions relative to home."""
  qpos = features[qpos_key]
  home_qpos = features[home_qpos_key]
  return qpos - home_qpos


def obs_ee_to_cube(features, ee_xpos_key: str, cube_xpos_key: str, **kwargs):
  """Vector from end-effector to cube."""
  ee_pos = features[ee_xpos_key]
  cube_pos = features[cube_xpos_key]
  return cube_pos - ee_pos


def obs_cube_to_goal(
    features, cube_xpos_key: str, target_xpos_key: str, **kwargs
):
  """Vector from cube to goal target."""
  cube_pos = features[cube_xpos_key]
  target_pos = features[target_xpos_key]
  return target_pos - cube_pos


def obs_goal_position_ee_frame(
    features, ee_xpos_key: str, ee_xmat_key: str, target_xpos_key: str, **kwargs
):
  """Goal position in EE frame for vision policy."""
  ee_pos = features[ee_xpos_key]
  ee_mat = features[ee_xmat_key].reshape(
      -1, 3, 3
  )  # (nworld, 9) -> (nworld, 3, 3)
  target_pos = features[target_xpos_key]
  target_rel_w = target_pos - ee_pos
  return torch.einsum("nij,nj->ni", ee_mat.transpose(-1, -2), target_rel_w)


def _center_crop_square(rgb: torch.Tensor) -> torch.Tensor:
  """Crops a (batch, C, H, W) image to its largest centered square."""
  height, width = rgb.shape[-2:]
  side = min(height, width)
  top = (height - side) // 2
  left = (width - side) // 2
  return rgb[..., top : top + side, left : left + side]


def obs_camera(
    features,
    camera_key: str,
    downsample_size: tuple[int, int] | None = None,
    pixel_noise_std: float = 0.0,
    **kwargs,
):
  """Camera observation normalized to [0, 1] in CHW format, optionally downsampled."""
  del kwargs
  rgb = features[camera_key]  # (batch, H, W, C) uint8
  rgb = rgb.permute(0, 3, 1, 2).float() / 255.0  # -> (batch, C, H, W)
  rgb = _center_crop_square(rgb)
  if downsample_size is not None and tuple(rgb.shape[-2:]) != tuple(
      downsample_size
  ):
    rgb = torch.nn.functional.interpolate(
        rgb, size=downsample_size, mode="area"  # pyrefly: ignore[bad-argument-type]
    )
  if pixel_noise_std > 0.0:
    noise = torch.randn_like(rgb) * pixel_noise_std
    rgb = torch.clamp(rgb + noise, 0.0, 1.0)
  return rgb


def override_flat_materials(
    model,
    spec,
    device: torch.device,
    *,
    patterns: tuple[str, ...],
    env_mask: torch.Tensor | None = None,
) -> None:
  """Override materials for matched geoms to be flat (no texture)."""
  del device  # Unused.
  for pattern in patterns:
    bound = model.bind_pattern(spec, mujoco.mjtObj.mjOBJ_GEOM, pattern)
    matid = bound.geom_matid
    new_matid = torch.full_like(matid, -1)
    bound.set("geom_matid", new_matid, env_mask=env_mask)


def reward_action_rate(
    features, action_key: str, history_key: str, penalty_scale: float = -0.01
):
  """Action rate penalty (penalize jerky actions)."""
  action_t0 = features[action_key]
  action_t1 = features[history_key][:, 0]
  delta = action_t0 - action_t1
  return penalty_scale * torch.sum(delta**2, dim=-1)


def reward_joint_vel_hinge(
    features, qvel_key: str, max_vel: float = 0.5, penalty_scale: float = -0.01
):
  """Joint velocity hinge penalty (penalize excess velocity)."""
  qvel = features[qvel_key]
  excess = (qvel.abs() - max_vel).clamp_min(0.0)
  return penalty_scale * torch.sum(excess**2, dim=-1)


def reward_joint_pos_limits(
    features,
    qpos_key: str,
    joint_range_key: str,
    penalty_scale: float = -10.0,
    margin: float = 0.9,
):
  """Joint position limits penalty (penalize near limits)."""
  qpos = features[qpos_key]
  joint_range = features[joint_range_key]  # (njoints, 2)
  lower_hard = joint_range[..., 0]
  upper_hard = joint_range[..., 1]

  range_center = (upper_hard + lower_hard) / 2
  range_half = (upper_hard - lower_hard) / 2
  lower_limits = range_center - margin * range_half
  upper_limits = range_center + margin * range_half

  out_of_lower = -(qpos - lower_limits).clamp(max=0.0)
  out_of_upper = (qpos - upper_limits).clamp(min=0.0)
  out_of_limits = out_of_lower + out_of_upper

  return penalty_scale * torch.sum(out_of_limits, dim=-1)


def reward_floor_contact(
    features,
    contact_key: str = "floor_contact/data",
    penalty_scale: float = -0.1,
) -> torch.Tensor:
  """Penalty for floor/table contact via ContactSensorDevice."""
  contact_data = features[contact_key]
  has_contact = contact_data.any(dim=(-2, -1)).float()
  return penalty_scale * has_contact


def reward_arm_nominal_pose(
    features,
    qpos_key: str,
    home_qpos_key: str,
    num_arm_joints: int = 6,
    num_joints: int | None = None,
    std: float = 1.0,
):
  """Reward for staying close to nominal (home) pose for the arm only."""
  if num_joints is not None:
    num_arm_joints = num_joints
  qpos = features[qpos_key][:, :num_arm_joints]
  home_qpos = features[home_qpos_key]
  if home_qpos.ndim > 1:
    home_qpos = home_qpos[:, :num_arm_joints]
  else:
    home_qpos = home_qpos[:num_arm_joints]
  error = torch.sum(torch.square(qpos - home_qpos), dim=-1)
  return torch.exp(-error / std**2)


def reward_ee_nominal_orientation(
    features,
    xmat_key: str,
    nominal_xmat: tuple[float, ...] | torch.Tensor,
    std: float = 1.0,
):
  """Reward for staying close to nominal end-effector orientation."""
  xmat = features[xmat_key]  # Shape (nworld, 9)
  if isinstance(nominal_xmat, tuple):
    nominal = torch.as_tensor(
        nominal_xmat, dtype=xmat.dtype, device=xmat.device
    ).view(1, 9)
  else:
    nominal = nominal_xmat.view(1, 9).to(device=xmat.device, dtype=xmat.dtype)
  error = torch.sum(torch.square(xmat - nominal), dim=-1)
  return torch.exp(-error / std**2)


def reward_success(
    features,
    cube_xpos_key: str = "object/xpos",
    target_xpos_key: str = "target/mocap_pos",
    distance_threshold: float = 0.08,
    lift_height_threshold: float = 0.04,
    **kwargs,
) -> torch.Tensor:
  """True (1.0) if cube is lifted off table and reasonably close to mocap target."""
  del kwargs
  cube_pos = features[cube_xpos_key]
  target_pos = features[target_xpos_key]
  dist = torch.linalg.norm(target_pos - cube_pos, dim=-1)
  lifted = cube_pos[..., 2] > lift_height_threshold
  close = dist < distance_threshold
  return (lifted & close).float()


def reward_reaching(
    features,
    ee_xpos_key: str = "ee/xpos",
    cube_xpos_key: str = "object/xpos",
    reaching_std: float = 0.05,
    **kwargs,
) -> torch.Tensor:
  """Reward for gripper approaching the cube."""
  del kwargs
  ee_pos = features[ee_xpos_key]
  cube_pos = features[cube_xpos_key]
  reach_error = torch.sum(torch.square(ee_pos - cube_pos), dim=-1)
  return torch.exp(-reach_error / reaching_std**2)


def reward_gripper_close(
    features,
    ee_xpos_key: str = "ee/xpos",
    cube_xpos_key: str = "object/xpos",
    qpos_key: str = "robot_state/joint_pos",
    gripper_joint_idx: int = -1,
    open_pos: float = 0.80,
    closed_pos: float = 0.15,
    reaching_std: float = 0.025,
    **kwargs,
) -> torch.Tensor:
  """Reward for closing the gripper once the end-effector is at the cube."""
  del kwargs
  ee_pos = features[ee_xpos_key]
  cube_pos = features[cube_xpos_key]
  reach_error = torch.sum(torch.square(ee_pos - cube_pos), dim=-1)
  near_cube = torch.exp(-reach_error / reaching_std**2)

  gripper_qpos = features[qpos_key][..., gripper_joint_idx]
  closed_frac = torch.clamp(
      (open_pos - gripper_qpos) / (open_pos - closed_pos + 1e-6), 0.0, 1.0
  )
  return near_cube * closed_frac


def reward_cube_lift(
    features,
    cube_xpos_key: str = "object/xpos",
    target_xpos_key: str = "target/mocap_pos",
    table_z: float = 0.025,
    **kwargs,
) -> torch.Tensor:
  """Reward for lifting the cube off table towards target mocap height."""
  del kwargs
  cube_pos = features[cube_xpos_key]
  target_pos = features[target_xpos_key]
  cube_z = cube_pos[..., 2]
  target_z = target_pos[..., 2]
  return torch.clamp((cube_z - table_z) / (target_z - table_z + 1e-6), 0.0, 1.0)


def reward_cube_to_target(
    features,
    cube_xpos_key: str = "object/xpos",
    target_xpos_key: str = "target/mocap_pos",
    table_z: float = 0.025,
    std: float = 0.10,
    **kwargs,
) -> torch.Tensor:
  """Reward bringing cube to target mocap position, active only when lifted."""
  del kwargs
  cube_pos = features[cube_xpos_key]
  target_pos = features[target_xpos_key]
  cube_z = cube_pos[..., 2]
  lifted = (cube_z > table_z).float()
  dist_sq = torch.sum(torch.square(target_pos - cube_pos), dim=-1)
  return lifted * torch.exp(-dist_sq / std**2)
