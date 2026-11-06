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

"""MDP terms (observations and success checks) for the agentic tasks.

Robot-specific numbers (gripper range, base position) are passed in as term
parameters by the environment config, so these terms work for any arm.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence

from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
import torch

# Arm state device name and its joint-position measurement; the real SO101
# driver publishes the same key.
ROBOT_STATE = embodiment_lib.ROBOT_STATE
QPOS_KEY = embodiment_lib.QPOS_KEY
EE_XPOS_KEY = embodiment_lib.EE_XPOS_KEY
EE_XMAT_KEY = embodiment_lib.EE_XMAT_KEY

# Pinch-center distance from an object above which it is considered released.
RELEASE_CLEARANCE = 0.045


def obs_gripper_state(
    features: Mapping[str, torch.Tensor],
    *,
    gripper_range: tuple[float, float],
    qpos_key: str = QPOS_KEY,
) -> torch.Tensor:
  """Returns normalized gripper opening in [0.0, 1.0] (0=closed, 1=open).

  Args:
    features: Simulator or hardware features.
    gripper_range: Gripper joint positions (closed, open).
    qpos_key: Joint positions whose last entry is the gripper joint.
  """
  gripper = features[qpos_key][:, -1:]
  low, high = gripper_range
  return torch.clamp((gripper - low) / (high - low), 0.0, 1.0)


def obs_ee_ori(
    features: Mapping[str, torch.Tensor],
    *,
    base_xy: tuple[float, float],
    free_azimuth: bool = False,
) -> torch.Tensor:
  """Returns the pinch-site orientation `[yaw, pitch, azimuth_off]` in radians.

  `yaw` is the world yaw of the jaw closing axis (site x) folded into
  `[-pi/2, pi/2)`; `pitch` is the tilt of the approach axis (site z) below
  horizontal; `azimuth_off` is the horizontal azimuth offset of the approach
  axis relative to the radial heading from `base_xy`. See `embodiment.py` for
  the convention; this is its batched inverse.

  Args:
    features: Simulator or hardware features.
    base_xy: World `(x, y)` of the robot base pan axis.
    free_azimuth: Whether the embodiment has 6+ arm DoFs to steer horizontal
      approach azimuth independently of the radial heading.

  Returns:
    Tensor of shape `[nworld, 3]`.
  """
  pos = features[EE_XPOS_KEY]
  rot = features[EE_XMAT_KEY].reshape(-1, 3, 3)
  closing, approach = rot[..., 0], rot[..., 2]
  heading = torch.atan2(pos[:, 1] - base_xy[1], pos[:, 0] - base_xy[0])
  horiz = torch.hypot(approach[:, 0], approach[:, 1])
  tilted = (
      (horiz > 0.05)
      if free_azimuth
      else torch.zeros_like(horiz, dtype=torch.bool)
  )
  azimuth = torch.where(
      tilted, torch.atan2(approach[:, 1], approach[:, 0]), heading
  )
  cos_a, sin_a = torch.cos(azimuth), torch.sin(azimuth)
  proj = torch.where(
      tilted, horiz, approach[:, 0] * cos_a + approach[:, 1] * sin_a
  )
  pitch = torch.atan2(-approach[:, 2], proj)
  ortho_dot = (
      torch.sin(pitch) * (closing[:, 0] * cos_a + closing[:, 1] * sin_a)
      + torch.cos(pitch) * closing[:, 2]
  )
  tang_dot = -closing[:, 0] * sin_a + closing[:, 1] * cos_a
  yaw = azimuth + torch.atan2(tang_dot, ortho_dot)
  yaw = torch.remainder(yaw + torch.pi / 2, torch.pi) - torch.pi / 2
  az_off = (
      torch.remainder(azimuth - heading + torch.pi, 2 * torch.pi) - torch.pi
  )
  return torch.stack([yaw, pitch, az_off], dim=-1)


def _released(
    features: Mapping[str, torch.Tensor],
    object_pos: torch.Tensor,
    gripper_range: tuple[float, float],
    released_norm: float,
) -> torch.Tensor:
  """True when the gripper is open or the pinch center clears the object."""
  gripper_norm = obs_gripper_state(
      features, gripper_range=gripper_range
  ).squeeze(-1)
  ee_dist = torch.linalg.norm(features[EE_XPOS_KEY] - object_pos, dim=-1)
  return (gripper_norm > released_norm) | (ee_dist > RELEASE_CLEARANCE)


def check_contained(
    features: Mapping[str, torch.Tensor],
    *,
    object_keys: Sequence[str],
    anchor_key: str,
    xy_radius: float,
    z_min: float,
    z_max: float,
    gripper_range: tuple[float, float],
    released_norm: float,
) -> torch.Tensor:
  """Returns bool [nworld]: every object rests in/on the container, released.

  An object counts when its XY lies within `xy_radius` of the container
  position `anchor_key` and its height relative to it is within
  `[z_min, z_max]`.
  """
  anchor = features[anchor_key]
  ok = torch.ones_like(anchor[:, 0], dtype=torch.bool)
  for key in object_keys:
    pos = features[key]
    xy_dist = torch.linalg.norm(pos[:, :2] - anchor[:, :2], dim=-1)
    dz = pos[:, 2] - anchor[:, 2]
    ok &= (xy_dist < xy_radius) & (dz > z_min) & (dz < z_max)
    ok &= _released(features, pos, gripper_range, released_norm)
  return ok


def check_inserted(
    features: Mapping[str, torch.Tensor],
    *,
    peg_key: str,
    socket_key: str,
    xy_tol: float,
    z_max: float,
    gripper_range: tuple[float, float],
    released_norm: float,
) -> torch.Tensor:
  """Returns bool [nworld]: peg is inserted into the socket cavity, released.

  The peg must be aligned within `xy_tol` of the socket center, with its bottom
  resting inside the cavity (relative height `dz <= z_max`), and the gripper
  open.

  Args:
    features: Environment feature dictionary.
    peg_key: Feature key for the peg 3D position.
    socket_key: Feature key for the socket 3D position.
    xy_tol: Maximum horizontal XY offset in meters.
    z_max: Maximum vertical offset `z_peg - z_socket` in meters.
    gripper_range: Normalized gripper range `(closed, open)` in radians.
    released_norm: Minimum normalized opening for the gripper to count as open.

  Returns:
    Boolean tensor of shape `[nworld]`.
  """
  socket_pos = features[socket_key]
  peg_pos = features[peg_key]
  xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
  dz = peg_pos[:, 2] - socket_pos[:, 2]
  ok = (xy_dist < xy_tol) & (dz <= z_max)
  return ok & _released(features, peg_pos, gripper_range, released_norm)


def check_capped(
    features: Mapping[str, torch.Tensor],
    *,
    cap_key: str,
    marker_key: str,
    xy_tol: float,
    dz_min: float,
    dz_max: float,
    gripper_range: tuple[float, float],
    released_norm: float,
) -> torch.Tensor:
  """Returns bool [nworld]: cap is coaxially placed over the marker tip and released.

  The cap must be aligned within `xy_tol` in XY of the marker center, with its
  vertical position `dz = z_cap - z_marker` within `[dz_min, dz_max]`, and the
  gripper open.

  Args:
    features: Environment feature dictionary.
    cap_key: Feature key for the cap 3D position.
    marker_key: Feature key for the marker 3D position.
    xy_tol: Maximum horizontal XY offset in meters.
    dz_min: Minimum vertical offset `z_cap - z_marker` in meters.
    dz_max: Maximum vertical offset `z_cap - z_marker` in meters.
    gripper_range: Normalized gripper range `(closed, open)` in radians.
    released_norm: Minimum normalized opening for the gripper to count as open.

  Returns:
    Boolean tensor of shape `[nworld]`.
  """
  cap_pos = features[cap_key]
  marker_pos = features[marker_key]
  xy_dist = torch.linalg.norm(cap_pos[:, :2] - marker_pos[:, :2], dim=-1)
  dz = cap_pos[:, 2] - marker_pos[:, 2]
  ok = (xy_dist < xy_tol) & (dz >= dz_min) & (dz <= dz_max)
  return ok & _released(features, cap_pos, gripper_range, released_norm)


def check_usb_plugged(
    features: Mapping[str, torch.Tensor],
    *,
    usb_key: str,
    port_key: str,
    dx_range: tuple[float, float],
    y_tol: float,
    dz_range: tuple[float, float],
    gripper_range: tuple[float, float],
    released_norm: float,
) -> torch.Tensor:
  """Returns bool [nworld]: USB drive is plugged horizontally/angled into the port slot and released.

  The USB center relative to the port fixture (`pos_usb - pos_port`) must lie
  within `dx_range` along X, within `y_tol` laterally in Y, and within
  `dz_range` vertically in Z, with the gripper open.

  Args:
    features: Environment feature dictionary.
    usb_key: Feature key for the USB body 3D position.
    port_key: Feature key for the port body 3D position.
    dx_range: Allowed `(min, max)` X offset `x_usb - x_port` in meters.
    y_tol: Maximum lateral Y offset `|y_usb - y_port|` in meters.
    dz_range: Allowed `(min, max)` Z offset `z_usb - z_port` in meters.
    gripper_range: Normalized gripper range `(closed, open)` in radians.
    released_norm: Minimum normalized opening for the gripper to count as open.

  Returns:
    Boolean tensor of shape `[nworld]`.
  """
  usb_pos = features[usb_key]
  port_pos = features[port_key]
  dx = usb_pos[:, 0] - port_pos[:, 0]
  dy = torch.abs(usb_pos[:, 1] - port_pos[:, 1])
  dz = usb_pos[:, 2] - port_pos[:, 2]
  ok = (
      (dx >= dx_range[0])
      & (dx <= dx_range[1])
      & (dy < y_tol)
      & (dz >= dz_range[0])
      & (dz <= dz_range[1])
  )
  return ok & _released(features, usb_pos, gripper_range, released_norm)


def check_stacked(
    features: Mapping[str, torch.Tensor],
    *,
    object_keys: Sequence[str],
    gripper_range: tuple[float, float],
    released_norm: float,
    xy_tol: float = 0.015,
    dz_range: tuple[float, float] = (0.022, 0.038),
) -> torch.Tensor:
  """Returns bool [nworld]: the objects form one vertical tower, released.

  Objects are ordered by height; each consecutive pair must be within `xy_tol`
  in XY and separated by a height in `dz_range`. Any stacking order counts.
  """
  pos = torch.stack([features[k] for k in object_keys], dim=1)  # (n, k, 3)
  order = torch.argsort(pos[..., 2], dim=1)
  pos = torch.gather(pos, 1, order.unsqueeze(-1).expand_as(pos))
  ok = torch.ones_like(pos[:, 0, 0], dtype=torch.bool)
  for i in range(pos.shape[1] - 1):
    lo, hi = pos[:, i], pos[:, i + 1]
    xy_dist = torch.linalg.norm(hi[:, :2] - lo[:, :2], dim=-1)
    dz = hi[:, 2] - lo[:, 2]
    ok &= (xy_dist < xy_tol) & (dz > dz_range[0]) & (dz < dz_range[1])
  return ok & _released(features, pos[:, -1], gripper_range, released_norm)


def success_reward(
    features: Mapping[str, torch.Tensor],
    *,
    check: Callable[..., torch.Tensor],
    **kwargs,
) -> torch.Tensor:
  """Binary reward wrapping a task success check."""
  return check(features, **kwargs).to(dtype=torch.float32)


def success_termination(
    features: Mapping[str, torch.Tensor],
    *,
    check: Callable[..., torch.Tensor],
    **kwargs,
) -> torch.Tensor:
  """Terminates the episode when the task success check passes."""
  return check(features, **kwargs)
