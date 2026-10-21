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

"""Math helper functions."""

import torch


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Multiply two quaternions. Supports batching."""
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return torch.stack(
      [
          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ],
      dim=-1,
  )


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vector v by unit quaternion q.

  Args:
    q: (..., 4) quaternion [w, x, y, z].
    v: (..., 3) vector.

  Returns:
    (..., 3) rotated vector.
  """
  v = v.to(q.dtype)
  s = q[..., 0:1]  # (..., 1)
  u = q[..., 1:4]  # (..., 3)
  uv = torch.sum(u * v, dim=-1, keepdim=True)  # dot(u, v)
  uu = torch.sum(u * u, dim=-1, keepdim=True)  # dot(u, u)
  cross = torch.linalg.cross(u, v)
  return 2.0 * uv * u + (s * s - uu) * v + 2.0 * s * cross


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vector v by the inverse of unit quaternion q.

  Equivalent to quat_rotate(quat_conjugate(q), v) but avoids allocating
  the conjugated quaternion.

  Args:
    q: (..., 4) quaternion [w, x, y, z].
    v: (..., 3) vector.

  Returns:
    (..., 3) rotated vector.
  """
  v = v.to(q.dtype)
  s = q[..., 0:1]  # (..., 1)
  u = q[..., 1:4]  # (..., 3)
  neg_u = -u
  uv = torch.sum(neg_u * v, dim=-1, keepdim=True)
  uu = torch.sum(neg_u * neg_u, dim=-1, keepdim=True)
  cross = torch.linalg.cross(neg_u, v)
  return 2.0 * uv * neg_u + (s * s - uu) * v + 2.0 * s * cross


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  """Return the conjugate of a unit quaternion (negate vector part).

  For unit quaternions this is the same as the inverse.

  Args:
    q: (..., 4) quaternion [w, x, y, z].

  Returns:
    (..., 4) conjugate quaternion.
  """
  return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  """Compute the inverse of a quaternion.

  For unit quaternions this equals the conjugate; the squared-norm
  division makes it correct for non-unit quaternions too.

  Args:
    q: (..., 4) quaternion [w, x, y, z].
    eps: Small value to avoid division by zero.

  Returns:
    (..., 4) inverse quaternion [w, x, y, z].
  """
  return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def axis_angle_from_quat(
    quat: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
  """Convert a quaternion to its axis-angle (rotation vector) form.

  The returned vector's direction is the rotation axis and its magnitude
  is the rotation angle in radians (the SO(3) log map). A first-order
  Taylor expansion is used near zero rotation for numerical stability.
  Inverse of `axis_angle_to_quat`.

  Args:
    quat: (..., 4) quaternion [w, x, y, z].
    eps: Angle threshold below which the Taylor approximation is used.

  Returns:
    (..., 3) axis-angle vector.
  """
  # Flip to the positive-w hemisphere so the angle is in [0, pi].
  quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
  mag = torch.linalg.norm(quat[..., 1:], dim=-1)
  half_angle = torch.atan2(mag, quat[..., 0])
  angle = 2.0 * half_angle
  # axis_angle = xyz / (sin(angle/2) / angle); near zero use the Taylor
  # expansion sin(x/2)/x ~= 1/2 - x^2/48.
  sin_half_angles_over_angles = torch.where(
      angle.abs() > eps,  # pyrefly: ignore[missing-attribute]
      torch.sin(half_angle) / angle,
      0.5 - angle * angle / 48,
  )
  return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Compute the box-minus operator (q1 minus q2) as an axis-angle vector.

  Returns the rotation vector of the relative rotation q1 * q2^-1, i.e.
  the tangent-space difference between the two orientations.

  Args:
    q1: (..., 4) quaternion [w, x, y, z].
    q2: (..., 4) quaternion [w, x, y, z].

  Returns:
    (..., 3) axis-angle difference vector.
  """
  quat_diff = quat_mul(q1, quat_conjugate(q2))
  return axis_angle_from_quat(quat_diff)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Compute the angular distance (in radians) between two quaternions.

  Args:
    q1: (..., 4) quaternion [w, x, y, z].
    q2: (..., 4) quaternion [w, x, y, z].

  Returns:
    (...,) angular distance in radians.
  """
  quat_dot = torch.sum(q1 * q2, dim=-1)
  dot_abs = torch.clamp(torch.abs(quat_dot), max=1.0 - 1e-6)
  return 2.0 * torch.acos(dot_abs)


@torch.jit.script
def quat_error_magnitude2(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Compute the angular distance (in radians) between two quaternions.

  Computed as the norm of the box-minus (axis-angle) difference, which
  is exact (returns 0.0 for identical orientations) and numerically
  stable for small errors, unlike acos-of-dot-product formulations.

  Args:
    q1: (..., 4) quaternion [w, x, y, z].
    q2: (..., 4) quaternion [w, x, y, z].

  Returns:
    (...,) angular distance in radians.
  """
  return torch.norm(quat_box_minus(q1, q2), dim=-1)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
  """Extract the yaw-only component of a quaternion.

  Projects the rotation onto the world z-axis: the result rotates by the
  input's heading (yaw) with no roll or pitch.

  Args:
    quat: (..., 4) quaternion [w, x, y, z].

  Returns:
    (..., 4) unit quaternion containing only the yaw rotation.
  """
  qw = quat[..., 0]
  qx = quat[..., 1]
  qy = quat[..., 2]
  qz = quat[..., 3]
  yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  quat_yaw = torch.zeros_like(quat)
  quat_yaw[..., 3] = torch.sin(yaw / 2)
  quat_yaw[..., 0] = torch.cos(yaw / 2)
  return torch.nn.functional.normalize(quat_yaw, p=2.0, dim=-1)


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute the pose of frame 2 in frame 1: T12 = T01^-1 * T02.

  Both input poses must be expressed in the same parent frame 0. Useful
  for turning world-frame poses into robot-local observations.

  Args:
    t01: (..., 3) position of frame 1 in frame 0.
    q01: (..., 4) orientation of frame 1 in frame 0, [w, x, y, z].
    t02: (..., 3) position of frame 2 in frame 0.
    q02: (..., 4) orientation of frame 2 in frame 0, [w, x, y, z].

  Returns:
    Tuple of ((..., 3) position, (..., 4) quaternion) of frame 2
    expressed in frame 1.
  """
  q10 = quat_inv(q01)
  q12 = quat_mul(q10, q02)
  t12 = quat_rotate(q10, t02 - t01)
  return t12, q12


@torch.jit.script
def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Convert axis-angle representation to quaternion.

  Args:
    axis: Rotation axis. Shape is (N, 3), assumed normalized.
    angle: Rotation angle in radians. Shape is (N,).

  Returns:
    Quaternion in (w, x, y, z). Shape is (N, 4).
  """
  s = torch.sin(angle * 0.5)
  c = torch.cos(angle * 0.5)
  return torch.stack(
      [c, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s], dim=-1
  )


@torch.jit.script
def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
  """Convert Euler angles (XYZ convention) to quaternion.

  Args:
    roll: Rotation around x-axis (in radians). Shape is (N,).
    pitch: Rotation around y-axis (in radians). Shape is (N,).
    yaw: Rotation around z-axis (in radians). Shape is (N,).

  Returns:
    Quaternion in (w, x, y, z). Shape is (N, 4).
  """
  n = roll.shape[0]
  device = roll.device

  x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)
  y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3)
  z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)

  qx = axis_angle_to_quat(x_axis, roll)
  qy = axis_angle_to_quat(y_axis, pitch)
  qz = axis_angle_to_quat(z_axis, yaw)

  return quat_mul(quat_mul(qz, qy), qx)


def quat_from_euler_extrinsic_zyx(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
  """Convert Euler angles (extrinsic ZYX convention, Yaw-Pitch-Roll) to quaternion.

  Args:
    roll: Rotation around x-axis (in radians). Shape is (N,).
    pitch: Rotation around y-axis (in radians). Shape is (N,).
    yaw: Rotation around z-axis (in radians). Shape is (N,).

  Returns:
    Quaternion in (w, x, y, z). Shape is (N, 4).
  """
  n = roll.shape[0]
  device = roll.device

  x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)
  y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3)
  z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)

  qx = axis_angle_to_quat(x_axis, roll)
  qy = axis_angle_to_quat(y_axis, pitch)
  qz = axis_angle_to_quat(z_axis, yaw)

  return quat_mul(quat_mul(qx, qy), qz)


@torch.jit.script
def sample_uniform(
    n: int,
    low: float,
    high: float,
    device: torch.device,
) -> torch.Tensor:
  """Sample n values uniformly from [low, high]."""
  return torch.rand(n, device=device) * (high - low) + low


@torch.jit.script
def sample_uniform_3d(
    n: int,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
  """Sample (N, 3) values uniformly from low[i] to high[i] per dimension.

  Args:
    n: Number of samples.
    low: Lower bounds for each dim. Shape is (3,).
    high: Upper bounds for each dim. Shape is (3,).
    device: Device for output tensor.

  Returns:
    Sampled values. Shape is (N, 3).
  """
  samples = torch.rand(n, 3, device=device)
  return samples * (high - low) + low


def sample_uniform_range(
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    size: tuple[int, ...] | int,
    device: torch.device | str,
) -> torch.Tensor:
  """Sample uniformly from [lower, upper) with an arbitrary output shape.

  Unlike `sample_uniform`, the bounds may be tensors that broadcast
  against the output shape (e.g. per-dimension ranges of shape (6,) with
  size (n, 6)).

  Args:
    lower: Scalar or tensor lower bound(s), broadcastable to `size`.
    upper: Scalar or tensor upper bound(s), broadcastable to `size`.
    size: Output shape (an int is treated as a 1-D shape).
    device: Device for the output tensor.

  Returns:
    Tensor of shape `size` sampled uniformly from [lower, upper).
  """
  if isinstance(size, int):
    size = (size,)
  return torch.rand(*size, device=device) * (upper - lower) + lower


@torch.jit.script
def sigmoids(x: torch.Tensor, value_at_1: float, sigmoid: str) -> torch.Tensor:
  """Returns 1-sigmoid(x) that hits value_at_1 at x=1."""
  if sigmoid == 'gaussian':
    scale = torch.sqrt(
        -2 * torch.log(torch.tensor(value_at_1, device=x.device, dtype=x.dtype))
    )
    return torch.exp(-0.5 * (x * scale) ** 2)
  elif sigmoid == 'quadratic':
    scale = (
        torch.sqrt(
            torch.tensor(1.0 - value_at_1, device=x.device, dtype=x.dtype)
        )
        + 1e-8
    )
    scaled_x = x * scale
    return torch.where(
        scaled_x < 1,
        1 - scaled_x**2,
        torch.tensor(0.0, dtype=x.dtype, device=x.device),
    )
  elif sigmoid == 'linear':
    scale = 1.0 - value_at_1
    scaled_x = x * scale
    return torch.where(
        scaled_x < 1,
        1 - scaled_x,
        torch.tensor(0.0, dtype=x.dtype, device=x.device),
    )
  else:
    raise ValueError(f'Unknown sigmoid type {sigmoid}')


@torch.jit.script
def tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = 'gaussian',
    value_at_margin: float = 0.1,
) -> torch.Tensor:
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise."""
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = torch.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = torch.where(in_bounds, 1.0, 0.0).to(dtype=x.dtype)
  else:
    d = torch.where(x < lower, lower - x, x - upper) / margin
    value = torch.where(in_bounds, 1.0, sigmoids(d, value_at_margin, sigmoid))

  return value


@torch.jit.script
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
  """Convert unit quaternions (w, x, y, z) to 3x3 rotation matrices."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  x2, y2, z2 = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z
  row0 = torch.stack(
      [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1
  )
  row1 = torch.stack(
      [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)], dim=-1
  )
  row2 = torch.stack(
      [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)], dim=-1
  )
  return torch.stack([row0, row1, row2], dim=-2)


@torch.jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert a 6D representation to a 3x3 rotation matrix using Gram-Schmidt."""
  shape = list(d6.shape)[:-1]
  shape.append(2)
  shape.append(3)
  x = d6.reshape(shape)
  x1, x2 = x[..., 0, :], x[..., 1, :]
  b1 = x1 / torch.linalg.norm(x1, dim=-1, keepdim=True)
  b2 = x2 - torch.sum(b1 * x2, dim=-1, keepdim=True) * b1
  b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack([b1, b2, b3], dim=-2)


@torch.jit.script
def rot_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
  """Extract first two rows of a 3x3 rotation matrix and flattens."""
  return R[..., :2, :].flatten(start_dim=-2)


@torch.jit.script
def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
  """Convert rotation matrices to quaternions (wxyz).

  Args:
    matrix: Rotation matrices of shape (B, 3, 3) or (3, 3).

  Returns:
    Quaternions of shape (B, 4) or (4,) in wxyz.
  """
  is_batched = matrix.ndim == 3
  if not is_batched:
    matrix = matrix.unsqueeze(0)

  batch_size = matrix.shape[0]

  t0 = 1.0 + matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
  t1 = 1.0 + matrix[:, 0, 0] - matrix[:, 1, 1] - matrix[:, 2, 2]
  t2 = 1.0 - matrix[:, 0, 0] + matrix[:, 1, 1] - matrix[:, 2, 2]
  t3 = 1.0 - matrix[:, 0, 0] - matrix[:, 1, 1] + matrix[:, 2, 2]

  t = torch.stack([t0, t1, t2, t3], dim=1)  # (B, 4)

  max_idx = torch.argmax(t, dim=1)  # (B,)
  max_val = torch.gather(t, 1, max_idx.unsqueeze(1)).squeeze(1)  # (B,)

  q = torch.zeros((batch_size, 4), dtype=matrix.dtype, device=matrix.device)

  mask0 = max_idx == 0
  if mask0.any():  # pyrefly: ignore[missing-attribute]
    s = 0.5 / torch.sqrt(max_val[mask0])
    q[mask0, 0] = 0.25 / s
    q[mask0, 1] = (matrix[mask0, 2, 1] - matrix[mask0, 1, 2]) * s
    q[mask0, 2] = (matrix[mask0, 0, 2] - matrix[mask0, 2, 0]) * s
    q[mask0, 3] = (matrix[mask0, 1, 0] - matrix[mask0, 0, 1]) * s

  mask1 = max_idx == 1
  if mask1.any():  # pyrefly: ignore[missing-attribute]
    s = 0.5 / torch.sqrt(max_val[mask1])
    q[mask1, 0] = (matrix[mask1, 2, 1] - matrix[mask1, 1, 2]) * s
    q[mask1, 1] = 0.25 / s
    q[mask1, 2] = (matrix[mask1, 0, 1] + matrix[mask1, 1, 0]) * s
    q[mask1, 3] = (matrix[mask1, 0, 2] + matrix[mask1, 2, 0]) * s

  mask2 = max_idx == 2
  if mask2.any():  # pyrefly: ignore[missing-attribute]
    s = 0.5 / torch.sqrt(max_val[mask2])
    q[mask2, 0] = (matrix[mask2, 0, 2] - matrix[mask2, 2, 0]) * s
    q[mask2, 1] = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) * s
    q[mask2, 2] = 0.25 / s
    q[mask2, 3] = (matrix[mask2, 1, 2] + matrix[mask2, 2, 1]) * s

  mask3 = max_idx == 3
  if mask3.any():  # pyrefly: ignore[missing-attribute]
    s = 0.5 / torch.sqrt(max_val[mask3])
    q[mask3, 0] = (matrix[mask3, 1, 0] - matrix[mask3, 0, 1]) * s
    q[mask3, 1] = (matrix[mask3, 0, 2] + matrix[mask3, 2, 0]) * s
    q[mask3, 2] = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) * s
    q[mask3, 3] = 0.25 / s

  q = q / torch.linalg.norm(q, dim=1, keepdim=True)

  if not is_batched:
    return q.squeeze(0)
  return q
