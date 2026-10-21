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

"""Tests for math helper functions."""

import math

from absl.testing import absltest
import numpy as np
from reaf.simulation.torch._src.common import math as pg_math
import scipy.spatial.transform
import torch

R = scipy.spatial.transform.Rotation


class MathTest(absltest.TestCase):

  def test_quat_mul(self):
    # Identity quaternion
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    res = pg_math.quat_mul(q1, q2)
    torch.testing.assert_close(res, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Rotation 90 deg around Z:
    # q = [cos(theta/2), 0, 0, sin(theta/2)]
    #   = [cos(pi/4), 0, 0, sin(pi/4)] = [0.7071, 0, 0, 0.7071]
    q_90_z = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
    # Rotate twice should be 180 deg around Z -> [0, 0, 0, 1]
    res = pg_math.quat_mul(q_90_z, q_90_z)
    torch.testing.assert_close(
        res, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6
    )

    # Batch support (2D batch)
    q_batch_1 = torch.stack([q1, q_90_z])
    q_batch_2 = torch.stack([q2, q_90_z])
    res_batch = pg_math.quat_mul(q_batch_1, q_batch_2)
    expected = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    ])
    torch.testing.assert_close(res_batch, expected, atol=1e-6, rtol=1e-6)

    # 3D Batch support (N, M, 4)
    q_3d_1 = torch.stack([q_batch_1, q_batch_1])  # (2, 2, 4)
    q_3d_2 = torch.stack([q_batch_2, q_batch_2])  # (2, 2, 4)
    res_3d = pg_math.quat_mul(q_3d_1, q_3d_2)
    expected_3d = torch.stack([expected, expected])  # (2, 2, 4)
    torch.testing.assert_close(res_3d, expected_3d, atol=1e-6, rtol=1e-6)

  def test_quat_rotate(self):
    # Identity quaternion should leave vector unchanged.
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    res = pg_math.quat_rotate(q_id, v)
    torch.testing.assert_close(res, v, atol=1e-6, rtol=1e-6)

    # 90 deg rotation around Z: [1, 0, 0] -> [0, 1, 0]
    q_90_z = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    v_x = torch.tensor([[1.0, 0.0, 0.0]])
    res = pg_math.quat_rotate(q_90_z, v_x)
    torch.testing.assert_close(
        res, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=1e-6
    )

    # Batch support.
    q_batch = torch.stack([q_id.squeeze(), q_90_z.squeeze()])
    v_batch = torch.stack([v.squeeze(), v_x.squeeze()])
    res_batch = pg_math.quat_rotate(q_batch, v_batch)
    expected = torch.stack([v.squeeze(), torch.tensor([0.0, 1.0, 0.0])])
    torch.testing.assert_close(res_batch, expected, atol=1e-6, rtol=1e-6)

    # Mixed precision support.
    q_float = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    v_double = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    res_mixed = pg_math.quat_rotate(q_float, v_double)
    self.assertEqual(res_mixed.dtype, torch.float32)
    torch.testing.assert_close(
        res_mixed, v_double.float(), atol=1e-6, rtol=1e-6
    )

  def test_quat_rotate_inverse(self):
    # Identity quaternion inverse should leave vector unchanged.
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    res = pg_math.quat_rotate_inverse(q_id, v)
    torch.testing.assert_close(res, v, atol=1e-6, rtol=1e-6)

    # 90 deg rotation around Z: inverse should rotate [0, 1, 0] -> [1, 0, 0]
    q_90_z = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    v_y = torch.tensor([[0.0, 1.0, 0.0]])
    res = pg_math.quat_rotate_inverse(q_90_z, v_y)
    torch.testing.assert_close(
        res, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6
    )

    # Mixed precision support.
    q_float = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    v_double = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    res_mixed = pg_math.quat_rotate_inverse(q_float, v_double)
    self.assertEqual(res_mixed.dtype, torch.float32)
    torch.testing.assert_close(
        res_mixed, v_double.float(), atol=1e-6, rtol=1e-6
    )

  def test_quat_rotate_inverse_consistency(self):
    # quat_rotate_inverse(q, quat_rotate(q, v)) == v
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # unit quaternion
    v = torch.tensor([[1.0, 2.0, 3.0]])
    rotated = pg_math.quat_rotate(q, v)
    recovered = pg_math.quat_rotate_inverse(q, rotated)
    torch.testing.assert_close(recovered, v, atol=1e-5, rtol=1e-5)

  def test_quat_conjugate(self):
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    conj = pg_math.quat_conjugate(q)
    torch.testing.assert_close(
        conj, torch.tensor([0.5, -0.5, -0.5, -0.5]), atol=1e-6, rtol=1e-6
    )

    # Batch support.
    q_batch = torch.stack([q, torch.tensor([1.0, 0.0, 0.0, 0.0])])
    conj_batch = pg_math.quat_conjugate(q_batch)
    expected = torch.stack([
        torch.tensor([0.5, -0.5, -0.5, -0.5]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    ])
    torch.testing.assert_close(conj_batch, expected, atol=1e-6, rtol=1e-6)

  def test_quat_error_magnitude(self):
    # Same quaternion => near-zero error (clamped to ~0.00283 by 1-1e-6 floor).
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    err = pg_math.quat_error_magnitude(q, q)
    torch.testing.assert_close(err, torch.tensor([0.0]), atol=3e-3, rtol=1.0)

    # 90 deg rotation around Z vs identity => pi/2 error.
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_90_z = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    err = pg_math.quat_error_magnitude(q_id, q_90_z)
    torch.testing.assert_close(
        err, torch.tensor([math.pi / 2]), atol=1e-5, rtol=1e-5
    )

  def test_quat_and_rotation_conversions(self):
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    r = pg_math.quat_to_matrix(q)
    expected_r = torch.eye(3).unsqueeze(0)
    torch.testing.assert_close(r, expected_r)

    d6 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    r_from_6d = pg_math.rotation_6d_to_matrix(d6)
    torch.testing.assert_close(r_from_6d, expected_r)

    d6_recon = pg_math.rot_matrix_to_6d(r_from_6d)
    torch.testing.assert_close(d6_recon, d6)

  def test_matrix_to_quat(self):
    r = torch.eye(3)
    q = pg_math.matrix_to_quat(r)
    expected_q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(q, expected_q)

    r_batched = torch.eye(3).repeat(2, 1, 1)
    q_batched = pg_math.matrix_to_quat(r_batched)
    expected_q_batched = expected_q.repeat(2, 1)
    torch.testing.assert_close(q_batched, expected_q_batched)

    q_orig = torch.randn(10, 4)
    q_orig = q_orig / torch.linalg.norm(q_orig, dim=-1, keepdim=True)
    q_orig = torch.where(q_orig[:, 0:1] < 0, -q_orig, q_orig)

    r_orig = pg_math.quat_to_matrix(q_orig)
    q_recon = pg_math.matrix_to_quat(r_orig)
    q_recon = torch.where(q_recon[:, 0:1] < 0, -q_recon, q_recon)

    torch.testing.assert_close(q_recon, q_orig, atol=1e-5, rtol=1e-5)

    r_np = r_orig.numpy()
    r_scipy = R.from_matrix(r_np)
    q_scipy = r_scipy.as_quat()
    q_scipy_wxyz = np.stack(
        [q_scipy[:, 3], q_scipy[:, 0], q_scipy[:, 1], q_scipy[:, 2]], axis=-1
    )
    q_scipy_wxyz = torch.from_numpy(q_scipy_wxyz).to(q_recon.dtype)
    q_scipy_wxyz = torch.where(
        q_scipy_wxyz[:, 0:1] < 0, -q_scipy_wxyz, q_scipy_wxyz
    )
    torch.testing.assert_close(q_recon, q_scipy_wxyz, atol=1e-5, rtol=1e-5)

  def test_quat_from_euler_extrinsic_zyx(self):
    roll = torch.tensor([0.0])
    pitch = torch.tensor([0.0])
    yaw = torch.tensor([0.0])
    q = pg_math.quat_from_euler_extrinsic_zyx(roll, pitch, yaw)
    expected_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(q, expected_q)

    half_pi = torch.tensor([torch.pi / 2.0])
    zero = torch.tensor([0.0])

    q_roll = pg_math.quat_from_euler_extrinsic_zyx(half_pi, zero, zero)
    expected_roll = torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0]])
    torch.testing.assert_close(q_roll, expected_roll)

    q_pitch = pg_math.quat_from_euler_extrinsic_zyx(zero, half_pi, zero)
    expected_pitch = torch.tensor([[0.70710678, 0.0, 0.70710678, 0.0]])
    torch.testing.assert_close(q_pitch, expected_pitch)

    q_yaw = pg_math.quat_from_euler_extrinsic_zyx(zero, zero, half_pi)
    expected_yaw = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    torch.testing.assert_close(q_yaw, expected_yaw)

    q_comb = pg_math.quat_from_euler_extrinsic_zyx(zero, half_pi, half_pi)
    expected_comb = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    torch.testing.assert_close(q_comb, expected_comb)

    roll_rand = torch.randn(10)
    pitch_rand = torch.randn(10)
    yaw_rand = torch.randn(10)
    q_rand = pg_math.quat_from_euler_extrinsic_zyx(
        roll_rand, pitch_rand, yaw_rand
    )

    angles = np.stack(
        [yaw_rand.numpy(), pitch_rand.numpy(), roll_rand.numpy()], axis=-1
    )
    r_scipy = R.from_euler('zyx', angles, degrees=False)
    q_scipy = r_scipy.as_quat()
    q_scipy_wxyz = np.stack(
        [q_scipy[:, 3], q_scipy[:, 0], q_scipy[:, 1], q_scipy[:, 2]], axis=-1
    )
    q_scipy_wxyz = torch.from_numpy(q_scipy_wxyz).to(q_rand.dtype)

    q_rand = torch.where(q_rand[:, 0:1] < 0, -q_rand, q_rand)
    q_scipy_wxyz = torch.where(
        q_scipy_wxyz[:, 0:1] < 0, -q_scipy_wxyz, q_scipy_wxyz
    )
    torch.testing.assert_close(q_rand, q_scipy_wxyz, atol=1e-5, rtol=1e-5)

  def test_quat_inv_identity(self):
    """Inverse of identity quaternion is identity."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q_inv = pg_math.quat_inv(q)
    torch.testing.assert_close(
        q_inv, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )

  def test_quat_inv_unit(self):
    """For unit quaternions, inv equals conjugate; q * q_inv = identity."""
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])  # unit quaternion
    q_inv = pg_math.quat_inv(q)
    torch.testing.assert_close(
        q_inv,
        pg_math.quat_conjugate(q),
        atol=1e-6,
        rtol=1e-6,
    )
    product = pg_math.quat_mul(q, q_inv)
    torch.testing.assert_close(
        product, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )

  def test_quat_inv_non_unit(self):
    """Inverse of non-unit quaternion: q * q_inv = identity (scaled)."""
    q = torch.tensor([2.0, 1.0, 0.0, 0.0])  # non-unit
    q_inv = pg_math.quat_inv(q)
    product = pg_math.quat_mul(q, q_inv)
    torch.testing.assert_close(
        product, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5
    )

  def test_quat_inv_batch(self):
    """Batch inverse."""
    q_batch = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
    ])
    q_inv_batch = pg_math.quat_inv(q_batch)
    product = pg_math.quat_mul(q_batch, q_inv_batch)
    expected = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    torch.testing.assert_close(product, expected, atol=1e-6, rtol=1e-6)

  def test_axis_angle_from_quat_identity(self):
    """Identity quaternion maps to zero rotation vector."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    aa = pg_math.axis_angle_from_quat(q)
    torch.testing.assert_close(
        aa, torch.tensor([0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )

  def test_axis_angle_from_quat_90_deg(self):
    """90 deg around Z => axis-angle = [0, 0, pi/2]."""
    q = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
    aa = pg_math.axis_angle_from_quat(q)
    expected = torch.tensor([0.0, 0.0, math.pi / 2])
    torch.testing.assert_close(aa, expected, atol=1e-5, rtol=1e-5)

  def test_axis_angle_from_quat_90_deg_x(self):
    """90 deg around X => axis-angle = [pi/2, 0, 0]."""
    q = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0])
    aa = pg_math.axis_angle_from_quat(q)
    expected = torch.tensor([math.pi / 2, 0.0, 0.0])
    torch.testing.assert_close(aa, expected, atol=1e-5, rtol=1e-5)

  def test_axis_angle_from_quat_roundtrip(self):
    """axis_angle_to_quat(axis_angle_from_quat(q)) ≈ q."""
    n = 8
    q = torch.randn(n, 4)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    q = torch.where(q[:, 0:1] < 0, -q, q)  # positive w hemisphere
    aa = pg_math.axis_angle_from_quat(q)
    angle = torch.linalg.norm(aa, dim=-1)
    axis = aa / angle.unsqueeze(-1).clamp(min=1e-8)
    q_recon = pg_math.axis_angle_to_quat(axis, angle)
    q_recon = torch.where(q_recon[:, 0:1] < 0, -q_recon, q_recon)
    torch.testing.assert_close(q_recon, q, atol=1e-5, rtol=1e-5)

  def test_axis_angle_from_quat_scipy(self):
    """Cross-validate against scipy rotvec."""
    q_wxyz = torch.tensor([
        [0.70710678, 0.70710678, 0.0, 0.0],  # 90° X
        [0.70710678, 0.0, 0.70710678, 0.0],  # 90° Y
        [0.70710678, 0.0, 0.0, 0.70710678],  # 90° Z
    ])
    # scipy uses xyzw convention
    q_xyzw = np.stack(
        [
            q_wxyz[:, 1].numpy(),
            q_wxyz[:, 2].numpy(),
            q_wxyz[:, 3].numpy(),
            q_wxyz[:, 0].numpy(),
        ],
        axis=-1,
    )
    expected_aa = torch.from_numpy(R.from_quat(q_xyzw).as_rotvec()).float()
    aa = pg_math.axis_angle_from_quat(q_wxyz)
    torch.testing.assert_close(aa, expected_aa, atol=1e-5, rtol=1e-5)

  def test_quat_box_minus_identity(self):
    """Box-minus of identical quaternions is zero."""
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    diff = pg_math.quat_box_minus(q, q)
    torch.testing.assert_close(
        diff, torch.tensor([0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )

  def test_quat_box_minus_known_diff(self):
    """90° Z rotation box-minus identity = [0, 0, pi/2]."""
    q_id = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q_90z = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
    diff = pg_math.quat_box_minus(q_90z, q_id)
    expected = torch.tensor([0.0, 0.0, math.pi / 2])
    torch.testing.assert_close(diff, expected, atol=1e-5, rtol=1e-5)

  def test_quat_box_minus_antisymmetry(self):
    """box_minus(q1, q2) ≈ -box_minus(q2, q1)."""
    q1 = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0])
    q2 = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0])
    diff12 = pg_math.quat_box_minus(q1, q2)
    diff21 = pg_math.quat_box_minus(q2, q1)
    torch.testing.assert_close(diff12, -diff21, atol=1e-5, rtol=1e-5)

  def test_quat_error_magnitude2_same(self):
    """Same quaternion => zero error."""
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    err = pg_math.quat_error_magnitude2(q, q)
    torch.testing.assert_close(err, torch.tensor([0.0]), atol=1e-6, rtol=1e-6)

  def test_quat_error_magnitude2_known_angle(self):
    """Identity vs 90° Z should give pi/2."""
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_90z = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    err = pg_math.quat_error_magnitude2(q_id, q_90z)
    torch.testing.assert_close(
        err, torch.tensor([math.pi / 2]), atol=1e-5, rtol=1e-5
    )

  def test_quat_error_magnitude2_consistency(self):
    """quat_error_magnitude2 agrees with quat_error_magnitude."""
    q1 = torch.randn(5, 4)
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)
    q2 = torch.randn(5, 4)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True)
    torch.manual_seed(42)
    err1 = pg_math.quat_error_magnitude(q1, q2)
    err2 = pg_math.quat_error_magnitude2(q1, q2)
    torch.testing.assert_close(err1, err2, atol=1e-4, rtol=1e-4)

  def test_yaw_quat_pure_yaw(self):
    """A pure yaw rotation should be returned unchanged (up to sign)."""
    yaw_angle = math.pi / 3
    q_yaw = torch.tensor(
        [math.cos(yaw_angle / 2), 0.0, 0.0, math.sin(yaw_angle / 2)]
    )
    result = pg_math.yaw_quat(q_yaw.unsqueeze(0)).squeeze(0)
    # Normalize sign for comparison
    result = result * torch.sign(result[0])
    q_yaw = q_yaw * torch.sign(q_yaw[0])
    torch.testing.assert_close(result, q_yaw, atol=1e-5, rtol=1e-5)

  def test_yaw_quat_compound_rotation(self):
    """Compound rotation keeps only yaw, discarding roll/pitch."""
    # Build a quaternion with roll=30°, pitch=45°, yaw=60°
    roll = torch.tensor([math.radians(30)])
    pitch = torch.tensor([math.radians(45)])
    yaw = torch.tensor([math.radians(60)])
    q = pg_math.quat_from_euler_xyz(roll, pitch, yaw)
    q_yaw = pg_math.yaw_quat(q)
    # The yaw-only quaternion should have zero x and y components
    torch.testing.assert_close(
        q_yaw[..., 1], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        q_yaw[..., 2], torch.tensor([0.0]), atol=1e-6, rtol=1e-6
    )

  def test_yaw_quat_batch(self):
    """Batch yaw extraction."""
    q_batch = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # identity
        [0.70710678, 0.0, 0.0, 0.70710678],  # 90° Z
    ])
    q_yaw = pg_math.yaw_quat(q_batch)
    # Identity yaw is identity
    torch.testing.assert_close(
        q_yaw[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )
    # 90° Z yaw is 90° Z
    torch.testing.assert_close(
        q_yaw[1],
        torch.tensor([0.70710678, 0.0, 0.0, 0.70710678]),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_subtract_frame_transforms_identity(self):
    """When frame 1 is the origin, frame 2's world pose is returned."""
    t01 = torch.tensor([0.0, 0.0, 0.0])
    q01 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    t02 = torch.tensor([1.0, 2.0, 3.0])
    q02 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
    t12, q12 = pg_math.subtract_frame_transforms(t01, q01, t02, q02)
    torch.testing.assert_close(t12, t02, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q12, q02, atol=1e-5, rtol=1e-5)

  def test_subtract_frame_transforms_same_frame(self):
    """When both frames are the same, result is identity transform."""
    t = torch.tensor([1.0, 2.0, 3.0])
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    t12, q12 = pg_math.subtract_frame_transforms(t, q, t, q)
    torch.testing.assert_close(
        t12, torch.tensor([0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5
    )
    # q12 should be identity (up to sign)
    product = pg_math.quat_mul(q12, pg_math.quat_conjugate(q12))
    torch.testing.assert_close(
        product, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5
    )

  def test_subtract_frame_transforms_known(self):
    """Known transform: frame 1 at origin facing +Z yaw 90°, frame 2 at (1,0,0)."""
    t01 = torch.tensor([0.0, 0.0, 0.0])
    q01 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])  # 90° Z
    t02 = torch.tensor([1.0, 0.0, 0.0])
    q02 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity
    t12, _ = pg_math.subtract_frame_transforms(t01, q01, t02, q02)
    # In frame 1 (rotated 90° CCW around Z), world +x maps to local -y
    expected_t12 = torch.tensor([0.0, -1.0, 0.0])
    torch.testing.assert_close(t12, expected_t12, atol=1e-5, rtol=1e-5)

  def test_subtract_frame_transforms_batch(self):
    """Batch frame subtraction."""
    t01 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q01 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    t02 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    q02 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    t12, _ = pg_math.subtract_frame_transforms(t01, q01, t02, q02)
    expected_t = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]])
    torch.testing.assert_close(t12, expected_t, atol=1e-5, rtol=1e-5)

  def test_sample_uniform_range_shape(self):
    """Output shape matches the requested size."""
    result = pg_math.sample_uniform_range(0.0, 1.0, (4, 3), device='cpu')
    self.assertEqual(result.shape, (4, 3))

  def test_sample_uniform_range_bounds(self):
    """All samples fall within [lower, upper)."""
    lower, upper = -2.0, 5.0
    result = pg_math.sample_uniform_range(lower, upper, (1000,), device='cpu')
    self.assertTrue(torch.all(result >= lower).item())
    self.assertTrue(torch.all(result < upper).item())

  def test_sample_uniform_range_tensor_bounds(self):
    """Per-dimension tensor bounds broadcast correctly."""
    lower = torch.tensor([0.0, 10.0, -5.0])
    upper = torch.tensor([1.0, 20.0, 5.0])
    result = pg_math.sample_uniform_range(lower, upper, (500, 3), device='cpu')
    self.assertEqual(result.shape, (500, 3))
    self.assertTrue(torch.all(result[:, 0] >= 0.0).item())
    self.assertTrue(torch.all(result[:, 0] < 1.0).item())
    self.assertTrue(torch.all(result[:, 1] >= 10.0).item())
    self.assertTrue(torch.all(result[:, 1] < 20.0).item())
    self.assertTrue(torch.all(result[:, 2] >= -5.0).item())
    self.assertTrue(torch.all(result[:, 2] < 5.0).item())

  def test_sample_uniform_range_int_size(self):
    """Int size is treated as (size,)."""
    result = pg_math.sample_uniform_range(0.0, 1.0, 10, device='cpu')
    self.assertEqual(result.shape, (10,))

  def test_sigmoids(self):
    """Test sigmoid functions (gaussian, quadratic, linear)."""
    x = torch.tensor([0.0, 0.5, 1.0])

    # Gaussian sigmoid: value at 0 is 1.0, value at 1.0 matches value_at_1
    val_g = pg_math.sigmoids(x, value_at_1=0.1, sigmoid='gaussian')
    torch.testing.assert_close(val_g[0], torch.tensor(1.0))
    torch.testing.assert_close(val_g[2], torch.tensor(0.1))

    # Quadratic sigmoid
    val_q = pg_math.sigmoids(x, value_at_1=0.1, sigmoid='quadratic')
    torch.testing.assert_close(val_q[0], torch.tensor(1.0))

    # Linear sigmoid
    val_l = pg_math.sigmoids(x, value_at_1=0.1, sigmoid='linear')
    torch.testing.assert_close(val_l[0], torch.tensor(1.0))

  def test_tolerance(self):
    """Test tolerance computation for values in/out of bounds."""
    # Values inside bounds return 1.0
    x_in = torch.tensor([0.5])
    res_in = pg_math.tolerance(x_in, bounds=(0.0, 1.0))
    torch.testing.assert_close(res_in, torch.tensor([1.0]))

    # Values outside bounds return < 1.0 when margin > 0
    x_out = torch.tensor([1.5])
    res_out = pg_math.tolerance(
        x_out, bounds=(0.0, 1.0), margin=1.0, value_at_margin=0.1
    )
    self.assertLess(res_out.item(), 1.0)


if __name__ == '__main__':
  absltest.main()
