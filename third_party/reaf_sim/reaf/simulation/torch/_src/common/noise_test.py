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

"""Tests for noise module."""

import typing

from absl.testing import absltest
from reaf.simulation.torch._src.common import math
from reaf.simulation.torch._src.common import noise
import torch


class BuildModifierTest(absltest.TestCase):

  def test_uniform_modifier(self):
    torch.manual_seed(42)
    cfg = noise.Noise(dist=noise.Uniform(low=1.0, high=2.0), op="add")
    modifier = noise.build_modifier(cfg, device="cpu")
    data = torch.zeros(10, 3)
    result = modifier(data)
    self.assertEqual(result.shape, (10, 3))
    self.assertTrue((result >= 1.0).all() and (result < 2.0).all())

  def test_gaussian_modifier(self):
    torch.manual_seed(42)
    cfg = noise.Noise(dist=noise.Gaussian(mean=0.0, std=1.0), op="add")
    modifier = noise.build_modifier(cfg, device="cpu")
    data = torch.zeros(10, 3)
    result = modifier(data)
    self.assertEqual(result.shape, (10, 3))

  def test_log_uniform_modifier(self):
    torch.manual_seed(42)
    cfg = noise.Noise(dist=noise.LogUniform(low=0.1, high=10.0), op="add")
    modifier = noise.build_modifier(cfg, device="cpu")
    data = torch.ones(10, 3)
    result = modifier(data)
    self.assertEqual(result.shape, (10, 3))

  def test_unknown_op_raises(self):
    cfg = noise.Noise(
        dist=noise.Uniform(low=1.0, high=2.0),
        op=typing.cast(typing.Any, "unknown"),
    )
    with self.assertRaisesRegex(ValueError, "Unknown operation: unknown"):
      noise.build_modifier(cfg, device="cpu")

  def test_uniform_tensor_bounds(self):
    torch.manual_seed(42)
    low = torch.tensor([0.0, 1.0, 2.0])
    high = torch.tensor([1.0, 2.0, 3.0])
    result = noise.sample_uniform(
        (100, 3), torch.device("cpu"), low=low, high=high
    )
    self.assertEqual(result.shape, (100, 3))
    self.assertTrue((result[:, 0] >= 0.0).all() and (result[:, 0] < 1.0).all())
    self.assertTrue((result[:, 1] >= 1.0).all() and (result[:, 1] < 2.0).all())
    self.assertTrue((result[:, 2] >= 2.0).all() and (result[:, 2] < 3.0).all())

  def test_gaussian_tensor_params(self):
    torch.manual_seed(42)
    mean = torch.tensor([0.0, 10.0, 100.0])
    std = torch.tensor([0.1, 0.1, 0.1])
    result = noise.sample_gaussian(
        (1000, 3), torch.device("cpu"), mean=mean, std=std
    )
    self.assertEqual(result.shape, (1000, 3))
    # Check means are approximately correct.
    self.assertLess(result[:, 0].mean().abs(), 0.1)
    self.assertLess((result[:, 1].mean() - 10.0).abs(), 0.1)
    self.assertLess((result[:, 2].mean() - 100.0).abs(), 0.5)

  def test_log_uniform_tensor_bounds(self):
    torch.manual_seed(42)
    low = torch.tensor([0.1, 1.0, 10.0])
    high = torch.tensor([1.0, 10.0, 100.0])
    result = noise.sample_log_uniform(
        (100, 3), torch.device("cpu"), low=low, high=high
    )
    self.assertEqual(result.shape, (100, 3))
    self.assertTrue((result[:, 0] >= 0.1).all() and (result[:, 0] < 1.0).all())
    self.assertTrue((result[:, 1] >= 1.0).all() and (result[:, 1] < 10.0).all())
    self.assertTrue(
        (result[:, 2] >= 10.0).all() and (result[:, 2] < 100.0).all()
    )

  def test_so3_uniform_quaternion_modifier(self):
    torch.manual_seed(42)
    cfg = noise.Noise(dist=noise.SO3Uniform(low=-0.05, high=0.05))
    modifier = noise.build_modifier(cfg, device="cpu")
    data = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 100)
    result = modifier(data)
    self.assertEqual(result.shape, (100, 4))
    norms = torch.linalg.norm(result, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))
    angles = 2.0 * torch.acos(torch.clamp(result[:, 0], -1.0, 1.0))
    self.assertTrue((angles <= 0.05).all())  # pyrefly: ignore[missing-attribute]

  def test_so3_gaussian_rotation_6d_modifier(self):
    torch.manual_seed(42)
    cfg = noise.Noise(dist=noise.SO3Gaussian(mean=0.0, std=0.05))
    modifier = noise.build_modifier(cfg, device="cpu")
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    data = identity_6d.repeat(100, 1)
    result = modifier(data)
    self.assertEqual(result.shape, (100, 6))
    x = result.reshape(100, 2, 3)
    c1 = x[:, 0]
    c2 = x[:, 1]
    c1_norms = torch.linalg.norm(c1, dim=-1)
    c2_norms = torch.linalg.norm(c2, dim=-1)
    dots = torch.sum(c1 * c2, dim=-1)
    torch.testing.assert_close(c1_norms, torch.ones_like(c1_norms))
    torch.testing.assert_close(c2_norms, torch.ones_like(c2_norms))
    torch.testing.assert_close(dots, torch.zeros_like(dots))


class UniformTest(absltest.TestCase):

  def test_uniform(self):
    spec = noise.uniform(-0.5, 0.25)
    self.assertEqual(spec, noise.Noise(dist=noise.Uniform(-0.5, 0.25)))

  def test_uniform_op(self):
    self.assertEqual(noise.uniform(0.3, 1.2, op="replace").op, "replace")

  def test_uniform_per_axis(self):
    specs = noise.uniform_per_axis({0: (-1.0, 1.0), 2: (0.0, 0.5)}, op="scale")
    self.assertEqual(
        specs,
        {
            0: noise.Noise(dist=noise.Uniform(-1.0, 1.0), op="scale"),
            2: noise.Noise(dist=noise.Uniform(0.0, 0.5), op="scale"),
        },
    )


class BiasTest(absltest.TestCase):

  def test_sample_and_apply_bias_ops(self):
    torch.manual_seed(0)
    for op, low, high in [
        ("add", 0.5, 1.0),
        ("scale", 2.0, 3.0),
    ]:
      cfg = noise.Noise(dist=noise.Uniform(low=low, high=high), op=op)  # pyrefly: ignore[bad-argument-type]
      resolved_op = noise.resolve_op(cfg)
      sampler = noise.build_sampler(cfg, "cpu")
      bias = noise.sample_bias(
          sampler, resolved_op, (4, 3), torch.float32, "cpu"
      )
      self.assertTrue((bias >= low).all() and (bias <= high).all())
      out = noise.apply_bias(torch.ones(4, 3), bias, resolved_op)
      if op == "add":
        torch.testing.assert_close(out, 1.0 + bias)
      else:
        torch.testing.assert_close(out, bias)

    with self.assertRaisesRegex(ValueError, "Unknown bias operation: replace"):
      cfg_replace = noise.Noise(
          dist=noise.Uniform(low=0.0, high=1.0), op="replace"
      )
      sampler_replace = noise.build_sampler(cfg_replace, "cpu")
      noise.sample_bias(
          sampler_replace, "replace", (4, 3), torch.float32, "cpu"
      )

  def test_sample_and_apply_rotate_bias_4d_and_6d(self):
    torch.manual_seed(0)
    cfg = noise.Noise(dist=noise.SO3Uniform(low=-0.1, high=0.1))
    resolved_op = noise.resolve_op(cfg)
    self.assertEqual(resolved_op, "rotate")
    sampler = noise.build_sampler(cfg, "cpu")

    # 1. Identity quaternion rotated by q_bias equals q_bias.
    q_bias = noise.sample_bias(
        sampler, resolved_op, (4, 4), torch.float32, "cpu"
    )
    q_ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    torch.testing.assert_close(
        noise.apply_bias(q_ident, q_bias, resolved_op), q_bias
    )

    # 2. Non-identity 4D quaternion vs 6D rotation equivalence:
    # Applying equivalent 4D and 6D biases to an arbitrary non-identity
    # orientation must yield the exact same rotated orientation.
    q_in = noise.sample_so3_uniform(
        (4, 4), torch.device("cpu"), low=-1.0, high=1.0
    )
    d6_in = math.rot_matrix_to_6d(math.quat_to_matrix(q_in))
    d6_bias = math.rot_matrix_to_6d(math.quat_to_matrix(q_bias))

    q_out = noise.apply_bias(q_in, q_bias, "rotate")
    d6_out = noise.apply_bias(d6_in, d6_bias, "rotate")
    torch.testing.assert_close(
        math.rot_matrix_to_6d(math.quat_to_matrix(q_out)),
        d6_out,
        atol=1e-5,
        rtol=1e-5,
    )

    # 3. Direct 6D sample_bias + apply_bias preserves SO(3) orthonormality.
    cfg_g = noise.Noise(dist=noise.SO3Gaussian(mean=0.0, std=0.08))
    sampler_g = noise.build_sampler(cfg_g, "cpu")
    d6_sampled_bias = noise.sample_bias(
        sampler_g, "rotate", (8, 6), torch.float32, "cpu"
    )
    d6_rotated = noise.apply_bias(d6_in.repeat(2, 1), d6_sampled_bias, "rotate")
    r_mat = math.rotation_6d_to_matrix(d6_rotated)
    torch.testing.assert_close(
        torch.matmul(r_mat, r_mat.transpose(-1, -2)),
        torch.eye(3).expand(8, 3, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.det(r_mat), torch.ones(8), atol=1e-5, rtol=1e-5
    )

    # 4. Unsupported feature dimension raises ValueError.
    with self.assertRaisesRegex(
        ValueError, "Rotation bias is only supported for 4D or 6D"
    ):
      noise.sample_bias(sampler, "rotate", (4, 3), torch.float32, "cpu")
    with self.assertRaisesRegex(
        ValueError, "Rotation bias is only supported for 4D or 6D"
    ):
      noise.apply_bias(torch.zeros(4, 3), torch.zeros(4, 3), "rotate")


if __name__ == "__main__":
  absltest.main()
