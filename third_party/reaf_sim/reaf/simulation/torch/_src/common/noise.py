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

"""Noise/distribution sampling functions for runtime sampling."""

from collections.abc import Mapping
import dataclasses
import math as _math
from typing import Callable
from typing import Literal

from reaf.simulation.torch._src.common import math
import torch


@dataclasses.dataclass
class Uniform:
  low: float | torch.Tensor
  high: float | torch.Tensor


@dataclasses.dataclass
class LogUniform:
  low: float | torch.Tensor
  high: float | torch.Tensor


@dataclasses.dataclass
class Gaussian:
  mean: float | torch.Tensor
  std: float | torch.Tensor


@dataclasses.dataclass
class SO3Uniform:
  low: float | torch.Tensor
  high: float | torch.Tensor


@dataclasses.dataclass
class SO3Gaussian:
  mean: float | torch.Tensor
  std: float | torch.Tensor


@dataclasses.dataclass
class Outlier:
  low: float | torch.Tensor
  high: float | torch.Tensor
  prob: float
  outlier_low: float | torch.Tensor
  outlier_high: float | torch.Tensor


SamplerFn = Callable[..., torch.Tensor]
Distribution = (
    Uniform | LogUniform | Gaussian | SO3Uniform | SO3Gaussian | Outlier
)
ModifierFn = Callable[..., torch.Tensor]
OperationType = Literal["add", "scale", "replace", "rotate"]


@dataclasses.dataclass
class Noise:
  dist: Distribution
  op: OperationType = "add"


def uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    op: OperationType = "add",
) -> Noise:
  """Returns a `Noise` spec sampled uniformly from `[low, high)`.

  Args:
    low: Lower bound of the uniform distribution.
    high: Upper bound of the uniform distribution.
    op: How the sample is applied to the value (`"add"`, `"scale"`,
      `"replace"`, or `"rotate"`).

  Returns:
    A `Noise` with a `Uniform` distribution.
  """
  return Noise(dist=Uniform(low=low, high=high), op=op)


def log_uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    op: OperationType = "add",
) -> Noise:
  """Returns a `Noise` spec sampled log-uniformly from `[low, high)`.

  Args:
    low: Lower bound of the log-uniform distribution.
    high: Upper bound of the log-uniform distribution.
    op: How the sample is applied to the value (`"add"`, `"scale"`, `"replace"`,
      or `"rotate"`).

  Returns:
    A `Noise` with a `LogUniform` distribution.
  """
  return Noise(dist=LogUniform(low=low, high=high), op=op)


def uniform_per_axis(
    ranges: Mapping[int, tuple[float, float]],
    op: OperationType = "add",
) -> dict[int, Noise]:
  """Returns per-axis uniform `Noise` specs, e.g. for model randomization.

  Args:
    ranges: Map from field axis index to its `(low, high)` range.
    op: How each sample is applied to the corresponding axis.

  Returns:
    A dict mapping each axis to a uniform `Noise`.
  """
  return {axis: uniform(lo, hi, op=op) for axis, (lo, hi) in ranges.items()}


def sample_uniform(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
  """Sample uniformly from [low, high)."""
  return torch.rand(shape, device=device) * (high - low) + low


def sample_gaussian(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
) -> torch.Tensor:
  """Sample from Gaussian(mean, std)."""
  return torch.randn(shape, device=device) * std + mean


def sample_log_uniform(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
  """Sample from log-uniform in [low, high)."""
  log_low = torch.log(low) if isinstance(low, torch.Tensor) else _math.log(low)
  log_high = (
      torch.log(high) if isinstance(high, torch.Tensor) else _math.log(high)
  )
  return torch.exp(
      torch.rand(shape, device=device) * (log_high - log_low) + log_low
  )


def sample_so3_uniform(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
  """Samples unit quaternions representing random uniform rotations by volume.

  Args:
    shape: The desired output shape. The last dimension is stripped and
      hardcoded to 4, since this function returns quaternions.
    device: The device to allocate the output tensors on.
    low: Lower bound of rotation angle.
    high: Upper bound of rotation angle.

  Returns:
    A tensor of unit quaternions with the given batch dims: shape[:-1].
  """
  batch_shape = shape[:-1]
  axis = torch.randn(batch_shape + (3,), device=device)
  axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-9)

  low_val = torch.as_tensor(low, device=device)
  high_val = torch.as_tensor(high, device=device)
  spans_zero = (low_val < 0.0) & (high_val > 0.0)

  a = torch.where(
      spans_zero,
      torch.zeros_like(low_val),
      torch.minimum(low_val.abs(), high_val.abs()),
  )
  b = torch.maximum(low_val.abs(), high_val.abs())

  u = torch.rand(batch_shape + (1,), device=device)
  # sample uniform 3D rotation angles in [a, b] using the small angle
  # approximation of the Haar measure (p(x) ~ x^2). The CDF is
  # F(x) = (x^3 - a^3) / (b^3 - a^3), which we sample from below.
  angle = (u * (b**3 - a**3) + a**3) ** (1.0 / 3.0)
  sin_half = torch.sin(angle * 0.5)
  return torch.cat([torch.cos(angle * 0.5), axis * sin_half], dim=-1)


def sample_so3_gaussian(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
) -> torch.Tensor:
  """Samples unit quaternions representing random Gaussian rotations in SO(3).

  Args:
    shape: The desired output shape. The last dimension is stripped and
      hardcoded to 4, since this function returns quaternions.
    device: The device to allocate the output tensors on.
    mean: Unused (mean is not supported).
    std: Standard deviation of rotation in tangent space.

  Returns:
    A tensor of unit quaternions with the given batch dims: shape[:-1].
  """
  del mean  # Mean is not supported for isotropic tangent-space Gaussian noise.
  batch_shape = shape[:-1]
  std_tensor = torch.as_tensor(std, device=device)
  vec = torch.randn(batch_shape + (3,), device=device) * std_tensor
  angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
  axis = vec / angle.clamp(min=1e-9)
  sin_half = torch.sin(angle * 0.5)
  return torch.cat([torch.cos(angle * 0.5), axis * sin_half], dim=-1)


def sample_outlier(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    prob: float,
    outlier_low: float | torch.Tensor,
    outlier_high: float | torch.Tensor,
) -> torch.Tensor:
  """Apply baseline uniform noise + occasional per-env outlier blips."""
  noise = torch.rand(shape, device=device) * (high - low) + low
  if prob > 0.0:
    mask = torch.rand(shape[0], device=device) < prob
    mask_shape = (shape[0],) + (1,) * (len(shape) - 1)
    mask = mask.view(mask_shape).to(dtype=noise.dtype)
    blip = (
        torch.rand(shape, device=device) * (outlier_high - outlier_low)
        + outlier_low
    )
    noise = noise + mask * blip
  return noise


_DATACLASS_TO_SAMPLER: dict[type, SamplerFn] = {
    Uniform: sample_uniform,
    LogUniform: sample_log_uniform,
    Gaussian: sample_gaussian,
    SO3Uniform: sample_so3_uniform,
    SO3Gaussian: sample_so3_gaussian,
    Outlier: sample_outlier,
}


def _get_sample_modifier(
    sampler: SamplerFn,
    op: OperationType = "add",
) -> ModifierFn:
  """Get a function that samples noise and applies it to data.

  Args:
    sampler: Sampler function: (shape, device, **kwargs) -> Tensor.
    op: Operation to apply - "add", "scale", or "replace".

  Returns:
    Modifier function: (data, **noise_params) -> modified_data.
  """
  if op == "add":

    def modifier(data: torch.Tensor, **kwargs) -> torch.Tensor:
      noise = sampler(data.shape, data.device, **kwargs)
      return data + noise.to(dtype=data.dtype)

  elif op == "scale":

    def modifier(data: torch.Tensor, **kwargs) -> torch.Tensor:
      noise = sampler(data.shape, data.device, **kwargs)
      return data * noise.to(dtype=data.dtype)

  elif op == "replace":

    def modifier(data: torch.Tensor, **kwargs) -> torch.Tensor:
      return sampler(data.shape, data.device, **kwargs).to(dtype=data.dtype)

  elif op == "rotate":

    def modifier(data: torch.Tensor, **kwargs) -> torch.Tensor:
      quat_shape = data.shape[:-1] + (4,)
      q_noise = sampler(quat_shape, data.device, **kwargs).to(dtype=data.dtype)
      if data.shape[-1] == 4:
        q_noisy = math.quat_mul(data, q_noise)
        return q_noisy / torch.linalg.norm(q_noisy, dim=-1, keepdim=True)
      elif data.shape[-1] == 6:
        r = math.rotation_6d_to_matrix(data)
        r_noise = math.quat_to_matrix(q_noise)
        r_noisy = torch.matmul(r, r_noise)
        return math.rot_matrix_to_6d(r_noisy)
      else:
        raise ValueError(
            "Rotation noise is only supported for 4D or 6D features, got"
            f" {data.shape}"
        )

  else:
    raise ValueError(
        f"Unknown operation: {op}. Available: add, scale, replace, rotate"
    )

  return modifier


def resolve_op(config: Noise) -> OperationType:
  """Returns the effective operation for a `Noise` config."""
  if isinstance(config.dist, (SO3Uniform, SO3Gaussian)):
    return "rotate"
  return config.op


def build_modifier(
    config: Noise,
    device: torch.device | str,
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Builds a fully-bound modifier function from a Noise dataclass.

  Moves any tensor parameters to the target device.

  Args:
    config: A Noise dataclass containing distribution and op.
    device: The target device for the modifier.

  Returns:
    A bound modifier callable: (data: Tensor) -> Tensor.
  """
  torch_device = torch.device(device) if isinstance(device, str) else device

  dist = config.dist
  op = resolve_op(config)

  if type(dist) not in _DATACLASS_TO_SAMPLER:
    raise TypeError(
        "Expected Distribution (Uniform, LogUniform, Gaussian), got"
        f" {type(dist)}"
    )

  sampler = _DATACLASS_TO_SAMPLER[type(dist)]
  modifier_fn = _get_sample_modifier(sampler, op=op)
  cfg_dict = dataclasses.asdict(dist)

  kwargs = {}
  for k, v in cfg_dict.items():
    if isinstance(v, torch.Tensor):
      v = v.to(device=torch_device)
    kwargs[k] = v

  def _bound_modifier(data: torch.Tensor) -> torch.Tensor:
    return modifier_fn(data, **kwargs)

  return _bound_modifier


def build_sampler(
    config: Noise,
    device: torch.device | str,
) -> Callable[[tuple[int, ...], torch.device], torch.Tensor]:
  """Builds a bound sampler `(shape, device) -> Tensor` from a Noise config."""
  torch_device = torch.device(device) if isinstance(device, str) else device
  dist = config.dist
  if type(dist) not in _DATACLASS_TO_SAMPLER:
    raise TypeError(
        "Expected Distribution (Uniform, LogUniform, Gaussian), got"
        f" {type(dist)}"
    )
  sampler = _DATACLASS_TO_SAMPLER[type(dist)]
  kwargs = {
      k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v
      for k, v in dataclasses.asdict(dist).items()
  }
  return lambda shape, target_device: sampler(shape, target_device, **kwargs)


def sample_bias(
    sampler: Callable[[tuple[int, ...], torch.device], torch.Tensor],
    op: OperationType,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
  """Samples a per-world episodic bias tensor of the given shape."""
  torch_device = torch.device(device) if isinstance(device, str) else device
  if op in ("add", "scale"):
    return sampler(shape, torch_device).to(dtype=dtype)
  if op == "rotate":
    if shape[-1] == 4:
      return sampler(shape, torch_device).to(dtype=dtype)
    if shape[-1] == 6:
      q_bias = sampler(shape[:-1] + (4,), torch_device).to(dtype=dtype)
      return math.rot_matrix_to_6d(math.quat_to_matrix(q_bias))
    raise ValueError(
        f"Rotation bias is only supported for 4D or 6D features, got {shape}"
    )
  raise ValueError(
      f"Unknown bias operation: {op}. Available: add, scale, rotate"
  )


def apply_bias(
    value: torch.Tensor,
    bias: torch.Tensor,
    op: OperationType,
) -> torch.Tensor:
  """Applies a pre-sampled episodic bias tensor to `value`."""
  bias = bias.to(dtype=value.dtype, device=value.device)
  if value.shape != bias.shape:
    value = value.reshape(bias.shape)
  if op == "add":
    return value + bias
  if op == "scale":
    return value * bias
  if op == "rotate":
    if value.shape[-1] == 4:
      q_noisy = math.quat_mul(value, bias)
      return q_noisy / torch.linalg.norm(q_noisy, dim=-1, keepdim=True)
    if value.shape[-1] == 6:
      r = math.rotation_6d_to_matrix(value)
      r_bias = math.rotation_6d_to_matrix(bias)
      return math.rot_matrix_to_6d(torch.matmul(r, r_bias))
    raise ValueError(
        "Rotation bias is only supported for 4D or 6D features, got"
        f" {value.shape}"
    )
  raise ValueError(
      f"Unknown bias operation: {op}. Available: add, scale, rotate"
  )
