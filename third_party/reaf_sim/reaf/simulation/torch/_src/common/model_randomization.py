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

"""Model randomization utilities for domain randomization."""

import math as _math
from typing import Callable
from typing import Sequence
from typing import Union

import mujoco
from reaf.simulation.torch._src.common import math
from reaf.simulation.torch._src.common import noise as noise_module
from reaf.simulation.torch._src.core import mujoco_types
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types as reaf_types
import torch


def randomize_field(
    model: reaf_types.Model,
    spec: mujoco.MjSpec,
    device: torch.device,
    *,
    field: str,
    pattern: str | tuple[str, ...],
    noise: Union[noise_module.Noise, dict[int, noise_module.Noise]],
    env_mask: torch.Tensor | None = None,
    custom_post_processors: Union[
        None, Callable[..., None], Sequence[Callable[..., None]]
    ] = None,
) -> None:
  """Randomize model fields using pattern-based binding.

  Args:
    model: torch_wrapper.TorchWrapper[mjw.Model] from sim_device.model.
    spec: MjSpec for pattern matching.
    device: Torch device.
    field: Full field name (e.g., "geom_friction", "body_mass").
    pattern: Name pattern(s) to match entities (regex supported).
    noise: Noise configuration (e.g., Uniform, Gaussian) or a dict of noise
      configurations per axis.
    env_mask: Optional boolean mask of environments to randomize.
    custom_post_processors: Optional custom post-processors to apply to the
      field after randomization. If None, default post-processors are applied.
  """
  # Parse field name to get obj_type for pattern resolution.
  prefix = field.partition("_")[0]
  obj_type = mujoco_types.PREFIX_TO_OBJ_TYPE.get(prefix + "_")
  if obj_type is None:
    raise ValueError(f"Unknown field prefix in '{field}'")

  # Get bound accessor (caches pattern resolution).
  bound = model.bind_pattern(spec, obj_type, pattern)

  # Get current values using the full field name.
  current = getattr(bound, field)

  full_tensor = getattr(model, field)
  if full_tensor.ndim == 2:
    # field with shape (nworld, dim1)
    if isinstance(noise, dict):
      if 0 in noise and len(noise) == 1:
        noise = noise[0]  # convert {0: config} to config
      else:
        raise ValueError(
            f"Field '{field}' does not support per-axis noise dict: {noise}"
        )
    modifier = noise_module.build_modifier(noise, device)
    result = modifier(current)
  elif full_tensor.ndim == 3:
    # field with shape (nworld, dim1, dim2)
    if isinstance(noise, dict):
      result = current.clone()
      for axis, axis_noise in noise.items():  # randomization per-axis
        modifier = noise_module.build_modifier(axis_noise, device)
        result[..., axis] = modifier(current[..., axis])
    else:
      # same randomization for all axes
      modifier = noise_module.build_modifier(noise, device)
      result = modifier(current)
  else:
    raise ValueError(
        "randomize_field expects fields to have rank 2 or 3."
        f"Field '{field}' has rank: {full_tensor.ndim}"
    )

  # TODO: randomize option fields like gravity and wind.
  processors = custom_post_processors
  if processors is None:
    processors = _FIELD_POST_PROCESSORS.get(field, [])

  if not isinstance(processors, (list, tuple)):
    processors = [processors]

  for processor in processors:
    processor(  # pyrefly: ignore[not-callable]
        model, spec, bound, env_mask, current, result
    )

  # Write back.
  bound.set(field, result, env_mask=env_mask)


def randomize_actuator_gains(
    model: reaf_types.Model,
    spec: mujoco.MjSpec,
    device: torch.device,
    *,
    pattern: str | tuple[str, ...],
    kp_noise: noise_module.Noise | None = None,
    kd_noise: noise_module.Noise | None = None,
    env_mask: torch.Tensor | None = None,
) -> None:
  """Randomize kp/kd for affine-biastype actuators, maintaining biasprm[1]=-kp."""
  names = mujoco_types.resolve_names(
      spec, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern
  )
  biastypes = [spec.actuator(name).biastype for name in names]
  if any(b != mujoco.mjtBias.mjBIAS_AFFINE for b in biastypes):
    raise ValueError(
        "randomize_actuator_gains only supports affine biastype actuators."
    )

  bound = model.bind_pattern(spec, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern)

  gainprm, biasprm = bound.gainprm, bound.biasprm

  new_gainprm, new_biasprm = None, None
  if kp_noise is not None:
    current_kp = gainprm[..., 0]
    kp_modifier = noise_module.build_modifier(kp_noise, device)
    new_kp = kp_modifier(current_kp)
    new_gainprm = gainprm.clone()
    new_gainprm[..., 0] = new_kp
    new_biasprm = biasprm.clone()
    new_biasprm[..., 1] = -new_kp

  if kd_noise is not None:
    current_kd = -biasprm[..., 2]
    kd_modifier = noise_module.build_modifier(kd_noise, device)
    new_kd = kd_modifier(current_kd)

    new_biasprm = biasprm.clone() if new_biasprm is None else new_biasprm
    new_biasprm[..., 2] = -new_kd

  if new_gainprm is not None:
    bound.set("gainprm", new_gainprm, env_mask=env_mask)
  if new_biasprm is not None:
    bound.set("biasprm", new_biasprm, env_mask=env_mask)


# Post-processors.
def _recompute_geom_bounds(
    model: torch_wrapper.TorchWrapper,
    spec: mujoco.MjSpec,
    bound,
    env_mask: torch.Tensor | None,
    old_value: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
) -> None:
  """Recompute geom_rbound and geom_aabb from the new geom_size."""
  del model, spec, old_value  # Unused.
  if new_value is None:
    raise ValueError(
        "new_value must be provided for geom_size post-processing."
    )
  size = new_value  # (nworld, n_geoms, 3)
  is_single_geom = size.ndim == 2
  if is_single_geom:
    # TODO(reaf): Can we avoid squeeze logic by slicing differently in TorchWrapper.
    size = size.unsqueeze(1)

  s0, s1, s2 = size[..., 0], size[..., 1], size[..., 2]
  geom_type = bound.type
  is_sphere = geom_type == mujoco.mjtGeom.mjGEOM_SPHERE.value
  is_capsule = geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE.value
  is_ellipsoid = geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value
  is_cylinder = geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER.value
  is_box = geom_type == mujoco.mjtGeom.mjGEOM_BOX.value

  is_supported = is_sphere | is_capsule | is_ellipsoid | is_cylinder | is_box
  if not is_supported.all():
    raise ValueError("geom_size bounds recomputation only supports primitives")

  rbound = torch.zeros_like(s0)
  rbound = torch.where(is_sphere, s0, rbound)
  rbound = torch.where(is_capsule, s0 + s1, rbound)
  rbound = torch.where(is_ellipsoid, torch.max(size, dim=-1).values, rbound)
  rbound = torch.where(
      is_cylinder, torch.linalg.vector_norm(size[..., :2], dim=-1), rbound
  )
  rbound = torch.where(is_box, torch.linalg.vector_norm(size, dim=-1), rbound)

  # Compute aabb. Initialize with size (box, ellipsoid).
  aabb_half = size.clone()
  # Y-axis is s0 for radial primitives (Sphere, Capsule, Cylinder), otherwise s1
  aabb_half[..., 1] = torch.where(is_sphere | is_capsule | is_cylinder, s0, s1)
  # Z-axis depends on shape type
  z_val = s2.clone()
  z_val = torch.where(is_sphere, s0, z_val)
  z_val = torch.where(is_capsule, s0 + s1, z_val)
  z_val = torch.where(is_cylinder, s1, z_val)
  aabb_half[..., 2] = z_val

  aabb_full = torch.zeros(
      size.shape[0], bound.id.numel(), 2, 3, device=size.device
  )
  aabb_full[..., 1, :] = aabb_half

  if is_single_geom:
    rbound = rbound.squeeze(1)
    aabb_full = aabb_full.squeeze(1)

  bound.set("rbound", rbound, env_mask=env_mask)
  bound.set("aabb", aabb_full, env_mask=env_mask)


def _recompute_body_inertia(
    model: torch_wrapper.TorchWrapper,
    spec: mujoco.MjSpec,
    bound,
    env_mask: torch.Tensor | None,
    old_value: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
) -> None:
  """Recompute and scale body_inertia proportionally with body_mass changes."""
  del model, spec  # Unused.
  if old_value is None or new_value is None:
    raise ValueError(
        "old_value and new_value must be provided for body_mass"
        " post-processing."
    )

  has_mass = old_value > 0
  torch.where(has_mass, new_value, old_value, out=new_value)
  scale = torch.where(has_mass, new_value / old_value, 1.0)
  inertia = bound.inertia  # Shape: (nworld, n_bodies, 3)

  new_inertia = inertia * scale.unsqueeze(-1)
  bound.set("inertia", new_inertia, env_mask=env_mask)


def randomize_quat(
    model: torch_wrapper.TorchWrapper,
    spec: mujoco.MjSpec,
    device: torch.device,
    *,
    field: str,
    pattern: str | tuple[str, ...],
    max_angle_degrees: float = 3.0,
    env_mask: torch.Tensor | None = None,
) -> None:
  """Randomize quaternion orientation.

  Applies a small random rotation perturbation to the matched quaternions.

  Args:
    model: TorchWrapper[mjw.Model].
    spec: mjspec for pattern matching.
    device: torch device.
    field: quaternion field name (e.g., "cam_quat", "body_quat", "geom_quat").
    pattern: name pattern(s) to match entities.
    max_angle_degrees: maximum rotation perturbation range in degrees.
    env_mask: optional boolean mask of environments to randomize.
  """
  prefix = field.partition("_")[0]
  obj_type = mujoco_types.PREFIX_TO_OBJ_TYPE.get(prefix + "_")
  if obj_type is None:
    raise ValueError(f"Unknown field prefix in '{field}'")

  bound = model.bind_pattern(spec, obj_type, pattern)
  quat = getattr(bound, field)

  shape = quat.shape
  axis_shape = shape[:-1] + (3,)
  angle_shape = shape[:-1] + (1,)

  # Small rotation perturbation.
  q_rand_axis = torch.randn(axis_shape, device=device)
  q_rand_axis = q_rand_axis / torch.norm(
      q_rand_axis, dim=-1, keepdim=True
  ).clamp(min=1e-9)
  max_angle_rad = max_angle_degrees * torch.pi / 180.0
  u = torch.rand(angle_shape, device=device)
  # Use cube root to sample uniformly in 3D rotation space.
  q_rand_angle = (u ** (1.0 / 3.0)) * max_angle_rad * 0.5

  delta_q = torch.cat(
      [torch.cos(q_rand_angle), torch.sin(q_rand_angle) * q_rand_axis], dim=-1
  )

  new_quat = math.quat_mul(quat, delta_q)
  new_quat = new_quat / torch.norm(new_quat, dim=-1, keepdim=True)

  bound.set(field, new_quat, env_mask=env_mask)


_FIELD_POST_PROCESSORS: dict[str, list[Callable[..., None]]] = {
    "body_mass": [_recompute_body_inertia],
    "geom_size": [_recompute_geom_bounds],
}
