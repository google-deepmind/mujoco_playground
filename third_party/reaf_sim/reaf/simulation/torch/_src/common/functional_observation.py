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

"""FunctionalObservationAdapter that wraps callable functions."""

from collections.abc import Mapping
from typing import Any

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import noise as noise_module
from reaf.simulation.torch._src.common import tensors
from reaf.simulation.torch._src.core import observation_space_adapter
from reaf.simulation.torch._src.core import types
import torch


def _move_tensors_to_device(
    params: dict[str, Any], device: torch.device
) -> dict[str, Any]:
  """Recursively move torch tensors in params dict to device."""
  result = {}
  for k, v in params.items():
    if isinstance(v, torch.Tensor):
      result[k] = v.to(device)
    elif isinstance(v, dict):
      result[k] = _move_tensors_to_device(v, device)
    else:
      result[k] = v
  return result


class FunctionalObservationAdapter(
    observation_space_adapter.ObservationSpaceAdapter
):
  """Wraps a set of ObsGroups as an ObservationSpaceAdapter.

  Each observation ``Obs`` is evaluated and post-processed in order:
  ``func -> bias -> noise -> scale -> offset -> clip -> history``

  * ``bias`` — optional ``noise.Noise`` sampled once per episode on init/reset
    (only applied when ``ObsGroup.enable_noise=True``).
  * ``noise`` — optional ``noise.Noise`` sampled on every step
    (only applied when ``ObsGroup.enable_noise=True``).
  * ``scale`` — optional multiplicative factor.
  * ``offset`` — optional additive offset.
  * ``clip`` — optional ``{"min": ..., "max": ...}`` clamp bounds.
  * ``history_length`` — optional int number of past steps to stack.
  """

  def __init__(
      self,
      *,
      observations: dict[str, config.ObsGroup],
      xpu_device: torch.device | str = "cpu",
      nworld: int = 1,
  ):
    self._observations = observations
    self._xpu_device = torch.device(xpu_device)
    self._nworld = nworld

    # Nested structures: group_name -> term_name -> value
    self._scales = {g: {} for g in observations}
    self._offsets = {g: {} for g in observations}
    self._clips = {g: {} for g in observations}
    self._history_buffers = {g: {} for g in observations}
    self._history_lengths = {g: {} for g in observations}
    self._modifiers = {g: {} for g in observations}
    self._bias_samplers = {g: {} for g in observations}
    self._bias_buffers = {g: {} for g in observations}
    self._bias_ops = {g: {} for g in observations}

    self._specs = {}
    self._steps_since_reset = None

    has_history = False

    for group_name, term_group in observations.items():
      group_specs = {}
      for name, term in term_group.terms.items():
        # Move parameters to device.
        term.params = _move_tensors_to_device(term.params, self._xpu_device)

        scale = term.scale
        offset = term.offset
        clip = term.clip
        history_length = term.history_length

        term_dtype = term.dtype or "float32"
        torch_dtype = types.NUMPY_TO_TORCH_DTYPE.get(
            np.dtype(term_dtype), torch.float32
        )

        if scale is not None:
          self._scales[group_name][name] = torch.as_tensor(
              scale, dtype=torch_dtype, device=self._xpu_device
          )

        if offset is not None:
          self._offsets[group_name][name] = torch.as_tensor(
              offset, dtype=torch_dtype, device=self._xpu_device
          )

        if clip is not None:
          clip_min = clip.get("min")
          clip_max = clip.get("max")
          if clip_min is not None:
            clip_min = torch.as_tensor(
                clip_min, dtype=torch_dtype, device=self._xpu_device
            )
          if clip_max is not None:
            clip_max = torch.as_tensor(
                clip_max, dtype=torch_dtype, device=self._xpu_device
            )
          self._clips[group_name][name] = (clip_min, clip_max)

        # Compute spec shape (accounting for history expansion).
        base_shape = term.output_shape
        if base_shape is None:
          raise ValueError(
              f"output_shape is required for observation term '{name}' in"
              f" group '{group_name}'."
          )
        if history_length > 0:
          if nworld is None:
            raise ValueError(
                "nworld must be provided when using history_length > 0"
                f" (term '{name}' in group '{group_name}')"
            )
          self._history_lengths[group_name][name] = history_length
          self._history_buffers[group_name][name] = torch.zeros(
              (nworld, history_length, *base_shape),
              dtype=torch_dtype,
              device=self._xpu_device,
          )
          # If history_length > 0, we stack frames along channels and
          # preserve 2D spatial shape so CNNModel can initialize Conv2D layers
          # instead of flattening into a 1D vector.
          if len(base_shape) == 3:
            adjusted_shape = (
                (1 + history_length) * base_shape[0],
                *base_shape[1:],
            )
          else:
            base_size = int(np.prod(base_shape))
            adjusted_shape = ((1 + history_length) * base_size,)
          has_history = True
        else:
          adjusted_shape = base_shape or ()

        group_specs[name] = specs.Array(shape=adjusted_shape, dtype=term_dtype)

        # Apply noise and per-episode bias if enabled for this group.
        # NOTE: While DeviceLayer-level noise is possible, we support per-term observation
        # noise for asymmetric actor-critic setups. The critic requires privileged
        # observations while the policy receives noisy observations. Applying noise
        # at the DeviceLayer level would make it impossible for the critic to access
        # privileged measurements.
        if term_group.enable_noise:
          noise_config = term.noise
          if noise_config is not None:
            self._modifiers[group_name][name] = noise_module.build_modifier(
                noise_config, self._xpu_device
            )
          bias_config = term.bias
          if bias_config is not None:
            if nworld is None:
              raise ValueError(
                  "nworld must be provided when using 'bias'"
                  f" (term '{name}' in group '{group_name}')"
              )
            bias_sampler = noise_module.build_sampler(
                bias_config, self._xpu_device
            )
            bias_op = noise_module.resolve_op(bias_config)
            self._bias_samplers[group_name][name] = bias_sampler
            self._bias_ops[group_name][name] = bias_op
            self._bias_buffers[group_name][name] = noise_module.sample_bias(
                bias_sampler,
                bias_op,
                (nworld, *base_shape),
                torch_dtype,
                self._xpu_device,
            )

      # Build spec for the group.
      # If multiple terms in group, flatten and concatenate.
      if len(term_group.terms) > 1:
        total_size = sum(
            int(np.prod(spec.shape)) for spec in group_specs.values()
        )
        first_spec = next(iter(group_specs.values()))
        self._specs[group_name] = specs.Array(
            shape=(total_size,), dtype=first_spec.dtype
        )
      elif len(term_group.terms) == 1:
        # Single term: use its spec directly (preserves shape, e.g. HWC).
        self._specs[group_name] = next(iter(group_specs.values()))
      else:
        raise ValueError(f"ObsGroup '{group_name}' has no terms.")

    # Per-env step counter since reset.
    if has_history:
      self._steps_since_reset = torch.zeros(
          nworld, dtype=torch.long, device=self._xpu_device
      )

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset history and episodic bias buffers for the given environment mask.

    Zeros the history buffer, resamples per-episode biases for masked worlds,
    and resets the steps-since-reset counter. On the next
    ``observations_from_features`` call, envs with the first step after reset
    get their entire history filled with the current observation.
    """
    for group_buffers in self._history_buffers.values():
      for buf in group_buffers.values():
        tensors.masked_fill(buf, env_mask, 0.0)
    for group_name, group_biases in self._bias_buffers.items():
      for name, buf in group_biases.items():
        new_bias = noise_module.sample_bias(
            self._bias_samplers[group_name][name],
            self._bias_ops[group_name][name],
            tuple(buf.shape),
            buf.dtype,
            self._xpu_device,
        )
        tensors.masked_fill(buf, env_mask, new_bias)
    if self._steps_since_reset is not None:
      tensors.masked_fill(self._steps_since_reset, env_mask, 0)

  def task_features_keys(self) -> set[str]:
    return set()

  def observation_spec(self) -> Mapping[str, specs.Array]:
    return {
        k: v.replace(shape=(self._nworld,) + v.shape)
        for k, v in self._specs.items()
    }

  def observations_from_features(
      self, features: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Computes observations from features (`func -> bias -> noise -> scale -> offset -> clip -> history`)."""
    obs_dict = {}
    for group_name, term_group in self._observations.items():
      group_obs = []
      for name, term in term_group.terms.items():
        value = term.func(features, **term.params)

        if name in self._bias_buffers[group_name]:
          value = noise_module.apply_bias(
              value,
              self._bias_buffers[group_name][name],
              self._bias_ops[group_name][name],
          )

        if name in self._modifiers[group_name]:
          value = self._modifiers[group_name][name](value)

        if name in self._scales[group_name]:
          value = value * self._scales[group_name][name].to(dtype=value.dtype)

        if name in self._offsets[group_name]:
          value = value + self._offsets[group_name][name].to(dtype=value.dtype)

        if name in self._clips[group_name]:
          clip_min, clip_max = self._clips[group_name][name]
          if clip_min is not None:
            clip_min = clip_min.to(dtype=value.dtype)
          if clip_max is not None:
            clip_max = clip_max.to(dtype=value.dtype)
          value = value.clip(min=clip_min, max=clip_max)

        if name in self._history_buffers[group_name]:
          buf = self._history_buffers[group_name][name]

          # Fill all history slots with current obs after reset.
          assert self._steps_since_reset is not None
          is_first_step = self._steps_since_reset == 0
          mask_shape = (-1,) + (1,) * value.ndim
          mask = is_first_step.view(mask_shape)  # pyrefly: ignore[missing-attribute]
          buf = torch.where(mask, value.unsqueeze(1), buf)

          if value.ndim == 4:
            # (B, C, H, W) + (B, history_length, C, H, W) -> (B, (1+H)*C, H, W)
            value_with_history = torch.cat([value, buf.flatten(1, 2)], dim=1)
          else:
            # Output: [current, past_0, past_1, ...] flattened.
            value_with_history = torch.cat(
                [value.unsqueeze(1), buf], dim=1
            ).flatten(start_dim=1)
          # Update buffer: shift right, insert current at position 0.
          self._history_buffers[group_name][name] = torch.cat(
              [value.unsqueeze(1), buf[:, :-1]], dim=1
          )
          value = value_with_history

        # If multiple terms in group, flatten before concatenating.
        if len(term_group.terms) > 1:
          value = value.flatten(start_dim=1)

        group_obs.append(value)

      if len(group_obs) == 1:
        obs_dict[group_name] = group_obs[0]
      else:
        obs_dict[group_name] = torch.cat(group_obs, dim=-1)

    # Increment steps-since-reset counter after all terms processed.
    if self._steps_since_reset is not None:
      self._steps_since_reset += 1

    return obs_dict
