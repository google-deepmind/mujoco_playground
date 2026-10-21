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

"""Processors for commands and measurements in the device layer and actions in the task layer.

Device layer processors model imperfect communication and sensing:

* ``StochasticSubstepCommands`` — **delay -> bias -> noise** (command arrives
  late at the actuator, which has a per-episode calibration bias and per-substep
  actuation noise).
* ``StochasticSubstepMeasurements`` — **bias -> noise -> delay** (sensor has a
  per-episode calibration bias and per-substep readout noise, and the reading
  arrives with a communication delay).

Task-level processors apply policy-facing action transforms:

* ``TransformActions`` — **scale -> offset -> clip** on commands sent to the
  Device Layer.  Pair with ``ActionHistory`` for raw-action feature production.
* ``DeltaActions`` — stateful action integrator:
  **pos_t = clip(pos_{t-1} + action * scale, min, max)** with optional action
  clipping and smoothing.
"""

from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
import logging
from typing import Callable

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import commands_processor
from reaf.simulation.torch._src.common import delay_buffer
from reaf.simulation.torch._src.common import noise as noise_lib
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import substep_commands_processor
from reaf.simulation.torch._src.core import substep_measurements_processor
import torch

logger = logging.getLogger(__name__)


def _resolve_delay(
    delay: tuple[float, float],
    simulation_timestep: float | None,
) -> tuple[int, int]:
  """Convert (min_seconds, max_seconds) delay to (min_lag, max_lag) in substeps."""
  if simulation_timestep is None:
    raise ValueError(
        "delay uses (min_seconds, max_seconds) but 'simulation_timestep' was"
        " not provided. The builder should inject this automatically."
    )
  min_seconds, max_seconds = delay
  min_lag = round(min_seconds / simulation_timestep)
  max_lag = round(max_seconds / simulation_timestep)
  return min_lag, max_lag


def _build_noise_and_delay_modifiers(
    keys: Sequence[str],
    delay: tuple[float, float] | None,
    noise_cfg: noise_lib.Noise | None,
    bias_cfg: noise_lib.Noise | None,
    sim_dt: float | None,
    nworld: int,
    xpu_device: str,
    name: str,
) -> tuple[
    dict[str, delay_buffer.DelayBuffer],
    dict[str, Callable[[torch.Tensor], torch.Tensor]],
    dict[str, Callable[[tuple[int, ...], torch.device], torch.Tensor]],
    dict[str, noise_lib.OperationType],
]:
  """Builds delay buffers, noise modifiers, and bias samplers."""
  buffers = {}
  modifiers = {}
  bias_samplers = {}
  bias_ops = {}

  for key in keys:
    if delay is not None:
      min_lag, max_lag = _resolve_delay(delay, sim_dt)
      if max_lag == 0:
        logger.warning(
            "%s: delay config for '%s' resolved to max_lag=0 — delay is a"
            " no-op. Check that the delay seconds are larger than"
            " simulation_timestep (%.4gs).",
            name,
            key,
            sim_dt or 0.0,
        )
      else:
        buffers[key] = delay_buffer.DelayBuffer(
            min_lag=min_lag,
            max_lag=max_lag,
            nworld=nworld,
            device=xpu_device,
        )
    if noise_cfg is not None:
      modifiers[key] = noise_lib.build_modifier(noise_cfg, xpu_device)
    if bias_cfg is not None:
      bias_samplers[key] = noise_lib.build_sampler(bias_cfg, xpu_device)
      bias_ops[key] = noise_lib.resolve_op(bias_cfg)

  return buffers, modifiers, bias_samplers, bias_ops


def _reset_bias_buffers(
    bias_buffers: dict[str, torch.Tensor],
    bias_samplers: Mapping[
        str, Callable[[tuple[int, ...], torch.device], torch.Tensor]
    ],
    bias_ops: Mapping[str, noise_lib.OperationType],
    xpu_device: str,
    env_mask: torch.Tensor | None,
) -> None:
  """Resamples already-allocated bias buffers in-place for `env_mask`."""
  for key, buf in bias_buffers.items():
    new_bias = noise_lib.sample_bias(
        bias_samplers[key],
        bias_ops[key],
        tuple(buf.shape),
        buf.dtype,
        xpu_device,
    )
    if env_mask is None:
      buf.copy_(new_bias)
    else:
      mask = env_mask.to(device=buf.device).view(-1, *([1] * (buf.ndim - 1)))
      torch.where(mask, new_bias, buf, out=buf)


class StochasticSubstepCommands(
    substep_commands_processor.SubstepCommandsProcessor
):
  """Models device command processing in order: ``delay -> bias -> noise``.

  Example::

      StochasticSubstepCommands(
          keys="robot/ctrl",
          delay=(0.005, 0.01),
          bias=noise.uniform(-0.01, 0.01),
          noise=noise.uniform(-0.002, 0.002),
          nworld=128,
          xpu_device="cuda:0",
          simulation_timestep=0.002,  # injected by the builder
      )

  Supported modifiers (applied in order ``delay -> bias -> noise``):

  * ``delay`` — optional ``(min_seconds, max_seconds)`` tuple converted to
    substep lags using ``simulation_timestep``.
  * ``bias`` — optional `noise.Noise` sampled once per episode on reset.
  * ``noise`` — optional `noise.Noise` sampled on every substep.
  """

  def __init__(
      self,
      name: str = "StochasticSubstepCommands",
      *,
      keys: str | Sequence[str],
      delay: tuple[float, float] | None = None,
      noise: noise_lib.Noise | None = None,
      bias: noise_lib.Noise | None = None,
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      simulation_timestep: float | None = None,
  ):
    self._name = name
    self._keys = [keys] if isinstance(keys, str) else list(keys)
    self._xpu_device = xpu_device

    (
        self._buffers,
        self._modifiers,
        self._bias_samplers,
        self._bias_ops,
    ) = _build_noise_and_delay_modifiers(
        self._keys,
        delay,
        noise,
        bias,
        simulation_timestep,
        nworld,
        xpu_device,
        self._name,
    )
    self._bias_buffers: dict[str, torch.Tensor] = {}

  @property
  def name(self) -> str:
    return self._name

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    for buf in self._buffers.values():
      buf.reset(env_mask)
    _reset_bias_buffers(
        self._bias_buffers,
        self._bias_samplers,
        self._bias_ops,
        self._xpu_device,
        env_mask,
    )

  def produced_substep_commands_keys(self) -> set[str]:
    return set(self._keys)

  def consumed_substep_commands_spec(self) -> Mapping[str, specs.Array]:
    # We create a placeholder spec for each key.
    # TODO(reaf-owners): consider key-only specs more widely, or establishing a
    # better pattern for how to represent these.
    return {k: specs.Array(shape=(), dtype=float) for k in self._keys}

  def process_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      substep_commands: MutableMapping[str, torch.Tensor],
  ) -> None:
    del simulator
    for key, value in list(substep_commands.items()):
      # delay
      if key in self._buffers:
        buf = self._buffers[key]
        buf.append(value)
        value = buf.read()
      # per-episode bias
      if key in self._bias_samplers:
        if key not in self._bias_buffers:
          self._bias_buffers[key] = noise_lib.sample_bias(
              self._bias_samplers[key],
              self._bias_ops[key],
              tuple(value.shape),
              value.dtype,
              self._xpu_device,
          )
        value = noise_lib.apply_bias(
            value, self._bias_buffers[key], self._bias_ops[key]
        )
      # per-substep noise
      if key in self._modifiers:
        value = self._modifiers[key](value)
      substep_commands[key] = value


class StochasticSubstepMeasurements(
    substep_measurements_processor.SubstepMeasurementsProcessor
):
  """Models device measurement processing in order: ``bias -> noise -> delay``.

  Example::

      StochasticSubstepMeasurements(
          keys=("arm_state/joint_pos", "arm_state/joint_vel"),
          bias=noise.uniform(-0.01, 0.01),
          noise=noise.uniform(-0.002, 0.002),
          delay=(0.005, 0.01),
      )

  Supported modifiers (applied in order ``bias -> noise -> delay``):

  * ``bias`` — optional `noise.Noise` sampled once per episode on reset.
  * ``noise`` — optional `noise.Noise` sampled on every substep.
  * ``delay`` — optional ``(min_seconds, max_seconds)`` tuple converted to
    substep lags using ``simulation_timestep``.

  For asymmetric actor-critic setups, prefer to apply noise in the observation
  terms for the policy only.
  """

  def __init__(
      self,
      name: str = "StochasticSubstepMeasurements",
      *,
      keys: str | Sequence[str],
      delay: tuple[float, float] | None = None,
      noise: noise_lib.Noise | None = None,
      bias: noise_lib.Noise | None = None,
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      simulation_timestep: float | None = None,
  ):
    self._name = name
    self._keys = [keys] if isinstance(keys, str) else list(keys)
    self._xpu_device = xpu_device

    (
        self._buffers,
        self._modifiers,
        self._bias_samplers,
        self._bias_ops,
    ) = _build_noise_and_delay_modifiers(
        self._keys,
        delay,
        noise,
        bias,
        simulation_timestep,
        nworld,
        xpu_device,
        self._name,
    )
    self._bias_buffers: dict[str, torch.Tensor] = {}

  @property
  def name(self) -> str:
    return self._name

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    for buf in self._buffers.values():
      buf.reset(env_mask)
    _reset_bias_buffers(
        self._bias_buffers,
        self._bias_samplers,
        self._bias_ops,
        self._xpu_device,
        env_mask,
    )

  def produced_substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {k: specs.Array(shape=(), dtype=float) for k in self._keys}

  def consumed_substep_measurements_keys(self) -> set[str]:
    return set(self._keys)

  def process_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      substep_measurements: MutableMapping[str, torch.Tensor],
  ) -> None:
    del simulator
    for key, value in list(substep_measurements.items()):
      # per-episode bias
      if key in self._bias_samplers:
        if key not in self._bias_buffers:
          self._bias_buffers[key] = noise_lib.sample_bias(
              self._bias_samplers[key],
              self._bias_ops[key],
              tuple(value.shape),
              value.dtype,
              self._xpu_device,
          )
        value = noise_lib.apply_bias(
            value, self._bias_buffers[key], self._bias_ops[key]
        )
      # per-substep noise
      if key in self._modifiers:
        value = self._modifiers[key](value)
      # delay
      if key in self._buffers:
        buf = self._buffers[key]
        buf.append(value)
        value = buf.read()
      substep_measurements[key] = value


class TransformActions(commands_processor.CommandsProcessor):
  """Task-level action transform: scale -> offset -> clip.

  A ``CommandsProcessor`` that transforms raw policy actions before they
  reach the Device Layer.  Pair with ``ActionHistory`` (which handles raw-action
  bookkeeping and feature production) in the task config.

  Example::

      TransformActions(
          input_key="actions",
          output_key="robot/ctrl",
          scale=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.044],
          offset=[0.0, -1.1, 1.0, 0.0, 0.5, 0.0, 0.022],
          clip=(-3.14, 3.14),
      )

  * ``input_key`` — command key to consume (default ``"actions"``).
  * ``output_key`` — required, the command key to produce.
  * ``scale`` — optional (default 1.0), scalar or per-dimension sequence/tensor.
  * ``offset`` — optional, additive offset after scaling.
  * ``clip`` — optional ``(min, max)`` tuple of bounds (scalar, sequence,
    tensor, or ``None`` for one-sided clipping) applied after offset.
  """

  def __init__(
      self,
      name: str = "TransformActions",
      *,
      input_key: str = "actions",
      output_key: str,
      scale: float | Sequence[float] | torch.Tensor = 1.0,
      offset: float | Sequence[float] | torch.Tensor | None = None,
      clip: (
          tuple[
              float | Sequence[float] | torch.Tensor | None,
              float | Sequence[float] | torch.Tensor | None,
          ]
          | None
      ) = None,
      xpu_device: str = "cuda:0",
  ):
    self._name = name
    self._xpu_device = xpu_device
    self._input_key = input_key
    self._output_key = output_key
    self._scale = torch.as_tensor(scale, dtype=torch.float32, device=xpu_device)
    self._offset = (
        torch.as_tensor(offset, dtype=torch.float32, device=xpu_device)
        if offset is not None
        else None
    )
    if clip is not None:
      clip_min, clip_max = clip
      self._clip_min = (
          torch.as_tensor(clip_min, dtype=torch.float32, device=xpu_device)
          if clip_min is not None
          else None
      )
      self._clip_max = (
          torch.as_tensor(clip_max, dtype=torch.float32, device=xpu_device)
          if clip_max is not None
          else None
      )
    else:
      self._clip_min = None
      self._clip_max = None

  @property
  def name(self) -> str:
    return self._name

  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    # We create a placeholder spec for the input key.
    # TODO(reaf-owners): consider key-only specs more widely, or establishing a
    # better pattern for how to represent these.
    return {
        self._input_key: gdmr_types.UnboundedArraySpec(  # pyrefly: ignore[bad-assignment]
            shape=(), dtype=float
        )
    }

  def produced_commands_keys(self) -> set[str]:
    return {self._output_key}

  def process_commands(  # pyrefly: ignore[bad-override]
      self, consumed_commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Apply scale -> offset -> clip."""
    value = consumed_commands[self._input_key]
    value = value * self._scale.to(dtype=value.dtype)
    if self._offset is not None:
      value = value + self._offset.to(dtype=value.dtype)
    if self._clip_min is not None or self._clip_max is not None:
      value = value.clip(min=self._clip_min, max=self._clip_max)
    return {self._output_key: value}

  def reset(self, env_mask=None) -> None:
    """No state to reset."""
    pass


class DeltaActions(commands_processor.CommandsProcessor):
  """Task-level delta action integrator: pos_t = clip(pos_{t-1} + action * scale, min, max).

  A stateful ``CommandsProcessor`` that integrates incoming delta actions
  starting from a configured offset (e.g. home/hover pose) and bounds them
  within clip limits.

  Example::

      DeltaActions(
          input_key="actions",
          output_key="robot/ctrl",
          scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.10],
          offset=[0.0, -0.26, 0.25, 1.13, 0.08, 0.80],
          action_clip=(-1.0, 1.0),
          smoothing=0.5,
          clip=([-3.14, ...], [3.14, ...]),
          nworld=64,
          xpu_device="cuda:0",
      )

  * ``input_key`` — command key to consume (default ``"actions"``).
  * ``output_key`` — required, the command key to produce.
  * ``scale`` — optional (default 1.0), scalar or per-dimension sequence/tensor.
  * ``offset`` — optional (default 0.0), initial pose vector or scalar.
  * ``action_clip`` — optional, clamp delta action before scaling (tuple or
    scalar).
  * ``smoothing`` — optional (default 0.0), exponential smoothing factor in
    [0, 1).
  * ``clip`` — optional ``(min, max)`` tuple of bounds (scalar, sequence,
    tensor, or ``None`` for one-sided clipping).
  """

  # TODO(b/535753795): Clean up parameters not needed once sim2real ablation is
  # finalized.

  def __init__(
      self,
      name: str = "DeltaActions",
      *,
      input_key: str = "actions",
      output_key: str,
      scale: float | Sequence[float] | torch.Tensor = 1.0,
      offset: float | Sequence[float] | torch.Tensor | None = None,
      action_clip: float | tuple[float, float] | None = None,
      smoothing: float = 0.0,
      clip: (
          tuple[
              float | Sequence[float] | torch.Tensor | None,
              float | Sequence[float] | torch.Tensor | None,
          ]
          | None
      ) = None,
      nworld: int = 1,
      xpu_device: str = "cuda:0",
  ):
    self._name = name
    self._nworld = nworld
    self._xpu_device = xpu_device
    self._input_key = input_key
    self._output_key = output_key

    self._scale = torch.as_tensor(scale, dtype=torch.float32, device=xpu_device)
    if offset is None:
      init_pos = torch.zeros(
          (nworld, *self._scale.shape), dtype=torch.float32, device=xpu_device
      )
    else:
      offset_tensor = torch.as_tensor(
          offset, dtype=torch.float32, device=xpu_device
      )
      if offset_tensor.ndim < 2:
        init_pos = offset_tensor.expand(nworld, *offset_tensor.shape).clone()
      else:
        init_pos = offset_tensor.clone()
    self._initial_pos = init_pos
    self._current_pos = init_pos.clone()

    if clip is not None:
      clip_min, clip_max = clip
      self._clip_min = (
          torch.as_tensor(clip_min, dtype=torch.float32, device=xpu_device)
          if clip_min is not None
          else None
      )
      self._clip_max = (
          torch.as_tensor(clip_max, dtype=torch.float32, device=xpu_device)
          if clip_max is not None
          else None
      )
    else:
      self._clip_min = None
      self._clip_max = None

    if action_clip is not None and isinstance(action_clip, (int, float)):
      action_clip = (-float(action_clip), float(action_clip))
    self._action_clip = action_clip
    self._smoothing = float(smoothing)
    self._prev_delta = torch.zeros_like(init_pos)

  @property
  def name(self) -> str:
    return self._name

  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return {
        self._input_key: gdmr_types.UnboundedArraySpec(  # pyrefly: ignore[bad-assignment]
            shape=(), dtype=float
        )
    }

  def produced_commands_keys(self) -> set[str]:
    return {self._output_key}

  def process_commands(  # pyrefly: ignore[bad-override]
      self, consumed_commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Integrates delta action: current_pos += delta * scale, then clips."""
    delta = consumed_commands[self._input_key]
    if self._action_clip is not None:
      delta = delta.clamp(min=self._action_clip[0], max=self._action_clip[1])
    scaled_delta = delta * self._scale
    if self._smoothing > 0.0:
      scaled_delta = (
          1.0 - self._smoothing
      ) * scaled_delta + self._smoothing * self._prev_delta
      self._prev_delta = scaled_delta.detach()
    new_pos = self._current_pos + scaled_delta
    if self._clip_min is not None or self._clip_max is not None:
      new_pos = new_pos.clamp(min=self._clip_min, max=self._clip_max)
    self._current_pos = new_pos
    return {self._output_key: new_pos}

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset integrated positions to initial offsets for given envs."""
    mask = slice(None) if env_mask is None else env_mask
    self._current_pos[mask] = self._initial_pos[mask]
    self._prev_delta[mask] = 0.0
