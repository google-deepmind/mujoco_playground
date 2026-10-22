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

"""Action history: tracks commands and exposes them as features."""

from collections.abc import Mapping
from typing import Sequence

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_producer
from reaf.simulation.torch._src.common import tensors
import torch


class ActionHistory(
    commands_processor.CommandsProcessor,
    features_producer.FeaturesProducer,
):
  """Tracks commands and exposes them as features.

  Flow:
    1. ``process_commands()``: stores raw action, passes command through.
    2. ``produce_features()``: returns current raw action and (optionally)
       past raw actions.

  Args:
    name: Unique identifier for this component.
    key: The command key to consume and pass through (e.g. ``"actions"``). Also
      used as the feature key for the current raw action.
    feature_shape: Shape of the action tensor per environment (excluding the
      batch dimension).
    nworld: Number of parallel environments.
    device: Torch device for tensor allocations.
    history_length: Number of past actions to buffer.  0 means only the current
      action feature is produced (no history).
    reset_action: Action to fill at reset with shape ``(nworld,
      *feature_shape)``. If None, defaults to zeros.
  """

  def __init__(
      self,
      name: str = "ActionHistory",
      *,
      key: str,
      feature_shape: tuple[int, ...] | Sequence[int],
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      history_length: int = 1,
      reset_action: torch.Tensor | Sequence[float] | None = None,
  ):
    self._name = name
    self._key = key
    self._feature_shape = tuple(feature_shape)
    self._nworld = nworld
    self._xpu_device = torch.device(xpu_device)
    self._history_length = history_length

    # Current raw action (always tracked).
    self._current_action = torch.zeros(
        (nworld, *self._feature_shape), device=self._xpu_device
    )

    # Action to fill at reset: [nworld, *feature_shape].
    if reset_action is not None:
      self._reset_action = torch.as_tensor(
          reset_action, device=self._xpu_device, dtype=torch.float32
      ).clone()
      expected_shape = (nworld, *self._feature_shape)
      if self._reset_action.shape != expected_shape:
        raise ValueError(
            f"reset_action has shape {self._reset_action.shape}, expected"
            f" {expected_shape}."
        )
    else:
      self._reset_action = torch.zeros(
          (nworld, *self._feature_shape), device=self._xpu_device
      )

    # History buffer: [nworld, history_length, *feature_shape]
    # Stores only PAST raw actions (index 0 = t-1, 1 = t-2, …).
    if history_length > 0:
      self._buffer = torch.zeros(
          (nworld, history_length, *self._feature_shape),
          device=self._xpu_device,
      )
      self._history_key = f"{self._key}_history"
    else:
      self._buffer = None
      self._history_key = None

  @property
  def name(self) -> str:
    return self._name

  @property
  def reset_action(self) -> torch.Tensor:
    """Action tensor of shape ``(nworld, *feature_shape)`` filled on reset."""
    return self._reset_action

  # --- CommandsProcessor ---

  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return {
        self._key: gdmr_types.UnboundedArraySpec(  # pyrefly: ignore[bad-assignment]
            shape=self._feature_shape, dtype=np.float32
        )
    }

  def produced_commands_keys(self) -> set[str]:
    return {self._key}

  def process_commands(  # pyrefly: ignore[bad-override]
      self, consumed_commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Store raw action in history, pass command through unchanged."""
    raw = consumed_commands[self._key]

    # Push previous current_action into history, then update.
    if self._buffer is not None:
      self._buffer = torch.cat(
          [self._current_action.unsqueeze(1), self._buffer[:, :-1]], dim=1
      )
    self._current_action = raw.clone()

    return {self._key: raw}

  # --- FeaturesProducer ---

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    result = {
        self._key: specs.Array(shape=self._feature_shape, dtype=np.float32)
    }
    if self._history_key is not None:
      result[self._history_key] = specs.Array(
          shape=(self._history_length, *self._feature_shape), dtype=np.float32
      )
    return result

  def required_features_keys(self) -> set[str]:
    return set()

  def produce_features(  # pyrefly: ignore[bad-override]
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Return current raw action and (optionally) past raw action history."""
    del required_features
    result = {self._key: self._current_action.clone()}
    if self._buffer is not None:
      result[self._history_key] = self._buffer.clone()  # pyrefly: ignore[unsupported-operation]
    return result

  # --- Reset ---

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset current action and history buffer for the given environments."""
    tensors.masked_fill(self._current_action, env_mask, self._reset_action)
    if self._buffer is not None:
      tensors.masked_fill(
          self._buffer, env_mask, self._reset_action[:, None, ...]
      )
