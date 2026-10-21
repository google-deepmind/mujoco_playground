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
"""Task layer for the Robotics Environment Authoring Framework."""

from collections.abc import Mapping
from collections.abc import Sequence
import itertools
from typing import Any
from typing import Protocol

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor as core_commands_processor
from reaf.core import features_observer as core_features_observer
from reaf.core import features_producer as core_features_producer
from reaf.core import logger as logger_module
from reaf.simulation.torch._src.common import default_discount_provider
from reaf.simulation.torch._src.common import zero_reward_provider
from reaf.simulation.torch._src.core import discount_provider as torch_discount_provider
from reaf.simulation.torch._src.core import reward_provider as torch_reward_provider
from reaf.simulation.torch._src.core import termination_checker as core_termination_checker
import torch


class _ResettableObject(Protocol):
  """Protocol for an object that can be reset."""

  def reset(self, *args, **kwargs) -> None:
    ...


class TaskLayer:
  """Torch-compatible version of the REAF TaskLayer class."""

  def __init__(  # pylint: disable=useless-parent-delegation
      self,
      *,
      commands_processors: Sequence[core_commands_processor.CommandsProcessor],
      features_producers: Sequence[core_features_producer.FeaturesProducer],
      termination_checkers: Sequence[
          core_termination_checker.TerminationChecker
      ],
      reward_provider: torch_reward_provider.RewardProvider | None = None,
      discount_provider: torch_discount_provider.DiscountProvider | None = None,
      features_observers: Sequence[
          core_features_observer.FeaturesObserver
      ] = (),
      loggers: Sequence[logger_module.Logger] = (),
      xpu_device: str = "cuda:0",
      nworld: int = 1,
  ):
    self._commands_processors = commands_processors
    self._features_producers = features_producers
    self._reward_provider = (
        reward_provider
        if reward_provider
        else zero_reward_provider.ZeroRewardProvider(
            nworld=nworld, xpu_device=xpu_device, name="zero_reward"
        )
    )
    self._termination_checkers = termination_checkers
    self._discount_provider = (
        discount_provider
        if discount_provider
        else default_discount_provider.DefaultDiscountProvider(
            nworld=nworld, xpu_device=xpu_device
        )
    )
    self._features_observers = features_observers
    self._loggers = list(loggers)

    # We make a set of all resettable objects so that these objects only get
    # their resets called once. This is important for e.g. when having a single
    # object that derives from two interfaces.
    self._resettable_objects: list[_ResettableObject] = []
    unique_ids = set()
    for resettable_object in itertools.chain(
        self._commands_processors,
        self._features_producers,
        self._termination_checkers,
        [self._reward_provider],
        [self._discount_provider],
    ):
      resettable_object_id = id(resettable_object)
      if resettable_object_id not in unique_ids:
        unique_ids.add(resettable_object_id)
        self._resettable_objects.append(resettable_object)

    self._nworld = nworld
    self._termination_buffer: Any = torch.full(
        (nworld,), fill_value=0, dtype=torch.int32, device=xpu_device
    )

    self._components_by_name = {}
    all_components = (
        list(features_producers)
        + list(commands_processors)
        + list(termination_checkers)
        + list(features_observers)
        + [reward_provider]
    )

    for c in all_components:
      name_attr = getattr(c, "name", None)
      if name_attr:
        name = name_attr() if callable(name_attr) else name_attr
        self._components_by_name[name] = c

  def commands_spec(
      self,
      device_layer_commands_spec: (
          Mapping[str, gdmr_types.AnyArraySpec] | None
      ) = None,
  ) -> Mapping[str, gdmr_types.AnyArraySpec]:
    """Returns the commands spec exposed by the task layer."""
    spec: dict[str, gdmr_types.AnyArraySpec] = dict(
        device_layer_commands_spec or {}
    )
    for processor in reversed(self._commands_processors):
      processor_produced_keys = processor.produced_commands_keys()
      spec = {
          key: value
          for key, value in spec.items()
          if key not in processor_produced_keys
      }
      for key, proc_spec in processor.consumed_commands_spec().items():
        spec[key] = proc_spec.replace(shape=(self._nworld,) + proc_spec.shape)

    return spec  # pyrefly: ignore[bad-return]

  def features_spec(
      self,
      device_layer_measurements_spec: Mapping[str, specs.Array] | None = None,
  ) -> dict[str, specs.Array]:
    """Returns the features spec exposed by the task layer."""
    spec = dict(device_layer_measurements_spec or {})
    for features_producer in self._features_producers:
      for k, v in features_producer.produced_features_spec().items():
        spec[k] = v.replace(shape=(self._nworld,) + v.shape)
    return spec

  def validate_spec(
      self,
      *,
      device_layer_commands_spec: (
          Mapping[str, gdmr_types.AnyArraySpec] | None
      ) = None,
      device_layer_measurements_spec: Mapping[str, specs.Array] | None = None,
  ) -> None:
    """Checks that the specs have consistent keys."""
    pass

  def reward_spec(self):
    """Returns the reward spec exposed by the task layer."""
    return self._reward_provider.reward_spec()

  def discount_spec(self):
    """Returns the discount spec exposed by the task layer."""
    return self._discount_provider.discount_spec()

  def perform_reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Reset the internal state of the task layer."""
    for resettable_object in self._resettable_objects:
      resettable_object.reset(env_mask)

  def check_for_termination(
      self, features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    """Checks for termination."""
    self._termination_buffer[:] = (
        core_termination_checker.TerminationResult.DO_NOT_TERMINATE
    )
    for termination_checker in self._termination_checkers:
      term = termination_checker.check_termination(features)
      self._termination_buffer = (
          core_termination_checker.TerminationResult.combine(
              self._termination_buffer,  # pyrefly: ignore[bad-argument-type]
              term,  # pyrefly: ignore[bad-argument-type]
          )
      )
    return self._termination_buffer

  def compute_reward(
      self, features: Mapping[str, torch.Tensor]
  ) -> torch_reward_provider.RewardValue:
    """Computes the reward given the features."""
    return self._reward_provider.compute_reward(features)

  def compute_discount(
      self,
      features: Mapping[str, torch.Tensor],
      terminations: torch.Tensor,
  ) -> torch.Tensor:
    """Computes the discount given the features and terminations."""
    return self._discount_provider.compute_discount(features, terminations)

  def compute_all_features(
      self, measurements: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Computes all the task layer features given the current measurements."""
    for logger in self._loggers:
      logger.record_measurements(measurements)

    current_features = dict(measurements)
    for feature_producer in self._features_producers:
      required_features = {
          k: current_features[k]
          for k in feature_producer.required_features_keys()
      }
      produced_features = feature_producer.produce_features(required_features)
      current_features.update(produced_features)

    for observer in self._features_observers:
      observer.observe_features(current_features)
    for logger in self._loggers:
      logger.record_features(current_features)

    return current_features

  def compute_final_commands(
      self, commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Computes final commands by running commands through all processors."""
    current_commands = dict(commands)
    for processor in self._commands_processors:
      consumed_commands = {
          k: current_commands.pop(k)
          for k in processor.consumed_commands_spec().keys()
      }
      produced_commands = processor.process_commands(consumed_commands)
      current_commands.update(produced_commands)
    return current_commands

  def add_logger(self, logger: logger_module.Logger) -> None:
    """Adds a logger to the task layer."""
    self._loggers.append(logger)

  def remove_logger(self, logger: logger_module.Logger) -> None:
    """Removes a logger from the task layer."""
    self._loggers.remove(logger)

  def get_component_by_name(self, name: str):
    """Retrieves a component by its name."""
    return self._components_by_name[name]

  def has_component(self, name: str) -> bool:
    """Checks if a component with the given name exists."""
    return name in self._components_by_name
