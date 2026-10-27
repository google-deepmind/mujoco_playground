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
"""The Robotics Environment Authoring Framework (REAF) Environment class."""

import abc
from collections.abc import Mapping
import enum
import functools
from typing import Any
from typing import Generic

from absl import logging
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import action_space_adapter as reaf_action_space_adapter
from reaf.core import default_observation_space_adapter
from reaf.core import environment as core_environment
from reaf.core import logger as reaf_logger
from reaf.core import observation_space_adapter as core_observation_space_adapter
from reaf.core import pass_through_action_space_adapter
from reaf.core import termination_checker
from reaf.simulation.torch._src.core import profile
from reaf.simulation.torch._src.core import protocols
from reaf.simulation.torch._src.core import reward_provider
from reaf.simulation.torch._src.core import task_layer as task_layer_module
from reaf.simulation.torch._src.core import types
import torch
import tree


class EnvironmentReset(abc.ABC, Generic[gdmr_env.ResetOptions]):
  """Reset hooks for environment lifecycle with privileged sim access.

  Hooks:

  - on_startup(): Called once at startup (e.g. domain randomization)
  - on_reset(): Called on every reset (e.g. initialize new episode targets)
  - on_step(): Called on every step (e.g. goal resampling, curriculum)
  - check_termination(): Triggers termination using privileged sim state
    (e.g. NaN detection, joint limits) — unlike TaskLayer terminators,
    this has direct access to model/data. Called on every step.

  The resetter receives full Environment access via set_env(),
  called by Environment.__init__.
  """

  _env: Any = None
  _nworld: int = 0
  _xpu_device: str | None = None

  def set_env(self, env: Any) -> None:
    """Called by Environment.__init__."""
    self._env = env
    self._nworld = env.nworld
    self._xpu_device = env.xpu_device

  def do_reset(
      self,
      config: gdmr_env.ResetOptions,
      env_mask: torch.Tensor | None,
  ) -> None:
    """Restore model defaults, reset the device layer, and call `on_reset()`.

    Model fields written through `sim.model` are tracked by the model
    `TorchWrapper`. The `Environment` then decides if it should call
    `Simulator.set_const()`, followed by a hard-coded call to `forward()`.

    Args:
      config: The reset options.
      env_mask: Boolean mask of the worlds to reset (`None` resets all).

    Raises:
      RuntimeError: If `set_env()` was not called before `do_reset()`.
    """
    if self._env is None:
      raise RuntimeError("set_env() must be called before do_reset()")

    if env_mask is None:
      env_mask = self._env._all_env_mask

    device_layer = self._env.device_layer
    if device_layer.sim is not None:
      # Restore post-startup model defaults in-place (cloned once on first
      # post-startup write) so relative reset randomizations (e.g. add/scale)
      # do not accumulate across episodes.
      device_layer.sim.model.restore_defaults(env_mask)
    device_layer.reset(env_mask)
    self.on_reset(config, env_mask)

  def on_startup(self) -> None:
    """Called once after set_env(). Domain randomization."""
    pass

  def on_reset(
      self,
      config: gdmr_env.ResetOptions,
      env_mask: torch.Tensor,
  ) -> None:
    """User-implemented: called in do_reset."""

  def on_step(self) -> None:
    """Called each step. Mid-episode resampling, curriculum."""
    pass

  def check_termination(self) -> torch.Tensor | None:
    """Check for termination using privileged sim state."""
    return None

  def default_reset_configuration(self) -> gdmr_env.ResetOptions:
    """Returns the default reset configuration."""
    return gdmr_env.Options()  # pyrefly: ignore[bad-return]


class EndOfEpisodeHandler:
  """Handler called after the last episode step."""

  def on_end_of_episode_stepping(
      self,
      final_timestep: dm_env.TimeStep,
      env_mask: torch.Tensor | None,
  ) -> None:
    del final_timestep, env_mask  # Unused.
    ...


class AutoResetMode(enum.Enum):
  """Controls how `Environment.step()` handles episode auto resets.

  Attributes:
    SYNC: Default. Does a host-device sync check (`if is_last_timestep.any():`)
      before auto-resetting. This avoids calling `reset_with_options()` if no
      envs need resetting, avoiding expensive `mj_forward` calls that default
      resetters (e.g. `EnvironmentReset`) execute.
    NO_SYNC: Calls `self.reset_with_options(env_mask=...)` unconditionally at
      the end of every `step()`. This avoids a host-device sync, but may
      invoke an expensive `mj_forward` on every step (e.g. the default
      `EnvironmentReset` implementation).
    DISABLED: Disables automatic resetting inside `Environment.step()`. The
      caller is responsible for calling
      `reset_with_options(options=..., env_mask=...)` explicitly.
      Gym/VecEnv style wrappers should use this option to avoid the default
      auto-reset and CUDA syncs in `Environment.step()`.
  """

  SYNC = "sync"  # Default. Does a host-device sync check before auto-reset.
  NO_SYNC = "no_sync"  # Unconditionally auto-resets without host-device sync.
  DISABLED = "disabled"  # Disables auto-reset in `Environment.step()`.


class Environment(core_environment.Environment):
  """The Robotics Environment Authoring Framework (REAF) Environment class.

  TODO(b/513593452, btaba): Bake in TaskState protocol and get_state()/set_state()
  public APIs on Environment and TaskLayer to avoid ad-hoc goal state poking.
  """

  _task_layer: task_layer_module.TaskLayer
  _device_layer: protocols.TorchDeviceLayer
  _environment_reset: EnvironmentReset
  _end_of_episode_handler: EndOfEpisodeHandler
  _should_finalize_episode: torch.Tensor
  _last_timestep: dm_env.TimeStep

  def __init__(
      self,
      *,
      device_layer: protocols.TorchDeviceLayer,
      task_layer: task_layer_module.TaskLayer,
      environment_reset: EnvironmentReset,
      action_space_adapter: (
          reaf_action_space_adapter.ActionSpaceAdapter | None
      ) = None,
      observation_space_adapter: (
          core_observation_space_adapter.ObservationSpaceAdapter | None
      ) = None,
      end_of_episode_handler: EndOfEpisodeHandler | None = None,
      environment_closer: core_environment.EnvironmentCloser | None = None,
      action_spec_enforcement_option: core_environment.ActionSpecEnforcementOption = core_environment.ActionSpecEnforcementOption.RAISE_ERROR,
      auto_reset_mode: AutoResetMode = AutoResetMode.SYNC,
  ):
    self._device_layer = device_layer
    self._xpu_device = self._device_layer.xpu_device
    self._nworld = self._device_layer.nworld
    self._task_layer = task_layer
    self._end_of_episode_handler = (
        end_of_episode_handler or EndOfEpisodeHandler()
    )
    self._environment_reset = environment_reset
    self._environment_closer = environment_closer
    self._action_spec_enforcement_option = action_spec_enforcement_option
    self._auto_reset_mode = auto_reset_mode

    # Before assigning the adapters, validate the specs on the task layer
    # and the device layer.
    self._validate_device_and_task_layer_specs()

    task_layer_commands_spec = self._task_layer.commands_spec(
        self._device_layer.commands_spec()
    )
    task_layer_features_spec = self._task_layer.features_spec(
        self._device_layer.measurements_spec()
    )

    if action_space_adapter is None:
      action_space_adapter = (
          pass_through_action_space_adapter.PassThroughActionSpaceAdapter(
              commands_spec=task_layer_commands_spec
          )
      )
    self._action_space_adapter = action_space_adapter

    if observation_space_adapter is None:
      observation_space_adapter = (
          default_observation_space_adapter.DefaultObservationSpaceAdapter(
              task_features_spec=task_layer_features_spec,
              selected_features=None,
              renamed_features=None,
              observation_type_mapper=None,
          )
      )
    self._observation_space_adapter = observation_space_adapter

    # Now we can validate the adapters.
    self._validate_adapters_specs()

    self._should_finalize_episode = torch.zeros(
        self._nworld, device=self._xpu_device, dtype=torch.bool
    )

    self._timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=_batch_spec_tree(self._task_layer.reward_spec(), self._nworld),
        discount=_batch_spec_tree(
            self._task_layer.discount_spec(), self._nworld
        ),
        observation=self._observation_space_adapter.observation_spec(),
    )

    read_only_zeros_like_spec_p = functools.partial(
        _read_only_zeros_like_spec, xpu_device=self._xpu_device
    )
    self._zero_reward, self._zero_discount = tree.map_structure(
        read_only_zeros_like_spec_p,
        (self._timestep_spec.reward, self._timestep_spec.discount),
    )
    zero_observations = tree.map_structure(
        read_only_zeros_like_spec_p,
        self._timestep_spec.observation,
    )
    self._last_timestep = self._termination(
        self._zero_reward, zero_observations
    )
    self._all_env_mask = torch.ones(
        self.nworld, dtype=torch.bool, device=self._xpu_device
    )

    self._environment_reset.set_env(self)
    self._environment_reset.on_startup()
    if self._device_layer.sim is not None:
      self._device_layer.sim.set_const()
      # Clear pre-startup defaults recorded during on_startup() so post-startup
      # model values become the defaults cloned on the first reset write.
      self._device_layer.sim.model.clear_defaults()
    self._device_layer.forward()

  @property
  def xpu_device(self) -> str:
    """Returns the hardware device used by the environment."""
    return self._xpu_device

  @property
  def nworld(self) -> int:
    """Returns the number of worlds running in simulation."""
    return self._nworld

  @property
  def auto_reset_mode(self) -> AutoResetMode:
    """Returns the automatic reset mode for `step()`."""
    return self._auto_reset_mode

  @auto_reset_mode.setter
  def auto_reset_mode(self, mode: AutoResetMode) -> None:
    """Sets the automatic reset mode for `step()`."""
    self._auto_reset_mode = mode

  def default_reset_options(self) -> gdmr_env.ResetOptions:
    return self._environment_reset.default_reset_configuration()

  def action_spec(self) -> Any:
    """Defines the actions that should be provided to `step`."""
    return self._action_space_adapter.action_spec()

  def timestep_spec(self) -> gdmr_types.TimeStepSpec:
    """Returns the spec associated to the returned TimeStep."""
    return self._timestep_spec

  def _restart(  # pyrefly: ignore[bad-override]
      self,
      observation: tree.Structure[torch.Tensor],  # pyrefly: ignore[invalid-type-var]
      env_mask: torch.Tensor,
  ) -> dm_env.TimeStep:
    """Resets the TimeStep `step_type`, `reward` and `discount`."""
    return dm_env.TimeStep(
        step_type=torch.where(
            env_mask,
            dm_env.StepType.FIRST,
            self._last_timestep.step_type,
        ),
        reward=tree.map_structure(
            lambda z, r: torch.where(env_mask, z, r),
            self._zero_reward,
            self._last_timestep.reward,
        ),
        discount=torch.where(
            env_mask,
            self._zero_discount + 1.0,
            self._last_timestep.discount,
        ),
        observation=observation,
    )

  @torch.no_grad()
  def reset_with_options(
      self,
      *,
      options: gdmr_env.ResetOptions,
      env_mask: torch.Tensor | None = None,
      # TODO(btaba): Remove is_last_timestep; it is redundant with env_mask.
      is_last_timestep: torch.Tensor | None = None,
  ) -> dm_env.TimeStep:
    """Starts a new sequence and returns the first `TimeStep`."""
    if env_mask is None:
      env_mask = (
          is_last_timestep
          if is_last_timestep is not None
          else self._all_env_mask
      )

    finalizing_env_mask = self._should_finalize_episode & env_mask
    self._finalize_episode(finalizing_env_mask)

    self._environment_reset.do_reset(options, env_mask)

    if self._device_layer.sim is not None:
      # The sim optionally runs set_const if any model fields were modified
      # during reset.
      self._device_layer.sim.set_const()
    # A blanket call to forward is done since resetters modify mjState.
    self._device_layer.forward()

    self._task_layer.perform_reset(env_mask)
    obs_space_reset_fn = getattr(self._observation_space_adapter, "reset", None)
    if obs_space_reset_fn is not None:
      obs_space_reset_fn(env_mask)
    measurements = self._device_layer.begin_stepping(env_mask)
    features = self._task_layer.compute_all_features(measurements)
    observations = self._compute_observations_from_features(features)

    # Update the observations for reset environments.
    observations_w_reset = self._last_timestep.observation
    if isinstance(observations_w_reset, dict):
      for k in observations:
        mask = env_mask.view(-1, *([1] * (observations[k].ndim - 1)))
        observations_w_reset[k][:] = torch.where(
            mask, observations[k], observations_w_reset[k]
        )
    else:
      mask = env_mask.view(-1, *([1] * (observations.ndim - 1)))
      observations_w_reset[:] = torch.where(
          mask, observations, observations_w_reset
      )

    # Update step_type, reward, and discount for reset environments.
    self._last_timestep = self._restart(
        observation=observations_w_reset,
        env_mask=env_mask,
    )

    # Make sure any early reset after this one triggers `_finalize_episode`.
    self._should_finalize_episode |= env_mask
    return self._last_timestep

  @torch.no_grad()
  @profile.trace(color=profile.Section.ENVIRONMENT)
  def step(
      self, action: gdmr_types.ActionType  # pyrefly: ignore[invalid-type-var]
  ) -> dm_env.TimeStep:
    """Updates the environment according to action and returns a `TimeStep`."""
    is_last_timestep: Any = None
    if self._auto_reset_mode is not AutoResetMode.DISABLED:
      # To abide by the `dm_env.Environment` contract (`FIRST -> MID -> ... ->
      # LAST -> FIRST`), `Environment.step()` captures `is_last_timestep =
      # self._last_timestep.last()` at the start of `step()`, steps all worlds,
      # and then calls `self.reset_with_options(env_mask=is_last_timestep, ...)`
      # at the end of `step()` to reset terminated worlds and splice their
      # `StepType.FIRST` and initial reset observations into
      # `self._last_timestep`.
      is_last_timestep = self._last_timestep.last()

    # Call step reset before the actual step.
    self._environment_reset.on_step()

    # Process the action to obtain a command, then step the environment.
    commands = self._compute_commands_from_agent_action(action)
    commands = self._task_layer.compute_final_commands(commands)
    measurements = self._device_layer.step(commands)

    # Compute all the features.
    features = self._task_layer.compute_all_features(measurements)

    # Compute the elements of the timestep.
    reward = self._task_layer.compute_reward(features)
    self._task_layer.update_metrics(features)

    # Combine termination from task layer (features-based) and resetter (sim-state).
    tl_termination = self._task_layer.check_for_termination(features)
    resetter_termination = self._environment_reset.check_termination()
    if resetter_termination is not None:
      termination_state: Any = termination_checker.TerminationResult.combine(
          tl_termination,  # pyrefly: ignore[bad-argument-type]
          resetter_termination,  # pyrefly: ignore[bad-argument-type]
      )
    else:
      termination_state = tl_termination

    discount = self._task_layer.compute_discount(features, termination_state)
    observations = self._compute_observations_from_features(features)

    ts_ = termination_state
    is_terminated = ts_ == termination_checker.TerminationResult.TERMINATE
    is_truncated = ts_ == termination_checker.TerminationResult.TRUNCATE
    is_terminated_or_truncated = torch.logical_or(is_terminated, is_truncated)
    self._last_timestep.step_type[:] = torch.where(
        is_terminated_or_truncated,
        dm_env.StepType.LAST,
        dm_env.StepType.MID,
    )

    self._last_timestep = dm_env.TimeStep(
        step_type=self._last_timestep.step_type,
        reward=reward,
        discount=torch.where(is_terminated, 0.0, discount),
        observation=observations,
    )

    # Reset environments that were LAST prior to the device_layer.step,
    # splicing their FIRST reset observations and StepType.FIRST into
    # self._last_timestep to abide by the dm_env contract across vectorized
    # worlds.
    if self._auto_reset_mode is AutoResetMode.NO_SYNC:
      assert is_last_timestep is not None
      self.reset_with_options(
          options=self.default_reset_options(),
          env_mask=is_last_timestep,
      )
    elif self._auto_reset_mode is AutoResetMode.SYNC:
      assert is_last_timestep is not None
      # CUDA sync, avoids forward calls if no envs need resetting.
      if is_last_timestep.any():
        self.reset_with_options(
            options=self.default_reset_options(),
            env_mask=is_last_timestep,
        )

    return self._last_timestep

  def _finalize_episode(  # pyrefly: ignore[bad-override]
      self, env_mask: torch.Tensor
  ) -> None:
    self._device_layer.end_stepping(env_mask)
    # It's crucial to call `end_stepping` on the device layer before invoking
    # the end of episode handler. This ensures no further `set_command` or
    # `get_measurements` calls are made. In contrast, the end of episode
    # handler might interact with devices, requiring them to be informed
    # beforehand.
    self._end_of_episode_handler.on_end_of_episode_stepping(
        self._last_timestep,
        env_mask,
    )
    self._should_finalize_episode &= ~env_mask

  @property
  def device_layer(
      self,
  ) -> Any:
    # TODO(b/513593452, btaba): Migrate downstream callers
    # to protocols.TorchDeviceLayer and re-tighten this type annotation.
    return self._device_layer

  @property
  def task_layer(  # pyrefly: ignore[bad-override]
      self,
  ) -> task_layer_module.TaskLayer:
    return self._task_layer

  @property
  def environment_reset(  # pyrefly: ignore[bad-override]
      self,
  ) -> EnvironmentReset:
    return self._environment_reset

  @environment_reset.setter
  def environment_reset(self, environment_reset: EnvironmentReset) -> None:
    self._environment_reset = environment_reset

  def add_logger(self, logger: reaf_logger.Logger) -> None:
    self._task_layer.add_logger(logger)

  def remove_logger(self, logger: reaf_logger.Logger) -> None:
    self._task_layer.remove_logger(logger)

  def _validate_device_and_task_layer_specs(self) -> None:
    """Validates the specs on the task layer."""
    # Validate the spec on the task layer.
    self._task_layer.validate_spec(
        device_layer_commands_spec=(self._device_layer.commands_spec()),
        device_layer_measurements_spec=(self._device_layer.measurements_spec()),
    )

  def _validate_adapters_specs(self) -> None:
    # Collect the full commands and features spec and validate them against
    # the adapters.
    commands_spec = set(
        self._task_layer.commands_spec(
            self._device_layer.commands_spec()
        ).keys()
    )
    features_spec = set(
        self._task_layer.features_spec(self._device_layer.measurements_spec())
    )

    # Check the action space adapter.
    adapter_keys = self._action_space_adapter.task_commands_keys()

    if adapter_keys != commands_spec:
      raise ValueError(
          "Mismatch between commands exposed by the action space adapter:"
          f" {adapter_keys} and commands spec expected by the task layer:"
          f" {commands_spec}."
      )

    # Check the observation spec adapter.
    adapter_keys = self._observation_space_adapter.task_features_keys()
    if not adapter_keys.issubset(features_spec):
      raise ValueError(
          "Failed to validate observation space adapter specs. Missing keys:"
          f" {adapter_keys - features_spec}"
      )

  def _compute_observations_from_features(  # pyrefly: ignore[bad-override]
      self, features: Mapping[str, torch.Tensor]
  ) -> Any:
    return self._observation_space_adapter.observations_from_features(
        features  # pyrefly: ignore[bad-argument-type]
    )

  def _compute_commands_from_agent_action(  # pyrefly: ignore[bad-override]
      self,
      agent_action: gdmr_types.ActionType,  # pyrefly: ignore[invalid-type-var]
  ) -> Mapping[str, torch.Tensor]:
    return self._action_space_adapter.commands_from_environment_action(  # pyrefly: ignore[bad-return]
        agent_action
    )

  def _termination(  # pyrefly: ignore[bad-override]
      self,
      reward: reward_provider.RewardValue,
      observation: tree.Structure[torch.Tensor],  # pyrefly: ignore[invalid-type-var]
  ) -> dm_env.TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return dm_env.TimeStep(
        step_type=torch.full(
            (self.nworld,),
            fill_value=dm_env.StepType.LAST,
            dtype=torch.uint8,
            device=self.xpu_device,
        ),
        observation=observation,
        reward=reward,
        discount=self._zero_discount,
    )

  def _enforce_action_spec(
      self, action: gdmr_types.ActionType  # pyrefly: ignore[invalid-type-var]
  ) -> gdmr_types.ActionType:  # pyrefly: ignore[invalid-type-var]
    """Enforces the action spec."""
    match self._action_spec_enforcement_option:
      case core_environment.ActionSpecEnforcementOption.IGNORE:
        pass
      case core_environment.ActionSpecEnforcementOption.CLIP_TO_SPEC:
        try:

          def clip_to_spec(a, s):
            if isinstance(s, specs.BoundedArray):
              return np.clip(a, s.minimum, s.maximum)
            return a

          action = tree.map_structure(
              clip_to_spec,
              action,
              self._action_space_adapter.action_spec(),
          )
        except ValueError as e:
          raise ValueError(
              "Failed to clip action to spec. Action:"
              f" {action} and spec: {self._action_space_adapter.action_spec()}"
          ) from e
      case core_environment.ActionSpecEnforcementOption.WARNING:

        def _validate_without_raising(a, s):
          dtype_ok = s.dtype == a.dtype
          shape_ok = s.shape == a.shape
          minimum_ok = True
          maximum_ok = True
          if isinstance(s, specs.BoundedArray):
            minimum_ok = (s.minimum <= a).all()
            maximum_ok = (a <= s.maximum).all()
          return dtype_ok and shape_ok and minimum_ok and maximum_ok

        if not all(
            tree.flatten(
                tree.map_structure(
                    _validate_without_raising,
                    action,
                    self._action_space_adapter.action_spec(),
                )
            )
        ):
          logging.warning(
              "Failed to validate action against spec. Action: %r and spec: %r",
              action,
              self._action_space_adapter.action_spec(),
          )
      case core_environment.ActionSpecEnforcementOption.RAISE_ERROR:
        action = tree.map_structure(
            lambda a, spec: spec.validate(a), action, self.action_spec()
        )
      case _:
        raise ValueError(
            "Unknown action spec enforcement option:"
            f" {self._action_spec_enforcement_option}"
        )
    return action


def _read_only_zeros_like_spec(
    spec: specs.Array, xpu_device: str
) -> torch.Tensor | np.ndarray:
  """Returns a zero array matching the specified spec."""
  torch_dtype = types.NUMPY_TO_TORCH_DTYPE.get(spec.dtype)
  # If the dtype is not supported by torch, return a numpy array.
  # This will occur for specs with non-numeric dtypes (such as 'object').
  if torch_dtype is None:
    return np.zeros(spec.shape, dtype=spec.dtype)
  arr = torch.full(
      spec.shape, fill_value=0, dtype=torch_dtype, device=xpu_device
  )
  return arr


def _batch_spec(spec: specs.Array, nworld: int) -> specs.Array:
  """Prepends batch dimension to a spec."""
  new_shape = (nworld,) + spec.shape
  if isinstance(spec, specs.BoundedArray):
    return specs.BoundedArray(
        shape=new_shape,
        dtype=spec.dtype,
        minimum=spec.minimum,
        maximum=spec.maximum,
        name=spec.name,
    )
  return specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)


def _batch_spec_tree(spec_tree, nworld: int):
  """Prepends batch dimension to all specs in a tree structure."""
  return tree.map_structure(lambda s: _batch_spec(s, nworld), spec_tree)
