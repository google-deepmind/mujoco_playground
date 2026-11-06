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

"""Environment configuration dataclasses.

Defines the declarative configuration for building REAF environments.
Uses a flat device-list pattern without scene-embodiment attachment logic.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Concatenate
from typing import Generic
from typing import Literal
from typing import overload
from typing import ParamSpec
from typing import Protocol
from typing import Self

import mujoco
from reaf.core import action_space_adapter as core_action_space_adapter
from reaf.simulation.torch._src.builders.config import Sim

_P = ParamSpec("_P")

MetricReduceMode = Literal["last", "max", "mean", "sum"]


def _feature(features: Mapping[str, Any], *, key: str) -> Any:
  """Extracts a named feature tensor from the features dict."""
  return features[key]


@dataclass
class Term:
  """Universal term: function + params."""

  func: Callable[..., Any]
  params: dict[str, Any] = field(default_factory=dict)
  output_shape: tuple[int, ...] | None = None

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      output_shape: tuple[int, ...] | None = None,
      **params: Any,
  ):
    self.func = func
    self.params = dict(params)
    self.output_shape = output_shape


@dataclass
class Obs(Term):
  """Observation term with explicit post-processing modifiers and kwargs."""

  bias: Any = None
  noise: Any = None
  scale: Any = None
  offset: Any = None
  clip: dict[str, Any] | None = None
  history_length: int = 0
  dtype: Any = None
  output_shape: tuple[int, ...] | None = None

  @overload
  def __init__(self, key: str) -> None:
    ...

  @overload
  def __init__(
      self,
      func: Callable[Concatenate[Any, _P], Any],
      *args: _P.args,
      **params: _P.kwargs,
  ) -> None:
    ...

  def __init__(
      self,
      func_or_key: Callable[..., Any] | str,
      *args: Any,
      **params: Any,
  ):
    if args:
      raise TypeError("Obs() only accepts keyword arguments for `func`.")
    if isinstance(func_or_key, str):
      if params:
        raise TypeError(
            f"Obs({func_or_key!r}) does not accept keyword arguments; "
            "use .configure(...) for observation modifiers."
        )
      super().__init__(_feature, key=func_or_key)
    else:
      super().__init__(func_or_key, **params)
    self.bias = None
    self.noise = None
    self.scale = None
    self.offset = None
    self.clip = None
    self.history_length = 0
    self.dtype = None

  def configure(
      self,
      *,
      bias: Any = None,
      noise: Any = None,
      scale: Any = None,
      offset: Any = None,
      clip: dict[str, Any] | None = None,
      history_length: int = 0,
      dtype: Any = None,
      output_shape: tuple[int, ...] | None = None,
  ) -> Self:
    """Configures observation post-processing modifiers and metadata."""
    if bias is not None:
      self.bias = bias
    if noise is not None:
      self.noise = noise
    if scale is not None:
      self.scale = scale
    if offset is not None:
      self.offset = offset
    if clip is not None:
      self.clip = clip
    if history_length > 0:
      self.history_length = history_length
    if dtype is not None:
      self.dtype = dtype
    if output_shape is not None:
      self.output_shape = output_shape
    return self


@dataclass
class Reward(Term):
  """Reward term with explicit weight and kwargs."""

  weight: float = 1.0

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      weight: float = 1.0,
      **params: Any,
  ):
    super().__init__(func, **params)
    self.weight = weight


@dataclass
class Metric(Term):
  """Metric term evaluated at each step and reduced over episodes."""

  reduce: MetricReduceMode = "mean"

  def __init__(
      self,
      func_or_key: Callable[..., Any] | str,
      *,
      reduce: MetricReduceMode = "mean",
      **params: Any,
  ):
    if isinstance(func_or_key, str):
      if params:
        raise TypeError(
            f"Metric({func_or_key!r}) does not accept keyword arguments."
        )
      super().__init__(_feature, key=func_or_key)
    else:
      super().__init__(func_or_key, **params)
    self.reduce = reduce


@dataclass
class ObsGroup:
  """Group of observation terms with shared settings (e.g.

  per-episode bias and per-step noise).
  """

  terms: dict[str, Obs] = field(default_factory=dict)
  enable_noise: bool = False

  def __init__(
      self,
      terms: Mapping[str, Obs] | None = None,
      *,
      enable_noise: bool = False,
      **named_terms: Obs,
  ):
    merged: dict[str, Obs] = {}
    if terms is not None:
      merged.update(terms)
    merged.update(named_terms)
    self.terms = merged
    self.enable_noise = enable_noise


_P_init = ParamSpec("_P_init")
_P_spec = ParamSpec("_P_spec")


class _Configurable(Protocol[_P_spec]):
  """Protocol capturing a device's `configure_spec` parameter specification."""

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
      *args: _P_spec.args,
      **kwargs: _P_spec.kwargs,
  ) -> mujoco.MjSpec:
    ...


@dataclass
class Component(Generic[_P_spec]):
  """Unified configuration for REAF devices, processors, etc.."""

  # The class of the component.
  component_type: type[Any]  # e.g., ActuatorDevice, CameraDevice, etc.
  # Parameters sent to configure_spec.
  build_params: dict[str, Any] = field(default_factory=dict)
  # Parameters sent to object constructor.
  runtime_params: dict[str, Any] = field(default_factory=dict)

  # Optional scene spec and attachment info; set via `attach(...)`.
  spec_fn: Callable[[], mujoco.MjSpec] | None = None
  attachment_prefix: str = ""
  attachment_site: str | None = None
  attachment_pos: tuple[float, float, float] | None = None
  attachment_quat: tuple[float, float, float, float] | None = None
  # Whether this component is simulation/learner-only and should be pruned when
  # building a deployment config via `make_deployment_config`.
  privileged: bool = False

  @overload
  def __init__(
      self: "Component[_P_spec]",
      component_type: Callable[_P_init, _Configurable[_P_spec]],
      *args: _P_init.args,
      **runtime_kwargs: _P_init.kwargs,
  ) -> None:
    ...

  @overload
  def __init__(
      self: "Component[[]]",
      component_type: Callable[_P_init, Any],
      *args: _P_init.args,
      **runtime_kwargs: _P_init.kwargs,
  ) -> None:
    ...

  def __init__(
      self,
      component_type: Any,
      *args: Any,
      **runtime_kwargs: Any,
  ):
    if args:
      raise TypeError("Component() only accepts keyword arguments.")
    self.component_type = component_type
    self.build_params = {}
    self.runtime_params = dict(runtime_kwargs)
    self.spec_fn = None
    self.attachment_prefix = ""
    self.attachment_site = None
    self.attachment_pos = None
    self.attachment_quat = None
    self.privileged = False

  def configure_spec(
      self, *args: _P_spec.args, **build_kwargs: _P_spec.kwargs
  ) -> Self:
    """Configures MjSpec build-time arguments on this component."""
    if args:
      raise TypeError("configure_spec() only accepts keyword arguments.")
    self.build_params.update(build_kwargs)
    return self

  def attach(
      self,
      spec_fn: Callable[[], mujoco.MjSpec],
      *,
      prefix: str,
      site: str | None = None,
      pos: tuple[float, float, float] | None = None,
      quat: tuple[float, float, float, float] | None = None,
  ) -> Self:
    """Configures how this device's MjSpec attaches to the base scene."""
    if not prefix:
      raise ValueError(
          "prefix must be a non-empty string when attaching a spec."
      )
    self.spec_fn = spec_fn
    self.attachment_prefix = prefix
    if site is not None:
      self.attachment_site = site
    if pos is not None:
      self.attachment_pos = pos
    if quat is not None:
      self.attachment_quat = quat
    return self

  def mark_privileged(self) -> Self:
    """Marks this component as privileged so it is pruned in deployment configs."""
    self.privileged = True
    return self


@dataclass
class TaskLayer:
  """Task logic configuration."""

  rewards: dict[str, Reward] = field(default_factory=dict)
  reward_provider: Component | None = None  # defaults to weighted sum
  metrics: dict[str, Metric] = field(default_factory=dict)
  # Stateless functional checks, grouped into a FunctionalTerminationChecker.
  terminations: dict[str, Term] = field(default_factory=dict)
  # Stateful TerminationChecker components (e.g. with internal state / reset()).
  termination_checkers: dict[str, Component] = field(default_factory=dict)
  command_processors: dict[str, Component] = field(default_factory=dict)
  feature_producers: dict[str, Component] = field(default_factory=dict)
  command_feature_producers: dict[str, Component] = field(default_factory=dict)


@dataclass
class DeviceLayer:
  """Configuration for simulation devices."""

  devices: dict[str, Component] = field(default_factory=dict)
  command_processors: dict[str, Component] = field(default_factory=dict)
  substep_measurement_processors: dict[str, Component] = field(
      default_factory=dict
  )


@dataclass
class Environment:
  """Top-level environment configuration."""

  # Scene - the arena using our lightweight spec DSL or mjspec.
  base_scene: mujoco.MjSpec

  # Configuration for simulation devices (devices, processors).
  device_layer: DeviceLayer = field(default_factory=DeviceLayer)

  # Task - the logic shared between real and sim.
  task_layer: TaskLayer = field(default_factory=TaskLayer)

  action_space_adapter: core_action_space_adapter.ActionSpaceAdapter | None = (
      None
  )
  observation_adapter: dict[str, ObsGroup] = field(default_factory=dict)

  # The all-knowing sim Gods of the experiment.
  resetters: dict[str, Component] = field(default_factory=dict)

  # Simulation parameters.
  sim: Sim = field(default_factory=Sim)

  episode_length_secs: float | None = 20.0
  nan_termination: bool = False
