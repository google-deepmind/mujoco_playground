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

"""Build REAF Environment from Embodied Scene config."""

from collections.abc import Mapping
from collections.abc import Sequence
import copy
import inspect
from typing import Any

import mujoco
from reaf.simulation.torch._src.builders import config as builders_config
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import clock_device
from reaf.simulation.torch._src.common import default_discount_provider
from reaf.simulation.torch._src.common import functional_observation
from reaf.simulation.torch._src.common import functional_reward
from reaf.simulation.torch._src.common import functional_termination
from reaf.simulation.torch._src.common import resetter as resetter_module
from reaf.simulation.torch._src.common import timeout_termination_checker
from reaf.simulation.torch._src.core import device_layer as device_layer_module
from reaf.simulation.torch._src.core import environment
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import task_layer as task_layer_module
from reaf.simulation.torch._src.core import types
import torch
import tree

_TIME_NAME = "time"


def _instantiate_component(
    component_type: Any,
    runtime_params: Mapping[str, Any],
    **injected_kwargs: Any,
) -> Any:
  """Instantiates `component_type`, injecting framework kwargs only if accepted."""
  params = dict(runtime_params)
  sig = inspect.signature(component_type)  # pyrefly: ignore[bad-argument-type]
  accepts_var_keyword = any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
  )
  for key, value in injected_kwargs.items():
    if value is None:
      continue
    if accepts_var_keyword or key in sig.parameters:
      params[key] = value
  return component_type(**params)


def _move_tensors_to_device(
    params: dict[str, Any], device: str
) -> dict[str, Any]:
  """Recursively move torch tensors in params dict to device."""
  return tree.map_structure(
      lambda v: v.to(device) if isinstance(v, torch.Tensor) else v, params
  )


def _to_float32(v):
  """Cast float64 tensors to float32, pass through others."""
  if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
    return v.float()
  return v


def _wrap_method_for_float32(obj, method_name: str, cast_output: bool = False):
  """Wrap a method to cast inputs (and optionally outputs) to float32."""
  original = getattr(obj, method_name)

  if cast_output:
    wrapped = lambda features: tree.map_structure(
        _to_float32, original(features)
    )
  else:
    wrapped = lambda features: original(
        tree.map_structure(_to_float32, features)
    )

  setattr(obj, method_name, wrapped)
  return obj


def _attachment_prefix(component: config.Component) -> str:
  """The prefix a device's MJCF gets when it is attached to the arena."""
  prefix = component.attachment_prefix
  if not prefix:
    raise ValueError(
        "attachment_prefix must be a non-empty string when spec_fn is set."
    )
  if not prefix.endswith("/"):
    prefix += "/"
  return prefix


def make_deployment_config(
    env_config: config.Environment,
    *,
    observation_groups: Sequence[str],
) -> config.Environment:
  """Strips a training config down to what a deployed policy needs.

  Deployment keeps only the MDP boundary the policy was trained against.
  Everything that exists for the learner rather than for the robot is dropped,
  and the config is pinned to a single world.

  Args:
    env_config: The training config the policy was trained with.
    observation_groups: The observation groups the policy consumes.

  Returns:
    A deployment config that can be used to test the policy in sim,
    or splice on top of a real device layer for sim2real deployment.
  """
  task_layer = copy.deepcopy(env_config.task_layer)
  task_layer.reward_provider = None
  task_layer.rewards = {}
  task_layer.terminations = {}
  task_layer.termination_checkers = {}

  observation_adapter = {}
  for group in observation_groups:
    term_group = copy.deepcopy(env_config.observation_adapter[group])
    term_group.enable_noise = False
    observation_adapter[group] = term_group

  task_layer.feature_producers = {
      k: c for k, c in task_layer.feature_producers.items() if not c.privileged
  }

  sim = copy.deepcopy(env_config.sim)
  sim.nworld = 1

  return config.Environment(
      base_scene=env_config.base_scene,
      device_layer=env_config.device_layer,
      task_layer=task_layer,
      action_space_adapter=env_config.action_space_adapter,
      observation_adapter=observation_adapter,
      resetters=env_config.resetters,
      sim=sim,
      episode_length_secs=env_config.episode_length_secs,
      nan_termination=env_config.nan_termination,
  )


def build_scene(env_config: config.Environment) -> mujoco.MjSpec:
  """Builds the full spec by calling all devices spec functions."""
  mj_spec = env_config.base_scene.copy()
  mj_spec.copy_during_attach = True
  for component in env_config.device_layer.devices.values():
    if component.spec_fn is None:
      continue
    prefix = _attachment_prefix(component)
    spec = component.spec_fn()
    if component.attachment_site:
      site = mj_spec.site(component.attachment_site)
      if site is None:
        raise ValueError(f"Site not found: {component.attachment_site}")
      mj_spec.attach(spec, prefix=prefix, site=site)
    else:
      frame = mj_spec.worldbody.add_frame()
      frame.pos = component.attachment_pos or (0.0, 0.0, 0.0)
      frame.quat = component.attachment_quat or (1.0, 0.0, 0.0, 0.0)
      mj_spec.attach(spec, prefix=prefix, frame=frame)

  builders_config.apply_sim_config(mj_spec, env_config.sim)
  return mj_spec


def build_device_layer(
    env_config: config.Environment,
) -> device_layer_module.DeviceLayer:
  """Builds only the simulated device layer."""
  nworld = env_config.sim.nworld
  xpu_device = env_config.sim.xpu_device

  mj_spec = build_scene(env_config)

  # Instantiate devices and configure spec (sim only)
  devices = [
      clock_device.ClockDevice(name="clock", time_output_name=_TIME_NAME)
  ]
  for name, component in env_config.device_layer.devices.items():
    dev = _instantiate_component(
        component.component_type,
        component.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
        name=name,
    )
    devices.append(dev)

    mj_spec = dev.configure_spec(mj_spec, **component.build_params)

  # Instantiate substep commands processors (sim only)
  substep_commands_processors = []
  for name, c in env_config.device_layer.command_processors.items():
    processor = _instantiate_component(
        c.component_type,
        c.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
        simulation_timestep=env_config.sim.simulation_timestep,
        name=name,
    )
    substep_commands_processors.append(processor)

  # Instantiate substep measurements processors (sim only)
  substep_measurements_processors = []
  for (
      name,
      c,
  ) in env_config.device_layer.substep_measurement_processors.items():
    processor = _instantiate_component(
        c.component_type,
        c.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
        simulation_timestep=env_config.sim.simulation_timestep,
        name=name,
    )
    substep_measurements_processors.append(processor)

  # Create Simulator (compiles MjModel, allocates MjData and backend buffers)
  simulator = simulator_module.Simulator(
      mj_spec=mj_spec,
      nworld=nworld,
      xpu_device=xpu_device,
      impl=types.Impl.WARP if env_config.sim.impl == "warp" else types.Impl.CPP,
      nconmax=env_config.sim.nconmax,
      njmax=env_config.sim.njmax,
      nccdmax=env_config.sim.nccdmax,
      nvmax=env_config.sim.nvmax,
  )

  # Create DeviceLayer
  device_layer = device_layer_module.DeviceLayer(
      simulator=simulator,
      devices=devices,
      control_timestep=env_config.sim.control_timestep,
      substep_commands_processors=substep_commands_processors,
      substep_measurements_processors=substep_measurements_processors,
  )
  return device_layer


def build_task_layer(
    env_config: config.Environment,
) -> task_layer_module.TaskLayer:
  """Build only the task layer."""
  nworld = env_config.sim.nworld
  xpu_device = env_config.sim.xpu_device
  impl = types.Impl(env_config.sim.impl)

  termination_terms = list(env_config.task_layer.terminations.values())
  for term in env_config.task_layer.rewards.values():
    term.params = _move_tensors_to_device(term.params, xpu_device)
  for term in termination_terms:
    term.params = _move_tensors_to_device(term.params, xpu_device)
  for component in env_config.task_layer.termination_checkers.values():
    component.runtime_params = _move_tensors_to_device(
        component.runtime_params, xpu_device
    )
  for component in env_config.task_layer.command_processors.values():
    component.runtime_params = _move_tensors_to_device(
        component.runtime_params, xpu_device
    )
  for component in env_config.task_layer.command_feature_producers.values():
    component.runtime_params = _move_tensors_to_device(
        component.runtime_params, xpu_device
    )

  # Commands and features, can be shared between real and sim.
  command_processors = []
  feature_producers = []
  for name, c in env_config.task_layer.command_feature_producers.items():
    instance = _instantiate_component(
        c.component_type,
        c.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
        name=name,
    )
    command_processors.append(instance)
    feature_producers.append(instance)
  for name, c in env_config.task_layer.command_processors.items():
    command_processors.append(
        _instantiate_component(
            c.component_type,
            c.runtime_params,
            nworld=nworld,
            xpu_device=xpu_device,
            name=name,
        )
    )
  for name, c in env_config.task_layer.feature_producers.items():
    producer = _instantiate_component(
        c.component_type,
        c.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
        name=name,
    )
    if impl == types.Impl.CPP:
      _wrap_method_for_float32(producer, "produce_features", cast_output=True)
    feature_producers.append(producer)

  termination_checkers = []
  if env_config.episode_length_secs is not None:
    termination_checkers.append(
        timeout_termination_checker.TimeoutTerminationChecker(
            xpu_device=xpu_device,
            timeout=env_config.episode_length_secs,
            time_feature_name=_TIME_NAME,
        )
    )
  if termination_terms:
    checker = functional_termination.FunctionalTerminationChecker(
        name="task_terminations",
        terms=termination_terms,
        # TODO(reaf-sim): all keys are passed torch Task Layer, but we may
        # want to revisit.
        required_keys=[],
    )
    checker.initialize(nworld=nworld, device=torch.device(xpu_device))
    termination_checkers.append(checker)  # pyrefly: ignore[bad-argument-type]
  for name, c in env_config.task_layer.termination_checkers.items():
    termination_checkers.append(
        _instantiate_component(
            c.component_type,
            c.runtime_params,
            nworld=nworld,
            xpu_device=xpu_device,
            name=name,
        )
    )

  # Rewards, can be shared between real and sim.
  reward_provider = None
  if env_config.task_layer.reward_provider is not None:
    c = env_config.task_layer.reward_provider
    reward_provider = _instantiate_component(
        c.component_type,
        c.runtime_params,
        nworld=nworld,
        xpu_device=xpu_device,
    )
  elif env_config.task_layer.rewards:
    reward_provider = functional_reward.FunctionalRewardProvider(
        name="rewards",
        terms=env_config.task_layer.rewards,
        required_keys=[],  # not required for reaf-sim
    )

  if reward_provider is not None:
    if hasattr(reward_provider, "initialize"):
      reward_provider.initialize(
          device=torch.device(xpu_device),
          dtype=torch.float32,
          dt=env_config.sim.control_timestep,
      )
    if impl == types.Impl.CPP:
      _wrap_method_for_float32(reward_provider, "compute_reward")

  task_layer = task_layer_module.TaskLayer(
      commands_processors=command_processors,
      features_producers=feature_producers,
      reward_provider=reward_provider,
      termination_checkers=termination_checkers,
      discount_provider=default_discount_provider.DefaultDiscountProvider(
          nworld=nworld, xpu_device=xpu_device
      ),
      features_observers=[],
      loggers=[],
      xpu_device=xpu_device,
      nworld=nworld,
  )
  return task_layer


def build_env(
    env_config: config.Environment,
) -> environment.Environment:
  """Build a full REAF Environment from config."""
  nworld = env_config.sim.nworld
  xpu_device = env_config.sim.xpu_device
  impl = types.Impl(env_config.sim.impl)

  device_layer = build_device_layer(env_config)
  task_layer = build_task_layer(env_config)

  # Observations, can be shared between real and sim.
  # TODO(btaba): consider moving observations into the task layer instead.
  observation_provider = functional_observation.FunctionalObservationAdapter(
      observations=env_config.observation_adapter,
      xpu_device=xpu_device,
      nworld=nworld,
  )

  if impl == types.Impl.CPP:
    _wrap_method_for_float32(
        observation_provider, "observations_from_features", cast_output=True
    )

  # Resetter, sim-specific.
  resetters = []
  for component in env_config.resetters.values():
    child = component.component_type(**component.runtime_params)
    resetters.append(child)
  if env_config.nan_termination:
    resetters.append(resetter_module.NanTerminationResetter())
  resetter = resetter_module.CompositeResetter(resetters)

  env = environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      observation_space_adapter=observation_provider,
      action_space_adapter=env_config.action_space_adapter,
      environment_reset=resetter,
  )

  return env
