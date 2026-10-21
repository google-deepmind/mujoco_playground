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

"""Common resetters for reaf-sim."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Union

from gdm_robotics.interfaces import environment as gdmr_env
import mujoco
from reaf.core import termination_checker
from reaf.simulation.torch._src.core import environment as reaf_environment
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch


def _apply_randomization_terms(
    model: torch_wrapper.TorchWrapper,
    spec: mujoco.MjSpec,
    xpu_device: Union[torch.device, str],
    terms: Sequence[tuple[Callable[..., Any], dict[str, Any]]],
    env_mask: torch.Tensor | None = None,
) -> None:
  for func, params in terms:
    if env_mask is not None:
      func(model, spec, xpu_device, env_mask=env_mask, **params)
    else:
      func(model, spec, xpu_device, **params)


class DomainRandomizationResetter(reaf_environment.EnvironmentReset):
  """Model randomization at startup and/or reset using Term functions."""

  def __init__(self, terms: Sequence | Mapping):
    if isinstance(terms, Mapping):
      terms_list = list(terms.values())
    else:
      terms_list = list(terms)

    self._startup_terms = []
    self._reset_terms = []

    for term in terms_list:
      params = dict(term.params) if term.params else {}
      trigger = params.pop("trigger", "startup")
      if trigger == "startup":
        self._startup_terms.append((term.func, params))
      elif trigger == "reset":
        self._reset_terms.append((term.func, params))
      else:
        raise ValueError(f"Invalid domain randomization trigger: '{trigger}'")

  def on_startup(self) -> None:
    """Apply each randomization term to the model."""
    if self._env is None or not self._startup_terms:
      return
    sim_device = self._env.device_layer
    _apply_randomization_terms(
        sim_device.sim.model,
        sim_device.sim.mj_spec,
        self._env.xpu_device,
        self._startup_terms,
    )

  def on_reset(
      self,
      config: gdmr_env.Options | None,
      env_mask: torch.Tensor,
  ) -> None:
    if self._env is None or not self._reset_terms:
      return
    sim_device = self._env.device_layer
    _apply_randomization_terms(
        sim_device.sim.model,
        sim_device.sim.mj_spec,
        self._env.xpu_device,
        self._reset_terms,
        env_mask=env_mask,
    )


class CompositeResetter(reaf_environment.EnvironmentReset):
  """Orchestrates multiple child resetters."""

  def __init__(self, resetters: Sequence[reaf_environment.EnvironmentReset]):
    self._resetters = list(resetters)

  def set_env(self, env: reaf_environment.Environment) -> None:
    super().set_env(env)
    for r in self._resetters:
      r.set_env(env)

  def on_startup(self) -> None:
    for r in self._resetters:
      r.on_startup()

  def on_reset(
      self,
      config: gdmr_env.ResetOptions,
      env_mask: torch.Tensor,
  ) -> None:
    for r in self._resetters:
      r.on_reset(config, env_mask)

  def on_step(self) -> None:
    for r in self._resetters:
      r.on_step()

  def check_termination(self) -> torch.Tensor | None:
    """Combine termination: any TERMINATE wins."""
    if self._env is None:
      return None
    results = []
    for r in self._resetters:
      result = r.check_termination()
      if result is not None:
        results.append(result)
    if not results:
      return None
    if len(results) == 1:
      return results[0]
    return torch.stack(results).max(dim=0).values


class NanTerminationResetter(reaf_environment.EnvironmentReset):
  """Terminates worlds whose simulator physics state is non-finite."""

  def __init__(self, name: str = "nan_termination"):
    self._name = name
    self.metrics: dict[str, torch.Tensor] = {}

  @property
  def name(self) -> str:
    return self._name

  def check_termination(self) -> torch.Tensor | None:
    """Checks for NaN or Inf in `qpos`, `qvel`, `qacc`, or `qacc_warmstart`."""
    if self._env is None:
      return None
    sim = self._env.device_layer.sim
    if sim is None:
      return None
    data = sim.data
    has_nan = ~torch.isfinite(
        data.qpos.sum(dim=-1)
        + (data.qvel + data.qacc + data.qacc_warmstart).sum(dim=-1)
    ).to(device=self._xpu_device)
    self.metrics["nan_terminated"] = has_nan.float()
    return torch.where(
        has_nan,
        termination_checker.TerminationResult.TERMINATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    ).to(dtype=torch.int32)


class TerminationResetter(reaf_environment.EnvironmentReset):
  """Sim-only termination checks using Term functions."""

  def __init__(self, terms: Sequence[types.Term]):
    self._terms = list(terms)

  def on_reset(
      self,
      config: gdmr_env.ResetOptions,
      env_mask: torch.Tensor,
  ) -> None:
    pass

  def check_termination(self) -> torch.Tensor | None:
    """Check all termination conditions, any True triggers termination."""
    if self._env is None:
      return None
    device_layer = self._env.device_layer

    terminate = torch.zeros(
        self._nworld, dtype=torch.bool, device=self._xpu_device
    )
    for term in self._terms:
      result = term.func(device_layer, **term.params)
      terminate = terminate | result

    return torch.where(
        terminate,
        termination_checker.TerminationResult.TERMINATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    ).to(dtype=torch.int32, device=self._xpu_device)
