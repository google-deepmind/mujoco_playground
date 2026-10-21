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

"""FunctionalEnvironmentReset with term-based startup/reset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from gdm_robotics.interfaces import environment as gdmr_env
from reaf.simulation.torch._src.core import environment as reaf_environment
from reaf.simulation.torch._src.core.types import Term
import torch

if TYPE_CHECKING:
  from reaf.simulation.torch._src.core import environment

# Key in params to specify term type
TYPE_KEY = "type"
TYPE_STARTUP = "startup"
TYPE_RESET = "reset"


class FunctionalEnvironmentReset(reaf_environment.EnvironmentReset):
  """EnvironmentReset with term-based startup/reset.

  Terms are filtered by params[TYPE_KEY]:
  - "startup" -> called in on_startup()
  - "reset" -> called in on_reset()

  Term functions receive device_coordinator as first arg.

  For on_step(), subclass and override.

  Example::

    def randomize_friction(dc, range):
      model = dc.simulation_device.model
      ...

    reset = FunctionalEnvironmentReset(
        terms=[
            Term(randomize_friction, type="startup", range=(0.3, 1.5)),
            Term(reset_robot, type="reset", entity="robot"),
        ],
    )
  """

  def __init__(self, terms: Sequence[Term] | None = None):
    terms = list(terms or [])
    self._startup_terms = []
    self._reset_terms = []
    for t in terms:
      params = dict(t.params)
      term_type = params.pop(TYPE_KEY, None)
      if term_type == TYPE_STARTUP:
        self._startup_terms.append((t.func, params))
      elif term_type == TYPE_RESET:
        self._reset_terms.append((t.func, params))
      else:
        raise ValueError(
            f"Invalid reset term type: {term_type!r}. Expected"
            f" {TYPE_STARTUP!r} or {TYPE_RESET!r}."
        )
    self._device_layer = None

  def set_env(self, env: "environment.Environment") -> None:
    super().set_env(env)
    self._device_layer = env.device_layer

  def on_startup(self) -> None:
    """Call startup terms."""
    if self._device_layer is None:
      raise RuntimeError("set_env() must be called before do_startup_reset()")
    for func, params in self._startup_terms:
      func(self._device_layer, **params)

  def on_reset(
      self,
      config: gdmr_env.ResetOptions,
      env_mask: torch.Tensor,
  ) -> None:
    """Call reset terms."""
    del config
    if self._device_layer is None:
      raise RuntimeError("set_env() must be called before do_reset()")

    for func, params in self._reset_terms:
      func(self._device_layer, env_mask, **params)

  def on_step(self) -> None:
    """Override in subclass for timer-based resampling."""
    pass
