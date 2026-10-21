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

"""Shared configuration for REAF environment builders."""

from dataclasses import dataclass
from typing import Literal

import mujoco


@dataclass
class Sim:
  """Simulation parameters."""

  nworld: int = 1
  xpu_device: str = "cuda:0"
  impl: Literal["warp", "cpp"] = "warp"
  control_timestep: float = 0.02
  simulation_timestep: float = 0.005
  jacobian: mujoco.mjtJacobian | None = None
  integrator: mujoco.mjtIntegrator | None = None
  iterations: int | None = None
  ls_iterations: int | None = None
  impratio: float | None = None
  cone: mujoco.mjtCone | None = None
  solver: mujoco.mjtSolver | None = None
  gravity: tuple[float, float, float] | None = None
  ccd_iterations: int | None = None
  # warp only
  nconmax: int | None = None
  njmax: int | None = None
  nccdmax: int | None = None
  nvmax: int | None = None


def apply_sim_config(spec: mujoco.MjSpec, sim: Sim) -> mujoco.MjSpec:
  """Applies physics and solver parameters from Sim config to an MjSpec."""
  spec.option.timestep = sim.simulation_timestep
  if sim.jacobian is not None:
    spec.option.jacobian = sim.jacobian
  if sim.integrator is not None:
    spec.option.integrator = sim.integrator
  if sim.iterations is not None:
    spec.option.iterations = sim.iterations
  if sim.ls_iterations is not None:
    spec.option.ls_iterations = sim.ls_iterations
  if sim.impratio is not None:
    spec.option.impratio = sim.impratio
  if sim.cone is not None:
    spec.option.cone = sim.cone
  if sim.solver is not None:
    spec.option.solver = sim.solver
  if sim.gravity is not None:
    spec.option.gravity = sim.gravity
  if sim.ccd_iterations is not None:
    spec.option.ccd_iterations = sim.ccd_iterations
  return spec
