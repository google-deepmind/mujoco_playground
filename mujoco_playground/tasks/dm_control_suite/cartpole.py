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

"""Cartpole task configuration."""

import importlib.resources
from typing import Mapping

import mujoco
from reaf.simulation import torch as device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import functional_reset
from reaf.simulation.torch import math
import torch


def _create_base_scene() -> mujoco.MjSpec:
  xml_text = (
      importlib.resources.files(__package__)
      .joinpath("assets", "cartpole.xml")
      .read_text(encoding="utf-8")
  )
  return mujoco.MjSpec.from_string(xml_text)


def _obs_cartpole(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  qpos = features["cartpole/joint_pos"]
  qvel = features["cartpole/joint_vel"]
  return torch.stack(
      [
          qpos[:, 0],
          torch.cos(qpos[:, 1]),
          torch.sin(qpos[:, 1]),
          qvel[:, 0],
          qvel[:, 1],
      ],
      dim=-1,
  )


def _reward_dense(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  qpos = features["cartpole/joint_pos"]
  qvel = features["cartpole/joint_vel"]
  cart_pos = qpos[:, 0]
  pole_cos = torch.cos(qpos[:, 1])
  pole_vel = qvel[:, 1]
  action = features["slider_actuator/ctrl"][:, 0]

  upright = (pole_cos + 1) / 2
  centered = math.tolerance(cart_pos, margin=2.0)
  centered = (1 + centered) / 2

  small_control = math.tolerance(
      action, margin=1.0, value_at_margin=0.0, sigmoid="quadratic"
  )
  small_control = (4 + small_control) / 5

  small_velocity = math.tolerance(pole_vel, margin=5.0)
  small_velocity = (1 + small_velocity) / 2

  return upright * centered * small_control * small_velocity


def reset_cartpole(dc, env_mask: torch.Tensor, swing_up: bool) -> None:
  """Initializes cartpole episode state."""
  data = dc.sim.data
  device = data.qpos.device
  nworld = data.qpos.shape[0]

  if swing_up:
    qpos0 = 0.01 * torch.randn(nworld, device=device)
    qpos1 = torch.pi + 0.01 * torch.randn(nworld, device=device)
  else:
    qpos0 = 0.2 * torch.rand(nworld, device=device) - 0.1
    qpos1 = 0.068 * torch.rand(nworld, device=device) - 0.034
  data.qpos[:, 0] = torch.where(env_mask, qpos0, data.qpos[:, 0])
  data.qpos[:, 1] = torch.where(env_mask, qpos1, data.qpos[:, 1])

  data.qvel[:] = torch.where(
      env_mask.unsqueeze(1),
      0.01 * torch.randn(nworld, data.qvel.shape[1], device=device),
      data.qvel,
  )


def _cartpole_env(swing_up: bool) -> config.Environment:
  """Base cartpole environment configuration."""
  observations = {
      "cartpole": config.Obs(_obs_cartpole).configure(output_shape=(5,)),
  }

  resetters = {
      "resetter": config.Component(
          functional_reset.FunctionalEnvironmentReset,
          terms=[
              config.Term(
                  reset_cartpole,
                  type="reset",
                  swing_up=swing_up,
              )
          ],
      )
  }

  task_layer = config.TaskLayer(
      rewards={"dense": config.Reward(_reward_dense, weight=1.0 / 0.01)},
  )

  devices = {
      "cartpole": config.Component(
          device.ArticulatedDevice,
          measurements=["joint_pos", "joint_vel"],
      ),
      "slider_actuator": config.Component(
          device.ActuatorDevice,
          targets=["slider"],
          measurements=["ctrl"],
      ),
  }

  return config.Environment(
      base_scene=_create_base_scene(),
      device_layer=config.DeviceLayer(devices=devices),
      task_layer=task_layer,
      observation_adapter={
          "policy": config.ObsGroup(terms=observations),
          "critic": config.ObsGroup(terms=observations),
      },
      resetters=resetters,
      sim=config.Sim(
          control_timestep=0.01,
          simulation_timestep=0.01,
      ),
      episode_length_secs=10.0,
  )


def cartpole_balance_env() -> config.Environment:
  return _cartpole_env(swing_up=False)


def cartpole_swingup_env() -> config.Environment:
  return _cartpole_env(swing_up=True)
