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

"""Cartpole vision task configuration."""

from typing import Mapping

from mujoco_playground.tasks.dm_control_suite import cartpole
from reaf.simulation import torch as device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import resetter
import torch


def _reward_vision(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Additive dense reward for vision-based cartpole balance."""
  qpos = features["cartpole/joint_pos"]
  qvel = features["cartpole/joint_vel"]
  cart_pos = qpos[:, 0]
  pole_cos = torch.cos(qpos[:, 1])
  cart_vel = qvel[:, 0]
  pole_vel = qvel[:, 1]

  alive = 0.1 * torch.ones_like(cart_pos)
  pole_pos_penalty = -1.0 * (1.0 - pole_cos) ** 2
  cart_pos_penalty = -0.02 * cart_pos**2
  cart_vel_penalty = -0.01 * torch.abs(cart_vel)
  pole_vel_penalty = -0.005 * torch.abs(pole_vel)

  return (
      alive
      + pole_pos_penalty
      + cart_pos_penalty
      + cart_vel_penalty
      + pole_vel_penalty
  )


def _termination_nan_check(device_layer, **kwargs):
  data = device_layer.sim.data
  return torch.isnan(data.qpos).any(dim=-1) | torch.isnan(data.qvel).any(dim=-1)


def _termination_pole_angle(device_layer, **kwargs):
  data = device_layer.sim.data
  pole_angle = data.qpos[:, 1]
  cart_pos = data.qpos[:, 0]
  return (torch.abs(pole_angle) > torch.pi / 2) | (torch.abs(cart_pos) > 3.0)


def cartpole_balance_vision_env() -> config.Environment:
  """Cartpole vision environment configuration."""
  cfg = cartpole.cartpole_balance_env()

  cfg.observation_adapter = {
      "policy": config.ObsGroup(
          terms={
              "pixels": (
                  config.Obs(
                      lambda d, **_: (d["fixed_rgb"].float() / 255.0).permute(
                          0, 3, 1, 2
                      ),
                  ).configure(output_shape=(3, 64, 64))
              )
          },
          enable_noise=False,
      ),
      "dummy": config.ObsGroup(
          terms={
              "zeros": (
                  config.Obs(
                      lambda d, **_: torch.zeros(
                          next(iter(d.values())).shape[0],
                          1,
                          device=next(iter(d.values())).device,
                      ),
                  ).configure(output_shape=(1,))
              )
          },
          enable_noise=False,
      ),
  }
  cfg.task_layer.rewards = {
      "dense": config.Reward(_reward_vision, weight=1.0 / 0.02),
  }
  cfg.sim.control_timestep = 0.02
  cfg.episode_length_secs = 5.0

  cfg.device_layer.devices["camera"] = config.Component(
      device.CameraDevice,
      cam_name="fixed",
      cam_resolution=(64, 64),
      render_rate_hz=1.0 / cfg.sim.control_timestep,
      render_rgb=True,
      render_depth=False,
  )

  cfg.resetters["termination"] = config.Component(
      resetter.TerminationResetter,
      terms=[
          config.Term(_termination_nan_check),
          config.Term(_termination_pole_angle),
      ],
  )
  return cfg
