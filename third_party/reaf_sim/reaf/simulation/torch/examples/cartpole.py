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
"""Cartpole multi-environment rollout and video generation example."""

from typing import Literal

import imageio
import mujoco
import numpy as np
from reaf.simulation.torch import ActuatorDevice
from reaf.simulation.torch import CameraDevice
from reaf.simulation.torch import flat_scene
from reaf.simulation.torch import MujocoCameraDevice
import torch

# DeepMind Control Suite Cartpole XML
CARTPOLE_XML = """
<mujoco model="cart-pole">
  <option timestep="0.01" integrator="RK4">
    <flag contact="disable" energy="disable"/>
  </option>
  <default>
    <default class="pole">
      <joint type="hinge" axis="0 1 0" damping="2e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass=".1"/>
    </default>
  </default>
  <visual>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light name="light" pos="0 0 6"/>
    <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0"/>
    <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2"/>
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1"/>
      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slider" joint="slider" forcerange="-10 10" ctrlrange="-10 10" ctrllimited="true" forcelimited="true"/>
  </actuator>
</mujoco>
"""


def tile_grid(batch_images: torch.Tensor, cols: int = 4) -> np.ndarray:
  """Tiles batched camera tensors into an (N_rows, N_cols) RGB image."""
  imgs = batch_images.detach().cpu().numpy()
  rows = [
      np.concatenate(imgs[i : i + cols], axis=1)
      for i in range(0, len(imgs), cols)
  ]
  return np.concatenate(rows, axis=0)


def create_cartpole_env(
    nworld: int = 16,
    xpu_device: str = "cuda:0",
    impl: Literal["cpp", "warp"] = "warp",
):
  """Builds a batched Cartpole environment with actuators and camera."""
  spec = mujoco.MjSpec.from_string(CARTPOLE_XML)
  if impl == "cpp":
    camera_comp = flat_scene.Component(
        MujocoCameraDevice,
        camera_name="fixed",
        width=160,
        height=120,
        render_rate_hz=20.0,
        image_output_name_prefix="camera/fixed",
    )
  else:
    camera_comp = flat_scene.Component(
        CameraDevice,
        cam_name="fixed",
        cam_resolution=(160, 120),
        render_rate_hz=20.0,
        render_rgb=True,
        output_name_prefix="camera/",
    )

  env_config = flat_scene.Environment(
      base_scene=spec,
      sim=flat_scene.Sim(
          nworld=nworld,
          xpu_device=xpu_device,
          impl=impl,
          simulation_timestep=0.01,
          control_timestep=0.05,
      ),
      device_layer=flat_scene.DeviceLayer(
          devices={
              "slider_actuator": flat_scene.Component(
                  ActuatorDevice,
                  targets=["slider"],
                  measurements=["joint_pos", "joint_vel"],
              ),
              "camera": camera_comp,
          }
      ),
      observation_adapter={
          "slider_actuator/joint_pos": flat_scene.ObsGroup(
              terms={
                  "pos": (
                      flat_scene.Obs("slider_actuator/joint_pos").configure(
                          output_shape=(1,)
                      )
                  ),
              }
          ),
          "camera/fixed_rgb": flat_scene.ObsGroup(
              terms={
                  "rgb": (
                      flat_scene.Obs("camera/fixed_rgb").configure(
                          output_shape=(120, 160, 3),
                          dtype="uint8",
                      )
                  ),
              }
          ),
      },
  )
  return flat_scene.build_env(env_config)


def main():
  nworld = 16 if torch.cuda.is_available() else 1
  xpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
  impl = "warp" if torch.cuda.is_available() else "cpp"

  env = create_cartpole_env(nworld=nworld, xpu_device=xpu_device, impl=impl)
  timestep = env.reset()
  frames = []

  for _ in range(100):
    commands = (
        torch.rand((env.nworld, 1), device=torch.device(xpu_device)) * 2.0 - 1.0
    ) * 5.0
    timestep = env.step({"slider_actuator/ctrl": commands})
    rgb_tensor = timestep.observation["camera/fixed_rgb"]
    frames.append(tile_grid(rgb_tensor, cols=4))

  output_path = "cartpole_grid_rollout.mp4"
  imageio.mimsave(output_path, frames, fps=20)
  print(f"Saved {output_path} ({len(frames)} frames across {nworld} worlds)")


if __name__ == "__main__":
  main()
