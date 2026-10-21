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

"""Trossen WXAI robot configuration for pick task."""

from mujoco_playground.robots.trossen_wxai.sim import device as trossen_device
from mujoco_playground.robots.trossen_wxai.sim import scene as trossen_scene
from mujoco_playground.tasks.manipulation.pick_cube import mdp
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from mujoco_playground.tasks.manipulation.pick_cube.trossen_wxai import mdp as trossen_mdp
import numpy as np
from reaf.simulation import torch as device
from reaf.simulation.torch import action_history
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import noise
from reaf.simulation.torch import processors
import torch


def right_arm_pick_cube_env() -> config.Environment:
  """Pick cube env using only the right arm (left arm is passive)."""
  cfg = pick_task.pick_cube_env()
  cfg.sim.njmax = 450

  cfg.base_scene = trossen_scene.trossen_wxai_biarm_table()

  # Split robot_state into arm and gripper
  cfg.device_layer.devices["arm_state"] = config.Component(
      device.ArticulatedDevice,
      home_qpos=trossen_device.PICKUP_QPOS[:6],
      measurements=["home_qpos", "joint_pos", "joint_vel", "joint_range"],
      joint_names=trossen_device.RIGHT_ARM_JOINTS[:6],
  )
  cfg.device_layer.devices["gripper_state"] = config.Component(
      device.ArticulatedDevice,
      home_qpos=trossen_device.PICKUP_QPOS[6:],
      measurements=["home_qpos", "joint_pos", "joint_vel", "joint_range"],
      joint_names=[
          "right/right_carriage_joint",
          "right/left_carriage_joint",
      ],
  )

  # Update robot actuator
  cfg.device_layer.devices["robot"] = config.Component(
      device.ActuatorDevice,
      targets=trossen_device.RIGHT_ARM_JOINTS,
      transmission="joint",
      home_ctrl=trossen_device.PICKUP_CTRL,
  ).configure_spec(
      mode="position",
      stiffness=trossen_device.RIGHT_ARM_STIFFNESS,
      damping=trossen_device.RIGHT_ARM_DAMPING,
      forcerange=trossen_device.RIGHT_ARM_FORCERANGE,
      ctrlrange=trossen_device.RIGHT_ARM_CTRLRANGE,
  )

  # Update ee site
  cfg.device_layer.devices["ee"] = config.Component(
      device.SiteDevice,
      site_name="right/ee_gripper_link",
  )

  # Configure floor_contact to use table_collision instead of floor
  if "floor_contact" in cfg.device_layer.devices:
    cfg.device_layer.devices["floor_contact"].configure_spec(
        refname="table/table_collision",
        objname=("right/gripper_right", "right/gripper_left"),
    )

  # Action processing.
  action_scale_val = trossen_device.RIGHT_ARM_ACTION_SCALE
  action_offset = trossen_device.RIGHT_ARM_ACTION_OFFSET

  cfg.task_layer.command_processors = {
      "action_scale": config.Component(
          processors.TransformActions,
          output_key="robot/ctrl",
          scale=action_scale_val,
          offset=action_offset,
      ),
  }

  # Action history.
  ctrl_shape = (len(trossen_device.PICKUP_CTRL),)
  cfg.task_layer.command_feature_producers = {
      "action_history": config.Component(
          action_history.ActionHistory,
          key="actions",
          history_length=1,
          feature_shape=ctrl_shape,
      ),
  }

  # serial bus latency: 5–10 ms ≈ 1–2 substeps @ 5 ms dt.
  usb_delay = (0.005, 0.010)
  cfg.device_layer.command_processors = {
      "usb_command_delay": config.Component(
          processors.StochasticSubstepCommands,
          keys="robot/ctrl",
          delay=usb_delay,
          noise=noise.uniform(
              torch.tensor([-np.deg2rad(1.0)] * 6 + [-0.002]),
              torch.tensor([np.deg2rad(1.0)] * 6 + [0.002]),
          ),
      ),
  }

  # Measurement latency.
  cfg.device_layer.substep_measurement_processors = {
      "usb_measurement_delay": config.Component(
          processors.StochasticSubstepMeasurements,
          keys=(
              "arm_state/joint_pos",
              "arm_state/joint_vel",
              "gripper_state/joint_pos",
              "gripper_state/joint_vel",
              "ee/xpos",
              "ee/xmat",
          ),
          delay=usb_delay,
      ),
  }

  # Observations
  base_policy_terms = cfg.observation_adapter["policy"].terms
  assert isinstance(base_policy_terms, dict)
  trossen_obs = {
      "arm_pos": (
          config.Obs(
              mdp.obs_joint_pos_rel,
              qpos_key="arm_state/joint_pos",
              home_qpos_key="arm_state/home_qpos",
          ).configure(
              noise=noise.uniform(-np.deg2rad(1.0), np.deg2rad(1.0)),
              output_shape=(6,),
          )
      ),
      "arm_vel": (
          config.Obs("arm_state/joint_vel").configure(
              noise=noise.uniform(-0.05, 0.05),
              output_shape=(6,),
          )
      ),
      "gripper_pos": (
          config.Obs(
              mdp.obs_joint_pos_rel,
              qpos_key="gripper_state/joint_pos",
              home_qpos_key="gripper_state/home_qpos",
          ).configure(
              noise=noise.uniform(-0.0015, 0.0015),
              output_shape=(2,),
          )
      ),
      "gripper_vel": (
          config.Obs("gripper_state/joint_vel").configure(
              noise=noise.uniform(-0.01, 0.01),
              output_shape=(2,),
          )
      ),
      "ee_to_object": base_policy_terms["ee_to_object"],
      "object_to_goal": base_policy_terms["object_to_goal"],
      "last_action": config.Obs("actions").configure(output_shape=ctrl_shape),
  }

  cfg.observation_adapter = {
      "policy": config.ObsGroup(terms=trossen_obs, enable_noise=True),
      "critic": config.ObsGroup(terms=trossen_obs, enable_noise=False),
  }

  # Rewards
  rewards = cfg.task_layer.rewards
  rewards["reaching"].params["reaching_std"] = 0.1
  rewards["action_rate"].params["penalty_scale"] = -0.03
  # fmt: off
  rewards["joint_vel_hinge"].params["qvel_key"] = "arm_state/joint_vel"
  rewards["joint_pos_limits"].params["qpos_key"] = "arm_state/joint_pos"
  rewards["joint_pos_limits"].params["joint_range_key"] = "arm_state/joint_range"
  # fmt: on
  rewards["arm_nominal_pose"] = config.Reward(
      mdp.reward_arm_nominal_pose,
      qpos_key="arm_state/joint_pos",
      home_qpos_key="arm_state/home_qpos",
      num_joints=6,
      std=1.0,
      weight=0.05,
  )
  rewards["ee_nominal_orientation"] = config.Reward(
      mdp.reward_ee_nominal_orientation,
      xmat_key="ee/xmat",
      nominal_xmat=(
          # fmt: off
          -0.70738825, 0.0, -0.7068252, 0.0, -1.0, 0.0, -0.7068252, 0.0, 0.70738825,
          # fmt: on
      ),
      std=0.5,
      weight=0.05,
  )

  # Reset logic
  cfg.resetters["pick_cube"] = config.Component(
      trossen_mdp.TrossenPickCubeResetter,
      object_pos_range=(
          (0.10, 0.25),
          (-0.1, 0.1),
          # TABLE_HEIGHT (0.75) + max_DR_cube_half_size (0.025)
          # + safety_margin (0.007)
          (
              trossen_scene.TABLE_HEIGHT + 0.032,
              trossen_scene.TABLE_HEIGHT + 0.032,
          ),
      ),
      target_pos_range=((0.10, 0.20), (-0.1, 0.1), (0.85, 1.0)),
      resampling_time_range=(8.0, 12.0),
      joint_perturbation=0.1,
  )

  # Domain randomization
  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  assert isinstance(dr_terms, dict)
  gripper_pattern = r"right/gripper_(left|right)_pad"
  dr_terms["geom_friction"].params["pattern"] = gripper_pattern
  dr_terms["geom_size"].params["pattern"] = r"cube/cube"
  dr_terms["geom_size"].params["noise"] = noise.uniform(
      0.020, 0.025, op="replace"
  )
  dr_terms["actuator_gains"].params["pattern"] = r"right/joint_\d+"
  dr_terms["dof_damping"].params["pattern"] = r"right/joint_\d+"

  return cfg


def right_arm_pick_cube_pixels_env() -> config.Environment:
  """Pick cube env with camera observations."""
  cfg = right_arm_pick_cube_env()

  cfg.device_layer.devices["camera"] = config.Component(
      device.CameraDevice,
      cam_name=["right/wrist_cam"],
      cam_resolution=[(32, 32)],
      render_rate_hz=1.0 / cfg.sim.control_timestep,
      render_rgb=True,
      render_depth=False,
      enabled_geom_groups=(0, 2, 3),
      use_precomputed_rays=False,
      use_shadows=False,
      use_textures=False,
  )

  base_obs = cfg.observation_adapter["policy"].terms
  policy_obs = {
      "arm_pos": base_obs["arm_pos"],
      "arm_vel": base_obs["arm_vel"],
      "gripper_pos": base_obs["gripper_pos"],
      "gripper_vel": base_obs["gripper_vel"],
      "ee_pos": config.Obs("ee/xpos").configure(output_shape=(3,)),
      "ee_orientation": config.Obs("ee/xmat").configure(output_shape=(9,)),
      "goal_position": (
          config.Obs(
              mdp.obs_goal_position_ee_frame,
              ee_xpos_key="ee/xpos",
              ee_xmat_key="ee/xmat",
              target_xpos_key="target/mocap_pos",
          ).configure(output_shape=(3,))
      ),
      "last_action": base_obs["last_action"],
  }

  cfg.observation_adapter = {
      "policy": config.ObsGroup(terms=policy_obs, enable_noise=True),
      "critic": cfg.observation_adapter["critic"],
      "camera": config.ObsGroup(
          terms={
              "camera": (
                  config.Obs(
                      mdp.obs_camera,
                      camera_key="right/wrist_cam_rgb",
                  ).configure(output_shape=(3, 32, 32))
              )
          },
          enable_noise=False,
      ),
  }

  # Add visual domain randomization terms
  visual_dr_terms = {
      "override_flat_materials": config.Term(
          mdp.override_flat_materials,
          patterns=("cube/cube", "table/tabletop", "floor"),
      ),
  }
  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  assert isinstance(dr_terms, dict)
  dr_terms.update(visual_dr_terms)

  return cfg
