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

"""Pick cube task configuration using Term pattern."""

import mujoco
from mujoco_playground._src.objects import primitive as objects
from mujoco_playground.tasks.manipulation.pick_cube import mdp
from mujoco_playground.tasks.manipulation.pick_cube import reset
from reaf.simulation import torch as device
from reaf.simulation.torch import action_history
from reaf.simulation.torch import DomainRandomizationResetter
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import model_randomization
from reaf.simulation.torch import noise

TARGET = "target"
TARGET_MOCAP_POS = f"{TARGET}/mocap_pos"


def _create_base_scene() -> mujoco.MjSpec:
  spec = objects.plane()
  spec.copy_during_attach = True
  return spec


def pick_cube_env() -> config.Environment:
  """Base pick cube environment configuration."""
  observations = {
      "joint_pos": (
          config.Obs(
              mdp.obs_joint_pos_rel,
              qpos_key="robot_state/joint_pos",
              home_qpos_key="robot_state/home_qpos",
          ).configure(
              noise=noise.uniform(-0.0025, 0.0025),
              output_shape=(6,),  # TODO: robot specific
          )
      ),
      "joint_vel": (
          config.Obs("robot_state/joint_vel").configure(
              noise=noise.uniform(-0.2, 0.2),
              output_shape=(6,),  # TODO: robot specific
          )
      ),
      "ee_to_object": (
          config.Obs(
              mdp.obs_ee_to_cube,
              ee_xpos_key="ee/xpos",
              cube_xpos_key="object/xpos",
          ).configure(
              noise=noise.uniform(-0.0025, 0.0025),
              output_shape=(3,),
          )
      ),
      "object_to_goal": (
          config.Obs(
              mdp.obs_cube_to_goal,
              cube_xpos_key="object/xpos",
              target_xpos_key=TARGET_MOCAP_POS,
          ).configure(
              noise=noise.uniform(-0.0025, 0.0025),
              output_shape=(3,),
          )
      ),
      "last_action": (
          config.Obs("actions").configure(
              output_shape=(6,),  # TODO: robot specific
          )
      ),
  }

  rewards = {
      "reaching": config.Reward(
          mdp.reward_reaching,
          ee_xpos_key="ee/xpos",
          cube_xpos_key="object/xpos",
          reaching_std=0.05,
          weight=1.0,
      ),
      "lift": config.Reward(
          mdp.reward_cube_lift,
          cube_xpos_key="object/xpos",
          target_xpos_key="target/mocap_pos",
          table_z=0.025,
          weight=3.0,
      ),
      "bringing": config.Reward(
          mdp.reward_cube_to_target,
          cube_xpos_key="object/xpos",
          target_xpos_key="target/mocap_pos",
          table_z=0.025,
          std=0.15,
          weight=3.0,
      ),
      "success": config.Reward(
          mdp.reward_success,
          cube_xpos_key="object/xpos",
          target_xpos_key="target/mocap_pos",
          distance_threshold=0.10,
          lift_height_threshold=0.04,
          weight=2.0,
      ),
      "action_rate": config.Reward(
          mdp.reward_action_rate,
          action_key="actions",
          history_key="actions_history",
          penalty_scale=-0.01,
          weight=1.0,
      ),
      "joint_vel_hinge": config.Reward(
          mdp.reward_joint_vel_hinge,
          qvel_key="robot_state/joint_vel",
          max_vel=0.5,
          penalty_scale=-0.01,
          weight=1.0,
      ),
      "joint_pos_limits": config.Reward(
          mdp.reward_joint_pos_limits,
          qpos_key="robot_state/joint_pos",
          joint_range_key="robot_state/joint_range",
          penalty_scale=-10.0,
          margin=0.9,
          weight=1.0,
      ),
      "floor_contact": config.Reward(
          mdp.reward_floor_contact,
          contact_key="floor_contact/data",
          penalty_scale=-0.1,
          weight=1.0,
      ),
  }

  metrics = {
      "reaching_dist": config.Metric(
          mdp.metric_distance,
          pos_a_key="ee/xpos",
          pos_b_key="object/xpos",
          reduce="last",
      ),
      "bringing_dist": config.Metric(
          mdp.metric_distance,
          pos_a_key="object/xpos",
          pos_b_key=TARGET_MOCAP_POS,
          reduce="last",
      ),
      "cube_lifted": config.Metric(
          mdp.metric_cube_lifted,
          cube_xpos_key="object/xpos",
          lift_height_threshold=0.04,
          reduce="max",
      ),
      "success": config.Metric(
          mdp.reward_success,
          cube_xpos_key="object/xpos",
          target_xpos_key=TARGET_MOCAP_POS,
          distance_threshold=0.10,
          lift_height_threshold=0.04,
          reduce="last",
      ),
  }

  resetters = {
      "pick_cube": config.Component(
          reset.PickCubeResetter,
          object_pos_range=((0.20, 0.35), (-0.1, 0.1), (0.02, 0.02)),
          target_pos_range=((0.20, 0.30), (-0.1, 0.1), (0.08, 0.2)),
          resampling_time_range=(8.0, 12.0),
          joint_perturbation=0.1,
      ),
      "domain_randomization": config.Component(
          DomainRandomizationResetter,
          terms={
              "geom_friction": config.Term(
                  model_randomization.randomize_field,
                  field="geom_friction",
                  pattern="PLACEHOLDER",  # Robot-specific
                  noise={
                      0: noise.uniform(0.3, 1.5, op="replace"),
                      1: noise.log_uniform(1e-4, 2e-2, op="replace"),
                      2: noise.log_uniform(1e-5, 5e-3, op="replace"),
                  },
              ),
              "geom_size": config.Term(
                  model_randomization.randomize_field,
                  field="geom_size",
                  pattern="cube/cube",
                  noise=noise.uniform(0.012, 0.018, op="replace"),
              ),
              "actuator_gains": config.Term(
                  model_randomization.randomize_actuator_gains,
                  pattern="PLACEHOLDER",  # Robot-specific
                  kp_noise=noise.uniform(0.8, 1.2, op="scale"),
                  kd_noise=noise.uniform(0.8, 1.2, op="scale"),
              ),
              "dof_damping": config.Term(
                  model_randomization.randomize_field,
                  field="dof_damping",
                  pattern="PLACEHOLDER",  # Robot-specific
                  noise=noise.uniform(0.8, 1.2, op="scale"),
              ),
          },
      ),
  }

  task_layer = config.TaskLayer(
      command_feature_producers={
          "action_history": config.Component(
              action_history.ActionHistory,
              key="actions",
              history_length=2,
              feature_shape=(6,),  # TODO: robot specific
          ),
      },
      rewards=rewards,
      metrics=metrics,
  )

  devices = {
      "object": (
          config.Component(
              device.FreeDevice,
              body_name="cube/cube",
              joint_name="cube/cube_freejoint",
          ).attach(
              lambda: objects.cube("cube"),
              prefix="cube",
              pos=(0.25, 0.0, 0.025),
          )
      ),
      "target": (
          config.Component(
              device.MocapDevice,
              body_name="target/target",
          ).attach(
              lambda: objects.cube("target", mocap=True),
              prefix="target",
              pos=(0.25, 0.0, 0.15),
          )
      ),
      "floor_contact": (
          config.Component(
              device.ContactSensorDevice,
              fields=("found",),
          ).configure_spec(
              objtype="body",
              objname=(),  # Robot-specific: set in embodiment config
              reftype="geom",
              refname="floor",
          )
      ),
  }

  return config.Environment(
      base_scene=_create_base_scene(),
      device_layer=config.DeviceLayer(devices=devices),
      task_layer=task_layer,
      observation_adapter={
          "policy": config.ObsGroup(terms=observations, enable_noise=True),
          "critic": config.ObsGroup(terms=observations, enable_noise=False),
      },
      resetters=resetters,
      nan_termination=True,
      sim=config.Sim(
          nconmax=64,
          njmax=256,
      ),
  )


def pick_cube_pixels_env() -> config.Environment:
  """Pick cube environment with camera observations."""
  cfg = pick_cube_env()

  # Add camera device
  cfg.device_layer.devices["camera"] = config.Component(
      device.CameraDevice,
      cam_name=["PLACEHOLDER_CAM"],  # Override in robot-specific config
      cam_resolution=[(32, 32)],
      render_rate_hz=1.0 / cfg.sim.control_timestep,
      render_rgb=True,
      render_depth=False,
      enabled_geom_groups=(0, 1, 2),
  )

  base_obs = cfg.observation_adapter["policy"].terms
  policy_obs = {
      "joint_pos": base_obs["joint_pos"],
      "joint_vel": base_obs["joint_vel"],
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
                      camera_key="PLACEHOLDER_CAM_rgb",
                  ).configure(
                      history_length=2,
                      output_shape=(3, 32, 32),
                  )
              )
          },
          enable_noise=False,
      ),
  }

  return cfg
