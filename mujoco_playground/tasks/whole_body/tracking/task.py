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

"""Whole-body motion tracking environment for the Unitree G1 using REAF."""

from mujoco_playground.robots.g1.sim import device as g1_device
from mujoco_playground.robots.g1.sim import scene as g1_scene
from mujoco_playground.tasks.whole_body.tracking import ghost
from mujoco_playground.tasks.whole_body.tracking import mdp
from mujoco_playground.tasks.whole_body.tracking import motion_command
from mujoco_playground.tasks.whole_body.tracking import reset
from reaf.simulation import torch as device
from reaf.simulation.torch import action_history
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import model_randomization
from reaf.simulation.torch import mujoco_types
from reaf.simulation.torch import noise
from reaf.simulation.torch import processors
from reaf.simulation.torch import resetter as common_resetter
import torch

_CONTROL_TIMESTEP = 0.02
_SIMULATION_TIMESTEP = 0.005
_DECIMATION = int(round(_CONTROL_TIMESTEP / _SIMULATION_TIMESTEP))

##
# Task-specific body selection and randomization ranges.
##

ANCHOR_BODY_NAME = "torso_link"
TRACKING_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
EE_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
POSE_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}
JOINT_POSITION_RANGE: tuple[float, float] = (-0.1, 0.1)

COM_BODY_NAME = "torso_link"
COM_OFFSET_RANGES: dict[int, tuple[float, float]] = {
    0: (-0.025, 0.025),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
}
FOOT_FRICTION_RANGE: tuple[float, float] = (0.3, 1.2)


def g1_env(
    trajectory_path: str | None = None,
    episode_length_secs: float = 10.0,
    play: bool = False,
    has_state_estimation: bool = True,
    enable_ghost: bool = False,
    xml_uri: str | None = None,
) -> config.Environment:
  """Whole-body motion tracking environment configuration for the Unitree G1.

  Args:
    trajectory_path: Path to a reference motion `.npz` file (see
      `motion_loader.py`). If `None`, loads a standing dummy motion for tests.
    episode_length_secs: Episode timeout duration in seconds.
    play: If True, disables observation noise, velocity pushes, and RSI
      randomization, and starts episodes at frame 0.
    has_state_estimation: If False, omits `motion_anchor_pos_b` and
      `base_lin_vel` from actor observations.
    enable_ghost: If True, attaches a semi-transparent ghost robot that replays
      the reference trajectory for visualization.
    xml_uri: Optional asset URI override for the G1 MJCF model.

  Returns:
    A `config.Environment` ready to be built with `build.build_env`.
  """
  num_joints = g1_device.NUM_JOINTS
  body_names = TRACKING_BODY_NAMES
  num_bodies = len(body_names)
  anchor_index = body_names.index(ANCHOR_BODY_NAME)
  ee_indexes = [body_names.index(n) for n in EE_BODY_NAMES]
  default_joint_pos = torch.tensor(
      g1_device.KNEES_BENT_JOINT_POS, dtype=torch.float32
  )

  base_scene = g1_scene.g1_scene_spec(xml_uri)
  robot_model = g1_scene.g1_robot_spec(xml_uri).compile()
  soft_limits = torch.tensor(
      mujoco_types.soft_joint_limits(
          robot_model,
          g1_device.JOINT_NAMES,
          g1_device.SOFT_JOINT_POS_LIMIT_FACTOR,
      ),
      dtype=torch.float32,
  )
  # Motion npz body arrays exclude the worldbody, so index = body_id - 1.
  body_indexes = [robot_model.body(n).id - 1 for n in body_names]

  ##
  # DeviceLayer: actuators, body tracking, contact sensor, optional ghost.
  ##

  devices: dict[str, config.Component] = {
      "robot": (
          config.Component(
              device.ActuatorDevice,
              targets=list(g1_device.JOINT_NAMES),
              measurements=["joint_pos", "joint_vel"],
          ).configure_spec(use_existing=True,)
      ),
      "body_tracking": config.Component(
          motion_command.BodyTrackingDevice,
          body_names=body_names,
          imu_lin_vel_sensor=g1_device.IMU_LIN_VEL_SENSOR,
          imu_ang_vel_sensor=g1_device.IMU_ANG_VEL_SENSOR,
      ),
      "self_collision": (
          config.Component(
              device.ContactSensorDevice,
              fields=("found", "force"),
              history_length=_DECIMATION,
          ).configure_spec(
              objtype="xbody",
              objname=(g1_device.ROOT_BODY_NAME,),
              reftype="xbody",
              refname=g1_device.ROOT_BODY_NAME,
              reduce="none",
              num=1,
          )
      ),
  }

  shared_state = motion_command.SharedMotionState() if enable_ghost else None
  ghost_updater = None
  if enable_ghost:
    devices["ghost_robot"] = config.Component(
        ghost.GhostDevice,
    ).configure_spec(xml_path=xml_uri or g1_device.G1_XML_URI)
    ghost_updater = config.Component(
        ghost.GhostUpdater,
        ghost_device_name="ghost_robot",
        shared_state=shared_state,
    )

  ##
  # TaskLayer: ActionHistory, TransformActions, and MotionReferenceProducer.
  ##

  action_history_comp = config.Component(
      action_history.ActionHistory,
      key="action",
      history_length=1,
      feature_shape=(num_joints,),
  )

  action_scale_comp = config.Component(
      processors.TransformActions,
      input_key="action",
      output_key="robot/ctrl",
      scale=g1_device.ACTION_SCALE_VECTOR,
      offset=g1_device.KNEES_BENT_JOINT_POS,
  )

  motion_ref_comp = config.Component(
      motion_command.MotionReferenceProducer,
      motion_file=trajectory_path,
      body_indexes=body_indexes,
      num_joints=num_joints,
      num_bodies=num_bodies,
      anchor_body_index=anchor_index,
      ee_body_indexes=ee_indexes,
      sampling_mode="start" if play else "adaptive",
      adaptive_kernel_size=1,
      adaptive_lambda=0.8,
      adaptive_uniform_ratio=0.1,
      adaptive_alpha=0.001,
      step_dt=_CONTROL_TIMESTEP,
      anchor_pos_z_threshold=0.25,
      anchor_ori_threshold=0.8,
      ee_pos_z_threshold=0.25,
      shared_state=shared_state,
  )

  ##
  # Rewards.
  ##

  rewards = {
      "motion_global_root_pos": config.Reward(
          mdp.reward_motion_global_anchor_pos,
          std=0.3,
          weight=0.5,
      ),
      "motion_global_root_ori": config.Reward(
          mdp.reward_motion_global_anchor_ori,
          std=0.4,
          weight=0.5,
      ),
      "motion_body_pos": config.Reward(
          mdp.reward_motion_relative_body_pos,
          std=0.3,
          weight=1.0,
      ),
      "motion_body_ori": config.Reward(
          mdp.reward_motion_relative_body_ori,
          std=0.4,
          weight=1.0,
      ),
      "motion_body_lin_vel": config.Reward(
          mdp.reward_motion_global_body_lin_vel,
          std=1.0,
          weight=1.0,
      ),
      "motion_body_ang_vel": config.Reward(
          mdp.reward_motion_global_body_ang_vel,
          std=3.14,
          weight=1.0,
      ),
      "action_rate_l2": config.Reward(
          mdp.reward_action_rate_l2,
          action_key="action",
          history_key="action_history",
          weight=-1e-1,
      ),
      "joint_limit": config.Reward(
          mdp.reward_joint_pos_limits,
          soft_joint_pos_limits=soft_limits,
          qpos_key="robot/joint_pos",
          weight=-10.0,
      ),
      "self_collisions": config.Reward(
          mdp.reward_self_collision,
          sensor_key="self_collision/force_history",
          force_threshold=10.0,
          weight=-10.0,
      ),
  }

  ##
  # Terminations (timeout truncation is handled via episode_length_secs).
  ##

  terminations = {
      "bad_anchor_pos_z": config.Term(
          mdp.termination_bad_anchor_pos_z_only,
          threshold=0.25,
      ),
      "bad_anchor_ori": config.Term(
          mdp.termination_bad_anchor_ori,
          threshold=0.8,
      ),
      "bad_motion_body_pos_z": config.Term(
          mdp.termination_bad_motion_body_pos_z_only,
          threshold=0.25,
          body_indexes=ee_indexes,
      ),
  }

  metrics = {
      "motion/error_anchor_pos": config.Metric(
          mdp.metric_error_anchor_pos, reduce="mean"
      ),
      "motion/error_anchor_rot": config.Metric(
          mdp.metric_error_anchor_rot, reduce="mean"
      ),
      "motion/error_anchor_lin_vel": config.Metric(
          mdp.metric_error_anchor_lin_vel, reduce="mean"
      ),
      "motion/error_anchor_ang_vel": config.Metric(
          mdp.metric_error_anchor_ang_vel, reduce="mean"
      ),
      "motion/error_body_pos": config.Metric(
          mdp.metric_error_body_pos, reduce="mean"
      ),
      "motion/error_body_rot": config.Metric(
          mdp.metric_error_body_rot, reduce="mean"
      ),
      "motion/error_body_lin_vel": config.Metric(
          mdp.metric_error_body_lin_vel, reduce="mean"
      ),
      "motion/error_body_ang_vel": config.Metric(
          mdp.metric_error_body_ang_vel, reduce="mean"
      ),
      "motion/error_joint_pos": config.Metric(
          mdp.metric_error_joint_pos, reduce="mean"
      ),
      "motion/error_joint_vel": config.Metric(
          mdp.metric_error_joint_vel, reduce="mean"
      ),
      "motion/sampling_entropy": config.Metric(
          "motion/sampling_entropy", reduce="last"
      ),
      "motion/sampling_top1_prob": config.Metric(
          "motion/sampling_top1_prob", reduce="last"
      ),
      "motion/sampling_top1_bin": config.Metric(
          "motion/sampling_top1_bin", reduce="last"
      ),
  }

  ##
  # Observations (asymmetric actor / critic groups).
  ##

  actor_terms = {
      "command": (
          config.Obs(mdp.obs_motion_command).configure(
              output_shape=(2 * num_joints,)
          )
      ),
      "motion_anchor_pos_b": (
          config.Obs(mdp.obs_anchor_pos_b).configure(
              noise=noise.uniform(-0.25, 0.25),
              output_shape=(3,),
          )
      ),
      "motion_anchor_ori_b": (
          config.Obs(mdp.obs_anchor_ori_b).configure(
              noise=noise.uniform(-0.05, 0.05),
              output_shape=(6,),
          )
      ),
      "base_lin_vel": (
          config.Obs("body_tracking/base_lin_vel").configure(
              noise=noise.uniform(-0.5, 0.5),
              output_shape=(3,),
          )
      ),
      "base_ang_vel": (
          config.Obs("body_tracking/base_ang_vel").configure(
              noise=noise.uniform(-0.2, 0.2),
              output_shape=(3,),
          )
      ),
      "joint_pos": (
          config.Obs(
              mdp.obs_joint_pos_rel,
              default_joint_pos=default_joint_pos,
              qpos_key="robot/joint_pos",
          ).configure(
              noise=noise.uniform(-0.01, 0.01),
              output_shape=(num_joints,),
          )
      ),
      "joint_vel": (
          config.Obs("robot/joint_vel").configure(
              noise=noise.uniform(-0.5, 0.5),
              output_shape=(num_joints,),
          )
      ),
      "actions": config.Obs("action").configure(output_shape=(num_joints,)),
  }
  if not has_state_estimation:
    actor_terms.pop("motion_anchor_pos_b")
    actor_terms.pop("base_lin_vel")

  critic_terms = {
      "command": (
          config.Obs(mdp.obs_motion_command).configure(
              output_shape=(2 * num_joints,)
          )
      ),
      "motion_anchor_pos_b": (
          config.Obs(mdp.obs_anchor_pos_b).configure(output_shape=(3,))
      ),
      "motion_anchor_ori_b": (
          config.Obs(mdp.obs_anchor_ori_b).configure(output_shape=(6,))
      ),
      "body_pos": (
          config.Obs(mdp.obs_robot_body_pos_b).configure(
              output_shape=(num_bodies * 3,)
          )
      ),
      "body_ori": (
          config.Obs(mdp.obs_robot_body_ori_b).configure(
              output_shape=(num_bodies * 6,)
          )
      ),
      "base_lin_vel": (
          config.Obs("body_tracking/base_lin_vel").configure(output_shape=(3,))
      ),
      "base_ang_vel": (
          config.Obs("body_tracking/base_ang_vel").configure(output_shape=(3,))
      ),
      "joint_pos": (
          config.Obs(
              mdp.obs_joint_pos_rel,
              default_joint_pos=default_joint_pos,
              qpos_key="robot/joint_pos",
          ).configure(output_shape=(num_joints,))
      ),
      "joint_vel": (
          config.Obs("robot/joint_vel").configure(output_shape=(num_joints,))
      ),
      "actions": config.Obs("action").configure(output_shape=(num_joints,)),
  }

  observations = {
      "policy": config.ObsGroup(terms=actor_terms, enable_noise=not play),
      "critic": config.ObsGroup(terms=critic_terms, enable_noise=False),
  }

  ##
  # Resetters: DomainRandomizationResetter, MotionResetter, PushRobotResetter.
  ##

  resetters = {
      "domain_randomization": config.Component(
          common_resetter.DomainRandomizationResetter,
          terms=[
              config.Term(
                  model_randomization.randomize_field,
                  trigger="startup",
                  field="body_ipos",
                  pattern=COM_BODY_NAME,
                  noise=noise.uniform_per_axis(COM_OFFSET_RANGES),
              ),
              config.Term(
                  model_randomization.randomize_field,
                  trigger="startup",
                  field="geom_friction",
                  pattern=g1_device.FOOT_GEOM_REGEX,
                  noise=noise.uniform_per_axis(
                      {0: FOOT_FRICTION_RANGE}, op="replace"
                  ),
              ),
          ],
      ),
      "motion_reset": config.Component(
          reset.MotionResetter,
          joint_names=g1_device.JOINT_NAMES,
          soft_joint_pos_limits=soft_limits,
          pose_range={} if play else POSE_RANGE,
          velocity_range={} if play else VELOCITY_RANGE,
          joint_position_range=(0.0, 0.0) if play else JOINT_POSITION_RANGE,
          command_name="motion",
      ),
  }
  if not play:
    resetters["push_robot"] = config.Component(
        reset.PushRobotResetter,
        velocity_range=VELOCITY_RANGE,
        interval_range_s=(1.0, 3.0),
        control_timestep=_CONTROL_TIMESTEP,
    )

  return config.Environment(
      base_scene=base_scene,
      device_layer=config.DeviceLayer(
          devices=devices,
          command_processors=(
              {"ghost_updater": ghost_updater} if ghost_updater else {}
          ),
      ),
      task_layer=config.TaskLayer(
          command_processors={"action_scale": action_scale_comp},
          feature_producers={"motion_reference": motion_ref_comp},
          command_feature_producers={"action_history": action_history_comp},
          rewards=rewards,
          metrics=metrics,
          terminations=terminations,
      ),
      observation_adapter=observations,
      resetters=resetters,
      sim=config.Sim(
          control_timestep=_CONTROL_TIMESTEP,
          simulation_timestep=_SIMULATION_TIMESTEP,
          nconmax=35,
          njmax=350,
      ),
      episode_length_secs=episode_length_secs,
  )
