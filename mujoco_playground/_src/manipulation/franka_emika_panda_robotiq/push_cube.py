# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Panda robotiq push cube environment."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src import types

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.franka_emika_panda_robotiq import panda_robotiq
import numpy as np

WORKSPACE_MIN = (0.3, -0.5, 0.0)
WORKSPACE_MAX = (0.75, 0.7, 0.5)
OBJ_SAMPLE_MIN = (0.4, -0.2, -0.005)
OBJ_SAMPLE_MAX = (0.65, 0.2, 0.04)


def default_config():
  """Returns reward config for the environment."""

  return config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.005,
      episode_length=3000,
      action_repeat=4,
      action_scale=0.1,
      action_history_len=5,
      obs_history_len=30,
      noise_config=config_dict.create(
          action_min_delay=1,  # env steps
          action_max_delay=3,  # env steps
          obs_min_delay=6,  # env steps
          obs_max_delay=12,  # env steps
          noise_scales=config_dict.create(
              obj_pos=0.015,  # meters
              obj_angle=7.5,  # degrees
              robot_qpos=0.1,  # radians
              robot_qvel=0.1,  # radians/s
              eef_pos=0.02,  # meters
              eef_angle=5.0,  # degrees
          ),
      ),
      reward_config=config_dict.create(
          termination_reward=-50.0,
          success_reward=500.0,
          success_wait_reward=3.0,
          success_step_count=30,
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=2.0,
              # Box goes to the target mocap.
              box_target=8.0,
              box_orientation=6.0,
              # Gripper collides with side of object, instead of top.
              gripper_collision_side=1.0,
              # Arm stays close to target pose.
              robot_target_qpos=0.75,
              # Reduce joint velocity.
              joint_vel=1.0,
              # Avoid joint vel limits.
              joint_vel_limit=3.0,
              # Torque penalty of the arm.
              total_command=-0.1,
              # Reduce action rate.
              action_rate=-0.1,
          ),
      ),
      impl="jax",
      nconmax=32 * 8192,
      njmax=256,
  )


def get_rand_dir(rng: jax.Array) -> jax.Array:
  key1, key2 = jax.random.split(rng)
  theta = jax.random.normal(key1) * 2 * jp.pi
  phi = jax.random.normal(key2) * jp.pi
  x = jp.sin(phi) * jp.cos(theta)
  y = jp.sin(phi) * jp.sin(theta)
  z = jp.cos(phi)
  return jp.array([x, y, z])


class PandaRobotiqPushCube(panda_robotiq.PandaRobotiqBase):
  """Environment for pushing a cube with a Panda robot and Robotiq gripper."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,  # pylint: disable=unused-argument
  ):
    super().__init__(
        config,
        panda_robotiq._ENV_DIR / "xmls/scene_panda_robotiq_cube.xml",
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")

  def _get_rand_target_pos(
      self, rng: jax.Array, offset: jax.Array
  ) -> jax.Array:
    min_pos = jp.array([-offset * 0.4, -offset, -0.005]) + self._init_obj_pos
    max_pos = jp.array([offset * 0.4, offset, 0.005]) + self._init_obj_pos
    pos = jax.random.uniform(rng, (3,), minval=min_pos, maxval=max_pos)
    return jp.clip(pos, np.array(OBJ_SAMPLE_MIN), np.array(OBJ_SAMPLE_MAX))

  def _get_rand_target_quat(
      self, rng: jax.Array, max_angle: jax.Array
  ) -> jax.Array:
    perturb_axis = jp.array([0.0, 0.0, 1.0], dtype=float)
    max_angle_rad = max_angle * jp.pi / 180
    perturb_theta = jax.random.uniform(rng, maxval=max_angle_rad)
    target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)
    return target_quat

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_box1, rng_box2, rng_target, rng_robot_arm, rng_theta = (
        jax.random.split(rng, 6)
    )

    # intialize box position
    box_pos = self._get_rand_target_pos(rng_box1, jp.array(0.15))
    box_pos = box_pos.at[2].set(self._init_obj_pos[2])
    box_quat = self._get_rand_target_quat(rng_box2, jp.array(360))

    # initialize target position
    target_pos = self._get_rand_target_pos(rng_target, jp.array(0.05))

    # initialize target orientation
    target_quat = self._get_rand_target_quat(rng_theta, jp.array(45))
    target_quat = math.quat_mul(box_quat, target_quat)

    # initialize mjx.Data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 7]
        .set(jp.concatenate([box_pos, box_quat]))
    )
    # sample random joint position for robot arm
    init_q = init_q.at[self._robot_arm_qposadr].set(
        init_q[self._robot_arm_qposadr]
        + 0.3
        * jax.random.uniform(
            rng_robot_arm,
            (7,),
            minval=self._jnt_range[:, 0] * self._joint_range_init_percent_limit,
            maxval=self._jnt_range[:, 1] * self._joint_range_init_percent_limit,
        )
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        mocap_pos=jp.array([target_pos]),
        mocap_quat=jp.array([target_quat]),
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "success": jp.array(0.0, dtype=float),
        "success_1": jp.array(0.0, dtype=float),
        "success_2": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.reward_scales.keys()},
    }
    info = {
        "rng": rng,
        "last_action": jp.zeros(7, dtype=float),
        "action_history": jp.zeros(self._config.action_history_len * 7),
        "success_step_count": jp.array(0, dtype=int),
        "prev_step_success": jp.array(0, dtype=int),
        "curriculum_id": jp.array(0, dtype=int),
        "angle_curriculum": jp.array([45, 45, 90, 135, 180], dtype=float),
        "pos_curriculum": jp.array([0.05, 0.05, 0.1, 0.2, 0.2], dtype=float),
    }
    obs = self._get_single_obs(data, info)
    info["obs_history"] = jp.zeros(self._config.obs_history_len * obs.shape[0])

    reward, done = jp.zeros(2)
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action_history = jp.roll(state.info["action_history"], 7).at[:7].set(action)
    state.info["action_history"] = action_history

    # add action delay
    state.info["rng"], key = jax.random.split(state.info["rng"])
    action_idx = jax.random.randint(
        key,
        (1,),
        minval=self._config.noise_config.action_min_delay,
        maxval=self._config.noise_config.action_max_delay,
    )
    action_w_delay = action_history.reshape((-1, 7))[action_idx[0]]

    # get the ctrl
    ctrl = action_w_delay * self._config.action_scale
    ctrl = jp.clip(
        ctrl, -self._max_torque / self._gear, self._max_torque / self._gear
    )
    # close the gripper
    ctrl = jp.concat((ctrl, jp.array([0.82], dtype=jp.float32)), axis=0)
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    # step the physics
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    state = state.replace(data=data)

    # calculate rewards
    rewards = self._get_reward(state.data, state.info, action)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    reward_scale_sum = sum(
        self._config.reward_config.reward_scales[k] for k in rewards
    )
    reward /= reward_scale_sum

    # termination reward
    termination = self._get_termination(state.data)
    reward += self._config.reward_config.termination_reward * termination

    # success reward
    state, success, success_wait = self._get_success(state)
    reward += self._config.reward_config.success_wait_reward * success_wait
    reward += self._config.reward_config.success_reward * success

    # finalize reward
    reward *= self.dt

    # reset target mocap if success
    state = self._reset_if_success(state, success)

    # calculate done
    state.metrics.update(out_of_bounds=termination.astype(float), **rewards)
    done = (
        termination
        | jp.isnan(state.data.qpos).any()
        | jp.isnan(state.data.qvel).any()
    )
    done = done.astype(float)

    # get observations
    obs = self._get_obs(state)

    # store info for the next step
    state.info["last_action"] = action_w_delay
    state.info["rng"], _ = jax.random.split(state.info["rng"])

    state = mjx_env.State(
        state.data, obs, reward, done, state.metrics, state.info
    )
    return state

  def _get_termination(self, data: mjx.Data):
    box_pos = data.xpos[self._obj_body]
    box_oob = box_pos[2] < -0.01
    box_oob |= (box_pos[0] > WORKSPACE_MAX[0]) | (box_pos[0] < WORKSPACE_MIN[0])
    box_oob |= (box_pos[1] > WORKSPACE_MAX[1]) | (box_pos[1] < WORKSPACE_MIN[1])
    gripper_pos = data.site_xpos[self._gripper_site]
    eef_oob = gripper_pos[2] > WORKSPACE_MAX[2]
    eef_oob |= (gripper_pos[0] > WORKSPACE_MAX[0]) | (
        gripper_pos[0] < WORKSPACE_MIN[0]
    )
    eef_oob |= (gripper_pos[1] > WORKSPACE_MAX[1]) | (
        gripper_pos[1] < WORKSPACE_MIN[1]
    )
    hand_wall_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._hand_wall_found_sensor
    ]
    has_wall_collision = sum(hand_wall_collision) > 0
    joints_near_limits = jp.any(
        jp.logical_or(
            data.qpos[self._robot_arm_qposadr]
            > (self._jnt_range[:, 1] * self._joint_limit_percentage),
            data.qpos[self._robot_arm_qposadr]
            < (self._jnt_range[:, 0] * self._joint_limit_percentage),
        )
    )
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._hand_floor_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    return (
        box_oob
        | eef_oob
        | has_wall_collision
        | joints_near_limits
        | floor_collision
    )

  def _get_success(
      self, state: mjx_env.State
  ) -> Tuple[mjx_env.State, jax.Array, jax.Array]:
    data = state.data
    target_pos = data.mocap_pos[self._mocap_target, :].ravel()
    target_quat = data.mocap_quat[self._mocap_target, :].ravel()
    box_pos = data.xpos[self._obj_body]
    box_quat = data.xquat[self._obj_body]
    ori_error = self._orientation_error(box_quat, target_quat)

    # get success condition
    success_cond_1 = jp.linalg.norm(target_pos - box_pos) < 0.03  # 3cm
    success_cond_2 = ori_error < (10 / 180 * jp.pi)  # 10 degrees
    success_cond_3 = (
        state.info["success_step_count"]
        >= self._config.reward_config.success_step_count
    )
    success = success_cond_1 & success_cond_2 & success_cond_3

    # report metrics
    state.metrics["success"] = success.astype(float)
    state.metrics["success_1"] = success_cond_1.astype(float)
    state.metrics["success_2"] = success_cond_2.astype(float)

    # calculate success counter for next step
    state.info["prev_step_success"] = (success_cond_1 & success_cond_2).astype(
        int
    )
    state.info["success_step_count"] = jp.where(
        state.info["prev_step_success"], state.info["success_step_count"] + 1, 0
    )
    state.info["prev_step_success"] *= 1 - success
    state.info["success_step_count"] *= 1 - success

    sub_success = success_cond_1 & success_cond_2
    return state, success, sub_success

  def _reset_if_success(
      self, state: mjx_env.State, success: jax.Array
  ) -> mjx_env.State:
    target_pos = state.data.mocap_pos[self._mocap_target, :].ravel()
    target_quat = state.data.mocap_quat[self._mocap_target, :].squeeze()
    # increase curriculum step
    state.info["curriculum_id"] += success.astype(int)
    max_pos = state.info["pos_curriculum"][state.info["curriculum_id"]]
    max_angle = state.info["angle_curriculum"][state.info["curriculum_id"]]
    # sample new target position and orientation
    new_target_pos = jp.where(
        success,
        self._get_rand_target_pos(state.info["rng"], max_pos),
        target_pos,
    )
    new_target_quat = jp.where(
        success,
        math.quat_mul(
            state.data.xquat[self._obj_body],
            self._get_rand_target_quat(state.info["rng"], max_angle),
        ),
        target_quat,
    )
    data = state.data.replace(
        mocap_pos=jp.array([new_target_pos]),
        mocap_quat=jp.array([new_target_quat]),
    )
    return state.replace(data=data)

  def _get_reward(
      self, data: mjx.Data, info: dict[str, Any], action: jax.Array
  ) -> dict[str, jax.Array]:
    # Target, gripper, and object rewards.
    target_pos = data.mocap_pos[self._mocap_target, :].ravel()
    box_pos = data.xpos[self._obj_body]
    side_dir = box_pos - target_pos
    side_dir = math.normalize(side_dir) * 0.1 * (math.norm(side_dir) > 1e-3)
    box_side_pos = side_dir + box_pos
    gripper_pos = data.site_xpos[self._gripper_site]

    gripper_box = reward_util.tolerance(
        jp.linalg.norm(box_side_pos - gripper_pos),
        (0, 0.1),
        margin=1.0,
        sigmoid="linear",
    )

    box_target = reward_util.tolerance(
        jp.linalg.norm(box_pos[:2] - target_pos[:2]),
        (0, 0.005),
        margin=0.4,
        sigmoid="reciprocal",
    )

    target_quat = data.mocap_quat[self._mocap_target, :].squeeze()
    ori_error = self._orientation_error(data.xquat[self._obj_body], target_quat)
    box_orientation = reward_util.tolerance(
        ori_error, (0, 0.2), margin=jp.pi, sigmoid="reciprocal"
    )

    hand_box_normal = []
    for sensorid in self._gripper_obj_normal_sensor:
      adr = self._mj_model.sensor_adr[sensorid]
      dim = self._mj_model.sensor_dim[sensorid]
      hand_box_normal.append(data.sensordata[adr : adr + dim])

    hand_box_normal = jp.mean(jp.array(hand_box_normal), axis=0)
    hand_box_normal = math.normalize(hand_box_normal)
    hand_box_normal_side = jp.cross(jp.array([0.0, 0.0, 1.0]), hand_box_normal)
    gripper_collision_side = jp.linalg.norm(hand_box_normal_side)

    # Action regularization.
    robot_target_qpos = reward_util.tolerance(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        ),
        (0, 0.5),
        margin=4.5,
        sigmoid="linear",
    )
    joint_vel_mse = jp.linalg.norm(
        data.qvel[self._qd_low_joint_pos_index : self._qd_upper_joint_pos_index]
    )
    joint_vel = reward_util.tolerance(
        joint_vel_mse, (0, 0.5), margin=2.0, sigmoid="reciprocal"
    )
    total_command = jp.linalg.norm(action)
    action_rate = jp.linalg.norm(action - info["last_action"])

    joints_near_vel_limits = jp.any(
        jp.logical_or(
            data.qvel[self._robot_arm_qposadr]
            > (self._jnt_vel_range[:, 1] * self._joint_vel_limit_percentage),
            data.qvel[self._robot_arm_qposadr]
            < (self._jnt_vel_range[:, 0] * self._joint_vel_limit_percentage),
        )
    )

    return {
        "box_target": box_target,
        "box_orientation": box_orientation,
        "gripper_box": gripper_box,
        "gripper_collision_side": gripper_collision_side,
        "robot_target_qpos": robot_target_qpos,
        "joint_vel": joint_vel,
        "joint_vel_limit": 1 - joints_near_vel_limits,
        "total_command": total_command,
        "action_rate": action_rate,
    }

  def _orientation_error(self, object_quat, target_quat) -> jax.Array:
    quat_diff = math.quat_mul(object_quat, math.quat_inv(target_quat))
    quat_diff = math.normalize(quat_diff)
    ori_error = 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), a_max=1.0))
    return ori_error

  def _get_obs(self, state: mjx_env.State) -> jax.Array:
    obs = self._get_single_obs(state.data, state.info)
    obs_size = obs.shape[0]

    # fill the buffer
    obs_history = (
        jp.roll(state.info["obs_history"], obs_size).at[:obs_size].set(obs)
    )
    state.info["obs_history"] = obs_history

    # add observation delay
    state.info["rng"], key = jax.random.split(state.info["rng"])
    obs_idx = jax.random.randint(
        key,
        (1,),
        minval=self._config.noise_config.obs_min_delay,
        maxval=self._config.noise_config.obs_max_delay,
    )
    obs = obs_history.reshape((-1, obs_size))[obs_idx[0]]

    return obs

  def _get_single_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    target_pos = data.mocap_pos[self._mocap_target, :].ravel()
    target_quat = data.mocap_quat[self._mocap_target, :].ravel()
    target_mat = math.quat_to_mat(target_quat)

    # Add noise to object position and orientation.
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    angle = jax.random.uniform(
        key1,
        minval=0,
        maxval=self._config.noise_config.noise_scales.obj_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key2), angle)
    obj_quat = data.xquat[self._obj_body]
    obj_quat_w_noise = math.quat_mul(rand_quat, obj_quat)
    obj_pos = data.xpos[self._obj_body]
    obj_pos_w_noise = obj_pos + jax.random.uniform(
        key3, (3,), minval=-1, maxval=1
    ) * self._config.noise_config.noise_scales.obj_pos

    # Add noise to robot proprio observation.
    info["rng"], key1, key2, key3, key4 = jax.random.split(info["rng"], 5)
    robot_qpos = data.qpos[
        self._q_low_joint_pos_index : self._q_upper_joint_pos_index
    ]
    robot_qpos_w_noise = robot_qpos + jax.random.uniform(
        key1, minval=0, maxval=self._config.noise_config.noise_scales.robot_qpos
    )
    robot_qvel = data.qvel[
        self._qd_low_joint_pos_index : self._qd_upper_joint_pos_index
    ]
    robot_qvel_w_noise = robot_qvel + jax.random.uniform(
        key2, minval=0, maxval=self._config.noise_config.noise_scales.robot_qvel
    )
    gripper_pos = data.site_xpos[self._gripper_site]
    # Consider running FK to get the noisy eef position/orientation.
    gripper_pos_w_noise = gripper_pos + jax.random.uniform(
        key3, minval=0, maxval=self._config.noise_config.noise_scales.eef_pos
    )
    gripper_mat = data.site_xmat[self._gripper_site]
    angle = jax.random.uniform(
        key1,
        minval=0,
        maxval=self._config.noise_config.noise_scales.eef_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key4), angle)
    rand_mat = math.quat_to_mat(rand_quat)
    gripper_mat_w_noise = rand_mat @ gripper_mat

    target_orientation = target_mat.ravel()[3:]
    obj_orientation_w_noise = math.quat_to_mat(obj_quat_w_noise).ravel()[3:]

    obs = jp.concatenate([
        target_pos,
        target_orientation,
        info["last_action"],
        # Robot joint angles and velocities.
        robot_qpos_w_noise,
        robot_qvel_w_noise,
        # End effector position and orientaiton.
        gripper_pos_w_noise,
        gripper_mat_w_noise.ravel()[3:],
        # Object position and orientation.
        obj_orientation_w_noise,
        obj_pos_w_noise,
    ])
    return obs

  @property
  def action_size(self):
    return 7
