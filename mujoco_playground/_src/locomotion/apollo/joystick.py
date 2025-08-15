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
"""Joystick task for Apollo."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import gait, mjx_env
from mujoco_playground._src.locomotion.apollo import base
from mujoco_playground._src.locomotion.apollo import constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking=1.0,
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              torques=0.0,
              action_rate=0.0,
              energy=0.0,
              feet_phase=1.0,
              alive=0.0,
              termination=0.0,
              pose=-1.0,
              collision=-1.0,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.12,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      command_config=config_dict.create(
          min=[-1.5, -0.8, -1.5],
          max=[1.5, 0.8, 1.5],
          zero_prob=[0.9, 0.25, 0.5],
      ),
      impl="jax",
      nconmax=8 * 8192,
      njmax=32 + 8 * 4,
  )


class Joystick(base.ApolloEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )

    self._cmd_min = jp.array(self._config.command_config.min)
    self._cmd_max = jp.array(self._config.command_config.max)
    self._cmd_zero_prob = jp.array(self._config.command_config.zero_prob)

    # fmt: off
    self._weights = jp.array([
        5.0, 5.0, 5.0,  # Torso.
        1.0, 1.0, 1.0,  # Neck.
        1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0,  # Left arm.
        1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0,  # Right arm.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # Left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # Right leg.
    ])
    # fmt: on

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize xy position and yaw, xy=+U(-0.5, 0.5), yaw=U(-pi, pi).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Perturb initial joint angles, qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:]
        * jax.random.uniform(
            key, (self.mjx_model.nq - 7,), minval=0.5, maxval=1.5
        )
    )

    # Perturb initial joint velocities, d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Sample gait frequency =U(1.25, 1.75).
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    # Sample command.
    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=self._cmd_min, maxval=self._cmd_max
    )

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "phase_dt": phase_dt,
        "phase": phase,
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "filtered_linvel": jp.zeros(3),
        "filtered_angvel": jp.zeros(3),
    }
    metrics = {
        "termination/fall_termination": jp.zeros(()),
    }
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state = self.apply_push(state)
    motor_targets = self._default_ctrl + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    linvel = self.get_local_linvel(data)
    state.info["filtered_linvel"] = (
        linvel * 1.0 + state.info["filtered_linvel"] * 0.0
    )
    angvel = self.get_gyro(data)
    state.info["filtered_angvel"] = (
        angvel * 1.0 + state.info["filtered_angvel"] * 0.0
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, state.metrics)
    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["phase"] = jp.where(
        jp.linalg.norm(state.info["command"]) > 0.01,
        state.info["phase"],
        jp.ones(2) * jp.pi,
    )
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(
      self, data: mjx.Data, metrics: dict[str, Any]
  ) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    metrics["termination/fall_termination"] = fall_termination.astype(
        jp.float32
    )
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _apply_noise(
      self, info: dict[str, Any], value: jax.Array, scale: float
  ) -> jax.Array:
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
    noisy_value = value + noise * self._config.noise_config.level * scale
    return noisy_value

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    # Ground-truth observations.
    gyro = self.get_gyro(data)
    gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    linvel = self.get_local_linvel(data)
    phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
    root_pos = data.qpos[:3]
    root_quat = data.qpos[3:7]
    actuator_torques = data.actuator_force
    # Noisy observations.
    noise_scales = self._config.noise_config.scales
    noisy_gyro = self._apply_noise(info, gyro, noise_scales.gyro)
    noisy_gravity = self._apply_noise(info, gravity, noise_scales.gravity)
    noisy_joint_angles = self._apply_noise(
        info, joint_angles, noise_scales.joint_pos
    )
    noisy_joint_vel = self._apply_noise(info, joint_vel, noise_scales.joint_vel)
    noisy_linvel = self._apply_noise(info, linvel, noise_scales.linvel)
    state = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        info["command"],
        noisy_joint_angles - self._init_q[7:],
        noisy_joint_vel,
        info["last_act"],
        phase,
    ])
    privileged_state = jp.hstack([
        state,
        # Unnoised.
        gyro,
        gravity,
        linvel,
        joint_angles - self._init_q[7:],
        joint_vel,
        # Extra.
        actuator_torques,
        root_pos,
        root_quat,
    ])
    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "termination": done,
        "alive": jp.array(1.0) - done,
        "tracking": self._reward_tracking(info["command"], info),
        "lin_vel_z": self._cost_lin_vel_z(info["filtered_linvel"]),
        "ang_vel_xy": self._cost_ang_vel_xy(info["filtered_angvel"]),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "feet_phase": self._reward_feet_phase(data, info["phase"]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "energy": self._cost_energy(data.qvel, data.actuator_force),
        "collision": self._cost_collision(data),
        "pose": self._cost_pose(data.qpos, info["command"]),
    }

  def _reward_tracking(
      self, commands: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    lin_vel_error = jp.sum(
        jp.square(commands[:2] - info["filtered_linvel"][:2])
    )
    r_linvel = jp.exp(
        -lin_vel_error / self._config.reward_config.tracking_sigma
    )
    ang_vel_error = jp.square(commands[2] - info["filtered_angvel"][2])
    r_angvel = jp.exp(
        -ang_vel_error / self._config.reward_config.tracking_sigma
    )
    return r_linvel + 0.5 * r_angvel

  def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
    return jp.square(local_linvel[2])

  def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
    return jp.sum(jp.square(local_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    torques = qfrc_actuator / self._actuator_torques
    return jp.sum(jp.abs(qvel[6:] * torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    adr = self._mj_model.sensor_adr
    # Hand - thigh.
    c = data.sensordata[adr[self._left_hand_left_thigh_found_sensor]] > 0
    c |= data.sensordata[adr[self._right_hand_right_thigh_found_sensor]] > 0
    # Foot - foot.
    c |= data.sensordata[adr[self._left_foot_right_foot_found_sensor]] > 0
    # Shin - shin.
    c |= data.sensordata[adr[self._left_shin_right_shin_found_sensor]] > 0
    # Thigh - thigh.
    c |= data.sensordata[adr[self._left_thigh_right_thigh_found_sensor]] > 0

    return jp.any(c)

  def _cost_pose(self, qpos: jax.Array, commands: jax.Array) -> jax.Array:
    # Uniform weights when standing still.
    weights = jp.where(
        jp.linalg.norm(commands) < 0.01,
        jp.ones_like(self._weights),
        self._weights,
    )
    # Reduce hip roll weight when lateral command is high.
    lateral_cmd = jp.abs(commands[1])
    hip_roll_weight = jp.where(lateral_cmd > 0.3, 0.01, 1.0)
    weights = weights.at[21].set(hip_roll_weight)
    weights = weights.at[27].set(hip_roll_weight)
    return jp.sum(jp.square(qpos[7:] - self._init_q[7:]) * weights)

  def _reward_feet_phase(self, data: mjx.Data, phase: jax.Array) -> jax.Array:
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    rz = gait.get_rz(
        phase, swing_height=self._config.reward_config.max_foot_height
    )
    error = jp.sum(jp.square(foot_z - rz))
    return jp.exp(-error / 0.01)

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=self._cmd_min, maxval=self._cmd_max
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_zero_prob, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    return x_k - w_k * (x_k - y_k * z_k)

  def apply_push(self, state: mjx_env.State) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    state.info["push"] = push
    state.info["push_step"] += 1
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)
    return state
