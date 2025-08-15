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
"""Joystick task for Spot."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import spot_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=300.0,
      Kd=1.0,
      early_termination=True,
      action_repeat=1,
      action_scale=0.3,
      history_len=3,
      obs_noise=config_dict.create(
          scales=config_dict.create(
              joint_pos=0.05,
              gyro=0.1,
              gravity=0.03,
              feet_pos=[0.01, 0.005, 0.02],
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.8,
              lin_vel_z=-2.0,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              termination=-1.0,
              posture=1.0,
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              feet_slip=-0.1,
              feet_clearance=-2.0,
              feet_height=-0.1,
              feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.12,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.8, 0.8],
          ang_vel_yaw=[-1.0, 1.0],
      ),
      impl="jax",
      nconmax=4 * 8192,
      njmax=12 + 4 * 4,
  )


class Joystick(spot_base.SpotEnv):
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
    self._post_init()
    self._pert_func = (
        self._maybe_apply_perturbation
        if config.pert_config.enable
        else lambda state, _: state
    )

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Weights for the posture cost.
    self._weights = jp.array([1.0, 1.0, 1.0] * 4)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    # Sample frequency and gait.
    # gait_freq = jax.random.uniform(
    #     rng,
    #     minval=self._config.gait_frequency[0],
    #     maxval=self._config.gait_frequency[1],
    # )
    # phase_dt = 2 * jp.pi * self.dt * gait_freq
    # gait = jax.random.randint(
    #     rng, minval=0, maxval=len(self._config.gaits), shape=()
    # )
    # phase = jp.array(_PHASES[gait])
    # foot_height = jax.random.uniform(
    #     rng,
    #     minval=self._config.foot_height[0],
    #     maxval=self._config.foot_height[1],
    # )

    rng, cmd_rng, noise_rng = jax.random.split(rng, 3)
    info = {
        "rng": rng,
        "step": 0,
        "command": self.sample_command(cmd_rng),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "qpos_error_history": jp.zeros(self._config.history_len * 12),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        # "gait_freq": gait_freq,
        # "gait": gait,
        # "phase": phase,
        # "phase_dt": phase_dt,
        # "foot_height": foot_height,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng, pert_rng = jax.random.split(state.info["rng"], 4)

    state = self._pert_func(state, pert_rng)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, noise_rng)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1

    # phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    # state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

    state.info["rng"] = rng
    state.info["command"] = jp.where(
        state.info["step"] > 200,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 200),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.85
    return jp.where(
        self._config.early_termination,
        fall_termination,
        jp.zeros((), dtype=fall_termination.dtype),
    )

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      rng: jax.Array,
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)  # (3,)
    rng, noise_rng = jax.random.split(rng)
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.obs_noise.scales.gyro
    )

    gravity = self.get_gravity(data)  # (3,)
    rng, noise_rng = jax.random.split(rng)
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.obs_noise.scales.gravity
    )

    joint_angles = data.qpos[7:]  # (12,)
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.scales.joint_pos
    )

    feet_pos = self.get_feet_pos(data)  # (4, 3)
    rng, noise_rng = jax.random.split(rng)
    noisy_feet_pos = feet_pos.at[..., 0].add(
        (2 * jax.random.uniform(noise_rng, shape=feet_pos[..., 0].shape) - 1)
        * self._config.obs_noise.scales.feet_pos[0]
    )
    noisy_feet_pos = noisy_feet_pos.at[..., 1].add(
        (2 * jax.random.uniform(noise_rng, shape=feet_pos[..., 1].shape) - 1)
        * self._config.obs_noise.scales.feet_pos[1]
    )
    noisy_feet_pos = noisy_feet_pos.at[..., 2].add(
        (2 * jax.random.uniform(noise_rng, shape=feet_pos[..., 2].shape) - 1)
        * self._config.obs_noise.scales.feet_pos[2]
    )
    feet_pos = feet_pos.ravel()  # (12,)
    noisy_feet_pos = noisy_feet_pos.ravel()  # (12,)

    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 12)
        .at[:12]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    # cos = jp.cos(info["phase"])
    # sin = jp.sin(info["phase"])
    # phase = jp.concatenate([cos, sin])

    state = jp.hstack([
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        qpos_error_history,
        noisy_feet_pos,
        info["last_act"],
        info["command"],
    ])
    privileged_state = jp.hstack([
        state,
        gyro,
        self.get_accelerometer(data),
        gravity,
        self.get_local_linvel(data),
        self.get_global_angvel(data),
        joint_angles - self._default_pose,
        feet_pos,
        data.qvel[6:],
        data.actuator_force,
        info["last_contact"],
        data.sensordata[self._foot_linvel_sensor_adr].ravel(),
        info["feet_air_time"],
        data.xfrc_applied[self._torso_body_id, :3],
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
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(
            self.get_gravity(data), data.qpos[2]
        ),
        "posture": self._reward_posture(data.qpos[7:], info["command"]),
        "termination": self._cost_termination(done),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info["command"]
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        # "feet_phase": self._cost_feet_phase(data, info["phase"],
        # info["foot_height"]),
    }

  # Tracking rewards.

  def _reward_posture(
      self, joint_angles: jax.Array, commands: jax.Array
  ) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose) * self._weights)
    cmd_norm = jp.linalg.norm(commands)
    weight = jp.where(
        cmd_norm < 0.01,
        -10.0,
        0.0,
    )
    return jp.exp(weight * cost)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(
      self, torso_zaxis: jax.Array, _: jax.Array
  ) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - info["last_act"]))
    c2 = jp.sum(jp.square(act - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2

  # Other rewards.

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  # Feet related rewards.

  def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # TODO(kevin): Desired foot height should be proportional to the command.
    # desired_z = 0.05 + jp.linalg.norm(command[:2]) * 0.1
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      command: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(command)
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    cost = jp.sum(jp.square(error) * first_contact)
    cost *= cmd_norm >= 0.01  # No penalty for zero commands.
    return cost

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm >= 0.01  # No reward for zero commands.
    return rew_air_time

  def _cost_feet_phase(self, data: mjx.Data, phase: jax.Array) -> jax.Array:
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(
        phase, swing_height=self._config.reward_config.max_foot_height
    )
    return jp.sum(jp.square(foot_z - rz))

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array) -> jax.Array:
    """Samples a random command with a 10% chance of being zero."""
    _, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
    cmd_config = self._config.command_config
    lin_vel_x = jax.random.uniform(
        rng1, minval=cmd_config.lin_vel_x[0], maxval=cmd_config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=cmd_config.lin_vel_y[0], maxval=cmd_config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=cmd_config.ang_vel_yaw[0],
        maxval=cmd_config.ang_vel_yaw[1],
    )
    cmd = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])
    return jp.where(jax.random.bernoulli(rng4, 0.1), jp.zeros(3), cmd)
