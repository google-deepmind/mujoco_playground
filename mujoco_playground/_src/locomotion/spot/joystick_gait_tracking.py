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
"""Joystick task with gait control for Spot."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import spot_constants as consts

_PHASES = np.array([
    [0, np.pi, np.pi, 0],  # trot
    [0, 0.5 * np.pi, np.pi, 1.5 * np.pi],  # walk
    [0, np.pi, 0, np.pi],  # pace
    [0, 0, np.pi, np.pi],  # bound
    [0, 0, 0, 0],  # pronk
])


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=400.0,
      Kd=10.0,
      early_termination=True,
      action_repeat=1,
      action_scale=0.6,
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
              # Rewards.
              feet_phase=2.0,
              tracking_lin_vel=0.5,
              tracking_ang_vel=0.5,
              # Costs.
              ang_vel_xy=-0.5,
              lin_vel_z=-0.5,
              hip_splay=-0.5,
          ),
          tracking_sigma=0.25,
      ),
      command_config=config_dict.create(
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.5, 0.5],
          ang_vel_yaw=[-1.0, 1.0],
      ),
      gait_frequency=[0.5, 4.0],
      gaits=["trot", "walk", "pace", "bound", "pronk"],
      foot_height=[0.08, 0.4],
      impl="jax",
      nconmax=4 * 8192,
      njmax=12 + 4 * 4,
  )


class JoystickGaitTracking(spot_base.SpotEnv):
  """Joystick task with gait control."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=str(consts.FEET_ONLY_FLAT_TERRAIN_XML),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._hx_idxs = jp.array([0, 3, 6, 9])
    self._hx_default_pose = self._default_pose[self._hx_idxs]
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

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

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, noise_rng, gait_freq_rng, gait_rng, foot_height_rng, cmd_rng = (
        jax.random.split(rng, 6)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Sample gait parameters.
    gait_freq = jax.random.uniform(
        gait_freq_rng,
        minval=self._config.gait_frequency[0],
        maxval=self._config.gait_frequency[1],
    )
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    gait = jax.random.randint(  # pylint: disable=redefined-outer-name
        gait_rng, minval=0, maxval=len(self._config.gaits), shape=()
    )
    phase = jp.array(_PHASES)[gait]
    foot_height = jax.random.uniform(
        foot_height_rng,
        minval=self._config.foot_height[0],
        maxval=self._config.foot_height[1],
    )

    info = {
        "command": self.sample_command(cmd_rng),
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "qpos_error_history": jp.zeros(self._config.history_len * 12),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "gait_freq": gait_freq,
        "gait": gait,
        "phase": phase,
        "phase_dt": phase_dt,
        "foot_height": foot_height,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    obs = self._get_obs(data, info, noise_rng, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng = jax.random.split(state.info["rng"], 3)

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
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, noise_rng, contact)
    done = self._get_termination(data)

    pos, neg = self._get_reward(data, action, state.info, state.metrics, done)
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg
    r_pos = sum(pos.values())
    r_neg = jp.exp(0.2 * sum(neg.values()))
    reward = r_pos * r_neg * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

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
      contact: jax.Array,
  ) -> jax.Array:
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

    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 12)
        .at[:12]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    # Concatenate final observation.
    return jp.hstack(
        [
            noisy_gyro,
            noisy_gravity,
            noisy_joint_angles,
            qpos_error_history,
            contact,
            phase,
            info["gait_freq"],
            info["gait"],
            info["foot_height"],
        ],
    )

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    del action, done, metrics  # Unused.
    pos = {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "feet_phase": self._reward_feet_phase(
            data, info["phase"], info["foot_height"]
        ),
    }
    neg = {
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data), info["gait"]
        ),
        "hip_splay": self._cost_hip_splay(data.qpos[7:]),
    }
    return pos, neg

  def _reward_feet_phase(
      self, data: mjx.Data, phase: jax.Array, foot_height: jax.Array
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    return jp.exp(-error / 0.1)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    reward = jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
    return reward

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _cost_hip_splay(self, joint_angles: jax.Array) -> jax.Array:
    current = joint_angles[self._hx_idxs]
    return jp.sum(jp.square(current - self._hx_default_pose))

  def _cost_lin_vel_z(self, global_linvel, gait: jax.Array) -> jax.Array:  # pylint: disable=redefined-outer-name
    # Penalize z axis base linear velocity unless pronk or bound.
    cost = jp.square(global_linvel[2])
    return cost * (gait > 2)

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

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
