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
"""Joystick for H1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.h1 import base as h1_base
from mujoco_playground._src.locomotion.h1 import h1_constants


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      obs_noise=0.0,
      max_foot_height=0.13,
      lin_vel_x=[-0.6, 1.5],
      lin_vel_y=[-0.8, 0.8],
      ang_vel_yaw=[-0.7, 0.7],
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.8,
              lin_vel_z=-2.0,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              torques=-0.0002,
              action_rate=-0.2,
              stand_still=-0.5,
              termination=-1.0,
              feet_slip=-0.1,
              feet_clearance=-0.5,
          ),
          tracking_sigma=0.25,
      ),
      impl="jax",
      nconmax=8 * 8192,
      njmax=19 + 8 * 4,
  )


class Joystick(h1_base.H1Env):
  """
  A class representing a joystick environment for locomotion control.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        h1_constants.FEET_ONLY_XML.as_posix(), config, config_overrides
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._lowers = self._mj_model.jnt_range[1:, 0]
    self._uppers = self._mj_model.jnt_range[1:, 1]
    self._torso_body_id = self._mj_model.body("torso_link").id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in h1_constants.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._left_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in h1_constants.LEFT_FEET_GEOMS]
    )
    self._right_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in h1_constants.RIGHT_FEET_GEOMS]
    )
    self._torso_mass = self._mj_model.body_mass[self._torso_body_id]
    foot_linvel_sensor_adr = []
    for site in h1_constants.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def sample_command(self, rng: jax.Array) -> jax.Array:
    _, rng1, rng2, rng3 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    return jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng = jax.random.split(rng, 3)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_vel": jp.zeros(self.mjx_model.nv - 6),
        "command": self.sample_command(cmd_rng),
        "step": 0,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(15 * 45)  # 15 steps of history.
    obs = self._get_obs(data, info, obs_history, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng = jax.random.split(state.info["rng"], 3)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self._n_frames  # pytype: disable=attribute-error
    )

    obs = self._get_obs(data, state.info, state.obs, noise_rng)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    torso_z = data.xpos[self._torso_body_id, -1]

    done = self._get_gravity(data)[-1] < 0
    done |= jp.any(joint_angles < self._lowers)
    done |= jp.any(joint_angles > self._uppers)
    done |= torso_z < 0.92

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Bookkeeping.
    state.info["last_act"] = action
    state.info["last_vel"] = joint_vel
    state.info["step"] += 1
    state.info["rng"] = rng
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      obs_history: jax.Array,
      rng: jax.Array,
  ) -> jax.Array:
    obs = jp.concatenate([
        self._get_localrpyrate(data)[-1].reshape(1),  # 1
        self._get_gravity(data),  # 3
        info["command"],  # 3
        data.qpos[7:] - self._default_pose,  # 19
        info["last_act"],  # 19
        # sum = 45
    ])
    if self._config.obs_noise >= 0.0:
      noise = self._config.obs_noise * jax.random.uniform(
          rng, obs.shape, minval=-1.0, maxval=1.0
      )
      obs = jp.clip(obs, -100.0, 100.0) + noise
    obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
    return obs

  def _get_gravity(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "upvector_torso")

  def _get_localrpyrate(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "localrpyrate_torso")

  def _get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_linvel_torso")

  def _get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_angvel_torso")

  def _get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "local_linvel_torso")

  def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
    sensor_id = self._mj_model.sensor(sensor_name).id
    sensor_adr = self._mj_model.sensor_adr[sensor_id]
    sensor_dim = self._mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

  # Reward functions.

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
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self._get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self._get_localrpyrate(data)
        ),
        # Regularization rewards.
        "lin_vel_z": self._cost_lin_vel_z(self._get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self._get_global_angvel(data)),
        "orientation": self._cost_orientation(self._get_gravity(data)),
        "torques": self._cost_torques(data.qfrc_actuator),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done, info["step"]),
        "feet_slip": self._cost_feet_slip(data),
        "feet_clearance": self._cost_feet_clearance(data),
    }

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

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions.
    return jp.sum(jp.square(act - last_act))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands.
    unit_cmd = commands[:2] / jp.linalg.norm(commands[:2])
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        unit_cmd[1] < 0.1
    )

  def _cost_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def _cost_feet_slip(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    left_feet_contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._left_foot_floor_found_sensor
    ])
    right_feet_contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._right_foot_floor_found_sensor
    ])
    feet_contact = jp.hstack(
        [left_feet_contact.any(), right_feet_contact.any()]
    )
    return jp.sum(vel_xy_norm_sq * feet_contact)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = (foot_z - self._config.max_foot_height) ** 2
    return jp.sum(delta * vel_norm)

  @property
  def xml_path(self) -> str:
    raise NotImplementedError()

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
