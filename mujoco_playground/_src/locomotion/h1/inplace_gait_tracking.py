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
"""Inplace gait tracking for H1."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.h1 import base as h1_base
from mujoco_playground._src.locomotion.h1 import h1_constants as consts

_PHASES = np.array([
    [0, np.pi],  # walk
    [0, 0],  # jump
])


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      early_termination=True,
      action_repeat=1,
      action_scale=0.6,
      history_len=3,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Rewards.
              feet_phase=2.0,
              # Costs.
              pose=-0.5,
              ang_vel=-0.5,
              lin_vel=-0.5,
          ),
      ),
      gait_frequency=[0.5, 4.0],
      gaits=["walk", "jump"],
      foot_height=[0.08, 0.4],
      impl="jax",
      nconmax=8 * 8192,
      njmax=19 + 8 * 4,
  )


class InplaceGaitTracking(h1_base.H1Env):
  """Inplace gait tracking for H1."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FEET_ONLY_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._config = config
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    self._hx_idxs = jp.array([
        0,
        1,
        4,  # left leg
        5,
        6,
        9,  # right leg
        10,  # torso
        11,
        12,
        13,
        14,  # left arm
        15,
        16,
        17,
        18,  # right arm
    ])
    self._weights = jp.array([
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])
    self._hx_default_pose = self._default_pose[self._hx_idxs]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._left_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
    )
    self._right_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
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
    rng, noise_rng, gait_freq_rng, gait_rng, foot_height_rng = jax.random.split(
        rng, 5
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

    # Initialize history buffers.
    qpos_error_history = jp.zeros(self._config.history_len * 19)
    qvel_history = jp.zeros(self._config.history_len * 19)

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
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "qpos_error_history": qpos_error_history,
        "qvel_history": qvel_history,
        "swing_peak": jp.zeros(2),
        "gait_freq": gait_freq,
        "gait": gait,
        "phase": phase,
        "phase_dt": phase_dt,
        "foot_height": foot_height,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    left_feet_contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._left_foot_floor_found_sensor
    ])
    right_feet_contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._right_foot_floor_found_sensor
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
    info["left_contact"] = jp.any(left_feet_contact)
    info["right_contact"] = jp.any(right_feet_contact)
    obs = self._get_obs(data, info, noise_rng, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, noise_rng = jax.random.split(state.info["rng"])

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    left_feet_contact = jp.array([
        data.sensordata[sensorid] > 0
        for sensorid in self._left_foot_floor_found_sensor
    ])
    right_feet_contact = jp.array([
        data.sensordata[sensorid] > 0
        for sensorid in self._right_foot_floor_found_sensor
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
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
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["rng"] = rng
    state.info["swing_peak"] *= ~contact
    state.info["left_contact"] = jp.any(left_feet_contact)
    state.info["right_contact"] = jp.any(right_feet_contact)

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Terminates if joint limits are exceeded or the robot falls.
    joint_angles = data.qpos[7:]
    joint_limit_exceed = jp.any(joint_angles < self._lowers)
    joint_limit_exceed |= jp.any(joint_angles > self._uppers)
    fall_termination = self.get_gravity(data)[-1] < 0.85
    return jp.where(
        self._config.early_termination,
        joint_limit_exceed | fall_termination,
        joint_limit_exceed,
    )

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      rng: jax.Array,
      contact: jax.Array,
  ) -> jax.Array:
    obs = jp.concatenate([
        self.get_gyro(data),  # 3
        self.get_gravity(data),  # 3
        data.qpos[7:] - self._default_pose,  # 19
        data.qvel[6:],  # 19
        info["last_act"],  # 19
    ])

    # Add noise.
    noise_vec = jp.zeros_like(obs)
    noise_vec = noise_vec.at[:3].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.gyro
    )
    noise_vec = noise_vec.at[3:6].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.gravity
    )
    noise_vec = noise_vec.at[6:25].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.joint_pos
    )
    noise_vec = noise_vec.at[25:44].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.joint_vel
    )
    obs = obs + (2 * jax.random.uniform(rng, shape=obs.shape) - 1) * noise_vec

    # Update history.
    qvel_history = jp.roll(info["qvel_history"], 19).at[:19].set(data.qvel[6:])
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 19)
        .at[:19]
        .set(data.qpos[7:] - info["motor_targets"])
    )
    info["qvel_history"] = qvel_history
    info["qpos_error_history"] = qpos_error_history

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    # Concatenate final observation.
    obs = jp.hstack(
        [
            obs,
            qvel_history,
            qpos_error_history,
            contact,
            phase,
            info["gait_freq"],
            info["gait"],
            info["foot_height"],
        ],
    )
    return obs

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: Dict[str, Any],
      metrics: Dict[str, Any],
      done: jax.Array,
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    del action, done, metrics  # Unused.
    pos = {
        "feet_phase": self._reward_feet_phase(
            data, info["phase"], info["foot_height"]
        ),
    }
    neg = {
        "ang_vel": self._cost_ang_vel(self.get_global_angvel(data)),
        "lin_vel": self._cost_lin_vel(self.get_global_linvel(data)),
        "pose": self._cost_pose(data.qpos[7:]),
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

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    # Penalize deviation from the default pose for certain joints.
    current = joint_angles[self._hx_idxs]
    return jp.sum(jp.square(current - self._hx_default_pose) * self._weights)

  def _cost_ang_vel(self, global_angvel: jax.Array) -> jax.Array:
    # Penalize angular velocity.
    return jp.sum(jp.square(global_angvel))

  def _cost_lin_vel(self, global_linvel: jax.Array) -> jax.Array:
    # Penalize xy linear velocity.
    return jp.sum(jp.square(global_linvel[:2]))
