# Copyright 2024 DeepMind Technologies Limited
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
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
import numpy as np


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      Kp=50.0,
      Kd=1.0,
      episode_length=300,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1,
      action_scale=0.6,
      obs_noise=0.05,
      obs_history_size=15,
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=1.0,
              torso_height=1.0,
              torques=-0.0002,
              action_rate=-0.001,
              energy=0.0,
          ),
      ),
  )


class Getup(go1_base.Go1Env):
  """Recover from a fall and stand up."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=str(consts.FULL_XML),
        Kp=config.Kp,
        Kd=config.Kd,
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
    self._settle_steps = np.round(
        self._config.settle_time / self.sim_dt
    ).astype(np.int32)
    self._z_des = 0.25
    self._up_vec = jp.array([0.0, 0.0, 1.0])

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    rng, height_rng, orientation_rng, qpos_rng = jax.random.split(rng, 4)

    qpos = jp.zeros(self.mjx_model.nq)

    # Initialize height and orientation of the root body.
    height = jax.random.uniform(height_rng, minval=0.6, maxval=0.7)
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    # Randomize the joint angles.
    # TODO(kevin): Switch to sampling a non-colliding pose within the full
    # joint range.
    noise = jax.random.uniform(qpos_rng, (12,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[7:].set(
        jp.clip(self._default_pose + noise, self._lowers, self._uppers)
    )

    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, noise_rng, reset_rng1, reset_rng2 = jax.random.split(rng, 4)

    # Randomly drop from height or initialize at default pose.
    qpos = jp.where(
        jax.random.bernoulli(reset_rng1, self._config.drop_from_height_prob),
        self._get_random_qpos(reset_rng2),
        self._init_q,
    )

    data = mjx_env.init(
        self.mjx_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv)
    )

    # Settle the robot.
    data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
    data = data.replace(time=0.0)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(self._config.obs_history_size * 33)
    obs = self._get_obs(data, info, obs_history, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, noise_rng = jax.random.split(state.info["rng"], 2)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info, state.obs, noise_rng)

    joint_angles = data.qpos[7:]
    done = jp.any(joint_angles < self._lowers)
    done |= jp.any(joint_angles > self._uppers)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Bookkeeping.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"] = rng
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
        self.get_gyro(data),  # 3
        self.get_accelerometer(data),  # 3
        self.get_gravity(data),  # 3
        data.qpos[7:] - self._default_pose,  # 12
        info["last_act"],  # 12
    ])  # total = 33
    if self._config.obs_noise >= 0.0:
      noise = self._config.obs_noise * jax.random.uniform(
          rng, obs.shape, minval=-1.0, maxval=1.0
      )
      obs = jp.clip(obs, -100.0, 100.0) + noise
    obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
    return obs

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics  # Unused.
    return {
        "orientation": self._reward_orientation(self.get_gravity(data)),
        "torso_height": self._reward_torso_height(data.qpos[2]),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
    }

  def _reward_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Reward for being upright.
    return (0.5 * torso_zaxis[2] + 0.5) ** 2

  def _reward_torso_height(self, torso_height: jax.Array) -> jax.Array:
    # Reward for torso beight at desired height.
    error = jp.clip((self._z_des - torso_height) / self._z_des, 0.0, 1.0)
    return 1.0 - error

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    # Penalize first and second derivative of actions.
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2
