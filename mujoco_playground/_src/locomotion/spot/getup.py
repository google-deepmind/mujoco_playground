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
"""Fall recovery task for Spot."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import spot_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      Kp=400.0,
      Kd=20.0,
      episode_length=300,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1,
      action_scale=0.6,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=1.0,
              torso_height=1.0,
              posture=1.0,
              stand_still=1.0,
              torques=0.0,
              action_rate=0.0,
          ),
      ),
      impl="jax",
      nconmax=30 * 8192,
      njmax=12 + 30 * 4,
  )


class Getup(spot_base.SpotEnv):
  """Recover from a fall and stand up."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=str(consts.FULL_FLAT_TERRAIN_XML),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    self._settle_steps = np.round(
        self._config.settle_time / self.sim_dt
    ).astype(np.int32)
    self._z_des = self._init_q[2]
    self._up_vec = jp.array([0.0, 0.0, 1.0])

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    rng, orientation_rng, qpos_rng = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)

    # Initialize height and orientation of the root body.
    height = 1.0
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    # Randomize joint angles.
    qpos = qpos.at[7:].set(
        jax.random.uniform(
            qpos_rng, (12,), minval=self._lowers, maxval=self._uppers
        )
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

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

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

    obs = self._get_obs(data, info, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, noise_rng = jax.random.split(state.info["rng"], 2)

    motor_targets = state.data.qpos[7:] + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info, noise_rng)

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
      rng: jax.Array,
  ) -> jax.Array:
    gyro = self.get_gyro(data)
    rng, noise_rng = jax.random.split(rng)
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.gyro
    )

    gravity = self.get_gravity(data)
    rng, noise_rng = jax.random.split(rng)
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.gravity
    )

    joint_angles = data.qpos[7:]
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    return jp.concatenate([
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        info["last_act"],  # 12
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics  # Unused.

    gravity = self.get_gravity(data)
    torso_height = data.qpos[2]
    joint_angles = data.qpos[7:]
    joint_torques = data.actuator_force

    is_upright = self._is_upright(gravity)
    is_at_desired_height = self._is_at_desired_height(torso_height)
    gate = is_upright * is_at_desired_height

    return {
        "orientation": self._reward_orientation(gravity),
        "torso_height": self._reward_torso_height(torso_height),
        "posture": self._reward_posture(joint_angles, is_upright),
        "stand_still": self._reward_stand_still(action, gate),
        "action_rate": self._cost_action_rate(action, info),
        "torques": self._cost_torques(joint_torques),
    }

  def _is_upright(self, gravity: jax.Array, ori_tol: float = 0.01) -> jax.Array:
    ori_error = jp.sum(jp.square(self._up_vec - gravity))
    return ori_error < ori_tol

  def _is_at_desired_height(
      self, torso_height: jax.Array, pos_tol: float = 0.005
  ) -> jax.Array:
    height_error = jp.clip((self._z_des - torso_height) / self._z_des, 0.0, 1.0)
    return height_error < pos_tol

  def _reward_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(self._up_vec - torso_zaxis))
    return jp.exp(-2.0 * error)

  def _reward_torso_height(self, torso_height: jax.Array) -> jax.Array:
    error = jp.clip((self._z_des - torso_height) / self._z_des, 0.0, 1.0)
    return 1.0 - error

  def _reward_posture(
      self, joint_angles: jax.Array, gate: jax.Array
  ) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _reward_stand_still(self, act: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(act))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - info["last_act"]))
    c2 = jp.sum(jp.square(act - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2
