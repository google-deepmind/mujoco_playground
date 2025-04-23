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
"""Quadruped environment."""

from itertools import product
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "quadruped.xml"

WALK_SPEED = 0.5
RUN_SPEED = 5.0


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      vision=False,
  )


def _find_non_contacting_height(
    mjx_model, data, orientation, x_pos=0.0, y_pos=0.0
):
  def body_fn(state):
    z_pos, num_contacts, num_attempts, _ = state
    qpos = data.qpos.at[:3].set(jp.array([x_pos, y_pos, z_pos]))
    qpos = qpos.at[3:7].set(jp.array(orientation))
    ndata = data.replace(qpos=qpos)
    ndata = mjx.forward(mjx_model, ndata)
    num_contacts = ndata.ncon
    z_pos += 0.01
    num_attempts += 1
    return (z_pos, num_contacts, num_attempts, ndata)

  initial_state = (0.0, 1, 0, data)  # (z_pos, num_contacts, num_attempts)
  *_, num_attemps, ndata = jax.lax.while_loop(
      lambda state: jp.greater(state[1], 0) & jp.less_equal(state[2], 10000),
      body_fn,
      initial_state,
  )
  ndata = jax.tree_map(
      lambda x, y: jp.where(jp.less(num_attemps, 10000), x, y), ndata, data
  )
  return ndata


class Quadruped(mjx_env.MjxEnv):
  """Quadruped environment."""

  def __init__(
      self,
      desired_speed: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError("Vision not implemented for Quadruped.")
    self._desired_speed = desired_speed
    self._xml_path = _XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), common.get_assets()
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

  def _post_init(self):
    self._force_torque_names = [
        f"{f}_toe_{pos}_{side}"
        for (f, pos, side) in product(
            ("force", "torque"), ("front", "back"), ("left", "right")
        )
    ]
    self._torso_id = self._mj_model.body("torso").id

  def reset(self, rng: jax.Array) -> mjx_env.State:
    data = mjx_env.init(self.mjx_model)
    metrics = {"reward/upright": jp.zeros(()), "reward/move": jp.zeros(())}
    info = {"rng": rng}
    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    lower, upper = (
        self._mj_model.actuator_ctrlrange[:, 0],
        self._mj_model.actuator_ctrlrange[:, 1],
    )
    action = (action + 1.0) / 2.0 * (upper - lower) + lower
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info
    ego = self._egocentric_state(data)
    torso_vel = self.torso_velocity(data)
    upright = self.torso_upright(data)
    imu = self.imu(data)
    force_torque = self.force_torque(data)
    return jp.hstack((ego, torso_vel, upright, imu, force_torque))

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del info, action
    move_reward = reward.tolerance(
        self.torso_velocity(data)[0],
        bounds=(self._desired_speed, float("inf")),
        sigmoid="linear",
        margin=self._desired_speed,
        value_at_margin=0.5,
    )
    upright_reward = self._upright_reward(data)
    metrics["reward/move"] = move_reward
    metrics["reward/upright"] = upright_reward
    return move_reward * upright_reward

  def _upright_reward(self, data: mjx.Data) -> jax.Array:
    upright = self.torso_upright(data)
    return reward.tolerance(
        upright,
        bounds=(1, float("inf")),
        sigmoid="linear",
        margin=2,
        value_at_margin=0,
    )

  def _egocentric_state(self, data: mjx.Data) -> jax.Array:
    return jp.hstack((data.qpos[7:], data.qvel[7:], data.act))

  def torso_upright(self, data: mjx.Data) -> jax.Array:
    return data.xmat[self._torso_id, 2, 2]

  def torso_velocity(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")

  def imu(self, data: mjx.Data) -> jax.Array:
    gyro = mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro")
    accelerometer = mjx_env.get_sensor_data(self.mj_model, data, "imu_accel")
    return jp.hstack((gyro, accelerometer))

  def force_torque(self, data: mjx.Data) -> jax.Array:
    return jp.hstack([
        mjx_env.get_sensor_data(self.mj_model, data, name)
        for name in self._force_torque_names
    ])

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
