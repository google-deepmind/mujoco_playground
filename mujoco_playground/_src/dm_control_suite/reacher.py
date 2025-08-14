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
"""Reacher environment."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "reacher.xml"
SMALL_TARGET = 0.015
BIG_TARGET = 0.05


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,  # 0.02 in DMC.
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=0,
  )


def _make_model(
    xml_path: epath.Path, target_size: float, assets: Dict[str, Any]
) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_string(xml_path.read_text(), assets)
  if mujoco.__version__ >= "3.3.0":
    target_body = spec.body("target")
  else:
    target_body = spec.find_body("target")
  target_geom = target_body.first_geom()
  target_geom.size[0] = target_size
  return spec.compile()


class Reacher(mjx_env.MjxEnv):
  """Reacher environment."""

  def __init__(
      self,
      target_size: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._target_size = target_size
    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = _make_model(_XML_PATH, target_size, self._model_assets)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._target_mocap_body_id = self.mj_model.body("target").mocapid[0]
    self._finger_geom_id = self.mj_model.geom("finger").id
    self._target_geom_id = self.mj_model.geom("target").id
    self._radii = self._mj_model.geom_size[
        [self._target_geom_id, self._finger_geom_id], 0
    ].sum()

    shoulder_joint_id = self.mj_model.joint("shoulder").id
    self._shoulder_qposadr = self.mj_model.jnt_qposadr[shoulder_joint_id]
    wrist_joint_id = self.mj_model.joint("wrist").id
    self._wrist_qposadr = self.mj_model.jnt_qposadr[wrist_joint_id]
    self._wrist_lower = self.mj_model.jnt_range[wrist_joint_id, 0]
    self._wrist_upper = self.mj_model.jnt_range[wrist_joint_id, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._shoulder_qposadr].set(
        jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi)
    )
    qpos = qpos.at[self._wrist_qposadr].set(
        jax.random.uniform(
            rng2, (), minval=self._wrist_lower, maxval=self._wrist_upper
        )
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    angle = jax.random.uniform(rng3, ()) * 2 * jp.pi
    radius = jax.random.uniform(rng4, (), minval=0.05, maxval=0.2)
    xy = radius * jp.array([jp.sin(angle), jp.cos(angle)])
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._target_mocap_body_id, :2].set(xy)
    )

    metrics = {}
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        data.qpos,
        self._finger_to_target(data),
        data.qvel,
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info, metrics  # Unused.
    return reward.tolerance(self._finger_to_target_dist(data), (0, self._radii))

  def _finger_to_target(self, data: mjx.Data) -> jax.Array:
    target_pos = data.geom_xpos[self._target_geom_id, :2]
    finger_pos = data.geom_xpos[self._finger_geom_id, :2]
    return target_pos - finger_pos

  def _finger_to_target_dist(self, data: mjx.Data) -> jax.Array:
    return jp.linalg.norm(self._finger_to_target(data))

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
