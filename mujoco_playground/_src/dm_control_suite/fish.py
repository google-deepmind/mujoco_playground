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
"""Fish environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "fish.xml"
_JOINTS = [
    "tail1",
    "tail_twist",
    "tail2",
    "finright_roll",
    "finright_pitch",
    "finleft_roll",
    "finleft_pitch",
]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.04,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=25,
  )


class Swim(mjx_env.MjxEnv):
  """Fish environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_body_id = self._mj_model.body("torso").id
    self._joints_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id] for j in _JOINTS
    ])
    self._target_geom_id = self._mj_model.geom("target").id
    self._target_body_id = self._mj_model.body("target").id
    self._mouth_geom_id = self._mj_model.geom("mouth").id
    self._target_mocap_body_id = self.mj_model.body("target").mocapid[0]

    self._radii = self._mj_model.geom_size[
        [self._mouth_geom_id, self._target_geom_id], 0
    ].sum()

    self._min_xyz = jp.array([-0.4, -0.4, 0.1])
    self._max_xyz = jp.array([0.4, 0.4, 0.3])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    qpos = jp.zeros(self.mjx_model.nq)

    quat = jax.random.normal(rng1, (4,))
    quat = quat / jp.linalg.norm(quat)
    qpos = qpos.at[3:7].set(quat)

    qpos = qpos.at[self._joints_qposadr].set(
        jax.random.uniform(
            rng2, (len(self._joints_qposadr),), minval=-0.2, maxval=0.2
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

    # Randomize target position.
    xyz = jax.random.uniform(
        rng3, (3,), minval=self._min_xyz, maxval=self._max_xyz
    )
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._target_mocap_body_id].set(xyz)
    )

    metrics = {
        "reward/in_target": jp.zeros(()),
        "reward/upright": jp.zeros(()),
    }
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
    upright = data.xmat[self._torso_body_id, 2, 2]
    joint_angles = data.qpos[self._joints_qposadr]
    mouth_to_target_global = (
        data.geom_xpos[self._target_geom_id]
        - data.geom_xpos[self._mouth_geom_id]
    )
    mouth_to_target_local = (
        mouth_to_target_global @ data.geom_xmat[self._mouth_geom_id]
    )
    return jp.concatenate([
        upright.reshape(1),
        joint_angles,
        mouth_to_target_local,
        data.qvel,
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    mouth_to_target_global = (
        data.geom_xpos[self._target_geom_id]
        - data.geom_xpos[self._mouth_geom_id]
    )
    mouth_to_target_local = (
        mouth_to_target_global @ data.geom_xmat[self._mouth_geom_id]
    )

    in_target = reward.tolerance(
        jp.linalg.norm(mouth_to_target_local),
        bounds=(0, self._radii),
        margin=2 * self._radii,
    )
    metrics["reward/in_target"] = in_target

    upright = data.xmat[self._torso_body_id, 2, 2]
    is_upright = 0.5 * (upright + 1)
    metrics["reward/upright"] = is_upright

    return (7 * in_target + is_upright) / 8

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
