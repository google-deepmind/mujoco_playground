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
"""Finger environment.

Changes from the dm_control implementation:

- Changed integrator to implicitfast.
- Reduced the timestep to 0.005 (from 0.01).
"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "finger.xml"
# For TURN tasks, the 'tip' geom needs to enter a spherical target of sizes:
EASY_TARGET_SIZE = 0.07
HARD_TARGET_SIZE = 0.03

# Spinning faster than this value (radian/second) is considered spinning.
_SPIN_VELOCITY = 15.0


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=25_000,
      njmax=25,
  )


def _make_turn_model(
    xml_path: epath.Path, target_radius: float, assets: Dict[str, Any]
) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_string(xml_path.read_text(), assets)
  target_site = None
  for site in spec.sites:
    if site.name == "target":
      target_site = site
      break
  assert target_site is not None
  target_site.size[0] = target_radius
  return spec.compile()


def _make_spin_model(
    xml_path: epath.Path, assets: Dict[str, Any]
) -> mujoco.MjModel:
  model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets)
  model.site_rgba[model.site("target").id, 3] = 0
  model.site_rgba[model.site("tip").id, 3] = 0
  model.dof_damping[model.joint("hinge").id] = 0.03
  return model


class Spin(mjx_env.MjxEnv):
  """Spin environment."""

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
    self._mj_model = _make_spin_model(_XML_PATH, self._model_assets)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._lowers = self._mj_model.jnt_range[:2, 0]
    self._uppers = self._mj_model.jnt_range[:2, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1 = jax.random.split(rng, 2)
    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[:2].set(
        jax.random.uniform(rng1, (2,), minval=self._lowers, maxval=self._uppers)
    )
    qpos = qpos.at[2].set(jax.random.uniform(rng1, minval=-jp.pi, maxval=jp.pi))

    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    metrics = {}
    info = {"rng": rng}
    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        self._bounded_position(data),
        data.qvel,
        self._touch(data),
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del metrics, action, info  # Unused.
    reward = self._hinge_velocity(data) <= -_SPIN_VELOCITY
    return reward.astype(float)

  def _hinge_velocity(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "hinge_velocity")[0]

  def _bounded_position(self, data: mjx.Data) -> jax.Array:
    """Returns the (x,z) position of the tip relative to the hinge."""
    proximal_pos = mjx_env.get_sensor_data(self.mj_model, data, "proximal")
    distal_pos = mjx_env.get_sensor_data(self.mj_model, data, "distal")
    return jp.concatenate([
        proximal_pos,
        distal_pos,
        self._tip_position(data),
    ])

  def _tip_position(self, data: mjx.Data) -> jax.Array:
    """Returns the (x,z) position of the tip relative to the hinge."""
    tip_xz = mjx_env.get_sensor_data(self.mj_model, data, "tip")[
        jp.array([0, 2])
    ]
    spinner_xz = mjx_env.get_sensor_data(self.mj_model, data, "spinner")[
        jp.array([0, 2])
    ]
    return tip_xz - spinner_xz

  def _touch(self, data: mjx.Data) -> jax.Array:
    """Returns logarithmically scaled signals from the two touch sensors."""
    top = mjx_env.get_sensor_data(self.mj_model, data, "touchtop")
    bottom = mjx_env.get_sensor_data(self.mj_model, data, "touchbottom")
    touch = jp.hstack([top, bottom])
    return jp.log1p(touch)

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def dt(self) -> float:
    return self._config.ctrl_dt

  @property
  def sim_dt(self) -> float:
    return self._config.sim_dt

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


class Turn(mjx_env.MjxEnv):
  """Turn environment."""

  def __init__(
      self,
      target_radius: float,
      vision: bool = False,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = _make_turn_model(
        _XML_PATH, target_radius, self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._lowers = self._mj_model.jnt_range[:2, 0]
    self._uppers = self._mj_model.jnt_range[:2, 1]
    self._hinge_joint_id = self._mj_model.joint("hinge").id
    self._radius = self._mj_model.geom("cap1").size.sum()
    self._target_size_size = self._mj_model.site("target").size[0]
    self._target_mocap_body_id = self.mj_model.body("target").mocapid[0]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2 = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[:2].set(
        jax.random.uniform(rng1, (2,), minval=self._lowers, maxval=self._uppers)
    )
    qpos = qpos.at[2].set(jax.random.uniform(rng1, minval=-jp.pi, maxval=jp.pi))

    data = mjx_env.make_data(self._mj_model, qpos=qpos, impl=self._config.impl)

    target_angle = jax.random.uniform(rng2, minval=-jp.pi, maxval=jp.pi)
    hinge_x = data.xanchor[self._hinge_joint_id, 0]
    hinge_z = data.xanchor[self._hinge_joint_id, 2]
    target_x = hinge_x + self._radius * jp.sin(target_angle)
    target_z = hinge_z + self._radius * jp.cos(target_angle)

    mocap_pos = data.mocap_pos
    mocap_pos = mocap_pos.at[self._target_mocap_body_id, 0].set(target_x)
    mocap_pos = mocap_pos.at[self._target_mocap_body_id, 2].set(target_z)
    data = data.replace(mocap_pos=mocap_pos)

    metrics = {}
    info = {"rng": rng}
    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        self._bounded_position(data),
        data.qvel,
        self._touch(data),
        self._target_position(data),
        self._dist_to_target(data).reshape(1),
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del metrics, action, info  # Unused.
    reward = self._dist_to_target(data) <= 0.0
    return reward.astype(float)

  def _hinge_velocity(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "hinge_velocity")[0]

  def _bounded_position(self, data: mjx.Data) -> jax.Array:
    """Returns the (x,z) position of the tip relative to the hinge."""
    proximal_pos = mjx_env.get_sensor_data(self.mj_model, data, "proximal")
    distal_pos = mjx_env.get_sensor_data(self.mj_model, data, "distal")
    return jp.concatenate([
        proximal_pos,
        distal_pos,
        self._tip_position(data),
    ])

  def _tip_position(self, data: mjx.Data) -> jax.Array:
    """Returns the (x,z) position of the tip relative to the hinge."""
    tip_xz = mjx_env.get_sensor_data(self.mj_model, data, "tip")[
        jp.array([0, 2])
    ]
    spinner_xz = mjx_env.get_sensor_data(self.mj_model, data, "spinner")[
        jp.array([0, 2])
    ]
    return tip_xz - spinner_xz

  def _touch(self, data: mjx.Data) -> jax.Array:
    """Returns logarithmically scaled signals from the two touch sensors."""
    top = mjx_env.get_sensor_data(self.mj_model, data, "touchtop")
    bottom = mjx_env.get_sensor_data(self.mj_model, data, "touchbottom")
    touch = jp.hstack([top, bottom])
    return jp.log1p(touch)

  def _target_position(self, data: mjx.Data) -> jax.Array:
    """Returns the (x,z) position of the target relative to the hinge."""
    target_pos = mjx_env.get_sensor_data(self.mj_model, data, "target")[
        jp.array([0, 2])
    ]
    spinner_pos = mjx_env.get_sensor_data(self.mj_model, data, "spinner")[
        jp.array([0, 2])
    ]
    return target_pos - spinner_pos

  def _to_target(self, data: mjx.Data) -> jax.Array:
    """Returns the vector from the tip to the target."""
    return self._target_position(data) - self._tip_position(data)

  def _dist_to_target(self, data: mjx.Data) -> jax.Array:
    """Returns the signed distance to the target surface, negative is inside."""
    return jp.linalg.norm(self._to_target(data)) - self._target_size_size

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
