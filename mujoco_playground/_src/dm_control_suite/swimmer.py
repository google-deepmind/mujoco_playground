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
"""Swimmer environment."""

from typing import Any, Dict, Optional, Union
import warnings

from etils import epath
import jax
import jax.numpy as jp
from lxml import etree
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "swimmer.xml"
_FILENAMES = [
    "materials.xml",
    "skybox.xml",
    "visual.xml",
]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.03,
      sim_dt=0.003,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=40,
  )


def _make_model(xml_path: str, n_bodies: int):
  """Generates an xml string defining a swimmer with `n_bodies` bodies."""
  if n_bodies < 3:
    raise ValueError(f"At least 3 bodies required. Received {n_bodies}")
  if n_bodies > 6:
    warnings.warn(
        "Globally setting jax_default_matmul_precision  to 'highest'. "
        "Higher matmul precision is required for longer chains in swimmer.py"
    )
    # Similar issue as https://github.com/google/brax/issues/386.
    jax.config.update("jax_default_matmul_precision", "highest")
  mjcf = etree.fromstring(epath.Path(xml_path).read_text())
  head_body = mjcf.find("./worldbody/body")
  actuator = etree.SubElement(mjcf, "actuator")
  sensor = etree.SubElement(mjcf, "sensor")

  parent = head_body
  for body_index in range(n_bodies - 1):
    site_name = f"site_{body_index}"
    child = _make_body(body_index=body_index)
    child.append(etree.Element("site", name=site_name))
    joint_name = f"joint_{body_index}"
    joint_limit = 360.0 / n_bodies
    joint_range = f"{-joint_limit} {joint_limit}"
    child.append(
        etree.Element("joint", {"name": joint_name, "range": joint_range})
    )
    motor_name = f"motor_{body_index}"
    actuator.append(etree.Element("motor", name=motor_name, joint=joint_name))
    velocimeter_name = f"velocimeter_{body_index}"
    sensor.append(
        etree.Element("velocimeter", name=velocimeter_name, site=site_name)
    )
    gyro_name = f"gyro_{body_index}"
    sensor.append(etree.Element("gyro", name=gyro_name, site=site_name))
    parent.append(child)
    parent = child

  # Move tracking cameras further away from the swimmer according to its length.
  cameras = mjcf.findall("./worldbody/body/camera")
  scale = n_bodies / 6.0
  for cam in cameras:
    if cam.get("mode") == "trackcom":
      old_pos = cam.get("pos").split(" ")
      new_pos = " ".join([str(float(dim) * scale) for dim in old_pos])
      cam.set("pos", new_pos)

  return etree.tostring(mjcf, pretty_print=True)


def _make_body(body_index):
  """Generates an xml string defining a single physical body."""
  body_name = f"segment_{body_index}"
  visual_name = f"visual_{body_index}"
  inertial_name = f"inertial_{body_index}"
  body = etree.Element("body", name=body_name)
  body.set("pos", "0 .1 0")
  etree.SubElement(body, "geom", {"class": "visual", "name": visual_name})
  etree.SubElement(body, "geom", {"class": "inertial", "name": inertial_name})
  return body


class Swim(mjx_env.MjxEnv):
  """Swimmer environment."""

  def __init__(
      self,
      n_links: int,
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
        _make_model(self.xml_path, n_links), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._nose_geom_id = self.mj_model.geom("nose").id
    self._head_body_id = self.mj_model.body("head").id
    self._target_geom_id = self.mj_model.geom("target").id
    self._target_size = self.mj_model.geom_size[self._target_geom_id, 0]
    self._target_mocap_body_id = self.mj_model.body("target").mocapid[0]

    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(
        jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi)
    )
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng2,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
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
    target_box = jp.where(jax.random.bernoulli(rng3, 0.2), 0.3, 2.0)
    xy = jax.random.uniform(rng4, (2,), minval=-target_box, maxval=target_box)
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
        self._joints(data),
        self._nose_to_target(data),
        self._body_velocities(data),
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info, metrics  # Unused.
    return reward.tolerance(
        self._nose_to_target_dist(data),
        bounds=(0, self._target_size),
        margin=5 * self._target_size,
        sigmoid="long_tail",
    )

  def _nose_to_target(self, data: mjx.Data) -> jax.Array:
    nose_to_target = (
        data.geom_xpos[self._target_geom_id]
        - data.geom_xpos[self._nose_geom_id]
    )
    head_orientation = data.xmat[self._head_body_id]
    nose_to_target_head = nose_to_target @ head_orientation
    return nose_to_target_head[:2]  # Ignore z component.

  def _nose_to_target_dist(self, data: mjx.Data) -> jax.Array:
    return jp.linalg.norm(self._nose_to_target(data))

  def _body_velocities(self, data: mjx.Data) -> jax.Array:
    xvel_local = data.sensordata[12:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def _joints(self, data: mjx.Data) -> jax.Array:
    return data.qpos[3:]  # Exclude root joints.

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
