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
"""Environment for training aloha to insert a peg into a socket."""

from typing import Any, Dict, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mj_utils
from mujoco_playground._src import mjx_env
from mujoco_playground._src import rewards as reward_util
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

_XML_PATH = (
    mjx_env.ROOT_PATH
    / "manipulation"
    / "aloha"
    / "xmls"
    / "mjx_single_peg_insertion.xml"
)

_ARM_JOINTS = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
]
_FINGER_GEOMS = [
    "left/left_finger_top",
    "left/left_finger_bottom",
    "left/right_finger_top",
    "left/right_finger_bottom",
    "right/left_finger_top",
    "right/left_finger_bottom",
    "right/right_finger_top",
    "right/right_finger_bottom",
]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.01,
      reward_config=config_dict.create(
          scales=config_dict.create(
              left_reward=1,
              right_reward=1,
              left_target_qpos=0.3,
              right_target_qpos=0.3,
              no_table_collision=0.3,
              socket_z_up=2,
              peg_z_up=2,
              socket_entrance_reward=4,
              peg_end2_reward=4,
              peg_insertion_reward=8,
          )
      ),
  )


def get_assets() -> Dict[str, bytes]:
  """Returns a dictionary of all assets used by the environment."""
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "aloha"
  mj_utils.update_assets(assets, path, "*.xml")
  mj_utils.update_assets(assets, path / "assets")
  path = mjx_env.ROOT_PATH / "manipulation" / "aloha" / "xmls"
  mj_utils.update_assets(assets, path, "*.xml")
  mj_utils.update_assets(assets, path / "assets")
  return assets


class SinglePegInsertion(mjx_env.MjxEnv):
  """Environment for training aloha to bring an object to target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = _XML_PATH.as_posix()
    mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), get_assets()
    )
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)
    self._action_scale = config.action_scale
    self._post_init()

  def _post_init(self):
    self._left_gripper_site = self._mj_model.site("left/gripper").id
    self._right_gripper_site = self._mj_model.site("right/gripper").id
    self._socket_entrance_site = self._mj_model.site("socket_entrance").id
    self._socket_rear_site = self._mj_model.site("socket_rear").id
    self._peg_end2_site = self._mj_model.site("peg_end2").id
    self._socket_body = self._mj_model.body("socket").id
    self._peg_body = self._mj_model.body("peg").id
    self._table_geom = self._mj_model.geom("table").id
    self._finger_geoms = [
        self._mj_model.geom(geom_id).id for geom_id in _FINGER_GEOMS
    ]
    self._socket_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._socket_body]
    ]
    self._peg_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._peg_body]
    ]
    arm_joint_ids = [self._mj_model.joint(j).id for j in _ARM_JOINTS]
    self._arm_qadr = jp.array(
        [self._mj_model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids]
    )
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._init_ctrl = jp.array(self._mj_model.keyframe("home").ctrl)
    self.lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self.uppers = self._mj_model.actuator_ctrlrange[:, 1]

    # lift goal: both in the air
    self._socket_entrance_goal_pos = jp.array([-0.05, 0, 0.15])
    self._peg_end2_goal_pos = jp.array([0.05, 0, 0.15])

  def reset(self, rng: jax.Array) -> State:
    rng, rng_peg, rng_socket = jax.random.split(rng, 3)

    peg_xy = jax.random.uniform(rng_peg, (2,), minval=-0.1, maxval=0.1)
    socket_xy = jax.random.uniform(rng_socket, (2,), minval=-0.1, maxval=0.1)
    init_q = self._init_q.at[self._peg_qadr : self._peg_qadr + 2].add(peg_xy)
    init_q = init_q.at[self._socket_qadr : self._socket_qadr + 2].add(socket_xy)

    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
    )

    info = {"rng": rng}
    obs = self._get_obs(data)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "peg_end2_dist_to_line": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    state = State(data, obs, reward, done, metrics, info)

    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self.lowers, self.uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    socket_rear_pos = data.site_xpos[self._socket_rear_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    # insertion reward: if peg end2 is aligned with hole entrance, then reward
    # distance from peg end to socket interior
    socket_ab = socket_entrance_pos - socket_rear_pos
    socket_t = jp.dot(peg_end2_pos - socket_rear_pos, socket_ab)
    socket_t /= jp.dot(socket_ab, socket_ab) + 1e-6
    nearest_pt = data.site_xpos[self._socket_rear_site] + socket_t * socket_ab
    peg_end2_dist_to_line = jp.linalg.norm(peg_end2_pos - nearest_pt)

    out_of_bounds = jp.any(jp.abs(data.xpos[self._socket_body]) > 1.0)
    out_of_bounds |= jp.any(jp.abs(data.xpos[self._peg_body]) > 1.0)

    raw_rewards = self._get_reward(
        data, use_peg_insertion_reward=(peg_end2_dist_to_line < 0.005)
    )
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = sum(rewards.values()) / sum(
        self._config.reward_config.scales.values()
    )

    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    state.metrics.update(
        **rewards,
        peg_end2_dist_to_line=peg_end2_dist_to_line,
        out_of_bounds=out_of_bounds.astype(float)
    )
    obs = self._get_obs(data)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    left_gripper_pos = data.site_xpos[self._left_gripper_site]
    socket_pos = data.xpos[self._socket_body]
    right_gripper_pos = data.site_xpos[self._right_gripper_site]
    peg_pos = data.xpos[self._peg_body]
    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    socket_z = data.xmat[self._socket_body].ravel()[6:]
    peg_z = data.xmat[self._peg_body].ravel()[6:]

    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        left_gripper_pos,
        socket_pos,
        right_gripper_pos,
        peg_pos,
        socket_entrance_pos,
        peg_end2_pos,
        socket_z,
        peg_z,
    ])

    return obs

  def _get_reward(
      self, data: mjx.Data, use_peg_insertion_reward: bool
  ) -> Dict[str, jax.Array]:
    left_socket_dist = jp.linalg.norm(
        data.xpos[self._socket_body] - data.site_xpos[self._left_gripper_site]
    )
    left_reward = reward_util.tolerance(
        left_socket_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )
    right_peg_dist = jp.linalg.norm(
        data.xpos[self._peg_body] - data.site_xpos[self._right_gripper_site]
    )
    right_reward = reward_util.tolerance(
        right_peg_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )

    robot_qpos_diff = data.qpos[self._arm_qadr] - self._init_q[self._arm_qadr]
    left_pose = jp.linalg.norm(robot_qpos_diff[:6])
    left_pose = reward_util.tolerance(left_pose, (0, 0.01), margin=2.0)
    right_pose = jp.linalg.norm(robot_qpos_diff[6:])
    right_pose = reward_util.tolerance(right_pose, (0, 0.01), margin=2.0)

    socket_dist = jp.linalg.norm(
        self._socket_entrance_goal_pos - data.xpos[self._socket_body]
    )
    socket_lift = reward_util.tolerance(
        socket_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    peg_dist = jp.linalg.norm(
        # self._peg_end2_goal_pos[2] - data.site_xpos[self._peg_end2_site][2]
        self._peg_end2_goal_pos
        - data.xpos[self._peg_body]
    )
    peg_lift = reward_util.tolerance(
        peg_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    # Check for collisions with the floor
    hand_table_collisions = [
        mj_utils.geoms_colliding(data, self._table_geom, g)
        for g in self._finger_geoms
    ]
    table_collision = (sum(hand_table_collisions) > 0).astype(float)

    socket_orientation = jp.dot(
        data.xmat[self._socket_body][2], jp.array([0.0, 0.0, 1.0])
    )
    socket_orientation = reward_util.tolerance(
        socket_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )
    peg_orientation = jp.dot(
        data.xmat[self._peg_body][2], jp.array([0.0, 0.0, 1.0])
    )
    peg_orientation = reward_util.tolerance(
        peg_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )

    peg_insertion_dist = jp.linalg.norm(
        data.site_xpos[self._peg_end2_site]
        - data.site_xpos[self._socket_rear_site]
    )
    peg_insertion_reward = (
        reward_util.tolerance(
            peg_insertion_dist, (0, 0.001), margin=0.1, sigmoid="linear"
        )
        * use_peg_insertion_reward
    )

    return {
        "left_reward": left_reward,
        "right_reward": right_reward,
        "left_target_qpos": left_pose * left_reward * right_reward,
        "right_target_qpos": right_pose * left_reward * right_reward,
        "no_table_collision": 1 - table_collision,
        "socket_entrance_reward": socket_lift,
        "peg_end2_reward": peg_lift,
        "socket_z_up": socket_orientation * socket_lift,
        "peg_z_up": peg_orientation * peg_lift,
        "peg_insertion_reward": peg_insertion_reward,
    }

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
