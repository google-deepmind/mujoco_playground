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


import functools
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

from brax.io import model as brax_loader
from brax.training.acme import running_statistics
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.ppo import networks as ppo_networks
import flax
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.aloha.s2r import base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
from mujoco_playground.config import manipulation_params


def default_config() -> config_dict.ConfigDict:  # TODO :Clean up.
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=160,
      action_repeat=1,
      action_scale=0.02,
      action_history_length=4,
      max_obs_delay=4,
      reset_buffer_size=10,
      obs_noise=config_dict.create(
          depth=True,
          brightness=[1.0, 3.0],
          grad_threshold=0.05,
          noise_multiplier=10,
          obj_pos=0.015,  # meters
          obj_vel=0.015,  # meters/s
          obj_angvel=0.2,
          gripper_box=0.015,  # meters
          obj_angle=7.5,  # degrees
          robot_qpos=0.1,  # radians
          robot_qvel=0.1,  # radians/s
          eef_pos=0.02,  # meters
          eef_angle=5.0,  # degrees
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(peg_insertion=8, obj_rot=0.5),
          sparse=config_dict.create(success=0, drop=-10, final_grasp=10),
          reg=config_dict.create(
              robot_target_qpos=1, joint_vel=1, grip_pos=0.5  # no sliding!
          ),
      ),
  )
  return config


def load_brax_policy(
    path: str, env_name, action_size, distill: bool = False
):  # Distillation requires extra inference metadata.
  ppo_params = manipulation_params.brax_ppo_config(env_name)
  # Pickcube 1-arm policy.
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **ppo_params.network_factory
  )
  network = network_factory(
      0,  # no params init required.
      action_size,
      preprocess_observations_fn=running_statistics.normalize,
  )
  make_policy = (
      bc_networks.make_inference_fn(network)
      if distill
      else ppo_networks.make_inference_fn(network)
  )
  trained_params = brax_loader.load_params(path)
  return make_policy(trained_params, deterministic=True)


def load_pick_policy(path, env_name):
  raw_policy = load_brax_policy(path, env_name, 7)

  def single2biarm_inference_fn(obs: jp.ndarray):
    l_obs, r_obs = jp.split(obs, 2, axis=-1)
    l_act, _ = raw_policy({"state": l_obs}, None)
    r_act, _ = raw_policy({"state": r_obs}, None)
    return jp.concatenate([l_act, r_act], axis=-1)

  return jax.jit(single2biarm_inference_fn)


class PegInsertion(base.S2RBase):

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      *,
      distill: bool = False,  # If true, this class just provides methods for the downstream distill class to use.
  ):
    self._distill = distill
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "aloha"
        / "xmls"
        / "s2r"
        / "mjx_peg_insertion.xml"
    )
    super().__init__(
        xml_path=xml_path,
        config=config,
        config_overrides=config_overrides,
    )

    if distill:
      self.pick_policy = lambda x: jp.zeros(self.action_size)
    else:
      self.pick_policy = load_pick_policy(
          pathlib.Path(__file__).parent / "params" / "AlohaS2RPick.prms",
          "AlohaS2RPick",
      )

    self.obj_names = ["socket", "peg"]
    self.hands = ["left", "right"]
    self.target_positions = jp.array(
        [[-0.10, 0.0, 0.25], [0.10, 0.0, 0.25]], dtype=float
    )
    self.target_quats = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    self._switch_height = 0.25 - 0.08  # Used for height-based switching.
    self.default_pre_insertion_qpos = self._mj_model.keyframe("preinsert").qpos
    self.default_pre_insertion_ctrl = self._mj_model.keyframe("preinsert").ctrl
    self._socket_end_site = self._mj_model.site("socket_rear").id
    self._peg_end_site = self._mj_model.site("peg_end1").id
    self._socket_entrance_site = self._mj_model.site("socket_entrance").id
    # hardcode because there's no mj_model.jnt_qveladr option.
    self._socket_qveladr = 16
    self._peg_qveladr = 22

    self.noise_config = config_dict.create(
        _peg_target_pos=np.array([0.00, 0.00, 0.00]),
        _socket_target_pos=np.array([0.00, 0.00, 0.00]),
        _left_waist_init_pos=np.array(0.1),
        _right_waist_init_pos=np.array(0.1),
    )

    fov_cam = 58
    self.noise_config["_peg_init_pos"] = config_dict.create(
        radius_min=0.27, radius_max=0.42, angle=jp.deg2rad(45 * fov_cam / 90)
    )
    self.noise_config["_socket_init_pos"] = config_dict.create(
        radius_min=0.27, radius_max=0.42, angle=jp.deg2rad(45 * fov_cam / 90)
    )

    self._finger_ctrladr = np.array([6, 13], dtype=int)
    self._skip_prob = 0.8 if not distill else 0.0
    self._post_init(keyframe="home")

  def reset(self, rng: jax.Array) -> State:
    data, info = self.init_objects(rng)
    metrics = {
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        **{k: 0.0 for k in self._config.reward_config.sparse.keys()},
        **{k: 0.0 for k in self._config.reward_config.reg.keys()},
    }

    info["has_switched"] = jp.array(0, dtype=int)
    info["preinsertion_buffer_qpos"] = jp.tile(
        self.default_pre_insertion_qpos, (self._config.reset_buffer_size, 1)
    )
    info["preinsertion_buffer_ctrl"] = jp.tile(
        self.default_pre_insertion_ctrl, (self._config.reset_buffer_size, 1)
    )
    info["time_of_switch"] = jp.array(0, dtype=int)
    metrics["peg_end2_dist_to_line"] = jp.array(0.0, dtype=float)

    obs = self._get_obs_insertion(data, info)
    if self._distill:
      self.reset_color_noise(info)
      obs = {**obs, **self._get_obs_distill(data, info)}

    # Random obs delay.
    _actor_obs, _ = flax.core.pop(obs, "has_switched")
    _actor_obs, _ = flax.core.pop(_actor_obs, "privileged")
    info["obs_history"] = base.init_obs_history(
        _actor_obs, self._config.max_obs_delay
    )

    # Assert that obs_history has same keys as obs minus popped keys
    obs_keys = set(obs.keys())
    history_keys = set(info["obs_history"].keys())
    expected_keys = obs_keys - {"privileged", "has_switched"}
    assert history_keys == expected_keys, (
        f"Mismatch between obs keys {expected_keys} and history keys"
        f" {history_keys}"
    )

    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    newly_reset = state.info["_steps"] == 0
    self._reset_info_if_needed(state, newly_reset)

    action = self._action_mux(action, state)
    state = self._state_mux(state)
    data = self._step(state, action)

    # Calculate rewards
    sparse_rewards, success, dropped, final_grasp = (
        self._calculate_sparse_rewards(state, data)
    )
    dense_rewards = self._calculate_dense_rewards(
        data, state.info, state.metrics
    )
    reg_rewards = self._calculate_reg_rewards(
        data, state.info, action, state.metrics
    )

    # Calculate total reward
    total_reward = self._calculate_total_reward(
        dense_rewards, sparse_rewards, reg_rewards, state
    )

    # Check done conditions
    done = self._check_done_conditions(data, state, dropped)

    # Update state
    state = self._update_state(state, data, total_reward, done)
    state.metrics.update({
        "success": success.astype(float),
        "drop": dropped.astype(float),
        "final_grasp": final_grasp,
    })
    return state

  def _reset_info_if_needed(self, state: State, newly_reset: bool):
    state.info["has_switched"] = jp.where(
        newly_reset, 0, state.info["has_switched"]
    )
    state.info["time_of_switch"] = jp.where(
        newly_reset, 0, state.info["time_of_switch"]
    )

  def _calculate_sparse_rewards(
      self, state: State, data: mjx.Data
  ) -> Tuple[Dict[str, float], bool, bool, float]:
    # Calculate success
    peg_insertion_dist = jp.linalg.norm(
        data.site_xpos[self._mj_model.site("socket_end1").id]
        - data.site_xpos[self._peg_end_site]
    )
    success = peg_insertion_dist < 0.01

    # Calculate dropped
    peg_height = data.xpos[self._peg_body][2]
    socket_height = data.xpos[self._socket_body][2]
    thresh = self._switch_height - 0.1

    dropped = jp.array(False)
    if not self._distill:
      dropped = (peg_height < thresh) | (socket_height < thresh)
      dropped = dropped & state.info["has_switched"].astype(bool)

    # Calculate final grasp
    l_grasped = (
        jp.linalg.norm(self.gripping_error(data, "left", "socket")) < 0.03
    )
    r_grasped = jp.linalg.norm(self.gripping_error(data, "right", "peg")) < 0.03

    final_grasp = 0.0
    grasped = l_grasped.astype(float) * r_grasped.astype(float)
    last_step = state.info["_steps"] >= (
        self._config.episode_length - self._config.action_repeat
    )
    final_grasp = grasped * last_step.astype(float)

    raw_sparse_rewards = {
        "success": success.astype(float),
        "drop": dropped.astype(float),
        "final_grasp": final_grasp,
    }

    sparse_rewards = {
        k: v * self._config.reward_config.sparse[k]
        for k, v in raw_sparse_rewards.items()
    }

    return sparse_rewards, success, dropped, final_grasp

  def _calculate_dense_rewards(
      self, data: mjx.Data, info: dict, metrics: Dict[str, float]
  ) -> Dict[str, float]:
    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    socket_rear_pos = data.site_xpos[self._socket_end_site]
    peg_end2_pos = data.site_xpos[self._peg_end_site]

    # Insertion reward: if peg end2 is aligned with hole entrance, then reward
    # distance from peg end to socket interior.
    socket_ab = socket_entrance_pos - socket_rear_pos
    socket_t = jp.dot(peg_end2_pos - socket_rear_pos, socket_ab)
    socket_t /= jp.dot(socket_ab, socket_ab) + 1e-6
    nearest_pt = data.site_xpos[self._socket_end_site] + socket_t * socket_ab
    peg_end2_dist_to_line = jp.linalg.norm(peg_end2_pos - nearest_pt)

    objects_aligned = peg_end2_dist_to_line < 0.01
    metrics["peg_end2_dist_to_line"] = peg_end2_dist_to_line

    peg_insertion_dist = jp.linalg.norm(
        data.site_xpos[self._mj_model.site("socket_end1").id]
        - data.site_xpos[self._peg_end_site]
    )

    peg_insertion_reward = reward_util.tolerance(
        peg_insertion_dist, (0, 0.001), margin=0.2, sigmoid="linear"
    ) * objects_aligned.astype(float)

    # Dense rotation reward
    rot_rewards = {}
    for obj, target in zip(self.obj_names, self.target_quats):
      obj_mat = data.xmat[getattr(self, f"_{obj}_body")]
      obj_target = math.quat_to_mat(jp.array(target))
      rot_err = jp.linalg.norm(obj_target.ravel()[:6] - obj_mat.ravel()[:6])
      rot_rewards[f"{obj}_rot"] = 1 - jp.tanh(5 * rot_err)

    raw_dense_rewards = {"peg_insertion": peg_insertion_reward, **rot_rewards}

    metrics.update({"peg_insertion": peg_insertion_reward})

    return {
        k: v * self._config.reward_config.scales.get(
            k, self._config.reward_config.scales.obj_rot
        )
        for k, v in raw_dense_rewards.items()
    }

  def _calculate_reg_rewards(
      self,
      data: mjx.Data,
      info: dict,
      action: jax.Array,
      metrics: Dict[str, float],
  ) -> Dict[str, float]:
    robot_target_qpos = self._robot_target_qpos(data)

    # Joint velocity regularization
    joint_vel_rewards = {}
    for side in self.hands:
      joint_vel_mse = jp.linalg.norm(
          data.qvel[getattr(self, f"_{side}_qposadr")]
      )
      joint_vel_rewards[f"{side}_joint_vel"] = reward_util.tolerance(
          joint_vel_mse, (0, 0.5), margin=2.0, sigmoid="reciprocal"
      )

    # Grip regularization
    e_l_grip = self.gripping_error(data, "left", "socket")
    e_r_grip = self.gripping_error(data, "right", "peg")
    r_l_grip = 1 - jp.tanh(5 * jp.linalg.norm(e_l_grip))
    r_r_grip = 1 - jp.tanh(5 * jp.linalg.norm(e_r_grip))

    raw_reg_rewards = {
        "robot_target_qpos": robot_target_qpos,
        "left_grip_pos": r_l_grip,
        "right_grip_pos": r_r_grip,
        **joint_vel_rewards,
    }

    metrics.update({"robot_target_qpos": jp.array(robot_target_qpos)})

    reg_rewards = {}
    for k, v in raw_reg_rewards.items():
      if k == "robot_target_qpos":
        reg_rewards[k] = v * self._config.reward_config.reg.robot_target_qpos
      elif k.endswith("_joint_vel"):
        reg_rewards[k] = (
            v * self._config.reward_config.reg.joint_vel / len(self.hands)
        )
      elif k.endswith("_grip_pos"):
        reg_rewards[k] = v * self._config.reward_config.reg.grip_pos / 2

    return reg_rewards

  def _calculate_total_reward(
      self,
      dense_rewards: Dict[str, float],
      sparse_rewards: Dict[str, float],
      reg_rewards: Dict[str, float],
      state: State,
  ) -> float:
    total_reward = (
        sum(dense_rewards.values())
        + sum(sparse_rewards.values())
        + sum(reg_rewards.values())
    )

    # Zero reward for when the other policy's taking action
    total_reward = jp.where(state.info["has_switched"], total_reward, 0.0)

    return total_reward

  def _check_done_conditions(
      self, data: mjx.Data, state: State, dropped: bool
  ) -> Tuple[bool, bool, bool, bool]:
    # Check if out of bounds
    out_of_bounds = jp.any(jp.abs(data.xpos[self._socket_body]) > 1.0)
    out_of_bounds |= jp.any(jp.abs(data.xpos[self._peg_body]) > 1.0)

    # Check if end of insertion
    end_of_insertion = jp.array(False)
    if not self._distill:
      end_of_insertion = (
          state.info["_steps"] - state.info["time_of_switch"] >= 60
      )
      end_of_insertion = end_of_insertion & state.info["has_switched"].astype(
          bool
      )

    # Check if rotated
    peg_mat = data.xmat[self._peg_body]
    z_axis = jp.array([0, 0, 1])
    peg_z = peg_mat[:3, 2]  # Z axis of peg is just last column.
    peg_z = peg_z / jp.linalg.norm(peg_z)
    angle = jp.arccos(jp.dot(z_axis, peg_z))
    rotated = angle > jp.deg2rad(80)

    socket_mat = data.xmat[self._socket_body]
    socket_z = socket_mat[:3, 2]
    socket_z = socket_z / jp.linalg.norm(socket_z)
    angle = jp.arccos(jp.dot(z_axis, socket_z))
    rotated |= angle > jp.deg2rad(80)

    # Combine all done conditions
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | dropped
    done = done | out_of_bounds | end_of_insertion
    done = done | rotated

    return done

  def _update_state(
      self, state: State, data: mjx.Data, total_reward: float, done: bool
  ) -> State:
    # Get observations
    obs = self._get_obs_insertion(data, state.info)
    if self._distill:
      obs = {
          **obs,
          **self._get_obs_distill(data, state.info),
      }

    # Update observation history
    state.info["rng"], key_obs = jax.random.split(state.info["rng"])
    base.use_obs_history(key_obs, state.info["obs_history"], obs)

    # Update step counter
    state.info["_steps"] += self._config.action_repeat
    state.info["_steps"] = jp.where(
        done | (state.info["_steps"] >= self._config.episode_length),
        0,
        state.info["_steps"],
    )

    # Return updated state
    return State(
        data,
        obs,
        jp.array(total_reward),
        done.astype(float),
        state.metrics,
        state.info,
    )

  def _action_mux(self, action: jp.array, state: mjx_env.State):
    """
    Chooses which policy to apply. If you've already toggled switched this round, always use the external policy.
    """

    data = state.data

    left_gripper_tip = self.gripper_tip_pos(data, "left")
    right_gripper_tip = self.gripper_tip_pos(data, "right")
    switch = (left_gripper_tip[2] > self._switch_height) & (
        right_gripper_tip[2] > self._switch_height
    )

    first_switch = jp.logical_and(state.info["has_switched"] == 0, switch)

    state.info["time_of_switch"] = jp.where(
        first_switch, state.info["_steps"], state.info["time_of_switch"]
    )

    #### Exploration Manager ####
    # If it's the first switch of the run, save the data to the buffer of states you can skip to at autoreset.
    def update_first_value(buf, val):
      buf = jp.roll(buf, 1, axis=0)
      buf = buf.at[0].set(val)
      return buf

    new_qpos_buf = update_first_value(
        state.info["preinsertion_buffer_qpos"], data.qpos
    )
    new_ctrl_buf = update_first_value(
        state.info["preinsertion_buffer_ctrl"], data.ctrl
    )

    state.info["preinsertion_buffer_qpos"] = jp.where(
        first_switch, new_qpos_buf, state.info["preinsertion_buffer_qpos"]
    )
    state.info["preinsertion_buffer_ctrl"] = jp.where(
        first_switch, new_ctrl_buf, state.info["preinsertion_buffer_ctrl"]
    )
    #### End Exploration Manager ####

    state.info["has_switched"] = jp.where(switch, 1, state.info["has_switched"])
    use_input = state.info["has_switched"].astype(bool) | self._distill
    return jp.where(
        use_input, action, self.pick_policy(state.obs["state_pickup"])
    )

  def _state_mux(self, state: mjx_env.State) -> mjx_env.State:

    state.info["rng"], key_skip, key_skip_index = jax.random.split(
        state.info["rng"], 3
    )
    i_buf = jax.random.randint(
        key_skip_index, (), minval=0, maxval=self._config.reset_buffer_size
    )
    preinsert_qpos = state.info["preinsertion_buffer_qpos"][i_buf]
    preinsert_ctrl = state.info["preinsertion_buffer_ctrl"][i_buf]

    preinsert_data = state.data.replace(
        qpos=preinsert_qpos, ctrl=preinsert_ctrl
    )

    newly_reset = state.info["_steps"] == 0
    to_skip = newly_reset * jax.random.bernoulli(key_skip, self._skip_prob)

    # The pre insert buffer is initialized with the home position, in which case you can't skip.
    to_skip = jp.logical_and(to_skip, jp.any(preinsert_qpos != self._init_q))
    state.info["has_switched"] = jp.where(
        to_skip, 1, state.info["has_switched"]
    )
    data = jax.tree_util.tree_map(
        lambda x, y: (1 - to_skip) * x + to_skip * y,
        state.data,
        preinsert_data,
    )

    #### Randomly hide ####
    qpos = data.qpos
    if self._distill:
      for obj in ["socket", "peg"]:
        state.info["rng"], key_hide = jax.random.split(state.info["rng"])
        hide = newly_reset * jax.random.bernoulli(key_hide, 0.07)
        obj_idx = getattr(self, f"_{obj}_qposadr")
        hidden_pos = jp.array([0.4, 0.33])
        if obj == "socket":
          hidden_pos = hidden_pos.at[0].set(hidden_pos[0] * -1)
        obj_hidden = qpos.at[obj_idx : obj_idx + 2].set(hidden_pos)
        qpos = jp.where(hide, obj_hidden, qpos)
    data = data.replace(qpos=qpos)
    ####

    return state.replace(data=data)

  def _get_obs_insertion(self, data: mjx.Data, info: dict) -> jax.Array:
    obs_pick = self._get_obs_pick(data, info)
    obs_insertion = jp.concatenate([obs_pick, self._get_obs_dist(data, info)])
    obs = {
        "state_pickup": obs_pick,
        "state": obs_insertion,
        "privileged": jp.concat([
            obs_insertion,
            (info["_steps"] / self._config.episode_length).reshape(1),
        ]),
        "has_switched": info["has_switched"].astype(float).reshape(1),
    }
    return obs

  def _get_obs_dist(self, data: mjx.Data, info: dict) -> jax.Array:
    delta = (
        data.site_xpos[self._socket_end_site]
        - data.site_xpos[self._peg_end_site]
    )
    info["rng"], key = jax.random.split(info["rng"])
    noise = jax.random.uniform(
        key,
        (2, 3),
        minval=-self._config.obs_noise.obj_pos,
        maxval=self._config.obs_noise.obj_pos,
    )
    return delta + (noise[1] - noise[0])
