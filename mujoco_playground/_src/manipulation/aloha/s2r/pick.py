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

from typing import Any, Dict, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.aloha import base
from mujoco_playground._src.manipulation.aloha.s2r import base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=160,
      action_repeat=1,
      action_scale=0.02,
      action_history_length=4,
      max_obs_delay=4,
      vision=False,
      dense_rot_weight=0.23,
      obs_noise=config_dict.create(
          depth=False,
          grad_threshold=0.05,
          noise_multiplier=10,
          obj_pos=0.015,  # meters
          obj_vel=0.015,  # meters/s
          obj_angvel=0.2,
          gripper_box=0.015,  # meters
          obj_angle=5.0,  # degrees
          robot_qpos=0.1,  # radians
          robot_qvel=0.1,  # radians/s
          eef_pos=0.02,  # meters
          eef_angle=5.0,  # degrees
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=16.0,
          ),
          sparse=config_dict.create(
              lift=0.5, grasped=0.5, success=0.5, success_time=0.1
          ),
          reg=config_dict.create(
              finger_force=0.007,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.005,
              joint_vel=0.005,
              # Arm stays close to target pose.
              robot_target_qpos=0.001,
          ),
      ),
  )
  return config


class Pick(base.S2RBase):

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "aloha"
        / "xmls"
        / "s2r"
        / "mjx_pick.xml"
    )
    super().__init__(xml_path, config, config_overrides)
    self.obj_names = ["box"]
    self.hands = ["left"]

    self.noise_config = config_dict.create(
        _box_target_pos=np.array([0.1, 0.1, 0.1]),
        _left_waist_init_pos=np.array(0.2),
    )
    self.noise_config["_box_init_pos"] = config_dict.create(
        radius_min=0.2,
        radius_max=0.45,  # Issue: overlapping paths.
        angle=jp.deg2rad(45),
    )

    self.target_positions = [jp.array([-0.1, 0.0, 0.25], dtype=float)]
    self.target_quats = [[1.0, 0.0, 0.0, 0.0]]
    self._finger_ctrladr = np.array([6], dtype=int)
    self._box_qveladr = 8

    self._post_init(keyframe="home")

  def reset(self, rng: jax.Array) -> State:
    data, info = self.init_objects(rng)
    metrics = {
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        **{k: 0.0 for k in self._config.reward_config.sparse.keys()},
        **{k: 0.0 for k in self._config.reward_config.reg.keys()},
    }
    obs = self._get_obs_pick(data, info)
    metrics["score"] = jp.array(0.0, dtype=float)
    info["score"] = jp.array(0, dtype=int)
    info["reached_box"] = jp.array(0.0, dtype=float)
    info["prev_reward"] = jp.array(0.0, dtype=float)
    info["success_time"] = jp.array(0, dtype=int)
    obs = {"state": obs, "privileged": obs}
    actor_obs, _ = flax.core.pop(
        obs, "privileged"
    )  # Privileged obs is not randomly shifted.
    info["obs_history"] = base.init_obs_history(
        actor_obs, self._config.max_obs_delay
    )
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    newly_reset = state.info["_steps"] == 0
    self._reset_info_if_needed(state, newly_reset)

    data = self._step(state, action)

    # Calculate rewards
    sparse_rewards, success = self._calculate_sparse_rewards(state, data)
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
    done, crossed_line = self._check_done_conditions(data, state)
    total_reward += jp.where(crossed_line, -1.0, 0.0)

    # Update score
    self._update_score(state, success, done)

    # Update state
    state = self._update_state(state, data, total_reward, done, newly_reset)

    return state

  def _reset_info_if_needed(self, state: State, newly_reset: bool):
    state.info["reached_box"] = jp.where(
        newly_reset, 0.0, state.info["reached_box"]
    )
    state.info["prev_reward"] = jp.where(
        newly_reset, 0.0, state.info["prev_reward"]
    )
    state.info["success_time"] = jp.where(
        newly_reset, 0, state.info["success_time"]
    )

  def _calculate_sparse_rewards(
      self, state: State, data: mjx.Data
  ) -> Tuple[Dict[str, float], bool, int]:
    grasped = self.is_grasped(data, "left")
    gripping_error = jp.linalg.norm(self.gripping_error(data, "left", "box"))
    grasped_correct = gripping_error < base.GRASP_THRESH
    grasped = grasped * grasped_correct.astype(float)

    box_pos = data.xpos[self._box_body]
    init_box_height = self._init_q[self._box_qposadr + 2]
    lifted = (box_pos[2] > (init_box_height + 0.05)).astype(float)

    success, success_time = self._calculate_success(state, box_pos, data)

    raw_sparse_rewards = {
        "grasped": grasped,
        "lift": lifted,
        "success": success.astype(float),
        "success_time": success.astype(float) * success_time,
    }
    state.metrics.update(**raw_sparse_rewards)
    sparse_rewards = {
        k: v * self._config.reward_config.sparse[k]
        for k, v in raw_sparse_rewards.items()
    }
    return sparse_rewards, success

  def _calculate_dense_rewards(
      self, data: mjx.Data, info: Dict[str, Any], metrics: Dict[str, float]
  ) -> Dict[str, float]:
    raw_rewards = self._get_dense_pick(data, info)
    metrics.update(**raw_rewards)
    return {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }

  def _calculate_reg_rewards(
      self,
      data: mjx.Data,
      info: Dict[str, Any],
      action: jax.Array,
      metrics: Dict[str, float],
  ) -> Dict[str, float]:
    raw_reg_rewards = self._get_reg_pick(data, info, action)
    f_lfing = self.get_finger_force(data, "left", "left")
    f_rfing = self.get_finger_force(data, "left", "right")
    f_fing = jp.mean(jp.linalg.norm(f_lfing) + jp.linalg.norm(f_rfing))
    max_f_fing = 7.0
    n_f_fing = jp.clip(f_fing, min=None, max=max_f_fing) / max_f_fing
    raw_reg_rewards.update(
        {"finger_force": n_f_fing * self.is_grasped(data, "left")}
    )
    metrics.update(**raw_reg_rewards)
    return {
        k: v * self._config.reward_config.reg[k]
        for k, v in raw_reg_rewards.items()
    }

  def _calculate_total_reward(
      self,
      dense_rewards: Dict[str, float],
      sparse_rewards: Dict[str, float],
      reg_rewards: Dict[str, float],
      state: State,
  ) -> float:
    total_reward = jp.clip(sum(dense_rewards.values()), -1e4, 1e4)
    total_reward += jp.clip(sum(sparse_rewards.values()), -1e4, 1e4)
    reward = jp.maximum(
        total_reward - state.info["prev_reward"], jp.zeros_like(total_reward)
    )
    state.info["prev_reward"] = jp.maximum(
        total_reward, state.info["prev_reward"]
    )
    reward = jp.where(state.info["_steps"] == 0, 0.0, reward)
    reward += jp.clip(sum(reg_rewards.values()), -1e4, 1e4)
    return reward

  def _check_done_conditions(
      self, data: mjx.Data, state: State
  ) -> Tuple[bool, bool, bool]:
    id_far_end = self.mj_model.site("box_end_2").id
    box_far_end = data.site_xpos[id_far_end]
    crossed_line = box_far_end[0] > (0.0 + 0.048 + 0.025)

    box_pos = data.xpos[self._box_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done | crossed_line

    return done, crossed_line

  def _update_score(self, state: State, success: bool, done: bool):
    last_step = (
        state.info["_steps"] + self._config.action_repeat
    ) >= self._config.episode_length
    state.info["score"] += success.astype(int) * last_step
    state.info["score"] = jp.clip(state.info["score"], min=0, max=(5 - 1))
    state.metrics["score"] = state.info["score"] * 1.0
    state.info["_steps"] += self._config.action_repeat
    state.info["_steps"] = jp.where(
        done | (state.info["_steps"] >= self._config.episode_length),
        0,
        state.info["_steps"],
    )

  def _update_state(
      self,
      state: State,
      data: mjx.Data,
      total_reward: float,
      done: bool,
      newly_reset: bool,
  ) -> State:
    obs = self._get_obs_pick(data, state.info)
    obs = {"state": obs, "privileged": obs}
    state.info["rng"], key_obs = jax.random.split(state.info["rng"])
    base.use_obs_history(key_obs, state.info["obs_history"], obs)
    return State(
        data, obs, total_reward, done.astype(float), state.metrics, state.info
    )

  def _calculate_thresholds(self, score: int) -> Tuple[float, float]:
    def map_to_range(val: int, a: float, b: float, num_vals=5):
      step = (b - a) / (num_vals - 1)  # Step size for the range
      index = jp.minimum(val // 1, num_vals - 1)
      return a + step * index

    pos_thresh = map_to_range(score, 0.04, 0.005)
    rot_thresh = map_to_range(score, 15, 2.5)
    return pos_thresh, rot_thresh

  def _calculate_success(
      self, state: State, box_pos: jax.Array, data: mjx.Data
  ) -> Tuple[bool, int]:
    pos_thresh, rot_thresh = self._calculate_thresholds(state.info["score"])
    success = (
        jp.linalg.norm(box_pos - state.info["_box_target_pos"]) < pos_thresh
    )
    box_mat = data.xmat[self._box_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._box_mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    rot_success = rot_err < jp.deg2rad(rot_thresh)
    success = success & rot_success

    state.info["success_time"] = jp.where(
        success & (state.info["success_time"] == 0),
        state.info["_steps"],
        state.info["success_time"],
    )

    success_time = state.info["_steps"] - state.info["success_time"]
    return success, success_time

  def _get_dense_pick(
      self, data: mjx.Data, info: Dict[str, Any]
  ) -> Dict[str, Any]:
    target_pos = info["_box_target_pos"]
    box_pos = data.xpos[self._box_body]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._box_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._box_mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    w_r = self._config.dense_rot_weight
    box_target = 1 - jp.tanh(5 * ((1 - w_r) * pos_err + w_r * rot_err))
    gripping_error = jp.linalg.norm(self.gripping_error(data, "left", "box"))
    gripper_box = 1 - jp.tanh(5 * gripping_error)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (
            jp.linalg.norm(self.gripping_error(data, "left", "box"))
            < base.GRASP_THRESH
        ),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
    }
    return rewards

  def _get_reg_pick(
      self, data: mjx.Data, info: Dict[str, Any], action: jp.ndarray
  ) -> Dict[str, Any]:
    rewards = {
        "robot_target_qpos": self._robot_target_qpos(data),
    }

    joint_vel_mse = jp.linalg.norm(data.qvel[self._left_qposadr])
    joint_vel = reward_util.tolerance(
        joint_vel_mse, (0, 0.5), margin=2.0, sigmoid="reciprocal"
    )
    rewards["joint_vel"] = joint_vel

    left_id = self._left_left_finger_geom_bottom
    right_id = self._left_right_finger_geom_bottom

    hand_floor_collision = [
        collision.geoms_colliding(data, getattr(self, f"_floor_geom"), g)
        for g in [
            left_id,
            right_id,
            self._left_hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)
    rewards[f"no_floor_collision"] = no_floor_collision

    return rewards


def domain_randomize(model: mjx.Model, rng: jax.Array):
  mj_model = Pick().mj_model
  obj_id = mj_model.geom("box").id
  obj_body_id = mj_model.body("box").id

  @jax.vmap
  def rand(rng):
    key, key_size, key_mass = jax.random.split(rng, 3)
    # geom size
    geom_size_sides = jax.random.uniform(key_size, (), minval=0.01, maxval=0.03)
    geom_size = model.geom_size.at[obj_id, 1:3].set(geom_size_sides)

    # mass
    mass = jax.random.uniform(key_mass, (), minval=0.03, maxval=0.1)
    mass = model.body_mass.at[obj_body_id].set(mass)

    return geom_size, mass

  geom_size, mass = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_size": 0,
      "body_mass": 0,
  })

  model = model.tree_replace({
      "geom_size": geom_size,
      "body_mass": mass,
  })

  return model, in_axes
