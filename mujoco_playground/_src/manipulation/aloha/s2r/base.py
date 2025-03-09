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
"""
This module contains the base class for the two phases of peg insertion.
It includes functionalities for initializing objects, calculating observations,
and managing the state of the environment during robotic manipulation tasks.
"""

import functools
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src.support import contact_force
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import aloha_constants
from mujoco_playground._src.manipulation.aloha import base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

QPOS_NOISE_MASK_SINGLE = [1] * 6 + [0] * 2  # 6 joints, 2 fingers.
ZIPF_S3 = [
    0.83,
    0.104,
    0.02,
    0.014,
    0.008,
    0.002,
]  # heavy-tailed zipf pmf evaluated at x=1, ..., 6 with s=3.
GRASP_THRESH = 0.015


def get_rand_dir(rng: jax.Array) -> jax.Array:
  key1, key2 = jax.random.split(rng)
  theta = jax.random.normal(key1) * 2 * jp.pi
  phi = jax.random.normal(key2) * jp.pi
  x = jp.sin(phi) * jp.cos(theta)
  y = jp.sin(phi) * jp.sin(theta)
  z = jp.cos(phi)
  return jp.array([x, y, z])


def init_obs_history(init_obs: Dict, history_len: int) -> Dict:
  """Initialize observation history dictionary.

  For each entry in init_obs, creates a history initialized to the same value.

  Args:
    init_obs: Initial observation dictionary
    history_len: Length of history to maintain

  Returns:
    Dictionary containing observation histories
  """
  obs_history = {}
  for k, v in init_obs.items():
    # for state and pixel obs
    obs_axes = (history_len,) + (1,) * len(v.shape)
    obs_history[k] = jp.tile(v, obs_axes)
  return obs_history


def use_obs_history(key, obs_history: Dict, obs: Dict) -> Tuple[Dict, Dict]:
  """Purely in-place.
  1. update obs history.
  2. update obs with value sampled from buffer.
  """
  key, key_sample = jax.random.split(key)  # all sub-obs share the same jitter.
  # Update obs history
  for k, v in obs_history.items():
    shifted = jp.roll(v, 1, axis=0)
    obs_history[k] = shifted.at[0].set(obs[k])
    # Sample
    logits = jp.log(jp.array(ZIPF_S3[: len(v)]))
    obs_idx = jax.random.categorical(key_sample, logits)
    obs[k] = obs_history[k][obs_idx]
  return obs_history, obs


class S2RBase(base.AlohaEnv):
  """Base class for Aloha S2R agent components."""

  def __init__(
      self,
      xml_path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(xml_path, config, config_overrides)
    self.base_info = {
        "left": {"pos": jp.array([-0.469, -0.019, 0.02])},
        "right": {"pos": jp.array([0.469, -0.019, 0.02])},
    }
    self._action_scale = config.action_scale

  def _post_init(self, keyframe: str):
    #### GLOBALS
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl

    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T
    self._init_q = self._mj_model.keyframe(keyframe).qpos

    #### PER OBJECT
    for obj in self.obj_names:
      setattr(
          self,
          f"_{obj}_qposadr",
          self._mj_model.jnt_qposadr[self._mj_model.body(obj).jntadr[0]],
      )
      setattr(self, f"_{obj}_body", self._mj_model.body(obj).id)
      setattr(
          self,
          f"_{obj}_init_pos",
          jp.array(
              self._init_q[
                  getattr(self, f"_{obj}_qposadr") : getattr(
                      self, f"_{obj}_qposadr"
                  )
                  + 3
              ],
              dtype=jp.float32,
          ),
      )
      setattr(
          self, f"_{obj}_grip_site", self._mj_model.site(f"{obj}_grip_here").id
      )
      setattr(
          self,
          f"_{obj}_mocap_target",
          self._mj_model.body(f"{obj}_mocap_target").mocapid,
      )

    #### PER HAND
    for hand in self.hands:
      setattr(
          self,
          f"_{hand}_left_finger_geom_bottom",
          self._mj_model.geom(f"{hand}/left_finger_bottom").id,
      )
      setattr(
          self,
          f"_{hand}_right_finger_geom_bottom",
          self._mj_model.geom(f"{hand}/right_finger_bottom").id,
      )
      # Fingertips
      setattr(
          self,
          f"_{hand}_left_fingertip",
          self._mj_model.site(f"{hand}/left_fingertip").id,
      )
      setattr(
          self,
          f"_{hand}_right_fingertip",
          self._mj_model.site(f"{hand}/right_fingertip").id,
      )
      setattr(
          self,
          f"_{hand}_hand_geom",
          self._mj_model.geom(f"{hand}/gripper_base").id,
      )
      setattr(
          self,
          f"_{hand}_gripper_site",
          self._mj_model.site(f"{hand}/gripper").id,
      )
      setattr(
          self,
          f"_{hand}_base_link",
          self._mj_model.body(f"{hand}/base_link").id,
      )
      setattr(
          self,
          f"_{hand}_qposadr",
          np.array([
              self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
              for j in getattr(aloha_constants, f"{hand.upper()}_JOINTS")
          ]),
      )

      match hand:
        case "left":
          self._left_ctrladr = np.linspace(0, 6, 7, dtype=int)
        case "right":
          self._right_ctrladr = np.linspace(7, 13, 7, dtype=int)
        case _:
          raise ValueError(f"Invalid hand: {hand}")
    self._floor_geom = self._mj_model.geom("table").id

  def _robot_target_qpos(self, data: mjx.Data) -> float:
    robot_target_qpos = 0.0
    for hand in self.hands:
      hand_ids = getattr(self, f"_{hand}_qposadr")
      robot_target_qpos += 1 - jp.tanh(
          jp.linalg.norm(data.qpos[hand_ids] - self._init_q[hand_ids])
      )
    return robot_target_qpos / len(self.hands)

  def sample_fan(self, rng: jax.Array, obj: str) -> Tuple[jax.Array, jax.Array]:
    """Sample a perturbation position and quaternion in a fan pattern.

    Args:
      rng: Random number generator key
      obj: Object name

    Returns:
      Tuple of position perturbation and quaternion
    """
    rng, rng_r, rng_angle = jax.random.split(rng, 3)
    r = jax.random.uniform(
        rng_r,
        shape=(),
        minval=self.noise_config[f"_{obj}_init_pos"].radius_min,
        maxval=self.noise_config[f"_{obj}_init_pos"].radius_max,
    )
    par = self.noise_config[f"_{obj}_init_pos"].angle
    # Can't be a symmetric fan or depth cameras can't distinguish objects
    angle = jax.random.uniform(
        rng_angle,
        shape=(),
        minval=-par / 2,
        maxval=par / 2,
    )
    dx = r * jp.cos(angle)
    dy = r * jp.sin(angle)
    # Jitter the angle so the object isn't perfectly aligned.
    angle_noise = jp.deg2rad(5)
    rng, rng_noise = jax.random.split(rng)
    angle += jax.random.uniform(
        rng_noise, shape=(), minval=-angle_noise, maxval=angle_noise
    )
    quat = jp.array([jp.cos(angle / 2), 0.0, 0.0, jp.sin(angle / 2)])
    return jp.array([dx, dy, 0.0]), quat

  def init_objects(self, rng: jax.Array) -> Tuple[mjx.Data, dict[str, Any]]:
    """Initialize object positions and targets.

    Args:
      rng: Random number generator key

    Returns:
      Tuple of MJX data and info dictionary
    """
    info = {}
    init_q = jp.array(self._init_q)

    for obj, targ, side in zip(
        self.obj_names, self.target_positions, self.hands
    ):  # Defined by child class.

      obj_idx = getattr(self, f"_{obj}_qposadr")

      # Object Position.
      rng, rng_offset = jax.random.split(rng)
      offset, quat_offset = self.sample_fan(rng_offset, obj)
      t = self.base_info[side]["pos"]
      idx_offset = 8 if side == "right" else 0
      base_angle = self._init_q[idx_offset]
      if side == "right":
        base_angle += np.deg2rad(180)
      base_quat = jp.array(
          [jp.cos(base_angle / 2), 0.0, 0.0, jp.sin(base_angle / 2)]
      )
      # rotation_matrix = self.base_info[side]["xmat"]
      rotation_matrix = math.quat_to_mat(base_quat)
      obj_pos = self.point2global(offset, rotation_matrix.T, t)
      init_q = init_q.at[obj_idx : obj_idx + 3].set(obj_pos)

      # Convert quat to mat
      obj_quat = math.quat_mul(base_quat, quat_offset)
      init_q = init_q.at[obj_idx + 3 : obj_idx + 7].set(obj_quat)

      # Target Position.
      rng, rng_target = jax.random.split(rng)
      range_val = self.noise_config[f"_{obj}_target_pos"]
      info[f"_{obj}_target_pos"] = targ + jax.random.uniform(
          rng_target, (3,), minval=-range_val, maxval=range_val
      )

    # Waist init.
    for hand in self.hands:
      rng, rng_waist = jax.random.split(rng)
      range_val = self.noise_config[f"_{hand}_waist_init_pos"]
      first_idx = getattr(self, f"_{hand}_qposadr")[0]
      # fan is assymmetrical. TODO: False?
      rand_setpoint = self._init_q[first_idx] + jax.random.uniform(
          rng_waist, (), minval=-range_val, maxval=range_val
      )
      init_q = init_q.at[first_idx].set(rand_setpoint)
      # Change for ctrl as well.
      first_idx_ctrl = getattr(self, f"_{hand}_ctrladr")[0]
      init_ctrl = (
          jp.array(self._init_ctrl).at[first_idx_ctrl].set(rand_setpoint)
      )

    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=init_ctrl,
    )

    for i, obj in enumerate(self.obj_names):
      target_quat = jp.array(self.target_quats[i])
      mocap_target = getattr(self, f"_{obj}_mocap_target")
      data = data.replace(
          mocap_pos=data.mocap_pos.at[mocap_target, :].set(
              info[f"_{obj}_target_pos"]
          ),
          mocap_quat=data.mocap_quat.at[mocap_target, :].set(target_quat),
      )

    info.update({
        "_steps": jp.array(0, dtype=int),
        "rng": rng,
        "action_history": jp.zeros(
            (self._config.action_history_length, self.action_size),
            dtype=jp.float32,
        ),
        "motor_targets": init_ctrl,
        "init_ctrl": init_ctrl,
    })

    return data, info

  def _step(self, state: State, action: jax.Array) -> mjx.Data:
    """
    Implements action scaling and random gripper delay.
    """
    # Reset if needed.
    newly_reset = state.info["_steps"] == 0
    state.info["action_history"] = jp.where(
        newly_reset,
        jp.zeros(
            (self._config.action_history_length, self.action_size),
            dtype=jp.float32,
        ),
        state.info["action_history"],
    )

    action_history = (
        jp.roll(state.info["action_history"], 1, axis=0).at[0].set(action)
    )
    state.info["action_history"] = action_history

    # Add action delay for all joints
    state.info["rng"], key_joints = jax.random.split(state.info["rng"])
    logits = jp.log(jp.array(ZIPF_S3[: self._config.action_history_length]))
    action_idx = jax.random.categorical(key_joints, logits)
    action = state.info["action_history"][action_idx]

    # Stronger noise to the grippers
    state.info["rng"], key_fingers = jax.random.split(state.info["rng"])
    action_idx = jax.random.randint(
        key_fingers, (), minval=0, maxval=self._config.action_history_length
    )
    action = action.at[self._finger_ctrladr].set(
        state.info["action_history"][action_idx][self._finger_ctrladr]
    )

    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    state.info["motor_targets"] = ctrl
    return data

  def gripping_error(self, data, hand, obj) -> float:
    """Calculate the error between gripper and object grip site.

    Args:
      data: MJX data
      hand: Hand name ('left' or 'right')
      obj: Object name

    Returns:
      Vector from gripper to grip site in local coordinates
    """
    rotation_matrix = data.xmat[getattr(self, f"_{hand}_base_link")]
    t = data.xpos[getattr(self, f"_{hand}_base_link")]
    point2local = functools.partial(
        self.point2local, rotation_matrix=rotation_matrix, t=t
    )
    p_lfing = data.site_xpos[getattr(self, f"_{hand}_left_fingertip")]
    p_rfing = data.site_xpos[getattr(self, f"_{hand}_right_fingertip")]
    p_mid = (p_lfing + p_rfing) / 2
    grip_here = data.site_xpos[getattr(self, f"_{obj}_grip_site")]
    gripper_obj = point2local(p_mid) - point2local(grip_here)
    return gripper_obj

  def gripper_tip_pos(self, data, hand) -> jax.Array:
    p_lfing = data.site_xpos[getattr(self, f"_{hand}_left_fingertip")]
    p_rfing = data.site_xpos[getattr(self, f"_{hand}_right_fingertip")]
    return (p_lfing + p_rfing) / 2

  def _get_obs_pick_helper(
      self, data: mjx.Data, info: dict[str, Any], side: str, obj: str
  ) -> jax.Array:
    """Calculate observations for pickup task between robot and object.

    Coordinates from Forward Kinematics are with respect to the side's base.

    Args:
      data: MJX data
      info: Info dictionary
      side: Robot side ('left' or 'right')
      obj: Object name

    Returns:
      Observation array
    """
    # Robot minimal coords
    i_rob_qpos = getattr(self, f"_{side}_qposadr")
    rob_qpos = data.qpos[i_rob_qpos]
    rob_qvel = data.qvel[i_rob_qpos]

    # Object minimal coords
    i_obj_qvel = getattr(self, f"_{obj}_qveladr")
    i_obj_qvel = np.linspace(i_obj_qvel, i_obj_qvel + 5, 6, dtype=int)
    g_obj_v, g_obj_angv = jp.split(data.qvel[i_obj_qvel], 2, axis=-1)

    # Derived quantities
    # g_gripper_pos = data.site_xpos[getattr(self, f"_{side}_gripper_site")]
    g_obj_pos = data.xpos[getattr(self, f"_{obj}_body")]
    g_target_pos = info[f"_{obj}_target_pos"]
    g_gripper_mat = data.site_xmat[getattr(self, f"_{side}_gripper_site")]
    g_obj_mat = data.xmat[getattr(self, f"_{obj}_body")]
    g_target_mat = math.quat_to_mat(
        data.mocap_quat[getattr(self, f"_{obj}_mocap_target")]
    )
    rotation_matrix = data.xmat[
        getattr(self, f"_{side}_base_link")
    ]  # world to local. Orientation.
    t = data.xpos[
        getattr(self, f"_{side}_base_link")
    ]  # world to local. Translation.

    frame2local = functools.partial(
        self.frame2local, rotation_matrix=rotation_matrix
    )
    point2local = functools.partial(
        self.point2local, rotation_matrix=rotation_matrix, t=t
    )

    obj_v, obj_angv = frame2local(g_obj_v), frame2local(g_obj_angv)
    # gripper_pos = point2local(g_gripper_pos)
    gripper_mat = frame2local(g_gripper_mat)
    obj_mat = frame2local(g_obj_mat)
    gripper_box = self.gripping_error(data, side, obj)  # local
    target_pos = point2local(g_target_pos)
    obj_pos = point2local(g_obj_pos)
    target_mat = frame2local(g_target_mat)

    #### ADD NOISE ####
    # QPOS, QVEL
    info["rng"], key_qpos, key_qvel = jax.random.split(info["rng"], 3)
    noise = jax.random.uniform(
        key_qpos,
        rob_qpos.shape,
        minval=-self._config.obs_noise.robot_qpos,
        maxval=self._config.obs_noise.robot_qpos,
    ) * jp.array(QPOS_NOISE_MASK_SINGLE)
    n_rob_qpos = rob_qpos + noise

    noise = jax.random.uniform(
        key_qvel,
        rob_qvel.shape,
        minval=-self._config.obs_noise.robot_qvel,
        maxval=self._config.obs_noise.robot_qvel,
    ) * jp.array(QPOS_NOISE_MASK_SINGLE)
    n_rob_qvel = rob_qvel + noise

    # OBJ V, ANGV
    info["rng"], key_obj_v, key_obj_angv = jax.random.split(info["rng"], 3)
    n_obj_v = obj_v + jax.random.uniform(
        key_obj_v,
        obj_v.shape,
        minval=-self._config.obs_noise.obj_vel,
        maxval=self._config.obs_noise.obj_vel,
    )
    n_obj_angv = obj_angv + jax.random.uniform(
        key_obj_angv,
        obj_angv.shape,
        minval=-self._config.obs_noise.obj_angvel,
        maxval=self._config.obs_noise.obj_angvel,
    )
    # GRIPPER, OBJ MAT
    info["rng"], key1, key2 = jax.random.split(info["rng"], 3)
    angle = jax.random.uniform(
        key1,
        minval=0,
        maxval=self._config.obs_noise.eef_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key2), angle)
    rand_mat = math.quat_to_mat(rand_quat)
    n_gripper_mat = rand_mat @ gripper_mat

    info["rng"], key1, key2 = jax.random.split(info["rng"], 3)
    angle = jax.random.uniform(
        key1,
        minval=0,
        maxval=self._config.obs_noise.obj_angle * jp.pi / 180,
    )
    rand_quat = math.axis_angle_to_quat(get_rand_dir(key2), angle)
    rand_mat = math.quat_to_mat(rand_quat)
    n_obj_mat = rand_mat @ obj_mat

    # GRIPPER BOX
    info["rng"], key_gripper_box = jax.random.split(info["rng"])
    noise_val = jax.random.uniform(
        key_gripper_box,
        (2, 3),
        minval=-self._config.obs_noise.gripper_box,
        maxval=self._config.obs_noise.gripper_box,
    )
    # Triangle distribution
    n_gripper_box = gripper_box + (noise_val[1] - noise_val[0])

    # OBJ POS
    info["rng"], key_obj = jax.random.split(info["rng"])
    n_obj_pos = obj_pos + jax.random.uniform(
        key_obj,
        obj_pos.shape,
        minval=-self._config.obs_noise.obj_pos,
        maxval=self._config.obs_noise.obj_pos,
    )

    #### DONE ADDING NOISE ####

    return jp.concatenate([
        n_rob_qpos,  # 0:8
        n_rob_qvel,  # 8:16
        n_obj_v,  # 16:19
        n_obj_angv,  # 19:22
        n_gripper_mat.ravel()[3:],  # 25:31 OLD
        n_obj_mat.ravel()[3:],  # 31:37
        n_gripper_box,  # 37:40
        target_pos - n_obj_pos,  # 40:43
        target_mat.ravel()[:6] - n_obj_mat.ravel()[:6],  # 43:49
        data.ctrl[getattr(self, f"_{side}_ctrladr")] - n_rob_qpos[:-1],  # 49:56
    ])

  def _get_obs_pick(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """
    Calculate the observations in local coordinates
    allowing the gripper to pick up an object.
    Returns left and right-hand observations.
    """
    all_obs = []
    for side, obj in zip(self.hands, self.obj_names):
      all_obs.append(self._get_obs_pick_helper(data, info, side, obj))
    return jp.concatenate(all_obs)

  def is_grasped(self, data, hand) -> float:
    # Grasped if both fingers have applied forces > 5.
    t_f = 2.5  # min force. Don't need to squeeze so hard!

    # 3D vec; top and bottom collision bodies
    f_lfing = self.get_finger_force(data, hand, "left")
    f_rfing = self.get_finger_force(data, hand, "right")

    d_lfing = self.get_finger_dir(data, hand, "left")
    d_rfing = -1 * d_lfing

    l_d_flag = self.check_dir(f_lfing, d_lfing)
    l_f_flag = (jp.linalg.norm(f_lfing) > t_f).astype(float)
    r_d_flag = self.check_dir(f_rfing, d_rfing)
    r_f_flag = (jp.linalg.norm(f_rfing) > t_f).astype(float)

    grasped = jp.all(jp.array([l_d_flag, l_f_flag, r_d_flag, r_f_flag])).astype(
        float
    )

    return grasped

  def get_finger_force(self, data, hand, finger):
    """
    Sum up the 3D force vectors across bottom and top collision primitives
    """
    ids = jp.array([
        self._mj_model.geom(f"{hand}/{finger}_finger_{pos}").id
        for pos in ["top", "bottom"]
    ])  # 2
    contact_forces = [
        contact_force(self._mjx_model, data, i, True)[None, :3]  # 1, 3
        for i in np.arange(data.ncon)
    ]
    contact_forces = jp.concat(contact_forces, axis=0)  # ncon, 3
    matches = jp.isin(data.contact.geom, ids).any(axis=1)  # ncon
    dist_mask = data.contact.dist < 0  # ncon

    # Sum
    return jp.sum(contact_forces * (matches * dist_mask)[:, None], axis=0)

  def get_finger_dir(self, data, hand, finger):
    """
    A vector pointing from `finger` to the other finger.
    """
    other = "left" if finger == "right" else "right"

    site_fing = mujoco.mj_name2id(
        self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f"{hand}/{finger}_finger"
    )
    site_ofing = mujoco.mj_name2id(
        self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f"{hand}/{other}_finger"
    )

    v = data.site_xpos[site_ofing] - data.site_xpos[site_fing]

    return v / (jp.linalg.norm(v) + 1e-7)

  def check_dir(self, v1, v2, t_align=jp.deg2rad(75)) -> float:
    m = jp.linalg.norm(v1) * jp.linalg.norm(v2)
    return (jp.arccos(jp.dot(v1, v2) / (m + 1e-7)) < t_align).astype(float)

  def frame2local(self, frame, rotation_matrix):
    """Convert frame from global to local coordinates.

    Args:
      frame: Frame in global coordinates
      rotation_matrix: Rotation matrix from global to local

    Returns:
      Frame in local coordinates
    """
    return rotation_matrix @ frame

  def point2local(self, point, rotation_matrix, t):
    """Convert point from global to local coordinates.

    Args:
      point: Point in global coordinates
      rotation_matrix: Rotation matrix from global to local
      t: Translation vector

    Returns:
      Point in local coordinates
    """
    return self.frame2local(point - t, rotation_matrix)

  def point2global(self, point, rotation_matrix, t):
    """Convert point from local to global coordinates.

    Args:
      point: Point in local coordinates
      rotation_matrix: Rotation matrix from global to local
      t: Translation vector

    Returns:
      Point in global coordinates
    """
    return rotation_matrix.T @ point + t
