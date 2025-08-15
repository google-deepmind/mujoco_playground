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
"""A simple task with demonstrating sim2real transfer for pixels observations.
Pick up a cube to a fixed location using a cartesian controller."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.manipulation.franka_emika_panda import pick


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=64,
      render_height=64,
      use_rasterizer=False,
      enabled_geom_groups=[0, 1, 2],
  )


def default_config():
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=200,
      action_repeat=1,
      # Size of cartesian increment.
      action_scale=0.005,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide cube with gripper
              no_box_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      vision=False,
      vision_config=default_vision_config(),
      obs_noise=config_dict.create(brightness=[1.0, 1.0]),
      box_init_range=0.05,
      success_threshold=0.05,
      action_history_length=1,
      impl='jax',
      nconmax=12 * 8192,
      njmax=128,
  )
  return config


def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


class PandaPickCubeCartesian(pick.PandaPickCube):
  """Environment for training the Franka Panda robot to pick up a cube in
  Cartesian space."""

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config=default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):

    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    self._vision = config.vision

    xml_path = (
        mjx_env.ROOT_PATH
        / 'manipulation'
        / 'franka_emika_panda'
        / 'xmls'
        / 'mjx_single_cube_camera.xml'
    )
    self._xml_path = xml_path.as_posix()
    self._model_assets = panda.get_assets()

    mj_model = self.modify_model(
        mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=self._model_assets
        )
    )
    mj_model.opt.timestep = config.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)

    # Set gripper in sight of camera
    self._post_init(obj_name='box', keyframe='low_home')
    self._box_geom = self._mj_model.geom('box').id

    # Contact sensor ID.
    self._box_hand_found_sensor = self._mj_model.sensor('box_hand_found').id

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

    if self._vision:
      try:
        # pylint: disable=import-outside-toplevel
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
      except ImportError:
        warnings.warn(
            'Madrona MJX not installed. Cannot use vision with'
            ' PandaPickCubeCartesian.'
        )
        return
      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_width=self._config.vision_config.render_width,
          batch_render_view_height=self._config.vision_config.render_height,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=None,  # Use all cameras.
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def _post_init(self, obj_name, keyframe):
    super()._post_init(obj_name, keyframe)
    self._guide_q = self._mj_model.keyframe('picked').qpos
    self._guide_ctrl = self._mj_model.keyframe('picked').ctrl
    # Use forward kinematics to init cartesian control
    self._start_tip_transform = panda_kinematics.compute_franka_fk(
        self._init_ctrl[:7]
    )
    self._sample_orientation = False

  def modify_model(self, mj_model: mujoco.MjModel):
    # Expand floor size to non-zero so Madrona can render it
    mj_model.geom_size[mj_model.geom('floor').id, :2] = [5.0, 5.0]

    # Make the finger pads white for increased visibility
    mesh_id = mj_model.mesh('finger_1').id
    geoms = [
        idx
        for idx, data_id in enumerate(mj_model.geom_dataid)
        if data_id == mesh_id
    ]
    mj_model.geom_matid[geoms] = mj_model.mat('off_white').id
    return mj_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    x_plane = self._start_tip_transform[0, 3] - 0.03  # Account for finite gain

    # intialize box position
    rng, rng_box = jax.random.split(rng)
    r_range = self._config.box_init_range
    box_pos = jp.array([
        x_plane,
        jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
        0.0,
    ])

    # Fixed target position to simplify pixels-only training.
    target_pos = jp.array([x_plane, 0.0, 0.20])

    # initialize pipeline state
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data = data.replace(
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat)
    )
    if not self._vision:
      # mocap target should not appear in the pixels observation.
      data = data.replace(
          mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
      )

    # initialize env state and info
    metrics = {
        'out_of_bounds': jp.array(0.0),
        **{
            f'reward/{k}': 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
        'reward/success': jp.array(0.0),
        'reward/lifted': jp.array(0.0),
    }

    info = {
        'rng': rng,
        'target_pos': target_pos,
        'reached_box': jp.array(0.0, dtype=float),
        'prev_reward': jp.array(0.0, dtype=float),
        'current_pos': self._start_tip_transform[:3, 3],
        'newly_reset': jp.array(False, dtype=bool),
        'prev_action': jp.zeros(3),
        '_steps': jp.array(0, dtype=int),
        'action_history': jp.zeros((
            self._config.action_history_length,
        )),  # Gripper only
    }

    reward, done = jp.zeros(2)

    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(3)], axis=0)
    if self._vision:
      rng_brightness, rng = jax.random.split(rng)
      brightness = jax.random.uniform(
          rng_brightness,
          (1,),
          minval=self._config.obs_noise.brightness[0],
          maxval=self._config.obs_noise.brightness[1],
      )
      info.update({'brightness': brightness})

      render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
      info.update({'render_token': render_token})

      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, brightness)
      obs = {'pixels/view_0': obs}

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Runs one timestep of the environment's dynamics."""
    action_history = (
        jp.roll(state.info['action_history'], 1).at[0].set(action[2])
    )
    state.info['action_history'] = action_history
    # Add action delay
    state.info['rng'], key = jax.random.split(state.info['rng'])
    action_idx = jax.random.randint(
        key, (), minval=0, maxval=self._config.action_history_length
    )
    action = action.at[2].set(state.info['action_history'][action_idx])

    state.info['newly_reset'] = state.info['_steps'] == 0

    newly_reset = state.info['newly_reset']
    state.info['prev_reward'] = jp.where(
        newly_reset, 0.0, state.info['prev_reward']
    )
    state.info['current_pos'] = jp.where(
        newly_reset, self._start_tip_transform[:3, 3], state.info['current_pos']
    )
    state.info['reached_box'] = jp.where(
        newly_reset, 0.0, state.info['reached_box']
    )
    state.info['prev_action'] = jp.where(
        newly_reset, jp.zeros(3), state.info['prev_action']
    )

    # Ocassionally aid exploration.
    state.info['rng'], key_swap = jax.random.split(state.info['rng'])
    to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
    swapped_data = state.data.replace(
        qpos=self._guide_q, ctrl=self._guide_ctrl
    )  # help hit the terminal sparse reward.
    data = jax.tree_util.tree_map_with_path(
        lambda path, x, y: ((1 - to_sample) * x + to_sample * y).astype(x.dtype)
        if len(path) == 1
        else x,
        state.data,
        swapped_data,
    )

    # Cartesian control
    increment = jp.zeros(4)
    increment = increment.at[1:].set(action)  # set y, z and gripper commands.
    ctrl, new_tip_position, no_soln = self._move_tip(
        state.info['current_pos'],
        self._start_tip_transform[:3, :3],
        data.ctrl,
        increment,
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    state.info.update({'current_pos': new_tip_position})

    # Simulator step
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    # Dense rewards
    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in raw_rewards.items()
    }

    # Penalize collision with box.
    hand_box = (
        data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]]
        > 0
    )
    raw_rewards['no_box_collision'] = jp.where(hand_box, 0.0, 1.0)

    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    if not self._vision:
      # Vision policy cannot access the required state-based observations.
      da = jp.linalg.norm(action - state.info['prev_action'])
      state.info['prev_action'] = action
      total_reward += self._config.reward_config.action_rate * da
      total_reward += no_soln * self._config.reward_config.no_soln_reward

    # Sparse rewards
    box_pos = data.xpos[self._obj_body]
    lifted = (box_pos[2] > 0.05) * self._config.reward_config.lifted_reward
    total_reward += lifted
    success = self._get_success(data, state.info)
    total_reward += success * self._config.reward_config.success_reward

    # Reward progress
    reward = jp.maximum(
        total_reward - state.info['prev_reward'], jp.zeros_like(total_reward)
    )
    state.info['prev_reward'] = jp.maximum(
        total_reward, state.info['prev_reward']
    )
    reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
    state.metrics.update({f'reward/{k}': v for k, v in raw_rewards.items()})
    state.metrics.update({
        'reward/lifted': lifted.astype(float),
        'reward/success': success.astype(float),
    })

    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
        | success
    )

    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
    if self._vision:
      _, rgb, _ = self.renderer.render(state.info['render_token'], data)
      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, state.info['brightness'])
      obs = {'pixels/view_0': obs}

    return state.replace(
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=state.info,
    )

  def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    target_pos = info['target_pos']
    if (
        self._vision
    ):  # Randomized camera positions cannot see location along y line.
      box_pos, target_pos = box_pos[2], target_pos[2]
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

  def _move_tip(
      self,
      current_tip_pos: jax.Array,
      current_tip_rot: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate new tip position from cartesian increment."""
    # Discrete gripper action where a < 0 := closed
    close_gripper = jp.where(action[3] < 0, 1.0, 0.0)

    scaled_pos = action[:3] * self._config.action_scale
    new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

    new_ctrl = current_ctrl

    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, current_ctrl[6], current_ctrl[:7]
    )
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
    new_tip_pos = jp.where(
        jp.any(jp.isnan(out_jp)), current_tip_pos, new_tip_pos
    )

    new_ctrl = new_ctrl.at[:7].set(out_jp)
    jaw_action = jp.where(close_gripper, -1.0, 1.0)
    claw_delta = jaw_action * 0.02  # up to 2 cm movement per ctrl.
    new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)

    return new_ctrl, new_tip_pos, no_soln

  @property
  def action_size(self) -> int:
    return 3

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
