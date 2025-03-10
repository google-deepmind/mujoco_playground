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

"""Distillation module for sim-to-real
transfer of ALOHA peg insertion."""

import pathlib
from typing import Any, Dict, Optional, Union

from brax.io import model as brax_loader
from brax.training import networks
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src.manipulation.aloha.s2r import base
from mujoco_playground._src.manipulation.aloha.s2r import depth_noise
from mujoco_playground._src.manipulation.aloha.s2r import peg_insertion
from mujoco_playground._src.manipulation.franka_emika_panda.randomize_vision import perturb_orientation


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      randomization_fn=None,
      render_width=64,
      render_height=64,
      enabled_geom_groups=[1, 2, 5],
      use_rasterizer=False,
      enabled_cameras=[4, 5],
  )


def default_config() -> (
    config_dict.ConfigDict
):  # TODO :Clean up. Or just import?
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
      vision=True,
      vision_config=default_vision_config(),
      obs_noise=config_dict.create(
          depth=True,
          brightness=[0.5, 2.5],
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


def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


def get_frozen_encoder_fn():
  """Returns a function that encodes observations using a frozen vision MLP."""
  vision_mlp = networks.VisionMLP(layer_sizes=(0,), policy_head=False)

  fpath = (
      pathlib.Path(__file__).parent / 'params' / 'VisionMLP2ChanCIFAR10.prms'
  )
  params = brax_loader.load_params(fpath)

  def encoder_fn(obs: Dict):
    stacked = {}
    for i in range(2):
      stacked[f'pixels/view_{i}'] = obs[f'pixels/view_{i}'][None, ...]
    return vision_mlp.apply(params, stacked)[0]  # unbatch

  return encoder_fn


class DistillPegInsertion(peg_insertion.PegInsertion):
  """Distillation environment for peg insertion task with vision capabilities.

  This class extends the PegInsertion environment to support policy distillation
  with vision-based observations, including depth and RGB camera inputs.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, distill=True)
    self._vision = config.vision
    self.encoder_fn = get_frozen_encoder_fn()
    if self._vision:
      # Import here to avoid dependency issues when vision is disabled
      # pylint: disable=import-outside-toplevel
      from madrona_mjx.renderer import BatchRenderer

      render_height = self._config.vision_config.render_height
      render_width = self._config.vision_config.render_width

      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_height=render_height,
          batch_render_view_width=render_width,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=np.asarray(
              self._config.vision_config.enabled_cameras
          ),
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )
      self.max_depth = {'pixels/view_0': 0.4, 'pixels/view_1': 0.4}

      if self._config.obs_noise.depth:
        # color range based on max_depth values.
        # Pre-sample random lines for simplicity.
        max_depth = self.max_depth['pixels/view_0']
        self.line_bank = jp.array(
            depth_noise.np_get_line_bank(
                render_height,
                render_width,
                bank_size=16384,
                color_range=[max_depth * 0.2, max_depth * 0.85],
            )
        )

  def reset_color_noise(self, info):
    info['rng'], rng_brightness = jax.random.split(info['rng'])

    info['brightness'] = jax.random.uniform(
        rng_brightness,
        (),
        minval=self._config.obs_noise.brightness[0],
        maxval=self._config.obs_noise.brightness[1],
    )

    info['color_noise'] = {}
    info['shade_noise'] = {}  # Darkness of the colored object.

    color_noise_scales = {0: 0.3, 2: 0.05}
    shade_noise_mins = {0: 0.5, 2: 0.9}
    shade_noise_maxes = {0: 1.0, 2: 2.0}

    def generate_noise(chan):
      info['rng'], rng_noise, rng_shade = jax.random.split(info['rng'], 3)
      noise = jax.random.uniform(
          rng_noise, (1, 3), minval=0, maxval=color_noise_scales[chan]
      )
      noise = noise.at[0, chan].set(0)
      info['color_noise'][chan] = noise
      info['shade_noise'][chan] = jax.random.uniform(
          rng_shade,
          (),
          minval=shade_noise_mins[chan],
          maxval=shade_noise_maxes[chan],
      )

    for chan in [0, 2]:
      generate_noise(chan)

  def _get_obs_distill(self, data, info, init=False):
    obs_pick = self._get_obs_pick(data, info)
    obs_insertion = jp.concatenate([obs_pick, self._get_obs_dist(data, info)])
    if not self._vision:
      state_wt = jp.concatenate([
          obs_insertion,
          (info['_steps'] / self._config.episode_length).reshape(1),
      ])
      return {'state_with_time': state_wt}
    if init:
      info['render_token'], rgb, depth = self.renderer.init(
          data, self._mjx_model
      )
    else:
      _, rgb, depth = self.renderer.render(info['render_token'], data)
    # Process depth.
    info['rng'], rng_l, rng_r = jax.random.split(info['rng'], 3)
    dmap_l = self.process_depth(depth, 0, 'pixels/view_0', rng_l)
    r_dmap_l = jax.image.resize(dmap_l, (8, 8, 1), method='nearest')
    dmap_r = self.process_depth(depth, 1, 'pixels/view_1', rng_r)
    r_dmap_r = jax.image.resize(dmap_r, (8, 8, 1), method='nearest')

    rgb_l = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
    rgb_r = jp.asarray(rgb[1][..., :3], dtype=jp.float32) / 255.0

    info['rng'], rng_noise1, rng_noise2 = jax.random.split(info['rng'], 3)
    rgb_l = adjust_brightness(
        self.rgb_noise(rng_noise1, rgb_l, info), info['brightness']
    )
    rgb_r = adjust_brightness(
        self.rgb_noise(rng_noise2, rgb_r, info), info['brightness']
    )
    latent_rgb_l, latent_rgb_r = jp.split(
        self.encoder_fn({'pixels/view_0': rgb_l, 'pixels/view_1': rgb_r}),
        2,
        axis=-1,
    )

    # Required for supervision to stay still.
    socket_pos = data.xpos[self._socket_body]
    dist_from_hidden = jp.linalg.norm(socket_pos[:2] - jp.array([-0.4, 0.33]))
    socket_hidden = jp.where(dist_from_hidden < 3e-2, 1.0, 0.0).reshape(1)

    peg_pos = data.xpos[self._peg_body]
    dist_from_hidden = jp.linalg.norm(peg_pos[:2] - jp.array([0.4, 0.33]))
    peg_hidden = jp.where(dist_from_hidden < 3e-2, 1.0, 0.0).reshape(1)

    obs = {
        'proprio': self._get_proprio(data, info),
        'pixels/view_0': dmap_l,  # view_i for debugging only
        'pixels/view_1': dmap_r,
        'pixels/view_2': rgb_l,
        'pixels/view_3': rgb_r,
        'pixels/latent_0': latent_rgb_l,  # actual policy inputs
        'pixels/latent_1': latent_rgb_r,
        'pixels/latent_2': r_dmap_l.ravel(),
        'pixels/latent_3': r_dmap_r.ravel(),
        'socket_hidden': socket_hidden,
        'peg_hidden': peg_hidden,
    }
    return obs

  def _get_proprio(self, data: mjx.Data, info: Dict) -> jax.Array:
    """Get the proprio observations for the real sim2real."""
    info['rng'], rng = jax.random.split(info['rng'])
    # qpos_noise = jax.random.uniform(rng, data.qpos.shape) - 0.5
    qpos_noise = jax.random.uniform(
        rng, (16,), minval=0, maxval=self._config.obs_noise.robot_qpos
    )
    qpos_noise = qpos_noise * jp.array(base.QPOS_NOISE_MASK_SINGLE * 2)
    qpos = data.qpos[:16] + qpos_noise
    l_posobs = qpos[self._left_qposadr]
    r_posobs = qpos[self._right_qposadr]

    def dupll(arr):
      # increases size of array by 1 by dupLicating its Last element.
      return jp.concatenate([arr, arr[-1:]])

    assert info['motor_targets'].shape == (14,), print(
        info['motor_targets'].shape
    )

    l_velobs = l_posobs - dupll(info['motor_targets'][:7])
    r_velobs = r_posobs - dupll(info['motor_targets'][7:])
    proprio_list = [l_posobs, r_posobs, l_velobs, r_velobs]

    switcher = [info['has_switched'].astype(float).reshape(1)]

    proprio = jp.concat(proprio_list + switcher)
    return proprio

  def add_depth_noise(self, key, img: jp.ndarray):
    """Add realistic depth sensor noise to the depth image."""
    render_width = self._config.vision_config.render_width
    render_height = self._config.vision_config.render_height
    assert img.shape == (render_height, render_width, 1)
    # squeeze
    img = img.squeeze(-1)
    grad_threshold = self._config.obs_noise.grad_threshold
    noise_multiplier = self._config.obs_noise.noise_multiplier

    key_edge_noise, key = jax.random.split(key)
    img = depth_noise.edge_noise(
        key_edge_noise,
        img,
        grad_threshold=grad_threshold,
        noise_multiplier=noise_multiplier,
    )
    key_kinect, key = jax.random.split(key)
    img = depth_noise.kinect_noise(key_kinect, img)
    key_dropout, key = jax.random.split(key)
    img = depth_noise.random_dropout(key_dropout, img)
    key_line, key = jax.random.split(key)
    noise_idx = jax.random.randint(key_line, (), 0, len(self.line_bank))
    img = depth_noise.apply_line_noise(img, self.line_bank[noise_idx])

    # With a low probability, return an all-black image.
    p_blackout = 0.02  # once per 2.5 sec.
    key_blackout, key = jax.random.split(key)
    blackout = jax.random.bernoulli(key_blackout, p=p_blackout)
    img = jp.where(blackout, 0.0, img)

    return img[..., None]

  def process_depth(
      self,
      depth,
      chan: int,
      view_name: str,
      key: Optional[jp.ndarray] = None,
  ):
    """Process depth image with normalization and optional noise."""
    img_size = self._config.vision_config.render_width
    num_cams = len(self._config.vision_config.enabled_cameras)
    assert depth.shape == (num_cams, img_size, img_size, 1)
    depth = depth[chan]
    max_depth = self.max_depth[view_name]
    # max_depth = info['max_depth']
    too_big = jp.where(depth > max_depth, 0, 1)
    depth = depth * too_big
    if self._config.obs_noise.depth and key is not None:
      depth = self.add_depth_noise(key, depth)
    return depth / max_depth  # Normalize

  def rgb_noise(self, key, img, info):
    """Apply domain randomization noise to RGB images."""
    # Assumes images are already normalized.
    pixel_noise = 0.03

    # Add noise to all channels and clip
    key_noise, key = jax.random.split(key)
    noise = jax.random.uniform(
        key_noise, img.shape, minval=0, maxval=pixel_noise
    )
    img += noise
    img = jp.clip(img, 0, 1)

    return img

  @property
  def observation_size(self):
    """Return the observation space dimensions for each observation type."""
    # Manually set observation size; default method breaks madrona MJX.
    ret = {
        'has_switched': (1,),
        'proprio': (33,),
        'state': (109,),
        'state_pickup': (106,),
        'peg_hidden': (1,),
        'socket_hidden': (1,),
        'privileged': (110,),
    }
    if self._vision:
      ret.update({
          'pixels/view_0': (8, 8, 1),
          'pixels/view_1': (8, 8, 1),
          'pixels/view_2': (32, 32, 3),
          'pixels/view_3': (32, 32, 3),
          'pixels/latent_0': (64,),
          'pixels/latent_1': (64,),
          'pixels/latent_2': (64,),
          'pixels/latent_3': (64,),
      })
    else:
      ret['state_with_time'] = (110,)
    return ret


def make_teacher_policy():
  """Create a teacher policy for distillation from pre-trained models."""
  env = DistillPegInsertion(config_overrides={'vision': False})

  f_pick_teacher = (
      pathlib.Path(__file__).parent / 'params' / 'AlohaS2RPick.prms'
  )
  f_insert_teacher = (
      pathlib.Path(__file__).parent / 'params' / 'AlohaS2RPegInsertion.prms'
  )

  teacher_pick_policy = peg_insertion.load_brax_policy(
      f_pick_teacher.as_posix(),
      'AlohaS2RPick',
      int(env.action_size / 2),
      distill=True,
  )

  teacher_insert_policy = peg_insertion.load_brax_policy(
      f_insert_teacher.as_posix(),
      'AlohaS2RPegInsertion',
      env.action_size,
      distill=True,
  )
  trained_params = brax_loader.load_params(f_insert_teacher.as_posix())  # WLOG
  obs_keys = trained_params[0].mean.keys()

  @jax.jit
  def teacher_inference_fn(obs, rng):
    l_obs, r_obs = jp.split(obs['state_pickup'], 2, axis=-1)
    l_act, l_extras = teacher_pick_policy(
        {'state': l_obs}, None
    )  # l_extras: for example, loc: B x act_size.
    r_act, r_extras = teacher_pick_policy({'state': r_obs}, None)

    if 'socket_hidden' in obs:
      l_act = jp.where(obs['socket_hidden'], jp.zeros_like(l_act), l_act)
      l_extras = jax.tree_util.tree_map(
          lambda x: jp.where(obs['socket_hidden'], jp.zeros_like(x), x),
          l_extras,
      )
      r_act = jp.where(obs['peg_hidden'], jp.zeros_like(r_act), r_act)
      r_extras = jax.tree_util.tree_map(
          lambda x: jp.where(obs['peg_hidden'], jp.zeros_like(x), x), r_extras
      )
    act_1 = jp.concatenate([l_act, r_act], axis=-1)
    act_extras_1 = jax.tree_util.tree_map(
        lambda x, y: jp.concatenate([x, y], axis=-1), l_extras, r_extras
    )
    obs_2 = {k: obs[k] for k in obs_keys}
    act_2, act_extras_2 = teacher_insert_policy(obs_2, rng)

    # Select a pair based on condition.
    c = obs['has_switched'].reshape(-1, 1)  # 0 for policy 1; 1 for policy 2
    act, extras = jax.tree_util.tree_map(
        lambda x, y: (1 - c) * x + c * y,
        (act_1, act_extras_1),
        (act_2, act_extras_2),
    )
    return act, extras

  return teacher_inference_fn


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Apply domain randomization to camera positions, lights, and materials."""
  cam_ids = default_vision_config().enabled_cameras
  mj_model = DistillPegInsertion(config_overrides={'vision': False}).mj_model
  table_geom_id = mj_model.geom('table').id
  b_ids = [
      mj_model.geom(f'socket-{wall}').id for wall in ['B', 'T', 'L', 'R']
  ]  # blue geoms
  r_ids = [
      mj_model.geom('red_peg').id,
      mj_model.geom('socket-W').id,
  ]  # red geoms

  @jax.vmap
  def rand(rng):
    # Geom RGBA
    geom_rgba = model.geom_rgba

    # MatID needs to change to enable RGBA randomization.
    geom_matid = model.geom_matid.at[:].set(-1)
    for id in b_ids:
      rng_obj, rng = jax.random.split(rng)
      obj_hue = jax.random.uniform(rng_obj, (), minval=0.5, maxval=1.0)
      geom_rgba = geom_rgba.at[id, 2].set(obj_hue)  # randomize blue dim.
      # Add some noise to the other two dims.
      rng_color, rng = jax.random.split(rng)  # Doesn't work.
      color_noise = jax.random.uniform(rng_color, (2,), minval=0, maxval=0.12)
      geom_rgba = geom_rgba.at[id, :2].set(geom_rgba[id, :2] + color_noise)
      geom_matid = geom_matid.at[id].set(-2)
    for id in r_ids:
      rng_obj, rng = jax.random.split(rng)
      obj_hue = jax.random.uniform(rng_obj, (), minval=0.0, maxval=1.0)
      geom_rgba = geom_rgba.at[id, 0].set(obj_hue)
      rng_color, rng = jax.random.split(rng)
      color_noise = jax.random.uniform(rng_color, (2,), minval=0, maxval=0.07)
      geom_rgba = geom_rgba.at[id, 1:3].set(geom_rgba[id, 1:3] + color_noise)
      geom_matid = geom_matid.at[id].set(-2)

    # Set the floor to a random gray-ish color.
    gray_value = jax.random.uniform(rng, (), minval=0.0, maxval=0.1)
    floor_rgba = (
        geom_rgba[table_geom_id]
        .at[:3]
        .set(jp.array([gray_value, gray_value, gray_value]))
    )
    geom_rgba = geom_rgba.at[table_geom_id].set(floor_rgba)

    # geom_matid = geom_matid.at[peg_geom_id].set(-2)
    # geom_matid = geom_matid.at[table_geom_id].set(-2)

    # Cameras
    cam_pos = model.cam_pos
    cam_quat = model.cam_quat
    for cur_idx in cam_ids:
      rng, rng_pos, rng_ori = jax.random.split(rng, 3)
      offset_scales = jp.array([0.0125, 0.005, 0.005])
      cam_offset = (
          jax.random.uniform(rng_pos, (3,), minval=-1, maxval=1) * offset_scales
      )
      cam_pos = cam_pos.at[cur_idx].set(cam_pos[cur_idx] + cam_offset)
      cam_quat = cam_quat.at[cur_idx].set(
          perturb_orientation(rng_ori, cam_quat[cur_idx], 5)
      )

    n_lights = model.light_pos.shape[0]  # full: (n_lights, 3)

    # Light position
    rng, rng_pos = jax.random.split(rng)
    offset_scales = 10 * jp.array([0.1, 0.1, 0.1]).reshape(1, 3)
    light_offset = (
        jax.random.uniform(rng_pos, model.light_pos.shape, minval=-1, maxval=1)
        * offset_scales
    )
    light_pos = model.light_pos + light_offset

    assert model.light_dir.shape == (n_lights, 3)
    # Perturb the light direction
    light_dir = model.light_dir
    for i_light in range(n_lights):
      rng, rng_ldir = jax.random.split(rng)
      nom_dir = model.light_dir[i_light]
      light_dir = light_dir.at[i_light].set(
          perturb_orientation(rng_ldir, nom_dir, 10)
      )

    # Cast shadows
    rng, rng_lsha = jax.random.split(rng)
    light_castshadow = jax.random.bernoulli(
        rng_lsha, 0.75, shape=(n_lights,)
    ).astype(jp.float32)

    return (
        cam_pos,
        cam_quat,
        geom_rgba,
        geom_matid,
        light_pos,
        light_dir,
        light_castshadow,
    )

  (
      cam_pos,
      cam_quat,
      geom_rgba,
      geom_matid,
      light_pos,
      light_dir,
      light_castshadow,
  ) = rand(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      'cam_pos': 0,
      'cam_quat': 0,
      'geom_rgba': 0,
      'geom_matid': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_castshadow': 0,
  })

  model = model.tree_replace({
      'cam_pos': cam_pos,
      'cam_quat': cam_quat,
      'geom_rgba': geom_rgba,
      'geom_matid': geom_matid,
      'light_pos': light_pos,
      'light_dir': light_dir,
      'light_castshadow': light_castshadow,
  })

  return model, in_axes
