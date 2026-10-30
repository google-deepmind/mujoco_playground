# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SO101 robot configuration for pick task."""

import functools
from typing import Any
from typing import Literal

import mujoco
from mujoco_playground.robots.so101.real import constant as real_constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.robots.so101.sim import scene as so101_scene
from mujoco_playground.tasks.manipulation.pick_cube import mdp
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from reaf.simulation import torch as sim_device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import model_randomization
from reaf.simulation.torch import noise
from reaf.simulation.torch import processors
import torch


@functools.cache
def _hover_ee_xmat() -> tuple[float, ...]:
  """Computes end-effector rotation matrix at the nominal HOVER_POSE."""
  model = so101_scene.so101().compile()
  data = mujoco.MjData(model)
  for j_name, val in zip(real_constant.JOINTS, real_constant.HOVER_POSE):
    data.joint(j_name).qpos = val
  mujoco.mj_forward(model, data)
  return tuple(float(x) for x in data.site("gripperframe").xmat)


def _configure_so101_robot(cfg: config.Environment) -> None:
  """Add SO101-specific robot configuration to an environment config."""
  # Ensure the floor geom has a solid color without the checkerboard texture.
  floor_geom = cfg.base_scene.geom("floor")
  if floor_geom is not None:
    floor_geom.material = ""
    floor_geom.rgba = [0.25, 0.25, 0.25, 1.0]
  cfg.base_scene.stat.extent = 4.0

  robot_joints = [
      f"{sim_constant.ATTACHMENT_PREFIX}{j}" for j in real_constant.JOINTS
  ]
  cfg.device_layer.devices["robot_state"] = config.Component(
      sim_device.ArticulatedDevice,
      joint_names=robot_joints,
      home_qpos=list(real_constant.HOVER_POSE_OPEN_GRIPPER),
      measurements=[
          real_constant.HOME_QPOS,
          real_constant.JOINT_POS,
          real_constant.JOINT_VEL,
          "joint_range",
      ],
  ).attach(so101_scene.so101, prefix=sim_constant.ATTACHMENT_PREFIX)

  cfg.device_layer.devices["robot"] = config.Component(
      sim_device.ActuatorDevice,
      targets=robot_joints,
      home_ctrl=list(real_constant.HOVER_POSE_OPEN_GRIPPER),
  )

  cfg.device_layer.devices["ee"] = config.Component(
      sim_device.SiteDevice,
      name="ee",
      site_name=sim_constant.EE_SITE,
  )

  cfg.device_layer.devices["floor_contact"].configure_spec(
      objname=(
          f"{sim_constant.ATTACHMENT_PREFIX}gripper",
          f"{sim_constant.ATTACHMENT_PREFIX}moving_jaw_so101_v1",
          f"{sim_constant.ATTACHMENT_PREFIX}camera_mount",
      )
  )

  cfg.task_layer.command_feature_producers["action_history"].runtime_params[
      "feature_shape"
  ] = (len(real_constant.JOINTS),)

  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  assert isinstance(dr_terms, dict)
  fingertip_pattern = (
      rf"{sim_constant.ATTACHMENT_PREFIX}(fixed_jaw|moving_jaw).*"
  )
  # Randomize fingertip sliding, torsional, and rolling friction coefficients.
  dr_terms["geom_friction"].params["pattern"] = fingertip_pattern
  # Scale proportional (kp) and derivative (kd) PD gains across robot actuators.
  dr_terms["actuator_gains"].params[
      "pattern"
  ] = rf"{sim_constant.ATTACHMENT_PREFIX}.*"
  # Scale passive joint damping across all robot degrees of freedom.
  dr_terms["dof_damping"].params[
      "pattern"
  ] = rf"{sim_constant.ATTACHMENT_PREFIX}.*"
  # Randomize target cube half-extents.
  dr_terms["geom_size"].params["noise"] = noise.uniform(
      0.012, 0.018, op="replace"
  )

  resetter_params = cfg.resetters["pick_cube"].runtime_params
  resetter_params["object_pos_range"] = (
      # x: 20 cm to 35 cm in front of the robot base.
      (0.20, 0.35),
      # y: 10 cm left to 10 cm right of the centerline.
      (-0.1, 0.1),
      # z: fixed initial center height of 1.8 cm above the table (z = 0).
      (0.018, 0.018),
  )
  resetter_params["resampling_time_range"] = (4.0, 7.0)

  stiffnesses = [sim_constant.STIFFNESS[jn] for jn in real_constant.JOINTS]
  arm_scales = [12.5 * sim_constant.FORCERANGE / s for s in stiffnesses[:5]]
  gripper_scale = 25.0 * sim_constant.FORCERANGE / stiffnesses[5]
  action_scale_val = arm_scales + [gripper_scale]

  cfg.task_layer.command_processors = {
      "action_scale": config.Component(
          processors.TransformActions,
          output_key="robot/ctrl",
          scale=action_scale_val,
          offset=real_constant.HOVER_POSE_OPEN_GRIPPER,
      ),
  }

  cfg.device_layer.command_processors = {
      "control_delay": config.Component(
          processors.StochasticSubstepCommands,
          keys="robot/ctrl",
          delay=(0.000, 0.004),
      ),
  }

  # Ground rewards in grasping, lifting, bringing the cube to a fixed hover
  # pose, and hardware safety.
  rewards = cfg.task_layer.rewards
  rewards["grasp"] = config.Reward(
      mdp.reward_gripper_close,
      ee_xpos_key="ee/xpos",
      cube_xpos_key="object/xpos",
      qpos_key=f"robot_state/{real_constant.JOINT_POS}",
      gripper_joint_idx=-1,
      open_pos=0.80,
      closed_pos=0.15,
      reaching_std=0.025,
      weight=1.5,
  )
  # Lifting from table_z=0.018.
  rewards["lift"].params["table_z"] = 0.018
  rewards["bringing"].params["table_z"] = 0.018
  rewards["bringing"].params["std"] = 0.08
  rewards["success"].params["distance_threshold"] = 0.05
  # Heavy penalty on hitting the floor to prevent real robots continuously
  # slamming the table.
  rewards["floor_contact"].params["penalty_scale"] = -1.0
  rewards["action_rate"].params["penalty_scale"] = -0.02
  rewards["joint_vel_hinge"].params["max_vel"] = 0.4
  rewards["joint_vel_hinge"].params["penalty_scale"] = -0.015

  rewards["arm_nominal_pose"] = config.Reward(
      mdp.reward_arm_nominal_pose,
      qpos_key=f"robot_state/{real_constant.JOINT_POS}",
      home_qpos_key=f"robot_state/{real_constant.HOME_QPOS}",
      num_arm_joints=5,
      std=1.0,
      weight=0.05,
  )
  rewards["ee_nominal_orientation"] = config.Reward(
      mdp.reward_ee_nominal_orientation,
      xmat_key="ee/xmat",
      nominal_xmat=_hover_ee_xmat(),
      std=0.5,
      weight=0.05,
  )


def so101_pick_cube_env() -> config.Environment:
  """SO101 pick cube environment configuration."""
  cfg = pick_task.pick_cube_env()
  _configure_so101_robot(cfg)
  return cfg


def _rest_qpos() -> list[float]:
  """The arm's rest pose in simulation units."""
  low, high = so101_scene.so101().compile().joint("gripper").range
  gripper = real_constant.gripper_pct_to_rad(
      real_constant.GRIPPER_HOME, float(low), float(high)
  )
  return list(real_constant.FOLLOWER_HOME[:-1]) + [float(gripper)]


def _set_geom_rgb(
    bound: Any, rgb: torch.Tensor, env_mask: torch.Tensor | None
) -> None:
  """Overwrites the RGB of the bound geoms, preserving their alpha."""
  rgba = bound.geom_rgba
  new_rgba = rgba.clone()
  if new_rgba[..., :3].ndim < rgb.ndim:
    rgb = rgb.squeeze(1)
  new_rgba[..., :3] = rgb
  bound.set("geom_rgba", new_rgba, env_mask=env_mask)


def randomize_uniform_geom_rgba(
    model: Any,
    spec: mujoco.MjSpec,
    device: torch.device,
    *,
    pattern: str | tuple[str, ...],
    low: tuple[float, float, float] = (0.05, 0.05, 0.05),
    high: tuple[float, float, float] = (0.95, 0.95, 0.95),
    env_mask: torch.Tensor | None = None,
) -> None:
  """Randomizes geom_rgba uniformly across all matched geoms per world."""
  bound = model.bind_pattern(spec, mujoco.mjtObj.mjOBJ_GEOM, pattern)
  rgb = noise.sample_uniform(
      (model.nworld, 1, 3),
      device,
      low=torch.tensor(low, device=device),
      high=torch.tensor(high, device=device),
  )
  _set_geom_rgb(bound, rgb, env_mask)


def _sample_floor_rgb(n: int, device: torch.device) -> torch.Tensor:
  """Samples floor colors covering wood desks, gray mats, and general tables."""
  u = functools.partial(noise.sample_uniform, device=device)
  # Gray mats and desks.
  gray = u((n, 1), low=0.12, high=0.87) + u((n, 3), low=-0.02, high=0.02)
  # Wood desks.
  red = u((n, 1), low=0.30, high=0.95) * u((n, 1), low=0.80, high=1.00)
  green = red * u((n, 1), low=0.60, high=0.75)
  blue = green * u((n, 1), low=0.45, high=0.65)
  wood = torch.cat([red, green, blue], dim=-1).clamp(0.08, 0.92)
  # Anything else like white and black table tops.
  other = u((n, 3), low=0.10, high=0.90)
  p = u((n, 1), low=0.0, high=1.0)
  return torch.where(p < 0.40, gray, torch.where(p < 0.75, wood, other))


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
  """Converts HSV tensor with last dim 3 in [0, 1] to RGB in [0, 1]."""
  h, s, v = hsv[..., :1], hsv[..., 1:2], hsv[..., 2:3]
  # All three channels are the same triangular hue ramp, evaluated at offsets
  # of 5, 3 and 1 sextants for red, green and blue respectively.
  offsets = torch.tensor([5.0, 3.0, 1.0], device=hsv.device)
  k = (offsets + h * 6.0) % 6.0
  return v - v * s * torch.clamp(torch.minimum(k, 4.0 - k), 0.0, 1.0)


def randomize_contrasting_geom_rgba(
    model: Any,
    spec: mujoco.MjSpec,
    device: torch.device,
    *,
    cube_pattern: str = "cube/cube",
    floor_pattern: str = "floor",
    min_dist: float = 0.50,
    env_mask: torch.Tensor | None = None,
) -> None:
  """Randomizes cube and floor colors ensuring vibrant, high-contrast cube appearance."""
  n = model.nworld
  floor_rgb = _sample_floor_rgb(n, device)

  # Sample vibrant, highly recognizable cube colors:
  # Hue Uniform(0, 1), Saturation Uniform(0.80, 1.0),
  # Brightness Uniform(0.85, 1.0). Guarantees the cube is never dark like a
  # shadow or desaturated like the floor.
  h = torch.rand(n, device=device)
  s = noise.sample_uniform((n,), device, low=0.80, high=1.00)
  v = noise.sample_uniform((n,), device, low=0.85, high=1.00)
  cube_rgb = _hsv_to_rgb(torch.stack([h, s, v], dim=-1))

  # If contrast against floor is below min_dist, shift to complementary hue
  dist = torch.linalg.norm(cube_rgb - floor_rgb, dim=-1, keepdim=True)
  h_shifted = (h + 0.5) % 1.0
  cube_shifted = _hsv_to_rgb(torch.stack([h_shifted, s, v], dim=-1))
  cube_rgb = torch.where(dist < min_dist, cube_shifted, cube_rgb)

  cube_bound = model.bind_pattern(spec, mujoco.mjtObj.mjOBJ_GEOM, cube_pattern)
  _set_geom_rgb(cube_bound, cube_rgb.unsqueeze(1), env_mask)
  floor_bound = model.bind_pattern(
      spec, mujoco.mjtObj.mjOBJ_GEOM, floor_pattern
  )
  _set_geom_rgb(floor_bound, floor_rgb.unsqueeze(1), env_mask)


def so101_pick_cube_pixels_env(
    impl: Literal["cpp", "warp"] | None = None,
) -> config.Environment:
  """SO101 pick cube with camera observations.

  Args:
    impl: "warp" or "cpp". Defaults to "warp" when CUDA is available and "cpp"
      otherwise.

  Returns:
    The environment config.
  """
  impl = impl or ("warp" if torch.cuda.is_available() else "cpp")
  cfg = pick_task.pick_cube_pixels_env()
  _configure_so101_robot(cfg)
  cfg.sim.ccd_iterations = 300
  cfg.sim.control_timestep = 0.1  # 10 Hz control frequency
  cfg.sim.impl = impl

  # Delta position action space for smooth control & scalable deploy.
  cfg.task_layer.command_processors = {
      "delta_action": config.Component(
          processors.DeltaActions,
          output_key="robot/ctrl",
          scale=sim_constant.DELTA_ACTION_SCALE,
          offset=real_constant.HOVER_POSE_OPEN_GRIPPER,
          action_clip=(-1.0, 1.0),
          smoothing=sim_constant.ACTION_SMOOTHING,
          clip=(sim_constant.ACTION_CLIP_MIN, sim_constant.ACTION_CLIP_MAX),
      ),
  }

  # Static workspace center for room lights and realistic shadows.
  cfg.base_scene.worldbody.add_body(
      name="workspace_center", pos=[0.25, 0.0, 0.0]
  )

  # Dim headlight so shadows on table and under arm are clearly visible.
  headlight = cfg.base_scene.visual.headlight
  if headlight is not None:
    headlight.diffuse = [0.1, 0.1, 0.1]
    headlight.ambient = [0.15, 0.15, 0.15]

  # Key light targets fixed workspace center and casts realistic shadows.
  key_light = cfg.base_scene.light("light")
  key_light.pos = [0.2, 0.0, 1.4]
  key_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  key_light.targetbody = "workspace_center"
  key_light.diffuse = [0.6, 0.6, 0.6]
  key_light.castshadow = True

  # Front fill light casts directional shadow across table for arm/wrist shadow.
  fill_light = cfg.base_scene.light("front_fill")
  fill_light.pos = [0.1, -1.0, 1.2]
  fill_light.diffuse = [0.35, 0.35, 0.35]
  fill_light.castshadow = True

  # Left and right room lights target fixed workspace center.
  left_light = cfg.base_scene.worldbody.add_light(name="left_room_light")
  left_light.pos = [-0.7, 0.1, 1.2]
  left_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  left_light.targetbody = "workspace_center"
  left_light.diffuse = [0.25, 0.25, 0.25]
  left_light.specular = [0.05, 0.05, 0.05]
  left_light.castshadow = False

  right_light = cfg.base_scene.worldbody.add_light(name="right_room_light")
  right_light.pos = [0.7, 0.1, 1.2]
  right_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  right_light.targetbody = "workspace_center"
  right_light.diffuse = [0.25, 0.25, 0.25]
  right_light.specular = [0.05, 0.05, 0.05]
  right_light.castshadow = False

  cam_fullname = sim_constant.WRIST_CAM_FULLNAME
  render_rate_hz = 1.0 / cfg.sim.control_timestep
  if impl == "cpp":
    cfg.sim.xpu_device = "cpu"
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    cfg.device_layer.devices["camera"] = config.Component(
        sim_device.MujocoCameraDevice,
        camera_name=cam_fullname,
        width=64,
        height=64,
        render_rate_hz=render_rate_hz,
        scene_option=scene_option,
        image_output_name_prefix=cam_fullname,
        rgb_output_name_suffix="_rgb",
    )
  else:
    cfg.device_layer.devices["camera"] = config.Component(
        sim_device.CameraDevice,
        cam_name=[cam_fullname],
        cam_resolution=[(64, 64)],
        render_rate_hz=render_rate_hz,
        render_rgb=True,
        render_depth=False,
        enabled_geom_groups=(0, 2),
        # Activates camera domain randomization.
        use_precomputed_rays=False,
        use_shadows=True,
    )

  policy_terms = cfg.observation_adapter["policy"].terms

  # Per-scenario bias that mimics off-calibrated robots, plus step-wise noise.
  policy_terms["joint_pos"].configure(
      bias=noise.Noise(
          dist=noise.Uniform(low=-0.01, high=0.01),
          op="add",
      )
  )
  policy_terms["ee_pos"].configure(
      bias=noise.Noise(
          dist=noise.Uniform(low=-0.0025, high=0.0025),
          op="add",
      ),
      noise=noise.Noise(
          dist=noise.Uniform(low=-0.001, high=0.001),
          op="add",
      ),
  )
  policy_terms["ee_orientation"].configure(
      noise=noise.Noise(
          dist=noise.Uniform(low=-0.02, high=0.02),
          op="add",
      )
  )

  cfg.observation_adapter["camera"] = config.ObsGroup(
      terms={
          "camera": (
              config.Obs(
                  mdp.obs_camera,
                  camera_key=sim_constant.WRIST_CAM_RGB,
                  downsample_size=(16, 16),
                  pixel_noise_std=0.02,
              ).configure(output_shape=(3, 16, 16))
          )
      },
      enable_noise=False,
  )

  visual_dr_terms = {
      # Strip textures from robot, cube, and floor geoms for flat RGBA colors.
      "override_flat_materials": config.Term(
          mdp.override_flat_materials,
          patterns=(
              rf"{sim_constant.ATTACHMENT_PREFIX}.*",
              "cube/cube",
              "floor",
          ),
      ),
      # Randomize robot base z-height at startup for mounting height offsets.
      "robot_base_pos": config.Term(
          model_randomization.randomize_field,
          field="body_pos",
          pattern=f"{sim_constant.ATTACHMENT_PREFIX}base",
          noise={
              2: noise.Noise(
                  dist=noise.Uniform(low=-0.0025, high=0.0025),
                  op="replace",
              ),
          },
      ),
      # Randomize all robot link RGB colors on episode reset.
      "geom_rgba_robot": config.Term(
          randomize_uniform_geom_rgba,
          pattern=rf"{sim_constant.ATTACHMENT_PREFIX}.*",
          low=(0.05, 0.05, 0.05),
          high=(0.95, 0.95, 0.95),
          trigger="reset",
      ),
      # Randomize floor and cube RGB colors on reset with high visual contrast.
      "geom_rgba_contrast": config.Term(
          randomize_contrasting_geom_rgba,
          cube_pattern="cube/cube",
          floor_pattern="floor",
          min_dist=0.50,
          trigger="reset",
      ),
      # Perturb 3D light positions to vary shadow angles and highlights.
      "light_pos": config.Term(
          model_randomization.randomize_field,
          field="light_pos",
          pattern=".*",
          noise=noise.uniform(-0.25, 0.25, op="add"),
      ),
      # Perturb aiming directions of all scene lights.
      "light_dir": config.Term(
          model_randomization.randomize_field,
          field="light_dir",
          pattern=".*",
          noise=noise.uniform(-0.35, 0.35, op="add"),
      ),
      # Scale diffuse brightness of all scene lights for varying illumination.
      "light_diffuse": config.Term(
          model_randomization.randomize_field,
          field="light_diffuse",
          pattern=".*",
          noise=noise.uniform(0.70, 1.30, op="scale"),
      ),
      # Perturb wrist camera 3D mounting position relative to the wrist link.
      "cam_pos": config.Term(
          model_randomization.randomize_field,
          field="cam_pos",
          pattern=sim_constant.WRIST_CAM_FULLNAME,
          noise={
              0: noise.uniform(-0.005, 0.005, op="add"),
              1: noise.uniform(-0.005, 0.005, op="add"),
              2: noise.uniform(-0.005, 0.005, op="add"),
          },
      ),
      # Perturb wrist camera mounting orientation.
      "cam_quat": config.Term(
          model_randomization.randomize_quat,
          field="cam_quat",
          pattern=sim_constant.WRIST_CAM_FULLNAME,
          max_angle_degrees=4.0,
      ),
      # Scale wrist camera vertical FOV for lens intrinsics variance.
      "cam_fovy": config.Term(
          model_randomization.randomize_field,
          field="cam_fovy",
          pattern=sim_constant.WRIST_CAM_FULLNAME,
          noise=noise.uniform(0.9, 1.1, op="scale"),
      ),
  }

  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  assert isinstance(dr_terms, dict)
  dr_terms.update(visual_dr_terms)

  return cfg
