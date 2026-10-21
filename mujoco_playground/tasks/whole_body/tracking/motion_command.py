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

"""Body tracking device and motion reference feature producer for REAF."""

from collections.abc import Mapping
import math

from dm_env import specs
from mujoco_playground.tasks.whole_body.tracking import motion_loader
import numpy as np
from reaf.simulation.torch import device as reaf_device
from reaf.simulation.torch import feature_producer
from reaf.simulation.torch import math as pg_math
from reaf.simulation.torch import simulator as simulator_module
import torch


def _compute_relative_body_poses(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Re-anchor reference body poses to the robot anchor's xy and yaw.

  Translates the reference motion to the robot anchor's xy position (preserving
  the reference z height) and rotates by the yaw-only orientation difference
  between the robot anchor and reference anchor.

  Args:
    anchor_pos_w: Reference anchor position, (B, 3).
    anchor_quat_w: Reference anchor quaternion, (B, 4).
    robot_anchor_pos_w: Robot anchor position, (B, 3).
    robot_anchor_quat_w: Robot anchor quaternion, (B, 4).
    body_pos_w: Reference body positions, (B, N, 3).
    body_quat_w: Reference body quaternions, (B, N, 4).

  Returns:
    Tuple of (body_pos_relative_w, body_quat_relative_w) with shapes
    (B, N, 3) and (B, N, 4).
  """
  num_bodies = body_pos_w.shape[1]
  anchor_pos = anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
  anchor_quat = anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
  robot_anchor_pos = (
      robot_anchor_pos_w[:, None, :].clone().repeat(1, num_bodies, 1)
  )
  robot_anchor_quat = robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)

  delta_pos_w = robot_anchor_pos
  delta_pos_w[..., 2] = anchor_pos[..., 2]
  delta_ori_w = pg_math.yaw_quat(
      pg_math.quat_mul(robot_anchor_quat, pg_math.quat_inv(anchor_quat))
  )

  body_quat_relative_w = pg_math.quat_mul(delta_ori_w, body_quat_w)
  body_pos_relative_w = delta_pos_w + pg_math.quat_rotate(
      delta_ori_w, body_pos_w - anchor_pos
  )
  return body_pos_relative_w, body_quat_relative_w


class SharedMotionState:
  """Mutable holder shared between MotionReferenceProducer and GhostUpdater."""

  def __init__(self) -> None:
    self.root_pos: torch.Tensor | None = None
    self.root_quat: torch.Tensor | None = None
    self.joint_pos: torch.Tensor | None = None


class BodyTrackingDevice(reaf_device.Device):
  """DeviceLayer entity exposing tracked body poses, velocities, and IMU data.

  Reads world-frame body positions, orientations, linear velocities at body
  frame origins, angular velocities, and base IMU velocimeter/gyro readings
  from the simulator.
  """

  def __init__(
      self,
      *,
      name: str = "",
      body_names: tuple[str, ...],
      imu_lin_vel_sensor: str = "imu_lin_vel",
      imu_ang_vel_sensor: str = "imu_ang_vel",
  ) -> None:
    self._name = name
    self._body_names = tuple(body_names)
    self._num_bodies = len(self._body_names)
    self._imu_lin_vel_sensor = imu_lin_vel_sensor
    self._imu_ang_vel_sensor = imu_ang_vel_sensor

    self._body_ids: torch.Tensor | None = None
    self._body_rootids: torch.Tensor | None = None
    self._imu_lin_adr: int = 0
    self._imu_ang_adr: int = 0

  @property
  def name(self) -> str:
    return self._name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    mj_model = simulator.mj_model
    device = simulator.data.qpos.device
    body_ids = [int(mj_model.body(n).id) for n in self._body_names]
    self._body_ids = torch.tensor(body_ids, dtype=torch.long, device=device)
    self._body_rootids = torch.tensor(
        [int(mj_model.body_rootid[i]) for i in body_ids],
        dtype=torch.long,
        device=device,
    )
    self._imu_lin_adr = int(mj_model.sensor(self._imu_lin_vel_sensor).adr[0])
    self._imu_ang_adr = int(mj_model.sensor(self._imu_ang_vel_sensor).adr[0])

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    nb = self._num_bodies
    f32 = np.float32
    return {
        f"{self._name}/body_pos_w": specs.Array(shape=(nb, 3), dtype=f32),
        f"{self._name}/body_quat_w": specs.Array(shape=(nb, 4), dtype=f32),
        f"{self._name}/body_lin_vel_w": specs.Array(shape=(nb, 3), dtype=f32),
        f"{self._name}/body_ang_vel_w": specs.Array(shape=(nb, 3), dtype=f32),
        f"{self._name}/base_lin_vel": specs.Array(shape=(3,), dtype=f32),
        f"{self._name}/base_ang_vel": specs.Array(shape=(3,), dtype=f32),
    }

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      *,
      force: bool = False,
  ) -> Mapping[str, torch.Tensor]:
    del force
    data = simulator.data
    body_pos = data.xpos[:, self._body_ids].float()
    body_quat = data.xquat[:, self._body_ids].float()
    cvel = data.cvel[:, self._body_ids].float()
    ref_com = data.subtree_com[:, self._body_rootids].float()
    body_ang_vel = cvel[..., :3]
    body_lin_vel = cvel[..., 3:] - torch.cross(
        body_ang_vel, ref_com - body_pos, dim=-1
    )
    base_lin_vel = data.sensordata[
        :, self._imu_lin_adr : self._imu_lin_adr + 3
    ].float()
    base_ang_vel = data.sensordata[
        :, self._imu_ang_adr : self._imu_ang_adr + 3
    ].float()
    return {
        f"{self._name}/body_pos_w": body_pos,
        f"{self._name}/body_quat_w": body_quat,
        f"{self._name}/body_lin_vel_w": body_lin_vel,
        f"{self._name}/body_ang_vel_w": body_ang_vel,
        f"{self._name}/base_lin_vel": base_lin_vel,
        f"{self._name}/base_ang_vel": base_ang_vel,
    }


class MotionReferenceProducer(feature_producer.FeatureProducer):
  """TaskLayer feature producer for reference motion tracking.

  Pure tensor component with no direct simulator dependency. Consumes robot
  joint and body measurements from DeviceLayer, aligns the reference motion
  at `self.time_steps` to the robot anchor frame, updates adaptive failure-bin
  statistics, and emits reference features and tracking error metrics.
  """

  def __init__(
      self,
      *,
      nworld: int = 1,
      xpu_device: str = "cpu",
      motion_file: str | None = None,
      body_indexes: tuple[int, ...] | list[int],
      num_joints: int,
      num_bodies: int,
      anchor_body_index: int,
      ee_body_indexes: tuple[int, ...] | list[int] = (),
      sampling_mode: str = "adaptive",
      adaptive_kernel_size: int = 1,
      adaptive_lambda: float = 0.8,
      adaptive_uniform_ratio: float = 0.1,
      adaptive_alpha: float = 0.001,
      step_dt: float = 0.02,
      anchor_pos_z_threshold: float = 0.25,
      anchor_ori_threshold: float = 0.8,
      ee_pos_z_threshold: float = 0.25,
      body_tracking_device_name: str = "body_tracking",
      robot_device_name: str = "robot",
      shared_state: SharedMotionState | None = None,
  ) -> None:
    super().__init__()
    if sampling_mode not in ("adaptive", "uniform", "start"):
      raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    self._nworld = nworld
    self._device = torch.device(xpu_device)
    self._num_joints = num_joints
    self._num_bodies = num_bodies
    self._anchor_index = anchor_body_index
    self._ee_indexes = list(ee_body_indexes)
    self._sampling_mode = sampling_mode
    self._adaptive_kernel_size = adaptive_kernel_size
    self._adaptive_lambda = adaptive_lambda
    self._adaptive_uniform_ratio = adaptive_uniform_ratio
    self._adaptive_alpha = adaptive_alpha
    self._step_dt = step_dt
    self._anchor_pos_z_threshold = anchor_pos_z_threshold
    self._anchor_ori_threshold = anchor_ori_threshold
    self._ee_pos_z_threshold = ee_pos_z_threshold
    self._body_tracking_name = body_tracking_device_name
    self._robot_name = robot_device_name
    self._shared_state = shared_state

    indexes_tensor = torch.tensor(
        list(body_indexes), dtype=torch.long, device=self._device
    )
    if motion_file:
      self.motion = motion_loader.MotionLoader(
          motion_file, indexes_tensor, device=str(self._device)
      )
    else:
      self.motion = motion_loader.make_dummy_motion(
          self._num_bodies, self._num_joints, device=str(self._device)
      )

    self.time_steps = torch.zeros(nworld, dtype=torch.long, device=self._device)

    self.bin_count = int(self.motion.time_step_total // (1.0 / step_dt)) + 1
    self.bin_failed_count = torch.zeros(self.bin_count, device=self._device)
    self._current_bin_failed = torch.zeros(self.bin_count, device=self._device)
    kernel = torch.tensor(
        [adaptive_lambda**i for i in range(adaptive_kernel_size)],
        dtype=torch.float32,
        device=self._device,
    )
    self.kernel = kernel / kernel.sum()
    self.sampling_entropy = torch.ones(nworld, device=self._device)
    self.sampling_top1_prob = torch.full(
        (nworld,), 1.0 / self.bin_count, device=self._device
    )
    self.sampling_top1_bin = torch.full((nworld,), 0.5, device=self._device)

    self._fail_latch = torch.zeros(
        nworld, dtype=torch.bool, device=self._device
    )
    self._fail_bin = torch.zeros(nworld, dtype=torch.long, device=self._device)
    self.metrics: dict[str, torch.Tensor] = {}

  @property
  def name(self) -> str:
    return "motion"

  def _uniform_sampling(self, env_mask: torch.Tensor) -> None:
    new_time_steps = torch.randint(
        0,
        self.motion.time_step_total,
        (self._nworld,),
        device=self._device,
    )
    self.time_steps[:] = torch.where(env_mask, new_time_steps, self.time_steps)
    self.sampling_entropy[:] = 1.0
    self.sampling_top1_prob[:] = 1.0 / self.bin_count
    self.sampling_top1_bin[:] = 0.5

  def _adaptive_sampling(self, env_mask: torch.Tensor) -> None:
    episode_failed = self._fail_latch & env_mask
    if torch.any(episode_failed):
      fail_bins = self._fail_bin[episode_failed]
      self._current_bin_failed[:] = torch.bincount(
          fail_bins, minlength=self.bin_count
      ).to(self._current_bin_failed.dtype)

    sampling_probabilities = (
        self.bin_failed_count
        + self._adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
        sampling_probabilities.unsqueeze(0).unsqueeze(0),
        [0, self._adaptive_kernel_size - 1],
        mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
        sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)
    sampling_probabilities = (
        sampling_probabilities / sampling_probabilities.sum()
    )

    sampled_bins = torch.multinomial(
        sampling_probabilities, self._nworld, replacement=True
    )
    new_time_steps = (
        (
            sampled_bins
            + pg_math.sample_uniform_range(
                0.0, 1.0, (self._nworld,), device=self._device
            )
        )
        / self.bin_count
        * (self.motion.time_step_total - 1)
    ).long()
    self.time_steps[:] = torch.where(env_mask, new_time_steps, self.time_steps)

    entropy = -(
        sampling_probabilities * (sampling_probabilities + 1e-12).log()
    ).sum()
    entropy_norm = (
        entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    )
    pmax, imax = sampling_probabilities.max(dim=0)
    self.sampling_entropy[:] = entropy_norm
    self.sampling_top1_prob[:] = pmax
    self.sampling_top1_bin[:] = imax.float() / self.bin_count

  def sample_time_steps(self, env_mask: torch.Tensor) -> None:
    """Sample reference motion start steps for environments in `env_mask`."""
    if self._sampling_mode == "start":
      self.time_steps[:] = torch.where(env_mask, 0, self.time_steps)
    elif self._sampling_mode == "uniform":
      self._uniform_sampling(env_mask)
    else:
      self._adaptive_sampling(env_mask)

  def clear_fail_latch(self, env_mask: torch.Tensor) -> None:
    """Clear the failure latch for reset environments."""
    self._fail_latch[:] = torch.where(env_mask, False, self._fail_latch)

  def advance_time_steps(self) -> torch.Tensor:
    """Advance reference motion time by one step and fold adaptive EMA.

    Returns:
      Boolean mask of environments whose trajectory reached the end.
    """
    self.time_steps += 1
    if self._sampling_mode == "adaptive":
      self.bin_failed_count = (
          self._adaptive_alpha * self._current_bin_failed
          + (1.0 - self._adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()
    return self.time_steps >= self.motion.time_step_total

  def _latch_failures(
      self,
      anchor_pos_ref: torch.Tensor,
      anchor_quat_ref: torch.Tensor,
      anchor_pos_robot: torch.Tensor,
      anchor_quat_robot: torch.Tensor,
      body_pos_relative: torch.Tensor,
      body_pos_robot: torch.Tensor,
  ) -> None:
    gravity = torch.zeros_like(anchor_pos_ref)
    gravity[:, 2] = -1.0
    proj_ref = pg_math.quat_rotate_inverse(anchor_quat_ref, gravity)
    proj_robot = pg_math.quat_rotate_inverse(anchor_quat_robot, gravity)

    fail = (
        torch.abs(anchor_pos_ref[:, 2] - anchor_pos_robot[:, 2])
        > self._anchor_pos_z_threshold
    )
    fail |= (
        torch.abs(proj_ref[:, 2] - proj_robot[:, 2])
        > self._anchor_ori_threshold
    )
    if self._ee_indexes:
      ee_err = torch.abs(
          body_pos_relative[:, self._ee_indexes, 2]
          - body_pos_robot[:, self._ee_indexes, 2]
      )
      fail |= torch.any(ee_err > self._ee_pos_z_threshold, dim=-1)

    newly = fail & ~self._fail_latch
    if torch.any(newly):
      bins = torch.clamp(
          (self.time_steps * self.bin_count)
          // max(self.motion.time_step_total, 1),
          0,
          self.bin_count - 1,
      )
      self._fail_bin = torch.where(newly, bins, self._fail_bin)
      self._fail_latch = self._fail_latch | fail

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    nj = self._num_joints
    nb = self._num_bodies
    f32 = np.float32
    return {
        "motion/joint_pos_ref": specs.Array(shape=(nj,), dtype=f32),
        "motion/joint_vel_ref": specs.Array(shape=(nj,), dtype=f32),
        "motion/anchor_pos_ref": specs.Array(shape=(3,), dtype=f32),
        "motion/anchor_quat_ref": specs.Array(shape=(4,), dtype=f32),
        "motion/anchor_pos_robot": specs.Array(shape=(3,), dtype=f32),
        "motion/anchor_quat_robot": specs.Array(shape=(4,), dtype=f32),
        "motion/body_pos_relative": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_quat_relative": specs.Array(shape=(nb, 4), dtype=f32),
        "motion/body_pos_robot": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_quat_robot": specs.Array(shape=(nb, 4), dtype=f32),
        "motion/body_pos_ref": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_quat_ref": specs.Array(shape=(nb, 4), dtype=f32),
        "motion/body_lin_vel_ref": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_ang_vel_ref": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_lin_vel_robot": specs.Array(shape=(nb, 3), dtype=f32),
        "motion/body_ang_vel_robot": specs.Array(shape=(nb, 3), dtype=f32),
        "metrics/error_anchor_pos": specs.Array(shape=(), dtype=f32),
        "metrics/error_anchor_rot": specs.Array(shape=(), dtype=f32),
        "metrics/error_anchor_lin_vel": specs.Array(shape=(), dtype=f32),
        "metrics/error_anchor_ang_vel": specs.Array(shape=(), dtype=f32),
        "metrics/error_body_pos": specs.Array(shape=(), dtype=f32),
        "metrics/error_body_rot": specs.Array(shape=(), dtype=f32),
        "metrics/error_body_lin_vel": specs.Array(shape=(), dtype=f32),
        "metrics/error_body_ang_vel": specs.Array(shape=(), dtype=f32),
        "metrics/error_joint_pos": specs.Array(shape=(), dtype=f32),
        "metrics/error_joint_vel": specs.Array(shape=(), dtype=f32),
        "metrics/sampling_entropy": specs.Array(shape=(), dtype=f32),
        "metrics/sampling_top1_prob": specs.Array(shape=(), dtype=f32),
        "metrics/sampling_top1_bin": specs.Array(shape=(), dtype=f32),
    }

  def required_features_keys(self) -> set[str]:
    bt = self._body_tracking_name
    rb = self._robot_name
    return {
        f"{rb}/joint_pos",
        f"{rb}/joint_vel",
        f"{bt}/body_pos_w",
        f"{bt}/body_quat_w",
        f"{bt}/body_lin_vel_w",
        f"{bt}/body_ang_vel_w",
    }

  def produce_features(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    bt = self._body_tracking_name
    rb = self._robot_name
    joint_pos = required_features[f"{rb}/joint_pos"].float()
    joint_vel = required_features[f"{rb}/joint_vel"].float()
    body_pos = required_features[f"{bt}/body_pos_w"].float()
    body_quat = required_features[f"{bt}/body_quat_w"].float()
    body_lin_vel = required_features[f"{bt}/body_lin_vel_w"].float()
    body_ang_vel = required_features[f"{bt}/body_ang_vel_w"].float()

    anchor_pos = body_pos[:, self._anchor_index]
    anchor_quat = body_quat[:, self._anchor_index]

    t = torch.clamp(self.time_steps, 0, self.motion.time_step_total - 1)
    ref_joint_pos = self.motion.joint_pos[t]
    ref_joint_vel = self.motion.joint_vel[t]
    ref_body_pos_w = self.motion.body_pos_w[t]
    ref_body_quat_w = self.motion.body_quat_w[t]
    ref_body_lin_vel_w = self.motion.body_lin_vel_w[t]
    ref_body_ang_vel_w = self.motion.body_ang_vel_w[t]
    ref_anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
    ref_anchor_quat_w = ref_body_quat_w[:, self._anchor_index]

    body_pos_relative_w, body_quat_relative_w = _compute_relative_body_poses(
        ref_anchor_pos_w,
        ref_anchor_quat_w,
        anchor_pos,
        anchor_quat,
        ref_body_pos_w,
        ref_body_quat_w,
    )

    self._latch_failures(
        ref_anchor_pos_w,
        ref_anchor_quat_w,
        anchor_pos,
        anchor_quat,
        body_pos_relative_w,
        body_pos,
    )

    if self._shared_state is not None:
      self._shared_state.root_pos = ref_body_pos_w[:, 0]
      self._shared_state.root_quat = ref_body_quat_w[:, 0]
      self._shared_state.joint_pos = ref_joint_pos

    anchor_lin_ref = ref_body_lin_vel_w[:, self._anchor_index]
    anchor_ang_ref = ref_body_ang_vel_w[:, self._anchor_index]
    anchor_lin = body_lin_vel[:, self._anchor_index]
    anchor_ang = body_ang_vel[:, self._anchor_index]

    feats: dict[str, torch.Tensor] = {
        "motion/joint_pos_ref": ref_joint_pos,
        "motion/joint_vel_ref": ref_joint_vel,
        "motion/anchor_pos_ref": ref_anchor_pos_w,
        "motion/anchor_quat_ref": ref_anchor_quat_w,
        "motion/anchor_pos_robot": anchor_pos,
        "motion/anchor_quat_robot": anchor_quat,
        "motion/body_pos_relative": body_pos_relative_w,
        "motion/body_quat_relative": body_quat_relative_w,
        "motion/body_pos_robot": body_pos,
        "motion/body_quat_robot": body_quat,
        "motion/body_pos_ref": ref_body_pos_w,
        "motion/body_quat_ref": ref_body_quat_w,
        "motion/body_lin_vel_ref": ref_body_lin_vel_w,
        "motion/body_ang_vel_ref": ref_body_ang_vel_w,
        "motion/body_lin_vel_robot": body_lin_vel,
        "motion/body_ang_vel_robot": body_ang_vel,
        "metrics/error_anchor_pos": torch.norm(
            ref_anchor_pos_w - anchor_pos, dim=-1
        ),
        "metrics/error_anchor_rot": pg_math.quat_error_magnitude2(
            ref_anchor_quat_w, anchor_quat
        ),
        "metrics/error_anchor_lin_vel": torch.norm(
            anchor_lin_ref - anchor_lin, dim=-1
        ),
        "metrics/error_anchor_ang_vel": torch.norm(
            anchor_ang_ref - anchor_ang, dim=-1
        ),
        "metrics/error_body_pos": (
            torch.norm(body_pos_relative_w - body_pos, dim=-1).mean(dim=-1)
        ),
        "metrics/error_body_rot": (
            pg_math.quat_error_magnitude2(body_quat_relative_w, body_quat).mean(
                dim=-1
            )
        ),
        "metrics/error_body_lin_vel": (
            torch.norm(ref_body_lin_vel_w - body_lin_vel, dim=-1).mean(dim=-1)
        ),
        "metrics/error_body_ang_vel": (
            torch.norm(ref_body_ang_vel_w - body_ang_vel, dim=-1).mean(dim=-1)
        ),
        "metrics/error_joint_pos": torch.norm(
            ref_joint_pos - joint_pos, dim=-1
        ),
        "metrics/error_joint_vel": torch.norm(
            ref_joint_vel - joint_vel, dim=-1
        ),
        "metrics/sampling_entropy": self.sampling_entropy,
        "metrics/sampling_top1_prob": self.sampling_top1_prob,
        "metrics/sampling_top1_bin": self.sampling_top1_bin,
    }
    self.metrics = {
        k.removeprefix("metrics/"): v
        for k, v in feats.items()
        if k.startswith("metrics/")
    }
    return feats

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    del env_mask
