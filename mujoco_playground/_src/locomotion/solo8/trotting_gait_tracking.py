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
"""Straight trotting task for Solo8."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.solo8 import base as solo8_base
from mujoco_playground._src.locomotion.solo8 import solo8_constants as consts
from mujoco_playground._src.locomotion.solo8 import trotting_demonstration_trajectory as demo_traj

_PHASES = jp.array([
    [0, jp.pi, jp.pi, 0],  # trot
    #[0, 0.5 * jp.pi, jp.pi, 1.5 * jp.pi],  # walk
    #[0, jp.pi, 0, jp.pi],  # pace
    #[0, 0, jp.pi, jp.pi],  # bound
    #[0, 0, 0, 0],  # pronk
])


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=50.0,
      Kd=0.5,
      early_termination=True,
      action_repeat=1,
      action_scale=0.3,
      history_len=3,
      obs_noise=config_dict.create(
          scales=config_dict.create(
              joint_pos=0.05,
              gyro=0.1,
              gravity=0.03,
              linvel=0.1,
              feet_pos=[0.01, 0.005, 0.02],
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Rewards for demonstration trajectory tracking.
              feet_height=1.0,         # reward for tracking foot height (rz)
              feet_xy_pos=0.5,         # reward for tracking horizontal foot position
              joint_tracking=0.3,      # reward for tracking reference joint angles
              tracking_lin_vel=0.5,
              tracking_ang_vel=0.5,
              flight_phase=1.0,
              # Gait pattern enforcement (TROTTING)
              flight_phase=0.5,        # reward for flight phases (all feet in air)
              invalid_contact=-0.8,    # penalty for 1,3,4 feet contact (only 0,2 valid)
              # Costs.
              ang_vel_xy=-0.5,
              lin_vel_z=-0.5,
              orientation=-5.0,
              action_rate=-0.01,
              invalid_contact=-0.5,
              #hip_splay=-0.5,
          ),
          tracking_sigma=0.25,
      ),
      command_config=config_dict.create(
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.5, 0.5],
          ang_vel_yaw=[-1.0, 1.0],
      ),
      fixed_vx=0.6,
      gait_frequency=[2.0, 2.0],
      gaits=["trot"],
      foot_height=[0.12, 0.12],
      # Gait trajectory parameters (match trotting_demonstration_trajectory.py)
      demo_gait_params=config_dict.create(
          freq=0.6,
          swing_height=0.08,
          ramp_time=1.0,
          hip_amp=0.12,
          knee_swing_amp=0.35,
          knee_stance_amp=0.05,
          swing_threshold=0.25,
      ),
      foot_height=[0.06, 0.06],
      impl="jax",
      nconmax=4 * 8192,
      njmax=50,
  )


class TrottingGaitTracking(solo8_base.Solo8Env):
  """Trotting task."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._hx_idxs = jp.array([0, 2, 4, 6])
    self._hx_default_pose = self._default_pose[self._hx_idxs]
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, noise_rng, gait_freq_rng, gait_rng, foot_height_rng, cmd_rng = (
        jax.random.split(rng, 6)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Sample gait parameters.
    gait_freq = jax.random.uniform(
        gait_freq_rng,
        minval=self._config.gait_frequency[0],
        maxval=self._config.gait_frequency[1],
    )
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    gait = jax.random.randint(  # pylint: disable=redefined-outer-name
        gait_rng, minval=0, maxval=len(self._config.gaits), shape=()
    )
    phase = jp.array(_PHASES)[gait]
    foot_height = jax.random.uniform(
        foot_height_rng,
        minval=self._config.foot_height[0],
        maxval=self._config.foot_height[1],
    )

    info = {
        "command": self.sample_command(cmd_rng),
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "motor_targets": jp.array(self._default_pose),
        "qpos_error_history": jp.zeros(self._config.history_len * 8),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "gait_freq": gait_freq,
        "gait": gait,
        "phase": phase,
        "phase_dt": phase_dt,
        "foot_height": foot_height,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    # Diagnostic metrics (same keys as Stage 1/2 for cross-comparison).
    metrics["diag/forward_vel"] = jp.zeros(())
    metrics["diag/base_height"] = jp.zeros(())
    metrics["diag/roll"] = jp.zeros(())
    metrics["diag/pitch"] = jp.zeros(())
    metrics["diag/action_rate"] = jp.zeros(())
    metrics["diag/n_contact"] = jp.zeros(())
    metrics["diag/is_diagonal"] = jp.zeros(())
    metrics["diag/is_flight"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    obs = self._get_obs(data, info, noise_rng, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng = jax.random.split(state.info["rng"], 3)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, noise_rng, contact)
    done = self._get_termination(data)

    pos, neg = self._get_reward(data, action, state.info, state.metrics, done, contact)
    pos, neg = self._get_reward(data, action, state.info, state.metrics, done, contact)
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg
    r_pos = sum(pos.values())
    r_neg = jp.exp(0.2 * sum(neg.values()))
    reward = r_pos * r_neg * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

    state.info["rng"] = rng
    state.info["step"] = jp.where(
        done | (state.info["step"] > 200),
        0,
        state.info["step"],
    )
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    # Diagnostic metrics for cross-approach comparison.
    local_vel = self.get_local_linvel(data)
    up = self.get_upvector(data)
    contact_f = contact.astype(jp.float32)
    state.metrics["diag/forward_vel"] = local_vel[0]
    state.metrics["diag/base_height"] = data.qpos[2]
    state.metrics["diag/roll"] = up[0]
    state.metrics["diag/pitch"] = up[1]
    state.metrics["diag/action_rate"] = jp.sqrt(
        jp.sum(jp.square(action - state.info["last_act"]))
    )
    n_contact = jp.sum(contact_f)
    state.metrics["diag/n_contact"] = n_contact
    pair1 = contact_f[0] * contact_f[3]  # FL+HR
    pair2 = contact_f[1] * contact_f[2]  # FR+HL
    state.metrics["diag/is_diagonal"] = (
        (pair1 > 0) | (pair2 > 0)
    ).astype(jp.float32)
    state.metrics["diag/is_flight"] = (n_contact == 0).astype(jp.float32)

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return jp.where(
        self._config.early_termination,
        fall_termination,
        jp.zeros((), dtype=fall_termination.dtype),
    )

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      rng: jax.Array,
      contact: jax.Array,
  ) -> jax.Array:
    gyro = self.get_gyro(data)  # (3,)
    rng, noise_rng = jax.random.split(rng)
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.obs_noise.scales.gyro
    )

    gravity = self.get_gravity(data)  # (3,)
    rng, noise_rng = jax.random.split(rng)
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.obs_noise.scales.gravity
    )

    linvel = self.get_local_linvel(data)  # (3,)
    rng, noise_rng = jax.random.split(rng)
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.obs_noise.scales.linvel
    )

    joint_angles = data.qpos[7:]  # (12,)
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.scales.joint_pos
    )

    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 8)
        .at[:8]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    # Concatenate final observation.
    return jp.hstack(
        [
            noisy_linvel,        # (3,) local linear velocity (velocity feedback)
            noisy_gyro,          # (3,)
            noisy_gravity,       # (3,)
            noisy_joint_angles,  # (8,)
            qpos_error_history,  # (history_len * 8,)
            contact,             # (4,)
            phase,               # (8,) cos+sin of foot phases
            info["command"],     # (3,) target velocity command
            info["gait_freq"],   # (1,)
            info["gait"],        # (1,)
            info["foot_height"], # (1,)
        ],
    )

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      contact: jax.Array,
      contact: jax.Array,
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    del done, metrics  # Unused.
    n_contact = jp.sum(contact.astype(jp.float32))

    # Reward for alternating diagonal leg pair contacts and suspension phase.
    def alternating_contact_reward(contact):
        # Diagonal pairs: (0, 3) and (1, 2)
        diagonal_contact = contact[0] * contact[3] + contact[1] * contact[2]
        suspension_phase = (n_contact == 0).astype(jp.float32)  # No legs touching the ground.
        return diagonal_contact + suspension_phase

    # Reward for following the desired phase sequence.
    def phase_tracking_reward(data, phase, foot_height):
          foot_pos = data.site_xpos[self._feet_site_id]
          foot_z = foot_pos[..., -1]
          rz = gait.get_rz(phase, swing_height=foot_height)
          error = jp.sum(jp.square(foot_z - rz))

          # Enforce equal interval lengths for each phase.
          phase_intervals = jp.diff(phase)
          interval_error = jp.sum(jp.square(phase_intervals - jp.mean(phase_intervals)))

          # Ensure correct alternation of phases (0,3 → 1,2 → suspension).
          # This can be done by checking the order of contact states.
          contact_order_error = jp.sum(
              jp.abs(contact[0] * contact[3] - contact[1] * contact[2])
          )

          # Combine the errors into a single reward term.
          return jp.exp(-error / 0.1) * jp.exp(-interval_error / 0.1) * jp.exp(-contact_order_error / 0.1)

    # Penalize stiff leg movements (large changes in velocities or positions).
    def smooth_movement_penalty(data):
        foot_vel = data.sensordata[self._foot_linvel_sensor_adr]
        return jp.sum(jp.square(foot_vel))

    pos = {
        "feet_height": self._reward_feet_height(
            data, ref_dict["foot_heights_ref"]
        ),
        "feet_xy_pos": self._reward_feet_xy_position(
            data, ref_dict["foot_xy_ref"]
        ),
        "joint_tracking": self._reward_joint_tracking(
            data, ref_dict["ctrl_ref"]
        ),
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "alternating_contact": alternating_contact_reward(contact),
        "phase_tracking": phase_tracking_reward(data, info["phase"], info["foot_height"]),
    }
    neg = {
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data), info["gait"]
        ),
        "orientation": self._cost_orientation(data),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "invalid_contact": self._cost_invalid_contact(n_contact),
        "stiff_movements": smooth_movement_penalty(data),
    }
    return pos, neg

  def _compute_reference_foot_trajectory(
      self,
      phase: jax.Array,  # (4,) phase per leg
      foot_height: jax.Array,  # scalar foot swing height
  ) -> Dict[str, jax.Array]:
    """
    Compute reference foot trajectory using analytical reference from demo trajectory.
    
    NOTE: Delegates to reference_at_phases_jax() from the demo module
    to ensure both demo and RL training use the EXACT SAME trajectory.
    
    Args:
      phase: (4,) phase angles for each leg (radians)
      foot_height: scalar swing height
    
    Returns:
      dict with:
        - foot_heights_ref: (4,) reference z-heights
        - foot_xy_ref: (4, 2) reference x,y positions (relative stride)
        - ctrl_ref: (8,) reference joint angles
    """
    # Get reference using shared demo trajectory function (JAX-native)
    # This ensures exact consistency between demo visualization and RL rewards
    ref = demo_traj.reference_at_phases_jax(
        phase_array=phase,
        qpos0=self._default_pose,  # Use default pose from the environment
        freq=self._config.demo_gait_params.freq,
        swing_height=foot_height,
        ramp_time=self._config.demo_gait_params.ramp_time,
        hip_amp=self._config.demo_gait_params.hip_amp,
        knee_swing_amp=self._config.demo_gait_params.knee_swing_amp,
        knee_stance_amp=self._config.demo_gait_params.knee_stance_amp,
        swing_threshold=self._config.demo_gait_params.swing_threshold,
    )
    
    foot_heights_ref = ref['foot_heights']  # (4,)
    ctrl_ref = ref['ctrl']  # (8,)
    
    # For horizontal position reference, compute from phases
    # Normalized height (0 = ground, 1 = peak swing)
    z_norm = jp.clip(foot_heights_ref / (foot_height + 1e-8), 0.0, 1.0)
    
    # Forward stride estimate: ~0.08m max during swing
    stride_length = 0.08
    forward_pos = stride_length * jp.sin(phase) * z_norm
    
    # Lateral sway (small): negligible in straight trot
    lateral_pos = jp.zeros_like(phase)
    
    foot_xy_ref = jp.stack([forward_pos, lateral_pos], axis=-1)  # (4, 2)
    
    return {
        "foot_heights_ref": foot_heights_ref,
        "foot_xy_ref": foot_xy_ref,
        "ctrl_ref": ctrl_ref,
    }

  def _reward_feet_height(
      self,
      data: mjx.Data,
      foot_heights_ref: jax.Array,  # (4,)
  ) -> jax.Array:
    """Reward for tracking desired foot z-height (vertical position)."""
    foot_pos = data.site_xpos[self._feet_site_id]  # (4, 3)
    foot_z = foot_pos[..., -1]  # (4,)
    
    height_error = jp.sum(jp.square(foot_z - foot_heights_ref))
    reward = jp.exp(-height_error / 0.08)  # sigma = 0.08
    return reward

  def _reward_feet_xy_position(
      self,
      data: mjx.Data,
      foot_xy_ref: jax.Array,  # (4, 2)
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    return jp.exp(-error / 0.1)

  def _reward_flight_phase(self, n_contact: jax.Array) -> jax.Array:
    # Reward when all 4 feet are in the air (flight phase of trot).
    return (n_contact == 0).astype(jp.float32)

  def _cost_invalid_contact(self, n_contact: jax.Array) -> jax.Array:
    # Penalize when 1, 3, or 4 feet contact the ground.
    # Valid trot states: 0 (flight) or 2 (diagonal stance).
    invalid = (n_contact == 1) | (n_contact == 3) | (n_contact == 4)
    return invalid.astype(jp.float32)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    reward = jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
    return reward

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _cost_hip_splay(self, joint_angles: jax.Array) -> jax.Array:
    current = joint_angles[self._hx_idxs]
    return jp.sum(jp.square(current - self._hx_default_pose))

  def _cost_lin_vel_z(self, global_linvel, gait: jax.Array) -> jax.Array:  # pylint: disable=redefined-outer-name
    del gait
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _reward_flight_phase(self, contact: jax.Array) -> jax.Array:
    """
    Reward for flight phases (all 4 feet in the air).
    
    This encourages the true trotting gait pattern where there are
    moments of complete aerial phase between stance phases.
    
    Args:
      contact: (4,) boolean array indicating which feet have contact
    
    Returns:
      Scalar reward: 1.0 when all feet are in air, 0.0 otherwise
    """
    no_contact = jp.sum(contact) == 0
    return jp.where(no_contact, 1.0, 0.0)

  def _cost_invalid_contact_pattern(self, contact: jax.Array) -> jax.Array:
    """
    Penalize invalid contact patterns for trotting gait.
    
    In trotting, only 2 valid contact states exist:
    - 0 feet: flight phase (aerial)
    - 2 feet: stance phase (diagonal pair on ground)
    
    Invalid states (1, 3, or 4 feet) are penalized to enforce proper gait.
    
    Args:
      contact: (4,) boolean array indicating which feet have contact
    
    Returns:
      Scalar penalty: 0.0 when valid (0 or 2 feet), 1.0 when invalid
    """
    num_contact = jp.sum(contact)
    # Valid: 0 (flight) or 2 (stance with diagonal pair)
    is_valid = jp.logical_or(num_contact == 0, num_contact == 2)
    return jp.where(is_valid, 0.0, 1.0)

  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    # Penalize non-flat base orientation (roll and pitch).
    up = self.get_upvector(data)
    return jp.sum(jp.square(up[:2]))

  def _cost_action_rate(
      self, action: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    # Penalize large changes in action between consecutive steps.
    return jp.sum(jp.square(action - last_act))

  # Changed to only sample straight line
  def sample_command(self, rng: jax.Array) -> jax.Array:
    del rng
    return jp.array([self._config.fixed_vx, 0.0, 0.0])
