"""Stage 1: DRL policy to track demonstration trotting trajectory.

This implements the first DRL stage from the paper "Model-free RL for Robust
Locomotion using Demonstrations from Trajectory Optimization."

The policy learns to track a time-based demonstration trajectory for trotting.
Key design choices from the paper (Section 2.2):
  - Episodes initialize at a randomly chosen point on the demonstration
  - Reward tracks the demonstration state at each timestep (Eq. 2)
  - Policy outputs desired joint positions for a fixed PD controller
  - Early termination on unrecoverable states

Reward structure (paper Eq. 2):
  r_s1 = r_ti + r_tt
  where r_ti has tracking terms for base height, velocity, orientation,
  angular velocity, and joint positions/velocities.
  r_tt is the target tracking regularization (paper Eq. 1).

Observation space (50 dims):
  noisy_linvel(3), noisy_gyro(3), noisy_gravity(3), noisy_joint_angles(8),
  qpos_error_history(24), contact(4), phase_sin(1), phase_cos(1), command(3)

Usage:
  python learning/train_jax_ppo.py --env_name Solo8TrottingDemonstrationStage1
"""

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
from mujoco_playground._src.locomotion.solo8.trotting_demonstration import (
    demonstration_trajectory as demo_traj,
)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=50.0,
      Kd=0.5,
      early_termination=True,
      action_repeat=1,
      action_scale=0.4,
      history_len=3,
      obs_noise=config_dict.create(
          scales=config_dict.create(
              joint_pos=0.05,
              gyro=0.1,
              gravity=0.03,
              linvel=0.1,
          ),
      ),
      # Demonstration trajectory parameters.
      demo=config_dict.create(
          swing_height=0.06,
          stride_amp=0.03,
          body_height=0.27,
          gait_frequency=2.0,
      ),
      # Reward configuration (paper Eq. 2).
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Trajectory imitation terms (positive, exp-based).
              joint_pos=5.0,
              base_height=1.0,
              base_vel=1.0,
              orientation=1.0,
              angvel=0.5,
              # Regularization costs (negative scales).
              target_tracking=-0.1,
              action_rate=-0.01,
          ),
          tracking_sigma=0.25,
      ),
      fixed_vx=0.6,
      impl="jax",
      nconmax=4 * 8192,
      njmax=50,
  )


class TrottingDemonstrationStage1(solo8_base.Solo8Env):
  """Stage 1: Learn to track a trotting demonstration trajectory.

  The policy receives the current robot state plus a timing signal
  (sin/cos of global gait phase) and learns to reproduce the reference
  trot cycle. The reward is purely time-based: at each step, the robot
  state is compared against the demonstration at the current phase.
  """

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
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    # Precompute gait phase increment per control step.
    freq = self._config.demo.gait_frequency
    self._phase_dt = 2.0 * jp.pi * freq * self.dt

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, phase_rng, noise_rng = jax.random.split(rng, 3)

    # Paper Section 2.2: "We initialize each episode at a randomly
    # chosen point on the demonstration trajectory."
    global_phase = jax.random.uniform(
        phase_rng, minval=-jp.pi, maxval=jp.pi
    )

    # Compute reference joint positions at the sampled phase.
    q_joint_ref = demo_traj.compute_reference(
        global_phase,
        swing_height=self._config.demo.swing_height,
        stride_amp=self._config.demo.stride_amp,
        body_height=self._config.demo.body_height,
    )

    # Initialize robot at the reference configuration.
    qpos = self._init_q.at[7:].set(q_joint_ref)
    qvel = jp.zeros(self.mjx_model.nv)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "command": jp.array([self._config.fixed_vx, 0.0, 0.0]),
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "motor_targets": q_joint_ref,
        "qpos_error_history": jp.zeros(self._config.history_len * 8),
        "global_phase": global_phase,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    # Diagnostic metrics (same keys as Stage 2 for cross-comparison).
    metrics["diag/forward_vel"] = jp.zeros(())
    metrics["diag/base_height"] = jp.zeros(())
    metrics["diag/roll"] = jp.zeros(())
    metrics["diag/pitch"] = jp.zeros(())
    metrics["diag/action_rate"] = jp.zeros(())
    metrics["diag/n_contact"] = jp.zeros(())
    metrics["diag/joint_pos_error"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    obs = self._get_obs(data, info, noise_rng, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, noise_rng = jax.random.split(state.info["rng"], 2)

    # Apply action: policy outputs offsets from default pose.
    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # Advance global phase.
    new_phase = state.info["global_phase"] + self._phase_dt
    new_phase = jp.fmod(new_phase + jp.pi, 2 * jp.pi) - jp.pi

    # Compute reference at the new phase.
    q_joint_ref = demo_traj.compute_reference(
        new_phase,
        swing_height=self._config.demo.swing_height,
        stride_amp=self._config.demo.stride_amp,
        body_height=self._config.demo.body_height,
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    obs = self._get_obs(data, state.info, noise_rng, contact)
    done = self._get_termination(data)

    # Compute rewards (paper Eq. 2: r_s1 = r_ti + r_tt).
    rewards = self._get_reward(
        data, action, state.info, q_joint_ref, motor_targets
    )
    reward = sum(
        v * self._config.reward_config.scales[k]
        for k, v in rewards.items()
    ) * self.dt

    # Update info.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    state.info["global_phase"] = new_phase
    state.info["rng"] = rng

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v * self._config.reward_config.scales[k]

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
    state.metrics["diag/n_contact"] = jp.sum(contact_f)
    state.metrics["diag/joint_pos_error"] = jp.sqrt(
        jp.sum(jp.square(q_joint_ref - data.qpos[7:]))
    )

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Terminate if robot flips (upvector z < 0 means upside down).
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
    """Build 50-dim observation vector.

    Components:
      noisy_linvel (3): Local linear velocity with noise.
      noisy_gyro (3): Angular velocity with noise.
      noisy_gravity (3): Gravity direction in body frame with noise.
      noisy_joint_angles (8): Joint positions with noise.
      qpos_error_history (24): Rolling buffer of motor tracking errors.
      contact (4): Binary foot contact indicators.
      phase_sin (1): sin(global_phase) — timing signal.
      phase_cos (1): cos(global_phase) — timing signal.
      command (3): Target velocity [vx, vy, vyaw].
    """
    # Noisy linear velocity.
    linvel = self.get_local_linvel(data)
    rng, noise_rng = jax.random.split(rng)
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.obs_noise.scales.linvel
    )

    # Noisy gyroscope.
    gyro = self.get_gyro(data)
    rng, noise_rng = jax.random.split(rng)
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.obs_noise.scales.gyro
    )

    # Noisy gravity direction.
    gravity = self.get_gravity(data)
    rng, noise_rng = jax.random.split(rng)
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.obs_noise.scales.gravity
    )

    # Noisy joint angles.
    joint_angles = data.qpos[7:]
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.scales.joint_pos
    )

    # Rolling qpos error history (tracks motor tracking quality).
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 8)
        .at[:8]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    # Global phase timing signal (sin/cos encoding avoids wrap discontinuity).
    phase_sin = jp.sin(info["global_phase"])
    phase_cos = jp.cos(info["global_phase"])

    return jp.hstack([
        noisy_linvel,                         # (3,)
        noisy_gyro,                           # (3,)
        noisy_gravity,                        # (3,)
        noisy_joint_angles,                   # (8,)
        qpos_error_history,                   # (24,)
        contact,                              # (4,)
        jp.array([phase_sin]),                # (1,)
        jp.array([phase_cos]),                # (1,)
        info["command"],                      # (3,)
    ])
    # Total: 3+3+3+8+24+4+1+1+3 = 50

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      q_joint_ref: jax.Array,
      motor_targets: jax.Array,
  ) -> Dict[str, jax.Array]:
    """Compute Stage 1 rewards (paper Eq. 2).

    Tracking terms (r_ti): each uses exp(-error / sigma) form.
    The exponential naturally bounds each term to [0, 1].

    Regularization (r_tt): target tracking from paper Eq. 1.
    """
    sigma = self._config.reward_config.tracking_sigma

    rewards = {}

    # --- Trajectory imitation terms (r_ti, paper Eq. 2) ---

    # 1. Joint position tracking (MAIN term, highest weight).
    # Paper: k_ti_9 * exp(-k_ti_10 * ||q_joint_demo - q_joint||)
    q_joint = data.qpos[7:]
    rewards["joint_pos"] = jp.exp(
        -jp.sum(jp.square(q_joint_ref - q_joint)) / sigma
    )

    # 2. Base height tracking.
    # Paper: k_ti_1 * exp(-k_ti_2 * ||x_base_demo - x_base||)
    # For trotting, we track z-height only (x/y grow with forward motion).
    z_base = data.qpos[2]
    z_ref = self._config.demo.body_height
    rewards["base_height"] = jp.exp(-jp.square(z_ref - z_base) / sigma)

    # 3. Base velocity tracking.
    # Paper: k_ti_3 * exp(-k_ti_4 * ||x_dot_base_demo - x_dot_base||)
    local_vel = self.get_local_linvel(data)
    v_ref = info["command"]
    rewards["base_vel"] = jp.exp(
        -jp.sum(jp.square(v_ref - local_vel)) / sigma
    )

    # 4. Orientation tracking.
    # Paper: k_ti_5 * exp(-k_ti_6 * ||q_base_demo ⊖ q_base||)
    # We use the upvector as a proxy for quaternion error.
    up = self.get_upvector(data)
    up_ref = jp.array([0.0, 0.0, 1.0])
    rewards["orientation"] = jp.exp(
        -jp.sum(jp.square(up_ref - up)) / sigma
    )

    # 5. Angular velocity tracking.
    # Paper: k_ti_7 * exp(-k_ti_8 * ||omega_base_demo - omega_base||)
    angvel = self.get_global_angvel(data)
    rewards["angvel"] = jp.exp(-jp.sum(jp.square(angvel)) / sigma)

    # --- Regularization terms ---

    # 6. Target tracking regularization (paper Eq. 1).
    # r_tt = -k_tt * ||q_joint_des(t) - q_joint(t+1)||^2
    # Penalizes gap between commanded and achieved joint positions.
    rewards["target_tracking"] = jp.sum(
        jp.square(motor_targets - q_joint)
    )

    # 7. Action rate cost: penalize jerky control.
    rewards["action_rate"] = jp.sum(
        jp.square(action - info["last_act"])
    )

    return rewards

  def sample_command(self, rng: jax.Array) -> jax.Array:
    del rng
    return jp.array([self._config.fixed_vx, 0.0, 0.0])
