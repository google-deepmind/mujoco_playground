"""Stage 2: Robust time-independent trotting policy.

This implements the second DRL stage from the paper "Model-free RL for Robust
Locomotion using Demonstrations from Trajectory Optimization."

Key changes from Stage 1 (paper Section 2.3):

1. INITIALIZATION: Replace trajectory-based init with a wider range of states.
   Randomize base height, tilt angles, and joint positions to cover the range
   of states the policy might encounter during deployment.

2. TIME-INDEPENDENT TASK REWARD: Replace demonstration tracking with direct
   task performance reward. The policy is free to change behavior as needed to
   perform the task, without being penalized for deviating from the demo.

3. REGULARIZATION REWARDS: Add terms for torque smoothness, smooth contact
   transitions, and target tracking to ensure deployable behavior.

The observation space is kept identical to Stage 1 (50 dims) so that
Stage 1 weights can be loaded directly via --load_checkpoint_path.

Usage:
  python learning/train_jax_ppo.py \\
    --env_name Solo8TrottingDemonstrationStage2 \\
    --load_checkpoint_path logs/Solo8TrottingDemonstrationStage1-<ts>/checkpoints
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

# Trot phase offsets: diagonal pairs move together.
# Leg order in the gait module: FR, FL, RR(HR), RL(HL).
# Our actuator order: FL, FR, HL, HR → phase offsets match.
_TROT_PHASE = np.array([0, np.pi, np.pi, 0])


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
      # Gait parameters for foot coordination (not demo tracking).
      gait_frequency=[2.0, 2.0],
      foot_height=[0.06, 0.06],
      # Wider initialization (paper Section 2.3).
      init_randomization=config_dict.create(
          base_height_range=[0.22, 0.32],
          base_tilt_range=[-0.3, 0.3],
          joint_noise_scale=0.1,
      ),
      # Target phase durations for trotting gait cycle.
      # At 2 Hz: cycle = 0.5s. With ~50% duty cycle per pair:
      #   stance per pair ≈ 0.20s, flight ≈ 0.04s (brief suspension).
      # These targets are evaluated at phase transitions (touchdown/liftoff).
      target_air_time=0.04,     # Target flight duration (seconds).
      target_stance_time=0.20,  # Target per-pair stance duration (seconds).
      # Task reward (time-independent, paper Section 2.3).
      #
      # Design principles for trotting:
      #   1. TASK: velocity tracking is the primary objective.
      #   2. TROT PATTERN: reward correct diagonal contact pairs
      #      (FL+HR or FR+HL), NOT just "2 feet on ground."
      #   3. PHASE TIMING: reward air/stance durations matching targets.
      #      Evaluated at phase transitions (sparse but informative).
      #   4. SMOOTHNESS: multiple terms targeting different aspects
      #      of smooth motion (action rate, action acceleration,
      #      foot slip, base stability).
      #   5. TIME-INDEPENDENCE: no term references a trajectory
      #      timeline. The gait clock coordinates foot phases but
      #      the robot is free to adjust timing/amplitude.
      reward_config=config_dict.create(
          scales=config_dict.create(
              # --- Task performance (positive rewards) ---
              tracking_lin_vel=1.5,     # Primary task: forward velocity.
              tracking_ang_vel=0.5,     # Yaw rate tracking.
              diagonal_contact=2.0,     # Correct diagonal pair stance.
              air_time=1.0,            # Flight duration matches target.
              stance_time=1.0,         # Stance duration matches target.
              feet_phase=1.0,           # Gait clock foot coordination.
              # --- Regularization costs (negative scales) ---
              orientation=-5.0,         # Stay upright (roll/pitch).
              base_height=-2.0,         # Consistent body height.
              ang_vel_xy=-0.5,          # No undesired base rotation.
              lin_vel_z=-2.0,           # No bouncing.
              action_rate=-0.05,        # Smooth actions (1st deriv).
              action_smoothness=-0.02,  # Smooth torque (2nd deriv, Eq 7).
              target_tracking=-0.1,     # PD tracking quality (Eq 1).
              foot_slip=-0.5,           # No sliding during stance.
          ),
          tracking_sigma=0.25,
      ),
      fixed_vx=0.6,
      impl="jax",
      nconmax=4 * 8192,
      njmax=50,
  )


class TrottingDemonstrationStage2(solo8_base.Solo8Env):
  """Stage 2: Robust time-independent trotting policy.

  This env loads Stage 1 policy weights (via --load_checkpoint_path)
  and continues training with:
    - Wider state initialization (randomized height, tilt, joints)
    - Time-independent task reward (velocity tracking + foot coordination)
    - Regularization for torque smoothness and smooth contact transitions
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
    rng, height_rng, tilt_rng, joint_rng, noise_rng = (
        jax.random.split(rng, 5)
    )
    rng, gait_freq_rng, foot_height_rng = jax.random.split(rng, 3)

    # Paper Section 2.3 - Wider initialization.
    # Randomize base height.
    init_cfg = self._config.init_randomization
    base_height = jax.random.uniform(
        height_rng,
        minval=init_cfg.base_height_range[0],
        maxval=init_cfg.base_height_range[1],
    )

    # Randomize base tilt (pitch and roll via quaternion perturbation).
    tilt_angles = jax.random.uniform(
        tilt_rng,
        shape=(2,),
        minval=init_cfg.base_tilt_range[0],
        maxval=init_cfg.base_tilt_range[1],
    )
    # Small-angle quaternion approximation: q ≈ [1, roll/2, pitch/2, 0].
    roll, pitch = tilt_angles[0], tilt_angles[1]
    half_roll, half_pitch = roll / 2, pitch / 2
    quat_norm = jp.sqrt(1 + half_roll**2 + half_pitch**2)
    init_quat = jp.array([1.0, half_roll, half_pitch, 0.0]) / quat_norm

    # Randomize joint positions around home pose.
    joint_noise = jax.random.uniform(
        joint_rng,
        shape=(8,),
        minval=-init_cfg.joint_noise_scale,
        maxval=init_cfg.joint_noise_scale,
    )
    init_joints = jp.array(self._default_pose) + joint_noise

    # Build initial qpos.
    qpos = self._init_q.at[2].set(base_height)
    qpos = qpos.at[3:7].set(init_quat)
    qpos = qpos.at[7:].set(init_joints)
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

    # Sample gait parameters (frequency and foot height from ranges).
    gait_freq = jax.random.uniform(
        gait_freq_rng,
        minval=self._config.gait_frequency[0],
        maxval=self._config.gait_frequency[1],
    )
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    foot_height = jax.random.uniform(
        foot_height_rng,
        minval=self._config.foot_height[0],
        maxval=self._config.foot_height[1],
    )

    # Initialize gait phase randomly (time-independent: no trajectory link).
    phase = jp.array(_TROT_PHASE, dtype=jp.float32)

    info = {
        "command": jp.array([self._config.fixed_vx, 0.0, 0.0]),
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "motor_targets": jp.array(self._default_pose),
        "qpos_error_history": jp.zeros(self._config.history_len * 8),
        "last_contact": jp.zeros(4, dtype=bool),
        "phase": phase,
        "phase_dt": phase_dt,
        "gait_freq": gait_freq,
        "foot_height": foot_height,
        # Duration counters for trot phase timing.
        # air_time: consecutive time (s) with all feet off ground.
        # pair1/2_stance_time: consecutive time (s) each diagonal pair
        #   has been in stance. Resets when the pair lifts off.
        "air_time": jp.float32(0),
        "pair1_stance_time": jp.float32(0),  # FL+HR pair
        "pair2_stance_time": jp.float32(0),  # FR+HL pair
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    # Diagnostic metrics for comparing approaches.
    metrics["diag/forward_vel"] = jp.zeros(())
    metrics["diag/base_height"] = jp.zeros(())
    metrics["diag/roll"] = jp.zeros(())
    metrics["diag/pitch"] = jp.zeros(())
    metrics["diag/action_rate"] = jp.zeros(())
    metrics["diag/n_contact"] = jp.zeros(())
    metrics["diag/is_diagonal"] = jp.zeros(())
    metrics["diag/is_flight"] = jp.zeros(())
    metrics["diag/air_time"] = jp.zeros(())
    metrics["diag/pair1_stance_time"] = jp.zeros(())
    metrics["diag/pair2_stance_time"] = jp.zeros(())

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

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])
    contact_f = contact.astype(jp.float32)

    # --- Duration counter logic ---
    # Detect phase transitions BEFORE updating counters.
    # Flight: all feet off ground.
    is_flight = jp.sum(contact_f) == 0
    was_flight = state.info["air_time"] > 0

    # Diagonal pairs: FL(0)+HR(3), FR(1)+HL(2).
    pair1_active = contact_f[0] * contact_f[3]  # FL+HR both on ground
    pair2_active = contact_f[1] * contact_f[2]  # FR+HL both on ground
    was_pair1 = state.info["pair1_stance_time"] > 0
    was_pair2 = state.info["pair2_stance_time"] > 0

    # Transition events (used by reward to evaluate completed durations).
    # touchdown: was flying, now have contact → evaluate air_time.
    touchdown = was_flight & (~is_flight)
    # pair liftoff: pair was in stance, now not → evaluate stance_time.
    pair1_liftoff = was_pair1 & (pair1_active == 0)
    pair2_liftoff = was_pair2 & (pair2_active == 0)

    # Build transition info for reward computation.
    transition_info = {
        "touchdown": touchdown,
        "completed_air_time": state.info["air_time"],
        "pair1_liftoff": pair1_liftoff,
        "pair2_liftoff": pair2_liftoff,
        "completed_pair1_stance": state.info["pair1_stance_time"],
        "completed_pair2_stance": state.info["pair2_stance_time"],
    }

    obs = self._get_obs(data, state.info, noise_rng, contact)
    done = self._get_termination(data)

    # Compute time-independent task rewards + regularization.
    pos, neg = self._get_reward(
        data, action, state.info, done, contact, transition_info
    )
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg
    r_pos = sum(pos.values())
    r_neg = jp.exp(0.2 * sum(neg.values()))
    reward = r_pos * r_neg * self.dt

    # Update state info.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1

    # Advance per-leg gait phases.
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

    state.info["rng"] = rng
    state.info["last_contact"] = contact

    # Update duration counters AFTER reward computation.
    # air_time: increment during flight, reset on contact.
    state.info["air_time"] = jp.where(
        is_flight, state.info["air_time"] + self.dt, jp.float32(0)
    )
    # pair stance times: increment while pair is active, reset otherwise.
    state.info["pair1_stance_time"] = jp.where(
        pair1_active > 0,
        state.info["pair1_stance_time"] + self.dt,
        jp.float32(0),
    )
    state.info["pair2_stance_time"] = jp.where(
        pair2_active > 0,
        state.info["pair2_stance_time"] + self.dt,
        jp.float32(0),
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    # Diagnostic metrics for cross-approach comparison.
    local_vel = self.get_local_linvel(data)
    up = self.get_upvector(data)
    state.metrics["diag/forward_vel"] = local_vel[0]
    state.metrics["diag/base_height"] = data.qpos[2]
    state.metrics["diag/roll"] = up[0]
    state.metrics["diag/pitch"] = up[1]
    state.metrics["diag/action_rate"] = jp.sqrt(
        jp.sum(jp.square(action - state.info["last_act"]))
    )
    n_contact = jp.sum(contact_f)
    state.metrics["diag/n_contact"] = n_contact
    state.metrics["diag/is_diagonal"] = (
        (pair1_active > 0) | (pair2_active > 0)
    ).astype(jp.float32)
    state.metrics["diag/is_flight"] = is_flight.astype(jp.float32)
    state.metrics["diag/air_time"] = state.info["air_time"]
    state.metrics["diag/pair1_stance_time"] = state.info["pair1_stance_time"]
    state.metrics["diag/pair2_stance_time"] = state.info["pair2_stance_time"]

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
    """Build 50-dim observation vector (same shape as Stage 1).

    The only semantic change from Stage 1: the sin/cos values now represent
    the gait coordination clock (first leg's phase) rather than the demo
    tracking timeline. This makes the observation time-independent while
    keeping weight compatibility with Stage 1.
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

    # Rolling qpos error history.
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 8)
        .at[:8]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    # Gait phase as timing signal (first leg's phase for coordination).
    # This is time-independent: it cycles naturally, not tracking a demo.
    phase_val = info["phase"][0]  # FL leg phase
    phase_sin = jp.sin(phase_val)
    phase_cos = jp.cos(phase_val)

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
      done: jax.Array,
      contact: jax.Array,
      transition_info: dict[str, Any],
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Time-independent task reward + regularization (paper Section 2.3).

    Positive rewards encourage task performance:
      - tracking_lin_vel: Track forward velocity command.
      - tracking_ang_vel: Track angular velocity command.
      - diagonal_contact: Reward correct diagonal pair stance (FL+HR or FR+HL).
      - air_time: At touchdown, reward if flight duration matched target.
      - stance_time: At pair liftoff, reward if stance duration matched target.
      - feet_phase: Foot height follows gait.get_rz() profile.

    Negative costs regularize smooth motion:
      - orientation: Penalize roll/pitch deviation.
      - base_height: Penalize deviation from nominal body height.
      - ang_vel_xy: Penalize undesired base rotation.
      - lin_vel_z: Penalize vertical oscillation / bouncing.
      - action_rate: Penalize jerky control (1st derivative).
      - action_smoothness: Penalize action jerk (2nd derivative).
      - target_tracking: Paper Eq. 1 — PD tracking quality.
      - foot_slip: Penalize foot sliding during ground contact.
    """
    del done
    sigma = self._config.reward_config.tracking_sigma
    # Contact per foot: [FL, FR, HL, HR].
    contact_f = contact.astype(jp.float32)

    # --- Positive task rewards ---
    pos = {}

    # Forward velocity tracking.
    local_vel = self.get_local_linvel(data)
    lin_vel_error = jp.sum(jp.square(info["command"][:2] - local_vel[:2]))
    pos["tracking_lin_vel"] = jp.exp(-lin_vel_error / sigma)

    # Angular velocity tracking (yaw).
    ang_vel = self.get_gyro(data)
    ang_vel_error = jp.square(info["command"][2] - ang_vel[2])
    pos["tracking_ang_vel"] = jp.exp(-ang_vel_error / sigma)

    # Diagonal contact reward: trotting means diagonal pairs stance together.
    # Pair 1: FL (idx 0) + HR (idx 3), Pair 2: FR (idx 1) + HL (idx 2).
    pair1_score = contact_f[0] * contact_f[3]  # FL+HR both on ground
    pair2_score = contact_f[1] * contact_f[2]  # FR+HL both on ground
    # Penalize cross-pair contact (e.g., FL+FR both on ground).
    cross_penalty = contact_f[0] * contact_f[1] + contact_f[2] * contact_f[3]
    pos["diagonal_contact"] = (pair1_score + pair2_score) * jp.exp(
        -cross_penalty
    )

    # Air time reward: evaluated at touchdown (flight → contact transition).
    # Rewards flight durations close to target_air_time.
    # Sparse: only non-zero at the step when the robot lands.
    target_air = self._config.target_air_time
    air_time_error = jp.square(
        transition_info["completed_air_time"] - target_air
    )
    pos["air_time"] = jp.where(
        transition_info["touchdown"],
        jp.exp(-air_time_error / (sigma * 0.01)),  # tight sigma for timing
        jp.float32(0),
    )

    # Stance time reward: evaluated at pair liftoff (stance → swing).
    # Rewards stance durations close to target_stance_time per diagonal pair.
    target_stance = self._config.target_stance_time
    p1_error = jp.square(
        transition_info["completed_pair1_stance"] - target_stance
    )
    p2_error = jp.square(
        transition_info["completed_pair2_stance"] - target_stance
    )
    p1_rew = jp.where(
        transition_info["pair1_liftoff"],
        jp.exp(-p1_error / (sigma * 0.01)),
        jp.float32(0),
    )
    p2_rew = jp.where(
        transition_info["pair2_liftoff"],
        jp.exp(-p2_error / (sigma * 0.01)),
        jp.float32(0),
    )
    pos["stance_time"] = p1_rew + p2_rew

    # Foot height tracking using gait.get_rz (periodic foot coordination).
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(info["phase"], swing_height=info["foot_height"])
    feet_error = jp.sum(jp.square(foot_z - rz))
    pos["feet_phase"] = jp.exp(-feet_error / 0.1)

    # --- Negative regularization costs ---
    neg = {}

    # Orientation cost: penalize roll/pitch.
    up = self.get_upvector(data)
    neg["orientation"] = jp.sum(jp.square(up[:2]))

    # Base height: penalize deviation from nominal 0.27 m.
    z_base = data.qpos[2]
    neg["base_height"] = jp.square(z_base - 0.27)

    # Angular velocity xy: penalize undesired base rotation.
    global_angvel = self.get_global_angvel(data)
    neg["ang_vel_xy"] = jp.sum(jp.square(global_angvel[:2]))

    # Vertical velocity: penalize bouncing.
    global_linvel = self.get_global_linvel(data)
    neg["lin_vel_z"] = jp.square(global_linvel[2])

    # Action rate: penalize jerky control (1st derivative).
    neg["action_rate"] = jp.sum(jp.square(action - info["last_act"]))

    # Action smoothness: penalize action jerk (2nd derivative).
    # a'' ≈ a(t) - 2*a(t-1) + a(t-2)
    neg["action_smoothness"] = jp.sum(
        jp.square(action - 2 * info["last_act"] + info["last_last_act"])
    )

    # Target tracking regularization (paper Eq. 1, used in both stages).
    q_joint = data.qpos[7:]
    neg["target_tracking"] = jp.sum(
        jp.square(info["motor_targets"] - q_joint)
    )

    # Foot slip: penalize foot velocity when foot is in contact.
    # Uses foot linear velocity sensors; only penalize xy components.
    foot_vel = jp.array([
        data.sensordata[self._foot_linvel_sensor_adr[i]]
        for i in range(4)
    ])  # (4, 3)
    foot_xy_vel_sq = jp.sum(jp.square(foot_vel[:, :2]), axis=-1)  # (4,)
    neg["foot_slip"] = jp.sum(foot_xy_vel_sq * contact_f)

    return pos, neg

  def sample_command(self, rng: jax.Array) -> jax.Array:
    del rng
    return jp.array([self._config.fixed_vx, 0.0, 0.0])
