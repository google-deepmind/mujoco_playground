"""Analytical trotting demonstration trajectory for Solo8.

Generates a reference trotting gait cycle using inverse kinematics from
desired foot trajectories. This provides the demonstration that Stage 1
DRL training will learn to track.

The trajectory defines, for any global phase angle phi in [-pi, pi):
  - 8 joint reference positions (HFE + KFE for each of 4 legs)
  - Base height reference
  - Base velocity reference

Key design choices:
  - Foot z-height follows gait.get_rz() cubic Bezier profile
  - Foot x-position follows sinusoidal fore-aft swing
  - Joint angles computed via 2-link planar IK (hip + knee)
  - Trot phase offsets: [FL=0, FR=pi, HL=pi, HR=0] (diagonal pairs)
"""

import jax.numpy as jp
import numpy as np

from mujoco_playground._src import gait


# Solo8 leg link lengths (upper leg and lower leg).
_L1 = 0.16  # upper leg (hip to knee)
_L2 = 0.16  # lower leg (knee to foot)

# Trot phase offsets: diagonal pairs move together.
# Leg order: FL, FR, HL, HR (matching actuator ordering).
_TROT_OFFSETS = jp.array([0.0, jp.pi, jp.pi, 0.0])

# Home pose joint angles from the XML keyframe.
# [FL_HFE, FL_KFE, FR_HFE, FR_KFE, HL_HFE, HL_KFE, HR_HFE, HR_KFE]
_HOME_JOINTS = np.array([1.0, -1.3, 1.0, -1.3, 0.8, -1.5, 0.8, -1.5])

# HFE and KFE index arrays for assembling the 8-DOF joint vector.
_HFE_IDXS = jp.array([0, 2, 4, 6])
_KFE_IDXS = jp.array([1, 3, 5, 7])


def fk_2link(theta1, theta2, L1=_L1, L2=_L2):
  """Forward kinematics for a 2-link planar leg.

  Computes foot position relative to hip joint in the sagittal plane.
  Convention: positive x = forward, negative z = downward.
  theta1 = HFE angle, theta2 = KFE angle.

  Args:
    theta1: Hip flexion/extension angle (HFE). Shape: scalar or (N,).
    theta2: Knee flexion/extension angle (KFE). Shape: scalar or (N,).
    L1: Upper leg length.
    L2: Lower leg length.

  Returns:
    px: Foot x-position relative to hip (negative = behind hip).
    pz: Foot z-position relative to hip (negative = below hip).
  """
  px = -L1 * jp.sin(theta1) - L2 * jp.sin(theta1 + theta2)
  pz = -L1 * jp.cos(theta1) - L2 * jp.cos(theta1 + theta2)
  return px, pz


def ik_2link(px, pz, L1=_L1, L2=_L2):
  """Inverse kinematics for a 2-link planar leg.

  Given desired foot position relative to hip, compute joint angles.
  Uses elbow-back configuration (negative KFE) matching Solo8 home pose.

  Args:
    px: Desired foot x relative to hip. Shape: scalar or (N,).
    pz: Desired foot z relative to hip. Shape: scalar or (N,).
    L1: Upper leg length.
    L2: Lower leg length.

  Returns:
    theta1: HFE angle. Same shape as input.
    theta2: KFE angle (negative = elbow-back). Same shape as input.
  """
  d_sq = px ** 2 + pz ** 2
  cos_theta2 = (d_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
  cos_theta2 = jp.clip(cos_theta2, -1.0, 1.0)
  theta2 = -jp.arccos(cos_theta2)  # negative for elbow-back config

  gamma = jp.arctan2(-px, -pz)
  alpha = jp.arctan2(L2 * jp.sin(theta2), L1 + L2 * jp.cos(theta2))
  theta1 = gamma - alpha

  return theta1, theta2


def compute_neutral_foot_positions():
  """Compute neutral (home) foot positions for front and hind legs.

  Returns:
    front_x, front_z: Foot position for front legs relative to hip.
    hind_x, hind_z: Foot position for hind legs relative to hip.
  """
  front_x, front_z = fk_2link(
      jp.float32(_HOME_JOINTS[0]),
      jp.float32(_HOME_JOINTS[1]),
  )
  hind_x, hind_z = fk_2link(
      jp.float32(_HOME_JOINTS[4]),
      jp.float32(_HOME_JOINTS[5]),
  )
  return front_x, front_z, hind_x, hind_z


def compute_reference(
    global_phase,
    swing_height=0.06,
    stride_amp=0.03,
    body_height=0.27,
):
  """Compute 8-DOF reference joint positions at a given global phase.

  For each leg:
    1. Compute the leg's local phase (global_phase + trot offset)
    2. Compute desired foot z from gait.get_rz (swing profile)
    3. Compute desired foot x from sinusoidal fore-aft trajectory
    4. Use IK to convert foot position to (HFE, KFE) angles

  Args:
    global_phase: Global phase angle in [-pi, pi). Scalar or batched.
    swing_height: Maximum foot height during swing (meters).
    stride_amp: Fore-aft foot displacement amplitude (meters).
    body_height: Nominal body height above ground (meters).

  Returns:
    q_ref: (8,) JAX array of reference joint angles.
           Order: [FL_HFE, FL_KFE, FR_HFE, FR_KFE,
                   HL_HFE, HL_KFE, HR_HFE, HR_KFE]
  """
  # Compute neutral foot positions from home pose.
  front_x_neutral, _, hind_x_neutral, _ = compute_neutral_foot_positions()

  # Per-leg phases with trot offsets.
  leg_phases = global_phase + _TROT_OFFSETS  # (4,) for FL, FR, HL, HR

  # Desired foot z-height in world frame, from Bezier swing profile.
  foot_z_world = gait.get_rz(leg_phases, swing_height=swing_height)  # (4,)

  # Desired foot x-displacement (sinusoidal fore-aft swing).
  foot_x_stride = stride_amp * jp.sin(leg_phases)  # (4,)

  # Neutral x for each leg: front legs share one value, hind legs another.
  x_neutral = jp.array([
      front_x_neutral, front_x_neutral,
      hind_x_neutral, hind_x_neutral,
  ])

  # Foot position relative to hip.
  foot_x_rel = x_neutral + foot_x_stride  # (4,)
  foot_z_rel = foot_z_world - body_height  # (4,) negative = below hip

  # Inverse kinematics for all 4 legs.
  hfe_angles, kfe_angles = ik_2link(foot_x_rel, foot_z_rel)  # (4,), (4,)

  # Assemble into 8-DOF joint vector:
  # [FL_HFE, FL_KFE, FR_HFE, FR_KFE, HL_HFE, HL_KFE, HR_HFE, HR_KFE]
  q_ref = jp.zeros(8)
  q_ref = q_ref.at[_HFE_IDXS].set(hfe_angles)
  q_ref = q_ref.at[_KFE_IDXS].set(kfe_angles)

  return q_ref


def compute_reference_velocity(
    global_phase,
    gait_frequency=2.0,
    swing_height=0.06,
    stride_amp=0.03,
    body_height=0.27,
):
  """Compute reference joint velocities via finite differences.

  Uses central difference: dq/dt ≈ (q(φ+ε) - q(φ-ε)) / (2·dt_epsilon)
  where dt_epsilon corresponds to the phase epsilon at the given frequency.

  Args:
    global_phase: Global phase angle. Scalar.
    gait_frequency: Gait frequency in Hz.
    swing_height: Maximum foot height during swing.
    stride_amp: Fore-aft foot displacement amplitude.
    body_height: Nominal body height.

  Returns:
    qd_ref: (8,) reference joint velocities.
  """
  omega = 2.0 * jp.pi * gait_frequency
  eps = 1e-3  # small phase increment
  dt_eps = eps / omega  # corresponding time increment

  q_plus = compute_reference(
      global_phase + eps, swing_height, stride_amp, body_height
  )
  q_minus = compute_reference(
      global_phase - eps, swing_height, stride_amp, body_height
  )

  qd_ref = (q_plus - q_minus) / (2.0 * dt_eps)
  return qd_ref
