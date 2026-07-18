import numpy as np
from pathlib import Path
from mujoco import MjModel, MjData, viewer
from mujoco_playground._src import gait, mjx_env
import mujoco
import jax
import jax.numpy as jp

# Path to solo8 xmls
SOLO8_XML_DIR = Path(__file__).resolve().parent / "xmls"
WORLD_XML = SOLO8_XML_DIR / "world_unconstrained.xml"

def get_assets():
  """Load Solo8 assets (xmls and meshes)."""
  assets = {}
  mjx_env.update_assets(assets, SOLO8_XML_DIR, "*.xml")
  mjx_env.update_assets(assets, SOLO8_XML_DIR / "meshes" / "stl" / "without_foot", "*.stl")
  mjx_env.update_assets(assets, SOLO8_XML_DIR / "meshes" / "stl" / "with_foot", "*.stl")
  return assets


def reference_at_time(
    t: float,
    qpos0: np.ndarray,
    freq: float = 0.6,
    swing_height: float = 0.08,
    ramp_time: float = 1.0,
    hip_amp: float = 0.12,
    knee_swing_amp: float = 0.35,
    knee_stance_amp: float = 0.05,
    swing_threshold: float = 0.25,
):
  """
  Compute reference joint targets and foot heights for trotting gait at time t.
  
  Args:
    t: Simulation time (seconds).
    qpos0: Initial joint configuration (8 values for hip/knee pairs).
    freq: Gait frequency in Hz.
    swing_height: Maximum foot lift height in meters.
    ramp_time: Time to ramp up from 0 (seconds).
    hip_amp: Hip swing amplitude in radians.
    knee_swing_amp: Knee flex amplitude during swing.
    knee_stance_amp: Knee extension amplitude during stance.
    swing_threshold: Normalized height threshold for swing detection (0-1).
  
  Returns:
    Dictionary with:
      - 'ctrl': (8,) array of desired joint angles.
      - 'foot_heights': (4,) array of desired foot z-heights (rz values).
      - 'swing_flags': (4,) array of swing/stance flags (1.0 swing, 0.0 stance).
      - 'phases': (4,) array of phase per leg.
      - 'ramp': scalar ramp-up factor.
  """
  trot_phases = gait.GAIT_PHASES[0]  # [0, π, π, 0]
  
  # Leg indexing: FR(0,1), FL(2,3), HR(4,5), HL(6,7)
  leg_joint_indices = [
      [2, 3],  # FR
      [0, 1],  # FL
      [6, 7],  # HR
      [4, 5],  # HL
  ]
  
  ctrl = np.zeros(8, dtype=np.float32)
  foot_heights = np.zeros(4, dtype=np.float32)
  swing_flags = np.zeros(4, dtype=np.float32)
  leg_phases = np.zeros(4, dtype=np.float32)
  
  # Ramp-up phase
  ramp = np.clip((t - 0.05) / ramp_time, 0.0, 1.0)
  if t < 0.05:
    ramp = 0.0
  
  global_phase = 2.0 * np.pi * freq * t
  
  for leg in range(4):
    phase_val = global_phase + float(trot_phases[leg])
    leg_phases[leg] = phase_val
    
    # Foot height via gait function
    rz = float(np.asarray(gait.get_rz(phase_val, swing_height=swing_height)))
    foot_heights[leg] = rz
    
    # Normalized height (0-1)
    z = float(np.clip(rz / (swing_height + 1e-8), 0.0, 1.0))
    
    # Swing vs stance detection
    is_swing = 1.0 if z > swing_threshold else 0.0
    is_stance = 1.0 - is_swing
    swing_flags[leg] = is_swing
    
    s = np.sin(phase_val)
    
    hfe_i, kfe_i = leg_joint_indices[leg]
    hip0 = float(qpos0[hfe_i])
    knee0 = float(qpos0[kfe_i])
    
    # Hip control: swing motion only
    forward_sign = -1.0
    ctrl[hfe_i] = hip0 + ramp * forward_sign * hip_amp * s * is_swing
    
    # Knee control: flex in swing, extend in stance
    ctrl[kfe_i] = (
        knee0
        - ramp * knee_swing_amp * z * is_swing
        + ramp * knee_stance_amp * is_stance
    )
  
  return {
      'ctrl': ctrl,
      'foot_heights': foot_heights,
      'swing_flags': swing_flags,
      'phases': leg_phases,
      'ramp': ramp,
  }


def reference_at_phases_jax(
    phase_array,  # (4,) JAX array of leg phases
    qpos0,  # (8,) JAX array of initial joint config
    freq: float = 0.6,
    swing_height: float = 0.08,
    ramp_time: float = 1.0,
    hip_amp: float = 0.12,
    knee_swing_amp: float = 0.35,
    knee_stance_amp: float = 0.05,
    swing_threshold: float = 0.25,
):
  """
  JAX version that computes reference directly from per-leg phases.
  
  This is the RL-friendly version: takes phase_array instead of reconstructing
  from time. Ensures exact consistency between demo and RL training.
  
  Args:
    phase_array: (4,) JAX array with phase per leg (radians)
    qpos0: (8,) JAX array of initial joint configuration
    freq: Gait frequency (used for ramp calculation)
    swing_height: Maximum foot swing height
    Other args: Same as reference_at_time_jax()
  
  Returns:
    Dictionary with:
      - 'ctrl': (8,) JAX array of desired joint angles
      - 'foot_heights': (4,) JAX array of desired foot z-heights
      - 'swing_flags': (4,) JAX array of swing/stance flags
      - 'phases': (4,) JAX array (copy of input phases)
  """
  phases = phase_array  # (4,)
  
  # Leg indexing: FR(0,1), FL(2,3), HR(4,5), HL(6,7)
  leg_joint_indices = jp.array([
      [2, 3],  # FR
      [0, 1],  # FL
      [6, 7],  # HR
      [4, 5],  # HL
  ])
  
  # Estimate ramp from phase progression (simplified: assume continuous ramp-up)
  # Ramp ≈ 1.0 after first ~1.67s with freq=0.6Hz (3 cycles)
  ramp = jp.clip(1.0, 0.0, 1.0)  # Already ramped up in normal training
  
  # Compute foot heights via gait function (vectorized with vmap)
  def compute_rz_single(phase_val):
    return gait.get_rz(phase_val, swing_height=swing_height)
  
  rz = jax.vmap(compute_rz_single)(phases)  # (4,)
  foot_heights = rz
  
  # Normalized height (0-1)
  z = jp.clip(rz / (swing_height + 1e-8), 0.0, 1.0)  # (4,)
  
  # Swing vs stance detection
  is_swing = jp.where(z > swing_threshold, 1.0, 0.0)  # (4,)
  is_stance = 1.0 - is_swing  # (4,)
  
  s = jp.sin(phases)  # (4,)
  
  # Extract hip and knee indices for each leg
  hip_indices = leg_joint_indices[:, 0]  # [2, 0, 6, 4]
  knee_indices = leg_joint_indices[:, 1]  # [3, 1, 7, 5]
  
  # Get initial positions for hips and knees
  hip0 = qpos0[hip_indices]  # (4,)
  knee0 = qpos0[knee_indices]  # (4,)
  
  # Hip control: swing motion only
  forward_sign = -1.0
  hip_ctrl = hip0 + ramp * forward_sign * hip_amp * s * is_swing  # (4,)
  
  # Knee control: flex in swing, extend in stance
  knee_ctrl = (
      knee0
      - ramp * knee_swing_amp * z * is_swing
      + ramp * knee_stance_amp * is_stance
  )  # (4,)
  
  # Build full ctrl array (8,)
  ctrl = jp.zeros(8)
  ctrl = ctrl.at[hip_indices].set(hip_ctrl)
  ctrl = ctrl.at[knee_indices].set(knee_ctrl)
  
  return {
      'ctrl': ctrl,
      'foot_heights': foot_heights,
      'swing_flags': is_swing,
      'phases': phases,
  }


def main():
  model = MjModel.from_xml_path(str(WORLD_XML), assets=get_assets())
  print("✓ Loaded Solo8 with meshes")

  data = MjData(model)

  print("timestep:", model.opt.timestep)
  print("nu:", model.nu)
  print("actuator_forcerange (first 8):\n", model.actuator_forcerange[:model.nu])
  print("actuator_gear (first 8):\n", model.actuator_gear[:model.nu])
  print("actuator_gainprm (first 8):\n", model.actuator_gainprm[:model.nu])
  print("actuator_biasprm (first 8):\n", model.actuator_biasprm[:model.nu])

  # --- Reset into the keyframe pose (prevents initial "slam" to 0-rad targets) ---
  key_name = "initial_joint_positions"
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
  if key_id < 0:
    raise RuntimeError(f"Keyframe '{key_name}' not found in model.")
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  # Store initial pose; we'll command around it (ctrl = q0 + delta).
  qpos0 = data.qpos.copy()

  # --- Gait parameters (start gentle) ---
  gait_params = {
      'freq': 0.6,
      'swing_height': 0.08,
      'ramp_time': 1.0,
      'hip_amp': 0.12,
      'knee_swing_amp': 0.35,
      'knee_stance_amp': 0.05,
      'swing_threshold': 0.25,
  }

  def control_callback(model, data):
    t = float(data.time)
    ref = reference_at_time(t, qpos0, **gait_params)
    data.ctrl[:] = ref['ctrl']

  mujoco.set_mjcb_control(control_callback)
  viewer.launch(model, data)


if __name__ == "__main__":
  main()
