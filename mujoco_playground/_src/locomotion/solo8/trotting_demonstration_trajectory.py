import numpy as np
from pathlib import Path
from mujoco import MjModel, MjData, viewer
from mujoco_playground._src import gait, mjx_env
import mujoco

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
  trot_phases = gait.GAIT_PHASES[0]  # Foot order: FR, FL, RR(HR), RL(HL)
  freq = 0.6                         # Hz
  swing_height = 0.08                # meters (used by get_rz)
  ramp_time = 1.0                    # seconds
  hip_amp = 0.12                     # rad (fore-aft leg swing proxy)
  knee_swing_amp = 0.35     # flex during swing
  knee_stance_amp = 0.05    # small extension during stance
  swing_threshold = 0.25    # z > this => swing


  # Actuator order:
  # FL_HFE(0), FL_KFE(1), FR_HFE(2), FR_KFE(3), HL_HFE(4), HL_KFE(5), HR_HFE(6), HR_KFE(7)
  leg_joint_indices = [
      [2, 3],  # FR
      [0, 1],  # FL
      [6, 7],  # HR (RR phase slot)
      [4, 5],  # HL (RL phase slot)
  ]

  joint_names = ["FL_HFE", "FL_KFE", "FR_HFE", "FR_KFE", "HL_HFE", "HL_KFE", "HR_HFE", "HR_KFE"]
  jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
  if any(j < 0 for j in jids):
    missing = [joint_names[i] for i, j in enumerate(jids) if j < 0]
    raise RuntimeError(f"Missing joints in model: {missing}")
  qadr = [int(model.jnt_qposadr[j]) for j in jids]  # qpos index per joint

  def _to_float(x):
    # gait.get_rz returns a jax array; convert to python float safely
    return float(np.asarray(x))

  def control_callback(model, data):
    t = float(data.time)
    ctrl = np.zeros(model.nu, dtype=np.float32)

    # hold targets briefly
    if t < 0.05:
        for i in range(model.nu):
            ctrl[i] = float(data.qpos[qadr[i]])
        data.ctrl[:] = ctrl
        return

    ramp = float(np.clip((t - 0.05) / ramp_time, 0.0, 1.0))
    global_phase = 2.0 * np.pi * freq * t

    for leg in range(4):
        phase_val = global_phase + float(trot_phases[leg])

        rz = float(np.asarray(gait.get_rz(phase_val, swing_height=swing_height)))
        z = float(np.clip(rz / (swing_height + 1e-8), 0.0, 1.0))

        is_swing = 1.0 if z > swing_threshold else 0.0
        is_stance = 1.0 - is_swing

        s = np.sin(phase_val)

        hfe_i, kfe_i = leg_joint_indices[leg]
        hip0  = float(qpos0[qadr[hfe_i]])
        knee0 = float(qpos0[qadr[kfe_i]])

        # hip: move mostly during swing to avoid stance scraping
        forward_sign = -1.0  # flip to +1.0 if direction is wrong
        ctrl[hfe_i] = hip0 + ramp * forward_sign * hip_amp * s * is_swing

        # knee: flex only in swing; (slightly) extend in stance for support
        ctrl[kfe_i] = (
            knee0
            - ramp * knee_swing_amp * z * is_swing   # flex while lifting
            + ramp * knee_stance_amp * is_stance     # support in stance
        )

    data.ctrl[:] = ctrl

  mujoco.set_mjcb_control(control_callback)
  viewer.launch(model, data)


if __name__ == "__main__":
  main()
