'''
sampling (N, T, n_act) data for tuning some physical parameters in the sim to solve the sim to real gap problem
Suggestions about collecting the data: try to collect data of different signals: square, sine and so on in order to solve the generalization issue and noise issue
A CMA toolbox is used to help tue the paramters: actuator_gainprm, actuator_dof_damping
'''

import os
import io
import sys
import gc
import contextlib
import numpy as np
from tqdm import tqdm

import mujoco
import mujoco.viewer

import cma
import pandas as pd
import glob

# ----------------------------
# User-config
# ----------------------------
DATA_PATH = "./data/real_control.npz"  # data contains [1,2,5,8,11,14] tendon length change in a time interval obtained from real hand, multiple experiments, real_control_1.csv

USE_VIEWER = False           # Toggle viewer on/off
STEP_PER_CMD = 25             # MuJoCo substeps per command
SUBSAMPLE_N = None           # e.g., 32 to speed up; None = use all trajectories
HZ = 20                      # Dataset sampling rate (for velocity term)
USE_VEL_TERM = False         # Include velocity MSE term in loss

# CMA-ES bounds (normalized domain is [0,1], we map to these)
KP_MIN, KP_MAX = 0.5, 10.0    # tendon actuator gains range (6 actuators)
KD_MIN, KD_MAX = 0.0, 1.0     # dof damping range (length = # of dofs)

# CMA-ES hyperparameters
SIGMA0 = 0.2
POPSIZE = 24
MAXITER = 60

# Loss weights
ALPHA_POS = 1.0
BETA_VEL = 0.01

DEBUG_MODE = False

# ----------------------------
# Load hand model
# mj_data: holds the dynamic simulation state at runtime:
# generalized positions (d.qpos), velocities (d.qvel), forces and torques (d.qfrc), sensor readings (d.sensordata), contact information (d.contact), time (d.time)
# mj_model: MuJoCo model (is a static description of the simulation world, geometry (meshes, shapes, sizes), joints, actuators, tendons, physical params, sensors, cameras),
# think of mj_model as a blueprint, mj_data as the state of the world evolving in time while MuJoCo runs the physics
# ----------------------------
xml_path = "./xmls/PreGen1_RightHand_Tendon_thumb.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# With tendon configuration:
TENDON_NAMES = [
    "if_tendon0",    # Index finger tendon
    "mf_tendon0",    # Middle finger tendon
    "rf_tendon0",    # Ring finger tendon
    "pf_tendon0",    # Pinky finger tendon
    "th_tendon1",    # Thumb tendon 1
    "th_tendon2"     # Thumb tendon 2
]

# Get tendon IDs and their corresponding actuators
tendon_ids = []
actuator_ids = []
for tendon_name in TENDON_NAMES:
    tendon_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    if tendon_id != -1:
        # Find position actuator for this tendon
        for actuator_id in range(mj_model.nu):
            if (mj_model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON and 
                mj_model.actuator_trnid[actuator_id, 0] == tendon_id):
                tendon_ids.append(tendon_id)
                actuator_ids.append(actuator_id)
                break

NUM_ACT = len(actuator_ids)  # Should be 6
NUM_DOFS = mj_model.nv       # Number of DOFs in tendon model
NUM_Q = mj_model.nq          # Number of position variables
print(f"#actuator:{NUM_ACT}, #dofs:{NUM_DOFS}, #position: {NUM_Q}, actuator ids: {actuator_ids}, tendor ids: {tendon_ids}")  
# 6, 16, 16, [0, 4, 8, 12, 16, 17], [0,1,2,3,5,6]

# ----------------------------
# Load real control data
# ----------------------------
print("Loading real control data...")
real_data = np.load(DATA_PATH, allow_pickle=True)

# Load individual tendon arrays using tendon names
tendon_sent_arrays = {}
tendon_read_arrays = {}

# Load sent data for each tendon
for tendon_name in TENDON_NAMES:
    sent_key = f'{tendon_name}_sent'
    read_key = f'{tendon_name}_read'
    
    if sent_key in real_data and read_key in real_data:
        tendon_sent_arrays[tendon_name] = real_data[sent_key]
        tendon_read_arrays[tendon_name] = real_data[read_key]
        print(f"Loaded {sent_key}: {len(real_data[sent_key])} experiments")
        print(f"Loaded {read_key}: {len(real_data[read_key])} experiments")
    else:
        print(f"Warning: Missing {sent_key} or {read_key}")

# Get metadata
experiment_info = real_data['experiment_info']
num_experiments = real_data['num_experiments']
print(f"Total experiments: {num_experiments}")

# Close the data file
real_data.close()

# Verify data structure
print("\nData structure verification:")
for tendon_name in TENDON_NAMES:
    if tendon_name in tendon_sent_arrays:
        sent_data = tendon_sent_arrays[tendon_name]
        read_data = tendon_read_arrays[tendon_name]
        print(f"Tendon {tendon_name}:")
        for exp_idx in range(min(3, len(sent_data))):  # Show first 3 experiments
            print(f"  Experiment {exp_idx}: sent={sent_data[exp_idx].shape}, read={read_data[exp_idx].shape}")

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def unnormalize(x):
    """
    x in [0,1]^(NUM_ACT + NUM_DOFS) -> (kp[NUM_ACT], kd[NUM_DOFS])
    """
    x = clamp(np.asarray(x), 0.0, 1.0)
    kp_norm = x[:NUM_ACT]
    kd_norm = x[NUM_ACT:]
    kp = KP_MIN + kp_norm * (KP_MAX - KP_MIN)
    kd = KD_MIN + kd_norm * (KD_MAX - KD_MIN)
    return kp, kd

def set_pd_gains(kp, kd):
    """
    Apply gains to MuJoCo tendon model:
      - actuator_gainprm[:,0] <- kp (one per tendon actuator)
      - dof_damping[:]        <- kd (one per DOF)
    """
    #assert kp.shape == (NUM_ACT,)
    #assert kd.shape == (NUM_DOFS,)
    # Set gains for tendon actuators only
    for i, actuator_id in enumerate(actuator_ids):
        mj_model.actuator_gainprm[i, 0] = kp[i]            # where the p value influence the simulation, 0-9 means different coeff
                                                           # 0: proportional gain, 1: velocity gain, 2: acceleration gain, 3:bias gain, 4: force gain, 5: time constant, 6: damping gain, 7: couping gain, 8: stiffness gain, 9: reference gain 

    # DOF damping: length nv
    mj_model.dof_damping[:] = kd      # where the d value influence the simulation


def find_position_actuator_for_tendon(m, tid):
    for aid in range(m.nu):
        if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        if m.actuator_trnid[aid, 0] != tid:
            continue
        lo, hi = m.actuator_ctrlrange[aid]
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            return aid
    return None


def rollout_and_mse(kp, kd):
    """
    For given (kp, kd), run over the dataset and compute loss.
    Returns (pos_mse, vel_mse, total_mse)
    """
    set_pd_gains(kp, kd)

    if USE_VIEWER:
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)
    else:
        viewer_ctx = contextlib.nullcontext()

    dt = 1.0 / HZ
    pos_se_accum = 0.0
    vel_se_accum = 0.0
    denom_pos = 0
    denom_vel = 0

    # Get number of experiments from any tendon array
    num_experiments = len(tendon_sent_arrays[TENDON_NAMES[0]])
    
    print(f"Running {num_experiments} experiments...")
    
    with viewer_ctx as viewer:
        for exp_idx in range(num_experiments):
            mujoco.mj_forward(mj_model, mj_data)
            if USE_VIEWER:
                viewer.sync()
            
            # Get experiment data for this experiment
            exp_sent_data = {}
            exp_read_data = {}
            exp_lengths = {}
            
            # Extract data for each tendon for this experiment
            for tendon_name in TENDON_NAMES:
                if tendon_name in tendon_sent_arrays and exp_idx < len(tendon_sent_arrays[tendon_name]):
                    exp_sent_data[tendon_name] = tendon_sent_arrays[tendon_name][exp_idx]
                    exp_read_data[tendon_name] = tendon_read_arrays[tendon_name][exp_idx]
                    exp_lengths[tendon_name] = len(exp_sent_data[tendon_name])
                else:
                    print(f"Warning: No data for {tendon_name}, experiment {exp_idx}")
                    continue
            
            # Find minimum length across all tendons for this experiment
            if not exp_lengths:
                print(f"Skipping experiment {exp_idx}: no valid data")
                continue
                
            min_length = min(exp_lengths.values())
            print(f"Experiment {exp_idx}: min_length={min_length}, lengths={exp_lengths}")
            
            # Initialize simulation outputs
            sim_outputs = np.zeros((min_length, NUM_ACT), dtype=np.float64)
            real_outputs = np.zeros((min_length, NUM_ACT), dtype=np.float64)
            
            # Reset control signals
            mj_data.ctrl[:] = 0.0
            
            # Rollout using recorded command (6 tendon actuators)
            for i, tendon_name in enumerate(TENDON_NAMES):
                if tendon_name in exp_sent_data:
                    tendon_sent = exp_sent_data[tendon_name]
                    tendon_read = exp_read_data[tendon_name]
                    actuator_id = find_position_actuator_for_tendon(mj_model, tendon_name)
                    # Use only up to min_length to ensure all tendons have data
                    for t in range(min_length):
                        mj_data.ctrl[actuator_id] = tendon_sent[t]
                        for _ in range(STEP_PER_CMD):
                            mujoco.mj_step(mj_model, mj_data)
                        
                        tendon_mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
                        if tendon_mj_id != -1:
                            sim_outputs[t, i] = mj_data.ten_length[tendon_mj_id].copy()
                        else:
                            print(f"Warning: Could not find {tendon_name}")
                        
                        # Store real data
                        real_outputs[t, i] = tendon_read[t]
                    
                    # Compute MSE for this actuator
                    actuator_mse = np.mean((sim_outputs[:, i] - real_outputs[:, i])**2)
                    pos_se_accum += actuator_mse
                    print(f"  Actuator {actuator_id} ({tendon_name}) MSE: {actuator_mse:.6e}")
                else:
                    print(f"Warning: No data for actuator {actuator_id} ({tendon_name}) in experiment {exp_idx}")

    # Return average MSE across all actuators and experiments
    total_actuators = num_experiments * NUM_ACT
    pos_mse = pos_se_accum / total_actuators if total_actuators > 0 else 0.0
    
    print(f"Total Position MSE: {pos_mse:.6e}")
    return float(pos_mse)

def silent(func, *args, **kwargs):
    """
    Suppress stdout/stderr for a noisy function section (like many rollouts)
    """
    if DEBUG_MODE:
        return func(*args, **kwargs)
    else:
        # Use contextlib.redirect_stdout and redirect_stderr (Python 3.4+)
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')), \
                 contextlib.redirect_stderr(open(os.devnull, 'w')):
                return func(*args, **kwargs)
        except AttributeError:
            # Fallback for older Python versions
            with open(os.devnull, "w") as devnull:
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    return func(*args, **kwargs)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err

def objective(x):
    x = clamp(x, 0.0, 1.0)
    kp, kd = unnormalize(x)
    # pos_mse, vel_mse, total = silent(rollout_and_mse, kp, kd)
    pos_mse = silent(rollout_and_mse, kp, kd)
    print(f"Position MSE: {pos_mse:.6e}")
    # if USE_VEL_TERM:
    #     print(f"Velocity MSE: {vel_mse:.6e}")
    # print(f"Total Loss:   {total:.6e}")
    # clean up any lingering buffers
    gc.collect()
    return float(pos_mse)

# ----------------------------
# Optimize
# ----------------------------
def main():
    DIM = NUM_ACT + NUM_DOFS
    # Seed guess (normalized). Start from mid-range for 6 tendon actuators
    kp0 = np.ones(NUM_ACT) * 2.5         # 6 tendon actuators
    kd0 = np.ones(NUM_DOFS) * 0.2        # DOF damping
    x0 = np.concatenate([
        (kp0 - KP_MIN) / (KP_MAX - KP_MIN),
        (kd0 - KD_MIN) / (KD_MAX - KD_MIN),
    ])
    x0 = clamp(x0, 0.0, 1.0)
    
    opts = {
        "bounds": [0.0, 1.0],
        "popsize": POPSIZE,
        "verb_disp": 1,
        "maxiter": MAXITER,
    }
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)

    best_error = float("inf")
    best_params = None

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in tqdm(solutions, desc="Evaluating population"):
            f = objective(sol)
            fitnesses.append(f)
        es.tell(solutions, fitnesses)
        es.disp()

        min_idx = int(np.argmin(fitnesses))
        if fitnesses[min_idx] < best_error:
            best_error = float(fitnesses[min_idx])
            kp_best, kd_best = unnormalize(solutions[min_idx])
            best_params = (kp_best.copy(), kd_best.copy())
            print(f"New best loss: {best_error:.6e}")
            with open("best_pd_gains_cma.txt", "w") as f:
                f.write(f"MSE={best_error:.8e}\n")
                f.write(f"Kp={kp_best.tolist()}\n")
                f.write(f"Kd={kd_best.tolist()}\n")

    print("\nFinal best PD gains:")
    kp, kd = best_params
    print("Kp:", kp)
    print("Kd:", kd)
    print(f"Best loss: {best_error:.6e}")

if __name__ == "__main__":
    main()