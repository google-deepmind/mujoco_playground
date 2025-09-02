'''
Test the sim to real gap using the same motor control input
tested: step response
TODO: other signals
'''

import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Model path
MODEL = "./xmls/PreGen1_RightHand_Tendon_thumb.xml" 

# Tendons to test
TENDON_NAMES_ID_MAP = {
    "if_tendon0": 5,      # index
    "mf_tendon0": 8,      # middle
    "rf_tendon0": 11,     # ring
    "pf_tendon0": 14,     # pinky
    "th_tendon1": 1,      #  Right thumb CMC flexion
    "th_tendon2": 2,      # Right thumb tendon
}

DATA_PATH = './data/real_control.csv'

def find_position_actuator_for_tendon(m, tid):
    for aid in range(m.nu):
        if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON:  # it is used to distinguish tendon actuators from other types of actuators
            continue
        if m.actuator_trnid[aid, 0] != tid:  # the left side gives the tendon ID that this actuator controls
            continue                         # only proceeds if it matches our target tendon ID
        lo, hi = m.actuator_ctrlrange[aid]
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:  # return the actuator id if all conditions are met
            return aid
    return None


def step_response_real(model, data, viewer, tendon_name, real_ctrl, resolution=25):  # real_ctr is from the real motor
    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)

    if tendon_id == -1:
        raise RuntimeError(f"Tendon {tendon_name} not found")
    actuator_id = find_position_actuator_for_tendon(model, tendon_id)
    if actuator_id is None:
        raise RuntimeError(f"{tendon_name}: No <position tendon=...> actuator found")
    y = []
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0
    lo, hi = model.actuator_ctrlrange[actuator_id]
    real_ctrl_min, real_ctrl_max = min(real_ctrl), max(real_ctrl)
    for i in range(len(real_ctrl)):
        data.ctrl[actuator_id] = np.interp(real_ctrl[i], [real_ctrl_min, real_ctrl_max], [hi, lo]) # map real to sim
        #data.ctrl[actuator_id] = hi-real_ctrl[i]  # map real to sim
        for _ in range(resolution):               # this can solve the problem of different sampling frequency between sim (0.01) and real (0.2)    
            mujoco.mj_step(model, data)           # the control signal is processed during each simulation step
        viewer.sync()
        y.append(float(data.ten_length[tendon_id]))
    return y


def step_response(model, data, viewer, tendon_name, data_len=5, resolution=25, edge_margin=0.1):
    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    if tendon_id == -1:
        raise RuntimeError(f"Tendon {tendon_name} not found")
    actuator_id = find_position_actuator_for_tendon(model, tendon_id)
    if actuator_id is None:
        raise RuntimeError(f"{tendon_name}: No <position tendon=...> actuator found")

    mujoco.mj_resetData(model, data)    # reset
    data.ctrl[:] = 0
    lo, hi = model.actuator_ctrlrange[actuator_id]
    span = hi - lo
    lo_eff = lo + edge_margin * span
    hi_eff = hi - edge_margin * span
    y = []
    for i in range(data_len):
        if i<2:
            data.ctrl[actuator_id] = float(hi)
            for _ in range(resolution):
                mujoco.mj_step(model, data)
        else:
            data.ctrl[actuator_id] = float(lo)
            for _ in range(resolution):
                mujoco.mj_step(model, data)
        
        viewer.sync()
        y.append(float(data.ten_length[tendon_id]))
    return y


def main(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    real_ctrl = df.dropna()
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    curves = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for tendon_name in TENDON_NAMES_ID_MAP.keys():
            tendon_id = TENDON_NAMES_ID_MAP[tendon_name]
            y1 = step_response_real(model, data, viewer, tendon_name, real_ctrl[f'tendon_{tendon_id}_sent'])  # read/sent, real means control from the real motor 
            y2 = step_response(model, data, viewer, tendon_name, data_len=5, resolution=25, edge_margin=0.1) 
            curves.append((tendon_name, y1, y2))
        if curves:
            n = len(curves)
            ncols = 2
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
            axes = axes.flatten()
            for ax, (tendon_name, y1, y2) in zip(axes, curves):
                ax.plot(y1, label='real')
                ax.plot(y2, label='sim')
                y1_min, y1_max = min(y1), max(y1)
                y2_min, y2_max = min(y2), max(y2)
                ax.set_title(f'{tendon_name}: real:({y1_min:.2f}, {y1_max:.2f}), sim:({y2_min:.2f}, {y2_max:.2f})')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Tendon length (m)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

        # Keep viewer open until you manually close it
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main(DATA_PATH)
