#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Model path
MODEL = "./xmls/PreGen1_RightHand_Tendon_thumb.xml" 

# Tendons to test
TENDON_NAMES = [
    "if_tendon0",
    "mf_tendon0",
    "rf_tendon0",
    "pf_tendon0",
    "th_tendon1",
    "th_tendon2",
]

MOTOR_LIMIT = {
    "if_tendon0": (0, 0.02667),
    "mf_tendon0": (0, 0.02667),
    "rf_tendon0": (0, 0.02667),
    "pf_tendon0": (0, 0.02667),
    "th_tendon1": (0, 0.009942),
    "th_tendon2": (0, 0.012384)
}  # data obtained from real motor

shift = 0.07
# Parameters
SETTLE_TIME = 1  # Steady state time at low reference before step (s)
SIM_TIME = 3.00  # Recording duration after step (s)
EDGE_MARGIN = 0.10  # Margin from ctrlrange boundaries as proportion
N_COLS = 2  # Number of columns in subplot


def find_position_actuator_for_tendon(model, tendon_id):
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        if model.actuator_trnid[actuator_id, 0] != tendon_id:  # the left side gives the tendon ID that this actuator controls
            continue                         # only proceeds if it matches our target tendon ID
        lo, hi = model.actuator_ctrlrange[actuator_id]
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:  # return the actuator id if all conditions are met
            return actuator_id
    return None


def step_response(model, data, viewer, tendon_name, settle_time, sim_time, edge_margin):
    dt = float(model.opt.timestep) # 0.01
    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    if tendon_id == -1:
        raise RuntimeError(f"Tendon {tendon_name} not found")
    actuator_id = find_position_actuator_for_tendon(model, tendon_id)
    if actuator_id is None:
        raise RuntimeError(f"{tendon_name}: No <position tendon=...> actuator found")

    # ========== Initialization: Set all tendons to high reference ==========
    mujoco.mj_resetData(model, data)    # reset
    data.ctrl[:] = 0.0
    print("=====Set all tendons to high reference=====")
    for tendon_name in TENDON_NAMES:
        tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
        if tendon_id == -1:
            continue
        actuator_id = find_position_actuator_for_tendon(model, tendon_id)
        if actuator_id is None:
            continue
        lo_all, hi_all = model.actuator_ctrlrange[actuator_id]
        span_all = hi_all - lo_all
        hi_eff_all = hi_all
        data.ctrl[actuator_id] = float(hi_eff_all)
        #data.ctrl[actuator_id] = float(MOTOR_LIMIT[tendon_name][1])  # change this from real accept command + shift value, this is only for test purpose
        #data.ctrl[actuator_id] = 0.10

    # Stabilize all at high position for some time
    settle_steps = int(round(settle_time / dt))
    start = time.time()
    print("======Stabilize all at high positionfor some time=======")
    for i in range(settle_steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        target = start + (i + 1) * dt
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)

    # ========== Step: Drive current tendon to low reference only ==========
    lo, hi = model.actuator_ctrlrange[actuator_id]
    span = hi - lo
    lo_eff = lo + edge_margin * span
    hi_eff = hi - edge_margin * span
    data.ctrl[actuator_id] = float(lo_eff) # original code
    #data.ctrl[aid] = MOTOR_LIMIT[tname][0]
    #data.ctrl[aid] = 0.07

    # Record response
    steps = int(round(sim_time / dt))
    t_list, y_list = [], []
    start = time.time()
    print("======Drive current tendon to low reference only=======")
    for i in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        t_list.append((i + 1) * dt)
        y_list.append(float(data.ten_length[tendon_id]))
        target = start + (i + 1) * dt
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)

    return np.asarray(t_list), np.asarray(y_list), lo_eff


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    curves = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for tendon_name in TENDON_NAMES:
            print(f">>> Starting {tendon_name} step experiment (duration: {SIM_TIME:.2f}s)")
            t, y, sp = step_response(
                model, data, viewer, tendon_name, SETTLE_TIME, SIM_TIME, EDGE_MARGIN
            )
            curves.append((tendon_name, t, y, sp))
            print(f"[Completed] {tendon_name}")

        # Experiment finished, plot subplots
        if curves:
            n = len(curves)
            ncols = N_COLS
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
            axes = axes.flatten()
            for ax, (tendon_name, t, y, sp) in zip(axes, curves):
                ax.plot(t, y, label=f"{tendon_name} length")
                ax.axhline(sp, linestyle="--", color="r", label="setpoint")
                ax.set_title(tendon_name)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Tendon length (m)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            for ax in axes[len(curves) :]:
                ax.axis("off")
            fig.suptitle("Tendon step responses", y=0.995, fontsize=12)
            fig.tight_layout()
            plt.show()

        # Keep viewer open until you manually close it
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
