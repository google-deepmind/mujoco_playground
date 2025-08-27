#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Model path
MODEL = "/home/nan/CodeRepos/tetheria_rl/mujoco_playground/mujoco_playground/_src/manipulation/tetheria_hand_tendon/xmls/PreGen1_RightHand_Tendon_thumb.xml"

# Tendons to test
TENDON_NAMES = [
    "if_tendon0",
    "mf_tendon0",
    "rf_tendon0",
    "pf_tendon0",
    "th_tendon1",
    "th_tendon2",
]

# Parameters
SETTLE_TIME = 1  # Steady state time at low reference before step (s)
SIM_TIME = 3.00  # Recording duration after step (s)
EDGE_MARGIN = 0.10  # Margin from ctrlrange boundaries as proportion
N_COLS = 2  # Number of columns in subplot


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


def step_response(m, d, viewer, tendon_name, settle_time, sim_time, edge_margin):
    dt = float(m.opt.timestep)

    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    if tid == -1:
        raise RuntimeError(f"Tendon {tendon_name} not found")
    aid = find_position_actuator_for_tendon(m, tid)
    if aid is None:
        raise RuntimeError(f"{tendon_name}: No <position tendon=...> actuator found")

    # ========== Initialization: Set all tendons to high reference ==========
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = 0.0
    for tname in TENDON_NAMES:
        tid_all = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, tname)
        if tid_all == -1:
            continue
        aid_all = find_position_actuator_for_tendon(m, tid_all)
        if aid_all is None:
            continue
        lo_all, hi_all = m.actuator_ctrlrange[aid_all]
        span_all = hi_all - lo_all
        hi_eff_all = hi_all
        d.ctrl[aid_all] = float(hi_eff_all)

    # Stabilize all at high position for some time
    settle_steps = int(round(settle_time / dt))
    start = time.time()
    for i in range(settle_steps):
        mujoco.mj_step(m, d)
        viewer.sync()
        target = start + (i + 1) * dt
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)

    # ========== Step: Drive current tendon to low reference only ==========
    lo, hi = m.actuator_ctrlrange[aid]
    span = hi - lo
    lo_eff = lo + edge_margin * span
    hi_eff = hi - edge_margin * span
    d.ctrl[aid] = float(lo_eff)

    # Record response
    steps = int(round(sim_time / dt))
    t_list, y_list = [], []
    start = time.time()
    for i in range(steps):
        mujoco.mj_step(m, d)
        viewer.sync()
        t_list.append((i + 1) * dt)
        y_list.append(float(d.ten_length[tid]))
        target = start + (i + 1) * dt
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)

    return np.asarray(t_list), np.asarray(y_list), lo_eff


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    curves = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        for tname in TENDON_NAMES:
            print(f">>> Starting {tname} step experiment (duration: {SIM_TIME:.2f}s)")
            t, y, sp = step_response(
                m, d, viewer, tname, SETTLE_TIME, SIM_TIME, EDGE_MARGIN
            )
            curves.append((tname, t, y, sp))
            print(f"[Completed] {tname}")

        # Experiment finished, plot subplots
        if curves:
            n = len(curves)
            ncols = N_COLS
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
            axes = axes.flatten()
            for ax, (tname, t, y, sp) in zip(axes, curves):
                ax.plot(t, y, label=f"{tname} length")
                ax.axhline(sp, linestyle="--", color="r", label="setpoint")
                ax.set_title(tname)
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
