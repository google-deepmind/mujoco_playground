""""
Deploy Go2 ONNX policy to MuJoCo (Sim2Sim deployment, sensor-based)
Corrected version following play_go2_trot_jax.py exactly
"""

import mujoco
import numpy as np
import onnxruntime as rt
from mujoco import viewer, Renderer
from etils import epath

import mediapy as media

from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.TrotUtil import (
    cos_wave, make_kinematic_ref, rotate_inv,
)

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


# ---------------------------------------------------------------------
# ONNX Controller (corrected)
# ---------------------------------------------------------------------
class Go2OnnxController:

    def __init__(self, policy_path, default_qpos, ctrl_dt, n_substeps, action_scale):
        # Load ONNX policy
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self._default_angles = default_qpos[7:]
        self._default_qpos = default_qpos
        self._action_scale = action_scale

        # last_action should be raw action ∈ [-1,1], not the ctrl
        # self._last_action = np.zeros_like(self._default_angles, dtype=np.float32)
        self._last_action = self._default_angles.copy()

        self._counter = 0
        self._n_substeps = n_substeps

        # gait reference (same as JAX)
        step_k = 13
        kin_q = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=ctrl_dt)
        kin_q = np.array(kin_q) + np.array(self._default_angles)

        self._kinematic_ref_qpos = kin_q
        self._step_idx = 0
        self._l_cycle = kin_q.shape[0]

    # ---------------------------------------------------------------------
    # Build observation (STRICTLY same as play_go2_trot_jax)
    # ---------------------------------------------------------------------
    def get_obs(self, model, data) -> np.ndarray:
        # yaw_rate: sensor gyro z-axis * 0.25
        gyro = data.sensor("gyro").data
        yaw_rate = gyro[2] * 0.25

        # orientation: quaternion → g_local
        quat = data.sensor("orientation").data
        g_world = np.array([0.0, 0.0, -1.0])
        g_local = rotate_inv(g_world, quat)

        # joint angles: MUST follow the same order as JAX
        joint_names = [
            "abduction_front_left_pos", "hip_front_left_pos", "knee_front_left_pos",
            "abduction_front_right_pos", "hip_front_right_pos", "knee_front_right_pos",
            "abduction_hind_left_pos", "hip_hind_left_pos", "knee_hind_left_pos",
            "abduction_hind_right_pos", "hip_hind_right_pos", "knee_hind_right_pos",
        ]
        angles = np.array([data.sensor(n).data[0] for n in joint_names])

        # reference qpos
        kin_ref = self._kinematic_ref_qpos[self._step_idx]

        # obs structure SAME AS JAX
        obs = np.concatenate([
            [yaw_rate],
            g_local,
            angles - self._default_angles,
            self._last_action,   # raw action [-1,1]
            kin_ref,
        ])

        return np.clip(obs, -100, 100).astype(np.float32)

    # ---------------------------------------------------------------------
    # Control step
    # ---------------------------------------------------------------------
    def get_control(self, model, data):

        if self._counter % self._n_substeps == 0:
            # collect obs
            obs = self.get_obs(model, data)
            obs = obs.reshape(1, -1)

            # ONNX forward
            actions, std = self._policy.run(["actions", "std"], {"obs": obs})
            act = actions[0]
            act = np.clip(act, -1.0, 1.0)

            # PD target & update last_action
            ctrl = self._default_angles + act * self._action_scale
            data.ctrl[:] = ctrl
            self._last_action = ctrl.copy()


            # phase update
            self._step_idx = (self._step_idx + 1) % self._l_cycle

        self._counter += 1


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    xml_path = consts.MJX_XML_PATH.as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    # same dt as JAX
    ctrl_dt = 0.02
    sim_dt = 0.002
    model.opt.timestep = sim_dt
    n_substeps = int(ctrl_dt / sim_dt)

    # PD gains (same as JAX)
    Kp = 80.0
    Kd = 0.5
    model.dof_damping[6:] = Kd
    model.actuator_gainprm[:, 0] = Kp
    model.actuator_biasprm[:, 1] = -Kp

    controller = Go2OnnxController(
        policy_path=(_ONNX_DIR / "go2_apg_policy.onnx").as_posix(),
        default_qpos=np.array(model.keyframe("home").qpos),
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=np.array([0.2, 0.8, 0.8] * 4),
    )

    mujoco.set_mjcb_control(controller.get_control)

    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer


# ---------------------------------------------------------------------
# Main – run and save video
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model, data, renderer = load_callback()

    frames = []
    run_time = 10.0
    print("Running simulation...")

    while data.time < run_time:
        mujoco.mj_step(model, data)

        renderer.update_scene(data)
        frames.append(renderer.render())

        if len(frames) % 100 == 0:
            print(f"time={data.time:.2f}s, frames={len(frames)}")

    print("Saving video...")
    media.write_video("go2_trot_onnx.mp4", frames, fps=int(1/model.opt.timestep))
    print("Saved to go2_trot_onnx.mp4")

    renderer.close()
