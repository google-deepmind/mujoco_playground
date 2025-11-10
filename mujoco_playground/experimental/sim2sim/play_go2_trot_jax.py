"""
Deploy Go2 JAX policy directly in MuJoCo (Sim2Sim deployment)
"""

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import viewer, Renderer
from etils import epath
import functools
import os
os.environ['MUJOCO_GL'] = 'egl'

# 导入 Go2 常量与工具函数
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.TrotUtil import (
    cos_wave, make_kinematic_ref, rotate_inv,
    dcos_wave, quaternion_to_rotation_6d
)

# 从你的训练代码导入模块
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.acme import running_statistics
from brax.io import model

from brax.envs.wrappers import training as brax_training

import mediapy as media

_HERE = epath.Path(__file__).parent

total_rewards = 0.0

# ---------------------------------------------------------------------
# Normalize Definition
# ---------------------------------------------------------------------
def make_normalize_fn(mean, std, max_abs_value=None):
    def normalize_fn(batch, _unused_processor_params=None):
        def normalize_leaf(data, m, s):
            if not jnp.issubdtype(data.dtype, jnp.inexact):
                return data
            data = (data - m) / s
            if max_abs_value is not None:
                data = jnp.clip(data, -max_abs_value, +max_abs_value)
            return data

        return jax.tree_util.tree_map(normalize_leaf, batch, mean, std)
    return normalize_fn

# ---------------------------------------------------------------------
# Controller Definition
# ---------------------------------------------------------------------
class Go2JaxController:
    """JAX 控制器（使用原始 Brax 训练的 policy）"""

    def __init__(self, env_name, policy_path, default_qpos, ctrl_dt, n_substeps, action_scale, alg_name='apg'):
        # ======= 初始化环境与网络 =======
        demo_cfg = registry.get_default_config(env_name)
        demo_cfg['env']['reset2ref'] = False
        demo_cfg['env']['reference_state_init'] = False
        demo_cfg['pert_config']['enable'] = False
        self.reward_scales = demo_cfg['rewards']['scales']

        demo_env = registry.load(env_name, demo_cfg)
        demo_env = brax_training.VmapWrapper(demo_env)
        self.obs_size = demo_env.observation_size
        self.action_size = demo_env.action_size

        params = model.load_params(policy_path)

        if alg_name == 'apg':
            alg_params = locomotion_params.brax_apg_config(env_name)
            alg_training_params = dict(alg_params)
            network_factory = apg_networks.make_apg_networks
            if "network_factory" in alg_params:
                del alg_training_params["network_factory"]
                network_factory = functools.partial(
                    apg_networks.make_apg_networks,
                    **alg_params.network_factory,
            )
        else:
            alg_params = locomotion_params.brax_ppo_config(env_name)
            alg_training_params = dict(alg_params)
            network_factory = ppo_networks.make_ppo_networks
            if "network_factory" in alg_params:
                del alg_training_params["network_factory"]
                network_factory = functools.partial(
                    ppo_networks.make_ppo_networks,
                    **alg_params.network_factory,
            )

        normalize = lambda x, y: x
        if alg_params['normalize_observations']:
            # normalize = running_statistics.normalize
            mean, std = params[0].mean, params[0].std
            normalize = make_normalize_fn(mean, std)

        network = network_factory(
            self.obs_size, self.action_size, preprocess_observations_fn=normalize
        )

        if alg_name == 'apg':
            make_inference_fn = apg_networks.make_inference_fn(network)
        else:
            make_inference_fn = ppo_networks.make_inference_fn(network)

        self._policy = jax.jit(make_inference_fn(params, deterministic=True))
        self._rng_key = jax.random.PRNGKey(0)

        # ======= 低层控制参数 =======
        self._default_angles = default_qpos[7:]
        self._default_qpos = default_qpos
        self._action_scale = action_scale
        self._last_action = np.zeros_like(self._default_angles, dtype=np.float32)
        self._counter = 0
        self._n_substeps = n_substeps

        # gait 参考轨迹
        step_k = 13
        kin_q = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=ctrl_dt)
        kin_qvel = make_kinematic_ref(dcos_wave, step_k, scale=0.3, dt=ctrl_dt)
        kin_q = np.array(kin_q) + np.array(self._default_angles)
        self._kinematic_ref_qpos = kin_q
        self._kinematic_ref_qvel = kin_qvel
        self._step_idx = 0
        self._l_cycle = int(kin_q.shape[0])
        self._feet_ids = np.array([demo_env._mj_model.geom(name).id for name in consts.FEET_GEOMS])
        # print("Shape of kinematic ref qpos:", self._kinematic_ref_qpos.shape)

    # ----------------- 提取观测 -----------------
    def get_obs(self, model, data) -> np.ndarray:
        # ----------------- sensor -----------------
        gyro = data.sensor("gyro").data
        yaw_rate = gyro[2] * 0.25

        quat = data.sensor("orientation").data
        g_world = np.array([0.0, 0.0, -1.0])
        g_local = rotate_inv(g_world, quat)

        joint_angle_names = [
            # FL
            "abduction_front_left_pos", "hip_front_left_pos", "knee_front_left_pos",
            # FR
            "abduction_front_right_pos", "hip_front_right_pos", "knee_front_right_pos",
            # RL
            "abduction_hind_left_pos", "hip_hind_left_pos", "knee_hind_left_pos",
            # RR
            "abduction_hind_right_pos", "hip_hind_right_pos", "knee_hind_right_pos",
        ]

        # for i, name in enumerate(joint_angle_names):
        #     print(i, name, model.joint(i).name)

        angles = np.array([data.sensor(name).data[0] for name in joint_angle_names])
        kin_ref = self._kinematic_ref_qpos[self._step_idx]

        sensor_obs = np.concatenate([
            [yaw_rate],
            g_local,
            angles - self._default_angles,
            self._last_action,
            kin_ref,
        ])

        obs = sensor_obs

        # # ----------------- data -----------------
        # local_omega = data.cvel[1, :3]
        # yaw_rate = local_omega[2] * 0.25
        # g_world = jnp.array([0.0, 0.0, -1.0])
        # g_local = rotate_inv(g_world, data.xquat[1])
        # angles = data.qpos[7:19]
        # kin_ref = self._kinematic_ref_qpos[self._step_idx]

        # obs = np.concatenate([
        #     [yaw_rate],
        #     g_local,
        #     angles - self._default_angles,
        #     self._last_action,
        #     kin_ref,
        # ])

        # diff = np.abs(sensor_obs - obs)
        # if diff.max() > 1e-6:
        #     print("Observation difference detected! Max diff:", diff.max())

        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    # ----------------- 控制接口 -----------------
    def get_control(self, model, data):
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            obs_jax = jnp.asarray(obs).reshape(1, -1)
            self._rng_key, policy_key = jax.random.split(self._rng_key)
            act_jax, _ = self._policy(obs_jax, policy_key)
            act_jax = jnp.clip(act_jax, -1, 1)
            act = np.array(act_jax[0])
            ctrl = self._default_angles + act * self._action_scale
            data.ctrl[:] = ctrl
            self._last_action = ctrl.copy()
            
            ref_qpos = self._default_qpos.copy()
            ref_qpos[7:] = self._kinematic_ref_qpos[self._step_idx]
            ref_qvel = np.zeros_like(data.qvel)
            ref_qvel[6:] = self._kinematic_ref_qvel[self._step_idx]
            ref_data = mujoco.MjData(model)
            ref_data.qpos[:] = ref_qpos
            ref_data.qvel[:] = ref_qvel

            mujoco.set_mjcb_control(None)
            mujoco.mj_forward(model, ref_data)
            mujoco.set_mjcb_control(self.get_control)

            r1 = self._reward_reference_tracking(data, ref_data) * self.reward_scales['reference_tracking']
            r2 = self._reward_min_reference_tracking(ref_data.qpos, ref_data.qvel, data) * self.reward_scales['min_reference_tracking']
            r3 = self._reward_feet_height(
                data.geom_xpos[self._feet_ids, 2],
                ref_data.geom_xpos[self._feet_ids, 2],
            ) * self.reward_scales['feet_height']
            r4 = self._reward_base_tracking(data, ref_data) * self.reward_scales['base_tracking']
            step_reward = r1 + r2 + r3 + r4

            global total_rewards
            total_rewards += step_reward

            self._step_idx = (self._step_idx + 1) % self._l_cycle

        ##########################
        # test reference
        ##########################
        # data.qpos[7:] = self._kinematic_ref_qpos[self._step_idx]
        # data.qpos[:7] = self._default_qpos[:7]
        # if self._counter % self._n_substeps == 0:
        #     self._step_idx = (self._step_idx + 1) % self._l_cycle

        self._counter += 1

    # ----------------- Reward functions -----------------
    def _reward_reference_tracking(self, data, ref_data):
        f = lambda a, b: np.mean(np.sum((a - b) ** 2, axis=-1))
        mse_pos = f(data.xpos[1:], ref_data.xpos[1:])
        mse_rot = f(quaternion_to_rotation_6d(data.xquat[1:]), 
                    quaternion_to_rotation_6d(ref_data.xquat[1:]))
        vel = data.cvel[1:, 3:]
        ang = data.cvel[1:, :3]
        ref_vel = ref_data.cvel[1:, 3:]
        ref_ang = ref_data.cvel[1:, :3]
        mse_vel = f(vel, ref_vel)
        mse_ang = f(ang, ref_ang)
        return mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, data):
        pos = np.concatenate([data.qpos[:3], data.qpos[7:]])
        pos_targ = np.concatenate([ref_qpos[:3], ref_qpos[7:]])
        pos_err = np.linalg.norm(pos_targ - pos)
        vel_err = np.linalg.norm(data.qvel - ref_qvel)
        return pos_err + vel_err

    def _reward_feet_height(self, feet_z, feet_z_ref):
        return np.sum(np.abs(feet_z - feet_z_ref))

    def _reward_base_tracking(self, data, ref_data):
        pos_err = np.linalg.norm(data.xpos[1] - ref_data.xpos[1])
        q = data.xquat[1]
        q_ref = ref_data.xquat[1]
        dot = abs(np.dot(q, q_ref))
        dot = np.clip(dot, -1.0, 1.0)
        rot_err = np.arccos(2 * dot**2 - 1)
        vel_err = np.linalg.norm(data.cvel[1] - ref_data.cvel[1])
        return pos_err + 0.5 * rot_err + 0.1 * vel_err


# ---------------------------------------------------------------------
# MuJoCo Load Callback
# ---------------------------------------------------------------------
def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    xml_path = consts.MJX_XML_PATH.as_posix()
    # xml_path = consts.MUJOCO_XML_PATH.as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    Kp = 80.0
    Kd = 0.5
    model.opt.timestep = sim_dt
    model.dof_damping[6:] = Kd
    model.actuator_gainprm[:, 0] = Kp
    model.actuator_biasprm[:, 1] = -Kp

    controller = Go2JaxController(
        env_name="Go2Trot",
        policy_path="/tmp/trotting_apg_2hz_policy",
        # policy_path="/tmp/trotting_ppo_2hz_policy",
        default_qpos=np.array(model.keyframe("home").qpos),
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=np.array([0.2, 0.8, 0.8] * 4),
        alg_name='apg'
        # alg_name='ppo'
    )

    mujoco.set_mjcb_control(controller.get_control)
    renderer = mujoco.Renderer(model, height=480, width=640)

    return model, data, renderer


# ---------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # viewer.launch(loader=load_callback)
    model, data, renderer = load_callback()
    print("Starting headless simulation...")
    frames = []

    run_time = 10.0
    while data.time < run_time:  # 运行10秒
        # 控制回调会自动执行
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)
        if len(frames) % 100 == 0:
             print(f"Time: {data.time:.2f}/{run_time} s, Frames collected: {len(frames)}")

    print(f"Simulation finished. Total frames: {len(frames)}")
    print("Total reward collected:", total_rewards)
    print()
    media.write_video("go2_trot_server.mp4", frames, fps=int(1/model.opt.timestep))
    print("Video saved as go2_trot_server.mp4")

    renderer.close()
