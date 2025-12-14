from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import os
import mediapy as media
import tqdm
# Math
import jax.numpy as jp
import numpy as np
import jax

import mujoco

from mujoco_playground._src.locomotion.go2.TrotUtil import (
    cos_wave, dcos_wave, make_kinematic_ref,
    quaternion_to_matrix, matrix_to_rotation_6d,
    quaternion_to_rotation_6d,
    rotate, rotate_inv
)

from mujoco_playground._src.locomotion.go2 import go2_constants as consts

# Sim
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env

try:
    from mujoco_playground._src.mjx_env import make_data
except ImportError:
    from typing import Optional
    import jax
    import mujoco
    from mujoco import mjx
    def make_data(
        model: mujoco.MjModel,
        qpos: Optional[jax.Array] = None,
        qvel: Optional[jax.Array] = None,
        ctrl: Optional[jax.Array] = None,
        act: Optional[jax.Array] = None,
        mocap_pos: Optional[jax.Array] = None,
        mocap_quat: Optional[jax.Array] = None,
        impl: Optional[str] = None,
        nconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        device: Optional[jax.Device] = None, # type: ignore
    ) -> mjx.Data:
        """Initialize MJX Data."""
        data = mjx.make_data(
            model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
        )
        if qpos is not None:
            data = data.replace(qpos=qpos)
        if qvel is not None:
            data = data.replace(qvel=qvel)
        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        if act is not None:
            data = data.replace(act=act)
        if mocap_pos is not None:
            data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
        if mocap_quat is not None:
            data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
        return data

from mujoco_playground._src.locomotion.go2.base import Go2Env

# Supporting
from ml_collections import config_dict
from typing import Any, Dict


# ----------------- default config -----------------
def default_config() -> config_dict.ConfigDict:
    # 注意：MjxEnv 要求 config 里必须包含 sim_dt 和 ctrl_dt
    cfg = config_dict.ConfigDict()
    cfg.Kp = 80.0          # PD 控制器的比例增益
    cfg.Kd = 0.5           # PD 控制器的微分增益
    cfg.sim_dt = 0.002          # 物理仿真步长（s）
    cfg.ctrl_dt = 0.02          # 控制步长（s） => n_frames = ctrl_dt / sim_dt = 10
    cfg.episode_length=240
    # 环境超参
    cfg.env = config_dict.ConfigDict()
    cfg.env.termination_height = 0.1
    cfg.env.step_k = 13         # 每条腿抬起/落下的子步数量
    cfg.env.err_threshold = 0.1
    cfg.env.action_scale = [0.2, 0.8, 0.8] * 4  # 每条腿3个关节，共4条腿
    cfg.env.reset2ref = True
    cfg.env.reference_state_init = False # RSI: Deepmimic
    cfg.env.impratio = 100
    # 扰动配置
    cfg.pert_config = config_dict.ConfigDict()
    cfg.pert_config.enable = False
    cfg.pert_config.velocity_kick = [0.0, 3.0]
    cfg.pert_config.kick_durations = [0.05, 0.2]
    cfg.pert_config.kick_wait_times = [1.0, 3.0]
    # 奖励权重
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    cfg.rewards.scales.min_reference_tracking = -2.5 * 3e-3
    cfg.rewards.scales.reference_tracking = -10.0
    cfg.rewards.scales.feet_height = -10.0
    cfg.rewards.scales.base_tracking = -1.0
    # 其他
    cfg.impl = "jax"
    cfg.nconmax = 4 * 8192
    cfg.njmax = 40
    return cfg


# ----------------- Env -----------------
class TrotGo2(Go2Env):
    """
    MJX-based TrotGo2 environment.
    Signature required by locomotion: __init__(self, config, config_overrides=None)
    """

    def __init__(self,
                 task: str = None, 
                 config: config_dict.ConfigDict = default_config(), 
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        # default_xml = os.path.normpath(
        #     os.path.join(CURRENT_DIR, "xmls", "scene_mjx_collision_free.xml")
        # )
        default_xml = consts.MJX_XML_PATH.as_posix()
        super().__init__(
            xml_path=default_xml,
            config=config,
            config_overrides=config_overrides,
        )

        self._post_init()

    def _post_init(self):    
        # 基本姿态 / 初始 qpos
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos.copy())
        self._default_ap_pose = jp.array(self._mj_model.keyframe("home").qpos[7:].copy())

        # actions limits
        self.lowers, self.uppers = self.mj_model.jnt_range[1:].T

        # 动作中心与缩放（3 joints per leg）
        self.action_loc = jp.array(self._default_ap_pose)
        self.action_scale = jp.array(self._config.env.action_scale)

        # 其他参数
        self.termination_height = float(getattr(self._config.env, "termination_height", 0.1))
        self.err_threshold = self._config.env.err_threshold
        self.reward_config = self._config.rewards
        self.feet_inds = jp.array( [self._mj_model.geom(name).id for name in consts.FEET_GEOMS] )
        # print("Feet geom ids:", self.feet_inds)
        self.base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_mass = self.mj_model.body("base").mass

        # imitation reference
        step_k = int(getattr(self._config.env, "step_k", 25))
        kinematic_ref_qpos = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=self.dt)
        kinematic_ref_qvel = make_kinematic_ref(dcos_wave, step_k, scale=0.3, dt=self.dt)
        self.l_cycle = int(kinematic_ref_qpos.shape[0])

        kinematic_ref_qpos = np.array(kinematic_ref_qpos) + np.array(self._default_ap_pose)
        ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
        ref_qs[:, 7:] = kinematic_ref_qpos
        self.kinematic_ref_qpos = jp.array(ref_qs)

        ref_qvels = np.zeros((self.l_cycle, 18))
        ref_qvels[:, 6:] = np.array(kinematic_ref_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvels)

        self.reset2ref = self._config.env.reset2ref
        self.reference_state_init = self._config.env.reference_state_init

    # -------- Envs API: reset/step ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # RSI
        if self.reference_state_init:
            rng, step_rng = jax.random.split(rng)
            init_step = jax.random.randint(step_rng, (), 0, self.l_cycle)
            qpos = self.kinematic_ref_qpos[init_step]
            qvel = self.kinematic_ref_qvel[init_step]
        else:
            # Deterministic init
            init_step = 0
            qpos = self._init_q
            qvel = jp.zeros(self.mjx_model.nv)

        # 创建 mjx data
        data = make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # 将机器人放到地面上（和原始 reset 一样）
        pen = jp.where(data.ncon > 0, jp.min(data._impl.contact.dist), 0.0)
        qpos = qpos.at[2].set(qpos[2] - pen)

        data = make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self._config.pert_config.kick_wait_times[0],
            maxval=self._config.pert_config.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
            jp.int32
        )
        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self._config.pert_config.kick_durations[0],
            maxval=self._config.pert_config.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        )
        pert_mag = jax.random.uniform(
            key3,
            minval=self._config.pert_config.velocity_kick[0],
            maxval=self._config.pert_config.velocity_kick[1],
        )

        # state_info 保持和原版一致
        state_info = {
            'rng': rng,
            'step': jp.array(init_step, dtype=jp.float32),
            'reward_tuple': {
                'reference_tracking': 0.0,
                'min_reference_tracking': 0.0,
                'feet_height': 0.0,
                'base_tracking': 0.0
            },
            'last_action': jp.zeros(self.mjx_model.nu),  # 12 通道动作
            'kinematic_ref': qpos,

            "steps_until_next_pert": steps_until_next_pert,
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
        }

        # 生成 obs
        obs = self._get_obs(data, state_info)

        # 初始化 reward 和 metrics
        reward, done = jp.zeros(2)
        metrics = {}
        for k in state_info['reward_tuple']:
            metrics[k] = state_info['reward_tuple'][k]

        # 返回 mjx_env.State
        state = mjx_env.State(data, obs, reward, done, metrics, state_info)
        return jax.lax.stop_gradient(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # TODO 扰动
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)
    
        action = jp.clip(action, -1, 1)
        ctrl = self.action_loc + (action * self.action_scale)

        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )

        step_idx = jp.array(state.info["step"] % self.l_cycle, int)
        ref_qpos = self.kinematic_ref_qpos[step_idx]
        ref_qvel = self.kinematic_ref_qvel[step_idx]

        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.mjx_model, ref_data)

        state.info["kinematic_ref"] = ref_qpos

        obs = self._get_obs(data, state.info)

        # 结束条件
        base_z = data.xpos[self.base_id, 2]
        done = jp.where(base_z < self.termination_height, 1.0, 0.0)
        R_base = quaternion_to_matrix(data.xquat[1])
        up = jp.array([0.0, 0.0, 1.0])
        base_z_axis_world = R_base @ up
        done = jp.where(jp.dot(base_z_axis_world, up) < 0.0, 1.0, done)

        # 奖励
        reward_tuple = dict(
            reference_tracking=self._reward_reference_tracking(data, ref_data) 
            * self.reward_config.scales.reference_tracking,
            min_reference_tracking=self._reward_min_reference_tracking(ref_qpos, ref_qvel, data)
            * self.reward_config.scales.min_reference_tracking,
            feet_height=self._reward_feet_height(data.geom_xpos[self.feet_inds][:, 2], ref_data.geom_xpos[self.feet_inds][:, 2])
            * self.reward_config.scales.feet_height,
            base_tracking=self._reward_base_tracking(data, ref_data)
            * self.reward_config.scales.base_tracking,
        )

        state.info["last_action"] = ctrl

        if self.reset2ref:
            err = (((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5).mean()
            to_ref = err > self.err_threshold

            reward_tuple['reference_tracking'] *= jax.numpy.where(to_ref, 10.0, 1.0)
            reward_tuple['base_tracking'] *= jax.numpy.where(to_ref, 10.0, 1.0)
            reward = sum(reward_tuple.values())
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]
            state.info["reward_tuple"] = reward_tuple

            def safe_select(a, b):
                return jp.where(to_ref, b, a)
            data_blend = jax.tree_util.tree_map(safe_select, data, ref_data)

            obs = self._get_obs(data_blend, state.info)
            state.info["step"] = state.info["step"] + 1.0

            return state.replace(data=data_blend, obs=obs, reward=reward, done=done)
        
        else:
            reward = sum(reward_tuple.values())
            state.info["reward_tuple"] = reward_tuple
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]

            state.info["step"] = state.info["step"] + 1.0

            return state.replace(data=data, obs=obs, reward=reward, done=done)
        
    def play_ref_motion(self, render_every: int = 2, seed: int = 0):
        """
        Play the built-in kinematic reference trajectory as an animation.

        Args:
            render_every: render every Nth frame (for speed)
            seed: random seed used for reset (to initialize state)
        Returns:
            frames: list of rendered RGB frames
        """
        print("Playing reference motion...")

        rng = jax.random.PRNGKey(seed)
        state = self.reset(rng)
        data = state.data

        traj = []
        for i in range(0, self.l_cycle, render_every):
            qpos = self.kinematic_ref_qpos[i]
            qvel = self.kinematic_ref_qvel[i]

            data = data.replace(qpos=qpos, qvel=qvel)
            data = mjx.forward(self.mjx_model, data)
            ref_state = state.replace(
                data=data,
                obs=None,
                reward=0.0,
                done=0.0,
                info={"step": float(i)},
            )
            traj.append(ref_state)

        fps = 1.0 / (self.dt * render_every)
        frames = self.render(traj, height=480, width=640)
        media.show_video(frames, fps=fps, loop=True)

        print(f"Rendered {len(frames)} frames at {fps:.1f} FPS.")
        return frames


    # -------- obs & reward helpers ----------
    def _get_obs(self, data, state_info: Dict[str, Any]):
        local_omega = data.cvel[1, :3]
        yaw_rate = local_omega[2]
        g_world = jp.array([0.0, 0.0, -1.0])
        g_local = rotate_inv(g_world, data.xquat[1])
        angles = data.qpos[7:19]
        last_action = state_info["last_action"]
        step_idx = jp.array(state_info["step"] % self.l_cycle, int)
        kin_ref = self.kinematic_ref_qpos[step_idx][7:]
        obs_list = [jp.array([yaw_rate]) * 0.25, g_local, angles - jp.array(self._default_ap_pose), last_action, kin_ref]
        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)
        return obs

    def _reward_reference_tracking(self, data, ref_data):
        f = lambda a, b: ((a - b) ** 2).sum(-1).mean()
        mse_pos = f(data.xpos[1:], ref_data.xpos[1:])
        mse_rot = f(quaternion_to_rotation_6d(data.xquat[1:]), quaternion_to_rotation_6d(ref_data.xquat[1:]))
        vel = data.cvel[1:, 3:]
        ang = data.cvel[1:, :3]
        ref_vel = ref_data.cvel[1:, 3:]
        ref_ang = ref_data.cvel[1:, :3]
        mse_vel = f(vel, ref_vel)
        mse_ang = f(ang, ref_ang)
        return mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, data):
        pos = jp.concatenate([data.qpos[:3], data.qpos[7:]])
        pos_targ = jp.concatenate([ref_qpos[:3], ref_qpos[7:]])
        pos_err = jp.linalg.norm(pos_targ - pos)
        vel_err = jp.linalg.norm(data.qvel - ref_qvel)
        return pos_err + vel_err

    def _reward_feet_height(self, feet_z, feet_z_ref):
        return jp.sum(jp.abs(feet_z - feet_z_ref))
    
    def _reward_base_tracking(self, data, ref_data):
        pos_err = jp.linalg.norm(data.xpos[1] - ref_data.xpos[1])
        q = data.xquat[1]
        q_ref = ref_data.xquat[1]
        dot = jp.abs(jp.dot(q, q_ref))  # q and -q are same rotation
        dot = jp.clip(dot, -1.0, 1.0)
        rot_err = jp.arccos(2 * dot**2 - 1)
        vel_err = jp.linalg.norm(data.cvel[1] - ref_data.cvel[1])
        return pos_err + 0.5 * rot_err + 0.1 * vel_err
    
    # ----------------- Disturbance -----------------
    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        def apply_pert(state: mjx_env.State) -> mjx_env.State:
            t = state.info["pert_steps"] * self.dt
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self.base_mass  # kg
                * state.info["pert_mag"]  # m/s
                / state.info["pert_duration_seconds"]  # 1/s
            )
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_id, :3].set(
                force * state.info["pert_dir"]
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_last_pert"] = jp.where(
                state.info["pert_steps"] >= state.info["pert_duration"],
                0,
                state.info["steps_since_last_pert"],
            )
            state.info["pert_steps"] += 1
            return state

        def wait(state: mjx_env.State) -> mjx_env.State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state.info["pert_steps"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                0,
                state.info["pert_steps"],
            )
            state.info["pert_dir"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                gen_dir(rng),
                state.info["pert_dir"],
            )
            return state.replace(data=data)

        return jax.lax.cond(
            state.info["steps_since_last_pert"]
            >= state.info["steps_until_next_pert"],
            apply_pert,
            wait,
            state,
        )


# # ----------------- 注册到 playground -----------------
# locomotion.register_environment(
#     'TrotAnymal',   # 环境名字
#     TrotAnymal,     # 环境类
#     default_config      # 默认配置函数
# )

