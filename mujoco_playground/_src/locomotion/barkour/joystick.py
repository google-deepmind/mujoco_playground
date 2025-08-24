# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick environment for Barkour."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

_FEET_SITES = [
    "foot_front_left",
    "foot_hind_left",
    "foot_front_right",
    "foot_hind_right",
]
_FEET_GEOMS = [
    "foot_front_left",
    "foot_hind_left",
    "foot_front_right",
    "foot_hind_right",
]


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "google_barkour_vb"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.3,
      obs_noise=0.05,
      lin_vel_x=[-0.6, 1.5],
      lin_vel_y=[-0.8, 0.8],
      ang_vel_yaw=[-0.7, 0.7],
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards are computed using exp(-delta^2/sigma)
              # sigma can be a hyperparameters to tune.
              # Track the base x-y velocity (no z-velocity tracking.)
              tracking_lin_vel=1.5,
              # Track the angular velocity along z-axis, i.e. yaw rate.
              tracking_ang_vel=0.8,
              # Below are regularization terms, we roughly divide the
              # terms to base state regularizations, joint
              # regularizations, and other behavior regularizations.
              # Penalize the base velocity in z direction, L2 penalty.
              lin_vel_z=-2.0,
              # Penalize the base roll and pitch rate. L2 penalty.
              ang_vel_xy=-0.05,
              # Penalize non-zero roll and pitch angles. L2 penalty.
              orientation=-5.0,
              # L2 regularization of joint torques, |tau|^2.
              torques=-0.0002,
              # Penalize the change in the action and encourage smooth
              # actions. L2 regularization |action - last_action|^2
              action_rate=-0.1,
              # Encourage no motion at zero command, L2 regularization
              # |q - q_default|^2.
              stand_still=-0.5,
              # Early termination penalty.
              termination=-1.0,
              # Encourage long swing steps.  However, it does not
              # encourage high clearances.
              feet_air_time=0.2,
          ),
          # Tracking reward = exp(-error^2/sigma).
          tracking_sigma=0.25,
      ),
      # Velocity of perturbation kick applied to the base, in m/s.
      velocity_kick=[0.1, 1.0],
      # Minimum and maximum duration (env steps) for perturbation kicks.
      kick_duration_steps=[1, 10],
      # Minimum and maximum wait steps before next perturbation kick.
      # A duration will be sampled uniformly from this range at the beginning of
      # each episode.
      kick_wait_steps=[50, 150],
      impl="jax",
      nconmax=4 * 8192,
      njmax=12 + 4 * 4,
  )


class Joystick(mjx_env.MjxEnv):
  """Joystick environment for Barkour."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    xml_path = mjx_env.MENAGERIE_PATH / "google_barkour_vb" / "scene_mjx.xml"
    self._xml_path = xml_path.as_posix()
    self._model_assets = get_assets()
    mj_spec = mujoco.MjSpec.from_file(
        xml_path.as_posix(), assets=self._model_assets
    )
    # add contact sensors
    feet_floor_found_sensor = []
    for geom in _FEET_GEOMS:
      name = f"{geom}_floor_found"
      mj_spec.add_sensor(
          name=name,
          type=mujoco.mjtSensor.mjSENS_CONTACT,
          objtype=mujoco.mjtObj.mjOBJ_GEOM,
          objname=geom,
          reftype=mujoco.mjtObj.mjOBJ_GEOM,
          refname="floor",
          intprm=[1, 1, 1],  # data=found, reduce=mindist
          datatype=mujoco.mjtDataType.mjDATATYPE_REAL,
          needstage=mujoco.mjtStage.mjSTAGE_ACC,
          dim=1,
      )
      feet_floor_found_sensor.append(name)
    self._feet_floor_found_sensor = feet_floor_found_sensor
    # compile spec
    mj_model = mj_spec.compile()
    mj_model.vis.global_.offwidth = 3840
    mj_model.vis.global_.offheight = 2160
    mj_model.dof_damping[6:] = 0.5239
    mj_model.actuator_gainprm[:, 0] = 35.0
    mj_model.actuator_biasprm[:, 1] = -35.0

    self._mj_model = mj_model
    self._mj_model.opt.timestep = config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self._lowers = jp.array([-0.7, -1.0, 0.05] * 4)
    self._uppers = jp.array([0.52, 2.1, 2.1] * 4)
    self._torso_body_id = self._mj_model.body("torso").id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in _FEET_SITES]
    )
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in _FEET_GEOMS]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._foot_radius = 0.0175

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3 = jax.random.split(rng, 3)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    return jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng = jax.random.split(rng, 3)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    kick_wait_steps = jax.random.randint(
        key1,
        (1,),
        minval=self._config.kick_wait_steps[0],
        maxval=self._config.kick_wait_steps[1],
    )
    kick_duration_steps = jax.random.randint(
        key2,
        (1,),
        minval=self._config.kick_duration_steps[0],
        maxval=self._config.kick_duration_steps[1],
    )
    vel_kick = jax.random.uniform(
        key3,
        minval=self._config.velocity_kick[0],
        maxval=self._config.velocity_kick[1],
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_vel": jp.zeros(self.mjx_model.nv - 6),
        "command": self.sample_command(cmd_rng),
        "last_contact": jp.zeros(len(_FEET_SITES), dtype=bool),
        "feet_air_time": jp.zeros(len(_FEET_SITES)),
        "kick_wait_steps": kick_wait_steps,
        "kick_duration_steps": kick_duration_steps,
        "vel_kick": vel_kick,
        "kick_dir": jp.zeros(3, dtype=float),
        "last_kick_step": jp.array([-jp.inf], dtype=float),
        "step": 0,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(15 * 31)  # store 15 steps of history
    obs = self._get_obs(data, info, obs_history, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng, cmd_rng, noise_rng, pert_rng = jax.random.split(state.info["rng"], 4)

    state = self._maybe_apply_perturbation(state, pert_rng)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info, state.obs, noise_rng)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    torso_z = data.xpos[self._torso_body_id, -1]

    contact = jp.array([
        data.sensordata[
            self._mj_model.sensor_adr[self._mj_model.sensor(sensor).id]
        ]
        > 0
        for sensor in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    done = self._get_gravity(data)[-1] < 0
    done |= jp.any(joint_angles < self._lowers)
    done |= jp.any(joint_angles > self._uppers)
    done |= torso_z < 0.18

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Bookkeeping.
    state.info["last_act"] = action
    state.info["last_vel"] = joint_vel
    state.info["step"] += 1
    state.info["rng"] = rng
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      obs_history: jax.Array,
      rng: jax.Array,
  ) -> jax.Array:
    obs = jp.concatenate([
        self._get_localrpyrate(data)[-1].reshape(1) * 0.25,
        self._get_gravity(data),
        info["command"] * jp.array([2.0, 2.0, 0.25]),
        data.qpos[7:] - self._default_pose,
        info["last_act"],
    ])
    obs = jp.clip(obs, -100.0, 100.0)
    if self._config.obs_noise >= 0.0:
      noise = self._config.obs_noise * jax.random.uniform(
          rng, obs.shape, minval=-1.0, maxval=1.0
      )
      obs += noise
    obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
    return obs

  def _get_gravity(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "upvector")

  def _get_localrpyrate(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "gyro")

  def _get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_linvel")

  def _get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_angvel")

  def _get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "local_linvel")

  def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
    sensor_id = self._mj_model.sensor(sensor_name).id
    sensor_adr = self._mj_model.sensor_adr[sensor_id]
    sensor_dim = self._mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

  # Reward functions.

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self._get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self._get_localrpyrate(data)
        ),
        # Regularization rewards.
        "lin_vel_z": self._cost_lin_vel_z(self._get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self._get_global_angvel(data)),
        "orientation": self._cost_orientation(self._get_gravity(data)),
        "torques": self._cost_torques(data.qfrc_actuator),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done, info["step"]),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
    }

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions.
    return jp.sum(jp.square(act - last_act))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands.
    unit_cmd = commands[:2] / jp.linalg.norm(commands[:2])
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        unit_cmd[1] < 0.1
    )

  def _cost_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands[:2])
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.05  # No reward for zero commands.
    return rew_air_time

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def get_xfrc(
        state: mjx_env.State, kick_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["kick_duration_steps"])
      force = (
          u_t
          * self._torso_mass
          * state.info["vel_kick"]
          / (state.info["kick_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * kick_dir
      )
      return xfrc_applied

    step, last_kick_step = state.info["step"], state.info["last_kick_step"]
    start_kick = jp.mod(step, state.info["kick_wait_steps"]) == 0
    start_kick &= step != 0  # No kick at the beginning of the episode.
    last_kick_step = jp.where(start_kick, step, last_kick_step)
    duration = jp.clip(step - last_kick_step, 0, 100_000)
    in_kick_interval = duration < state.info["kick_duration_steps"]

    kick_dir = jp.where(start_kick, gen_dir(rng), state.info["kick_dir"])
    xfrc = get_xfrc(state, kick_dir, duration) * in_kick_interval

    state.info["kick_dir"] = kick_dir
    state.info["last_kick_step"] = last_kick_step
    return state.tree_replace({"data.xfrc_applied": xfrc})

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
