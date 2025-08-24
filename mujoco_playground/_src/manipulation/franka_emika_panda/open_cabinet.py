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
"""Open a cabinet."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import mujoco  # pylint: disable=unused-import
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


def default_config() -> config_dict.ConfigDict:
  """Returns the default configuration for the environment."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the barrier.
              no_barrier_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
          )
      ),
      impl="jax",
      nconmax=12 * 8192,
      njmax=96,
  )


class PandaOpenCabinet(panda.PandaBase):
  """Environment for training the Franka Panda robot to bring an object to a
  target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,  # pylint: disable=unused-argument
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_cabinet.xml"
    )
    super().__init__(xml_path, config, config_overrides)

    # Enable hand base collision to shape learning
    self.mj_model.geom("hand_capsule").conaffinity = 3
    self._mjx_model = mjx.put_model(self.mj_model, impl=self._config.impl)

    self._post_init(obj_name="handle", keyframe="upright")
    self._barrier_geom = self._mj_model.geom("barrier").id

    # Contact sensor IDs.
    self._barrier_hand_found_sensor = [
        self._mj_model.sensor(f"barrier_{geom}_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng_target = jax.random.split(rng)

    # Initialize target position
    target_pos = jp.array([0.3, 0.0, 0.5])
    target_pos = target_pos.at[0].set(
        target_pos[0] + jax.random.uniform(rng_target, minval=-0.1, maxval=0.1)
    )

    rng, rng_arm = jax.random.split(rng)

    # Set initial qpos. Arm joints only.
    eps = jp.deg2rad(30)
    perturb_mins = jp.array([-eps, -eps, -eps, -2 * eps, -eps, 0, -eps])
    perturb_maxs = jp.array([eps, eps, eps, 0, eps, 2 * eps, eps])

    perturb_arm = jax.random.uniform(
        rng_arm, (7,), minval=perturb_mins, maxval=perturb_maxs
    )

    init_q = jp.array(self._init_q).at[:7].set(self._init_q[:7] + perturb_arm)
    init_ctrl = (
        jp.array(self._init_ctrl).at[:7].set(self._init_ctrl[:7] + perturb_arm)
    )

    data: mjx.Data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv),
        ctrl=init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    info = {
        "rng": rng,
        "reached_box": 0.0,
        "target_pos": target_pos,
        "target_mat": data.xmat[self._obj_body],  # For visualisation only
        "previously_gripped": 0.0,
    }
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0),
        **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
    }
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    """Advances the environment by one timestep."""
    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data: mjx.Data = mjx_env.step(
        self._mjx_model, state.data, ctrl, self.n_substeps
    )

    raw_rewards = self._get_rewards(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_rewards(self, data: mjx.Data, info: dict):
    # Compute reward terms
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]

    box_target = 1 - jp.tanh(5 * jp.linalg.norm(target_pos - box_pos))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))

    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the barrier
    hand_barrier_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._barrier_hand_found_sensor
    ]
    barrier_collision = sum(hand_barrier_collision) > 0
    no_barrier_collision = 1 - barrier_collision

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 1.0 * 0.012),
    )

    return {
        "box_target": box_target * info["reached_box"],
        "gripper_box": gripper_box,
        "no_barrier_collision": no_barrier_collision.astype(float),
        "robot_target_qpos": robot_target_qpos,
    }

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs
