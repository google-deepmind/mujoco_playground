# Copyright 2026 OneRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OneRobotics A1 end-effector position reaching task.

The target is the FK position of a randomized, joint-limited A1 configuration.
The policy controls seven position actuators with incremental joint targets.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.onerobotics_a1 import a1


def default_config() -> config_dict.ConfigDict:
  """A1 position-only Reach task configuration."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.05,
      target_joint_range=0.35,
      success_threshold=0.03,
      reward_config=config_dict.create(
          scales=config_dict.create(position=1.0, success=2.0)
      ),
      impl="jax",
      naconmax=128,
      njmax=128,
  )


class A1Reach(a1.A1Base):
  """Move the A1 right-arm end effector to a reachable 3-D target.

  Rewards:
    position: 1 - tanh(5 * ||ee_position - target_position||)
    success: 1 if the distance is below success_threshold, else 0.

  Both terms are multiplied by reward_config.scales.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, target_rng = jax.random.split(rng)
    home_arm_q = self._home_qpos[self._arm_qpos_ids]
    target_arm_q = home_arm_q + jax.random.uniform(
        target_rng,
        (self.action_size,),
        minval=-self._config.target_joint_range,
        maxval=self._config.target_joint_range,
    )
    target_arm_q = jp.clip(
        target_arm_q,
        self._joint_low,
        self._joint_high,
    )
    goal_q = self._home_qpos.at[self._arm_qpos_ids].set(target_arm_q)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=self._home_qpos,
        qvel=jp.zeros(self._mjx_model.nv),
        ctrl=self._home_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    # Calculate a kinematically reachable target inside the jitted reset.
    goal_data = mjx.kinematics(self._mjx_model, data.replace(qpos=goal_q))
    target_pos = goal_data.site_xpos[self._ee_site_id]

    # The extra mocap body is visual-only and does not affect arm kinematics.
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target_pos)
    )
    data = mjx.kinematics(self._mjx_model, data)
    info = {"rng": rng, "target_pos": target_pos}
    distance = jp.linalg.norm(data.site_xpos[self._ee_site_id] - target_pos)
    metrics = {
        "distance": distance,
        "success": (distance < self._config.success_threshold).astype(float),
        "reward/position": jp.array(0.0),
        "reward/success": jp.array(0.0),
    }
    return mjx_env.State(
        data=data,
        obs=self._get_obs(data, info),
        reward=jp.array(0.0),
        done=jp.array(0.0),
        metrics=metrics,
        info=info,
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    desired = state.data.ctrl[self._actuator_ids] + (
        action * self._config.action_scale
    )
    desired = jp.clip(desired, self._control_low, self._control_high)
    ctrl = state.data.ctrl.at[self._actuator_ids].set(desired)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    distance = jp.linalg.norm(
        data.site_xpos[self._ee_site_id] - state.info["target_pos"]
    )
    position_reward = 1.0 - jp.tanh(5.0 * distance)
    success = (distance < self._config.success_threshold).astype(float)
    reward_position = (
        self._config.reward_config.scales.position * position_reward
    )
    reward_success = self._config.reward_config.scales.success * success
    reward = reward_position + reward_success
    done = (
        jp.any(~jp.isfinite(data.qpos))
        | jp.any(~jp.isfinite(data.qvel))
        | jp.any(~jp.isfinite(data.ctrl))
    ).astype(float)

    # Wrappers add their own metric keys (for example, episode reward).  Keep
    # the existing dictionary structure stable for Brax's jitted lax.scan.
    state.metrics.update({
        "distance": distance,
        "success": success,
        "reward/position": reward_position,
        "reward/success": reward_success,
    })
    return mjx_env.State(
        data=data,
        obs=self._get_obs(data, state.info),
        reward=reward,
        done=done,
        metrics=state.metrics,
        info=state.info,
    )

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    ee_position = data.site_xpos[self._ee_site_id]
    target_position = info["target_pos"]
    return jp.concatenate([
        data.qpos[self._arm_qpos_ids],
        data.qvel[self._arm_qvel_ids],
        ee_position,
        target_position,
        target_position - ee_position,
        data.ctrl[self._actuator_ids],
    ])
