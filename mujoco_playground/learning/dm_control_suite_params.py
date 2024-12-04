# Copyright 2024 DeepMind Technologies Limited
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
"""RL config for DM Control Suite."""

from ml_collections import config_dict
from mujoco_playground._src import dm_control_suite


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=60_000_000,
      num_evals=10,
      reward_scaling=10.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=30,
      num_minibatches=32,
      num_updates_per_batch=16,
      discounting=0.995,
      learning_rate=1e-3,
      entropy_cost=1e-2,
      num_envs=2048,
      batch_size=1024,
  )

  if env_name.startswith('AcrobotSwingup'):
    rl_config.num_timesteps = 100_000_000
  if env_name == 'BallInCup':
    rl_config.discounting = 0.95
  elif env_name.startswith('Swimmer'):
    rl_config.num_timesteps = 100_000_000
  elif env_name == 'WalkerRun':
    rl_config.num_timesteps = 100_000_000
  elif env_name == 'FingerSpin':
    rl_config.discounting = 0.95
  elif env_name == 'PendulumSwingUp':
    rl_config.action_repeat = 4
    rl_config.num_updates_per_batch = 4

  return rl_config


def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == 'PendulumSwingUp':
    rl_config.action_repeat = 4

  if (
      env_name.startswith('Acrobot')
      or env_name.startswith('Swimmer')
      or env_name.startswith('Finger')
      or env_name.startswith('Hopper')
      or env_name
      in ('CheetahRun', 'HumanoidWalk', 'PendulumSwingUp', 'WalkerRun')
  ):
    rl_config.num_timesteps = 10_000_000

  return rl_config
