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
"""RL config for Manipulation envs."""

from ml_collections import config_dict
from mujoco_playground._src import manipulation


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=env_config.action_repeat,
      reward_scaling=1.0,
      network_factory=config_dict.create(),
  )

  if env_name.startswith('Aloha'):
    rl_config.num_timesteps = 60_000_000
    rl_config.num_evals = 10
    rl_config.unroll_length = 40
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 1024
    rl_config.batch_size = 128
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256, 256, 256)
  elif env_name.startswith('PandaBring'):
    rl_config.num_timesteps = 20_000_000
    rl_config.num_evals = 4
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.batch_size = 512
    rl_config.network_factory.policy_hidden_layer_sizes = (32, 32, 32, 32)
  elif env_name == 'PandaRobotiqPushCube':
    rl_config.num_timesteps = 2_000_000_000
    rl_config.num_evals = 10
    rl_config.unroll_length = 100
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.994
    rl_config.learning_rate = 6e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 512
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory.policy_hidden_layer_sizes = (64, 64, 64, 64)
    return rl_config
  else:
    raise ValueError(f'Unsupported env: {env_name}')

  return rl_config
