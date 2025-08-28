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
"""RL config for Manipulation envs."""

from typing import Optional
from ml_collections import config_dict
from mujoco_playground._src import manipulation


def brax_ppo_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=env_config.action_repeat,
      reward_scaling=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(32, 32, 32, 32),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
      num_resets_per_eval=10,
  )
  if env_name == "AlohaHandOver":
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 25
    rl_config.unroll_length = 15
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.num_eval_envs = 128
    rl_config.batch_size = 512
    rl_config.max_grad_norm = 1.0
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256, 256)
  elif env_name == "AlohaSinglePegInsertion":
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 10
    rl_config.unroll_length = 40
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 1024
    rl_config.batch_size = 512
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256, 256, 256)
    if impl == "warp":
      rl_config.num_timesteps *= 3
      rl_config.num_evals *= 3
  elif env_name == "PandaOpenCabinet":
    rl_config.num_timesteps = 40_000_000
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
  elif env_name == "PandaPickCubeCartesian":
    rl_config.num_timesteps = 5_000_000
    rl_config.num_evals = 5
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 8
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 5.0e-4
    rl_config.entropy_cost = 7.5e-3
    rl_config.num_envs = 1024
    rl_config.batch_size = 256
    rl_config.reward_scaling = 0.1
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256)
    rl_config.num_resets_per_eval = 1
    rl_config.max_grad_norm = 1.0
    if impl == "warp":
      rl_config.num_timesteps *= 4
      rl_config.num_evals *= 4
  elif env_name.startswith("PandaPickCube"):
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
    if impl == "warp":
      rl_config.num_timesteps *= 4
      rl_config.num_evals *= 4
  elif env_name == "PandaRobotiqPushCube":
    rl_config.num_timesteps = 1_800_000_000
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
    rl_config.num_eval_envs = 32
    rl_config.network_factory.policy_hidden_layer_sizes = (64, 64, 64, 64)
    if impl == "warp":
      rl_config.num_resets_per_eval = 10
      rl_config.num_timesteps = int(rl_config.num_timesteps * 1.5)
      rl_config.num_evals = int(rl_config.num_evals * 1.5)
  elif env_name == "LeapCubeRotateZAxis":
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 10
    rl_config.num_minibatches = 32
    rl_config.unroll_length = 40
    rl_config.num_updates_per_batch = 4
    rl_config.discounting = 0.97
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 256
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "LeapCubeReorient":
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.num_minibatches = 32
    rl_config.unroll_length = 40
    rl_config.num_updates_per_batch = 4
    rl_config.discounting = 0.99
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 256
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    rl_config.num_resets_per_eval = 1
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config


def brax_vision_ppo_config(
    env_name: str, unused_impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax Vision PPO config for the given environment."""
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      madrona_backend=True,
      wrap_env=False,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=env_config.action_repeat,
      reward_scaling=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(32, 32, 32, 32)
      ),
      num_resets_per_eval=10,
  )

  if env_name == "PandaPickCubeCartesian":
    rl_config.num_timesteps = 5_000_000
    rl_config.num_evals = 5
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 8
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 5.0e-4
    rl_config.entropy_cost = 7.5e-3
    rl_config.num_envs = 1024
    rl_config.batch_size = 256
    rl_config.reward_scaling = 0.1
    rl_config.num_resets_per_eval = 1
  else:
    raise NotImplementedError(f"Vision PPO params not tested for {env_name}")

  return rl_config


def rsl_rl_config(env_name: str, unused_impl: Optional[str] = None) -> config_dict.ConfigDict:  # pylint: disable=unused-argument
  """Returns tuned RSL-RL PPO config for the given environment."""

  rl_config = config_dict.create(
      seed=1,
      runner_class_name="OnPolicyRunner",
      policy=config_dict.create(
          init_noise_std=1.0,
          actor_hidden_dims=[512, 256, 128],
          critic_hidden_dims=[512, 256, 128],
          # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
          activation="elu",
          class_name="ActorCritic",
      ),
      algorithm=config_dict.create(
          class_name="PPO",
          value_loss_coef=1.0,
          use_clipped_value_loss=True,
          clip_param=0.2,
          entropy_coef=0.001,
          num_learning_epochs=5,
          # mini batch size = num_envs*nsteps / nminibatches
          num_mini_batches=4,
          learning_rate=3.0e-4,  # 5.e-4
          schedule="adaptive",  # could be adaptive, fixed
          gamma=0.99,
          lam=0.95,
          desired_kl=0.01,
          max_grad_norm=1.0,
      ),
      num_steps_per_env=24,  # per iteration
      max_iterations=100000,  # number of policy updates
      empirical_normalization=True,
      # logging
      save_interval=50,  # check for potential saves every this many iterations
      experiment_name="test",
      run_name="",
      # load and resume
      resume=False,
      load_run="-1",  # -1 = last run
      checkpoint=-1,  # -1 = last saved model
      resume_path=None,  # updated from load_run and chkpt
  )

  return rl_config
