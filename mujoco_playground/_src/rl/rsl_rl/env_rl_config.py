# Copyright 2026 Google LLC
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

"""Environment-specific configurations."""

import dataclasses

from mujoco_playground._src.rl.rsl_rl import rsl_config as rsl_config


@dataclasses.dataclass
class CartpoleRslRlOnPolicyRunnerCfg(rsl_config.RslRlOnPolicyRunnerCfg):
  """Config for Cartpole."""

  max_iterations: int = 200

  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["policy"],
          "critic": ["policy"],
      }
  )

  def __post_init__(self):
    self.algorithm.entropy_coef = 0.001


@dataclasses.dataclass
class CartpoleVisionRslRlOnPolicyRunnerCfg(rsl_config.RslRlOnPolicyRunnerCfg):
  """Config for Cartpole Vision (balance) with CNN encoder."""

  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["dummy", "policy"],
          "critic": ["dummy", "policy"],
      }
  )
  nworld: int = 1024
  max_iterations: int = 500
  num_steps_per_env: int = 24
  save_interval: int = 50

  def __post_init__(self):
    cnn_cfg = rsl_config.RslRlCNNCfg(
        output_channels=(16, 32),
        kernel_size=(5, 3),
        stride=(2, 2),
        padding="zeros",
        activation="elu",
        norm="batch",
        max_pool=True,
        global_pool="none",
    )
    self.actor = rsl_config.RslRlCNNModelCfg(
        hidden_dims=(128, 64),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=rsl_config.RslRlDistributionCfg(init_std=1.0),
        cnn_cfg={"policy": cnn_cfg},
    )
    self.critic = rsl_config.RslRlCNNModelCfg(
        hidden_dims=(128, 64),
        activation="elu",
        obs_normalization=False,
        cnn_cfg={"policy": cnn_cfg},
    )

    self.algorithm.value_loss_coef = 1.0
    self.algorithm.use_clipped_value_loss = True
    self.algorithm.clip_param = 0.2
    self.algorithm.entropy_coef = 0.005
    self.algorithm.num_learning_epochs = 5
    self.algorithm.num_mini_batches = 4
    self.algorithm.learning_rate = 1e-3
    self.algorithm.schedule = "adaptive"
    self.algorithm.gamma = 0.997
    self.algorithm.lam = 0.95
    self.algorithm.desired_kl = 0.01
    self.algorithm.max_grad_norm = 1.0


@dataclasses.dataclass
class PickCubeRslRlOnPolicyRunnerCfg(rsl_config.RslRlOnPolicyRunnerCfg):
  """Config for state-based PickCube."""

  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["policy"],
          "critic": ["critic"],
      }
  )
  vecenv_wrapper_cfg: rsl_config.RslRlVecEnvWrappercfg = dataclasses.field(
      default_factory=lambda: rsl_config.RslRlVecEnvWrappercfg(
          is_finite_horizon=False
      )
  )
  nworld: int = 8192
  max_iterations: int = 1000
  num_steps_per_env: int = 24
  save_interval: int = 100

  def __post_init__(self):
    self.actor.hidden_dims = (512, 256, 128)
    self.actor.obs_normalization = True
    self.actor.activation = "elu"
    self.actor.distribution_cfg = rsl_config.RslRlDistributionCfg(init_std=1.0)

    self.critic.hidden_dims = (512, 256, 128)
    self.critic.obs_normalization = True
    self.critic.activation = "elu"

    self.algorithm.value_loss_coef = 1.0
    self.algorithm.use_clipped_value_loss = True
    self.algorithm.clip_param = 0.2
    self.algorithm.entropy_coef = 0.005
    self.algorithm.num_learning_epochs = 8
    self.algorithm.num_mini_batches = 4
    self.algorithm.learning_rate = 1e-3
    self.algorithm.schedule = "adaptive"
    self.algorithm.gamma = 0.99
    self.algorithm.lam = 0.95
    self.algorithm.desired_kl = 0.01
    self.algorithm.max_grad_norm = 1.0


@dataclasses.dataclass
class PickCubeVisionRslRlOnPolicyRunnerCfg(rsl_config.RslRlOnPolicyRunnerCfg):
  """Config for PickCube with a wrist camera."""

  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["policy", "camera"],
          "critic": ["critic"],
      }
  )
  vecenv_wrapper_cfg: rsl_config.RslRlVecEnvWrappercfg = dataclasses.field(
      default_factory=lambda: rsl_config.RslRlVecEnvWrappercfg(
          is_finite_horizon=False
      )
  )
  torch_compile_mode: str | None = "default"
  nworld: int = 4096
  max_iterations: int = 3000
  num_steps_per_env: int = 16
  save_interval: int = 50

  def __post_init__(self):
    cnn_cfg = rsl_config.RslRlCNNCfg(
        output_channels=(16, 32),
        kernel_size=(5, 3),
        stride=(2, 2),
        padding="zeros",
        activation="elu",
        max_pool=False,
        global_pool="none",
    )
    self.actor = rsl_config.RslRlCNNModelCfg(
        hidden_dims=(256, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=rsl_config.RslRlDistributionCfg(init_std=1.0),
        cnn_cfg={"camera": cnn_cfg},
    )
    self.critic = rsl_config.RslRlMLPModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )

    self.algorithm.value_loss_coef = 1.0
    self.algorithm.use_clipped_value_loss = True
    self.algorithm.clip_param = 0.2
    self.algorithm.entropy_coef = 0.005
    self.algorithm.num_learning_epochs = 4
    self.algorithm.num_mini_batches = 4
    self.algorithm.learning_rate = 1e-3
    self.algorithm.schedule = "adaptive"
    self.algorithm.gamma = 0.99
    self.algorithm.lam = 0.95
    self.algorithm.desired_kl = 0.01
    self.algorithm.max_grad_norm = 1.0


@dataclasses.dataclass
class G1WholeBodyTrackingRslRlOnPolicyRunnerCfg(
    rsl_config.RslRlOnPolicyRunnerCfg
):
  """Config for G1 WholeBodyTracking."""

  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["policy"],
          "critic": ["critic"],
      }
  )
  vecenv_wrapper_cfg: rsl_config.RslRlVecEnvWrappercfg = dataclasses.field(
      default_factory=lambda: rsl_config.RslRlVecEnvWrappercfg(
          is_finite_horizon=False
      )
  )
  nworld: int = 4096
  max_iterations: int = 30000
  num_steps_per_env: int = 24
  save_interval: int = 500

  def __post_init__(self):
    self.actor.hidden_dims = (512, 256, 128)
    self.actor.obs_normalization = True
    self.actor.activation = "elu"
    self.actor.distribution_cfg = rsl_config.RslRlDistributionCfg(init_std=1.0)

    self.critic.hidden_dims = (512, 256, 128)
    self.critic.obs_normalization = True
    self.critic.activation = "elu"

    self.algorithm.value_loss_coef = 1.0
    self.algorithm.use_clipped_value_loss = True
    self.algorithm.clip_param = 0.2
    self.algorithm.entropy_coef = 0.005
    self.algorithm.num_learning_epochs = 5
    self.algorithm.num_mini_batches = 4
    self.algorithm.learning_rate = 1e-3
    self.algorithm.schedule = "adaptive"
    self.algorithm.gamma = 0.99
    self.algorithm.lam = 0.95
    self.algorithm.desired_kl = 0.01
    self.algorithm.max_grad_norm = 1.0
