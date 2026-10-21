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

"""RSL-RL configuration as dataclasses."""

import dataclasses
from typing import Any
from typing import Literal
from typing import Tuple


@dataclasses.dataclass
class RslRlVecEnvWrappercfg:
  """Config for the RSL-RL VecEnv wrapper."""

  is_finite_horizon: bool = True


@dataclasses.dataclass
class RslRlDistributionCfg:
  """Config for the output distribution of an actor model.

  Attributes:
    class_name: Distribution class name.
    init_std: Initial standard deviation.
    std_type: Parameterization of the standard deviation: "scalar" or "log".
  """

  class_name: str = "GaussianDistribution"
  init_std: float = 1.0
  std_type: Literal["scalar", "log"] = "scalar"


@dataclasses.dataclass
class RslRlMLPModelCfg:
  """Config for an MLPModel (actor or critic).

  Attributes:
    class_name: Model class name.
    hidden_dims: Hidden dimensions of the MLP.
    activation: Activation function for the MLP.
    obs_normalization: Whether to normalize observations.
    distribution_cfg: Distribution config (actor only, None for critic).
  """

  class_name: str = "MLPModel"
  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  activation: str = "elu"
  obs_normalization: bool = False
  distribution_cfg: RslRlDistributionCfg | None = None


@dataclasses.dataclass
class RslRlCNNCfg:
  """Configuration for CNN layers.

  Attributes:
    output_channels: Number of output channels for each convolutional layer.
    kernel_size: Kernel size for each convolutional layer.
    stride: Stride for each convolutional layer.
    dilation: Dilation for each convolutional layer.
    padding: Padding type for convolutional layers.
    norm: Normalization type for each convolutional layer.
    activation: Activation function for convolutional layers.
    max_pool: Whether to use max pooling after each convolutional layer.
    global_pool: Global pooling type to apply at the end.
    flatten: Whether to flatten the output.
  """

  output_channels: Tuple[int, ...] | None = None
  kernel_size: int | Tuple[int, ...] = 3
  stride: int | Tuple[int, ...] = 1
  dilation: int | Tuple[int, ...] = 1
  padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
  norm: Literal["none", "batch", "layer"] | Tuple[str, ...] = "none"
  activation: str = "elu"
  max_pool: bool | Tuple[bool, ...] = False
  global_pool: Literal["none", "max", "avg"] = "none"
  flatten: bool = True


@dataclasses.dataclass
class RslRlCNNModelCfg(RslRlMLPModelCfg):
  """Config for a CNNModel (actor or critic) with CNN layers.

  Attributes:
    class_name: Model class name.
    cnn_cfg: CNN configuration, keyed by observation group name.
  """

  class_name: str = "CNNModel"
  cnn_cfg: dict[str, Any] | None = None


@dataclasses.dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm.

  Attributes:
    class_name: Alorithm class name.
    num_learning_epochs: The number of learning epochs per update.
    num_mini_batches: The number of mini-batches per update. mini batch size =
      num_envs * num_steps / num_mini_batches
    learning_rate: The learning rate.
    schedule: The learning rate schedule.
    gamma: The discount factor.
    lam: The lambda parameter for Generalized Advantage Estimation (GAE).
    entropy_coef: The coefficient for the entropy loss.
    desired_kl: The desired KL divergence between the new and old policies.
    max_grad_norm: The maximum gradient norm for the policy.
    value_loss_coef: The coefficient for the value loss.
    use_clipped_value_loss: Whether to use clipped value loss.
    clip_param: The clipping parameter for the policy.
    normalize_advantage_per_mini_batch: Whether to normalize the advantage per
      mini-batch. Default is False, where the advantage is normalized over the
      entire collection of trajectories. If True, the advantage is normalized
      over mini-batches only.
  """

  class_name: str = "PPO"
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.005
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False
  # TODO: maybe add RND and symmetry config.


@dataclasses.dataclass
class RslRlBaseRunnerCfg:
  """Base configuration for RSL-RL runners.

  Attributes:
    seed: The seed for the experiment.
    nworld: The number of worlds to simulate.
    max_iterations: The maximum number of iterations.
    num_steps_per_env: The number of steps per environment update.
    obs_groups: Mapping from observation groups to observation sets.
    clip_actions: The clipping range for action values.
    save_interval: The number of iterations between saves.
    experiment_name: The experiment name.
    run_name: The run name.
    logger: The logger to use. Only tensorboard is supported.
    resume: Whether to resume the experiment.
    load_run: The run directory to load.
    load_checkpoint: The checkpoint file to load.
  """

  seed: int = 42
  nworld: int = 4096
  max_iterations: int = 300
  num_steps_per_env: int = 24
  obs_groups: dict[str, list[str]] = dataclasses.field(
      default_factory=lambda: {
          "actor": ["policy"],
          "critic": ["policy", "critic"],
      },
  )
  clip_actions: float | None = None
  save_interval: int = 100
  experiment_name: str = "RslRl-Experiment"
  run_name: str = ""
  logger: Literal["tensorboard"] = "tensorboard"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  torch_compile_mode: str | None = None


@dataclasses.dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  """Config for the RSL-RL on-policy runner.

  Attributes:
    class_name: The runner class name.
    actor: The actor model configuration.
    critic: The critic model configuration.
    algorithm: The algorithm configuration.
    vecenv_wrapper_cfg: The VecEnv wrapper configuration.
  """

  class_name: str = "OnPolicyRunner"
  actor: RslRlMLPModelCfg = dataclasses.field(
      default_factory=lambda: RslRlMLPModelCfg(
          distribution_cfg=RslRlDistributionCfg(),
      )
  )
  critic: RslRlMLPModelCfg = dataclasses.field(default_factory=RslRlMLPModelCfg)
  algorithm: RslRlPpoAlgorithmCfg = dataclasses.field(
      default_factory=RslRlPpoAlgorithmCfg
  )
  vecenv_wrapper_cfg: RslRlVecEnvWrappercfg = dataclasses.field(
      default_factory=RslRlVecEnvWrappercfg
  )
