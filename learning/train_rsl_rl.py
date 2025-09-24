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
# pylint: disable=wrong-import-position
"""Train a PPO agent using RSL-RL for the specified environment."""

import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime
import json

from absl import app
from absl import flags
from absl import logging
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco
from rsl_rl.runners import OnPolicyRunner
import torch
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Suppress logs if you want
logging.set_verbosity(logging.WARNING)

# Define flags similar to the JAX script
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "BerkeleyHumanoidJoystickFlatTerrain",
    (
        "Name of the environment. One of: "
        f"{', '.join(mujoco_playground.registry.ALL_ENVS)}"
    ),
)
_LOAD_RUN_NAME = flags.DEFINE_string(
    "load_run_name", None, "Run name to load from (for checkpoint restoration)."
)
_CHECKPOINT_NUM = flags.DEFINE_integer(
    "checkpoint_num", -1, "Checkpoint number to load from."
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train."
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode).",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 4096, "Number of parallel envs.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean(
    "multi_gpu", False, "If true, use multi-GPU training (distributed)."
)
_CAMERA = flags.DEFINE_string(
    "camera", None, "Camera name to use for rendering."
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config for {env_name}")


def main(argv):
  """Run training and evaluation for the specified environment using RSL-RL."""
  del argv  # unused

  # Possibly parse the device for multi-GPU
  if _MULTI_GPU.value:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_rank = local_rank
    device = f"cuda:{local_rank}"
    print(f"Using multi-GPU: local_rank={local_rank}, device={device}")
  else:
    device = _DEVICE.value
    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

  # If play-only, use fewer envs
  num_envs = 1 if _PLAY_ONLY.value else _NUM_ENVS.value

  # Load default config from registry
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  print(f"Environment config:\n{env_cfg}")

  # Generate unique experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Logging directory
  logdir = os.path.abspath(os.path.join("logs", exp_name))
  os.makedirs(logdir, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Checkpoint directory
  ckpt_path = os.path.join(logdir, "checkpoints")
  os.makedirs(ckpt_path, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.tensorboard.patch(root_logdir=logdir)
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Save environment config to JSON
  with open(
      os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
  ) as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # Domain randomization
  randomizer = registry.get_domain_randomizer(_ENV_NAME.value)

  # We'll store environment states during rendering
  render_trajectory = []

  # Callback to gather states for rendering
  def render_callback(_, state):
    render_trajectory.append(state)

  # Create the environment
  raw_env = registry.load(_ENV_NAME.value, config=env_cfg)
  brax_env = wrapper_torch.RSLRLBraxWrapper(
      raw_env,
      num_envs,
      _SEED.value,
      env_cfg.episode_length,
      1,
      render_callback=render_callback,
      randomization_fn=randomizer,
      device_rank=device_rank,
  )

  # Build RSL-RL config
  train_cfg = get_rl_config(_ENV_NAME.value)

  obs_size = raw_env.observation_size
  if isinstance(obs_size, dict):
    train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
  else:
    train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}

  # Overwrite default config with flags
  train_cfg.seed = _SEED.value
  train_cfg.run_name = exp_name
  train_cfg.resume = _LOAD_RUN_NAME.value is not None
  train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
  train_cfg.checkpoint = _CHECKPOINT_NUM.value

  train_cfg_dict = train_cfg.to_dict()
  runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

  # If resume, load from checkpoint
  if train_cfg.resume:
    resume_path = wrapper_torch.get_load_path(
        os.path.abspath("logs"),
        load_run=train_cfg.load_run,
        checkpoint=train_cfg.checkpoint,
    )
    print(f"Loading model from checkpoint: {resume_path}")
    runner.load(resume_path)

  if not _PLAY_ONLY.value:
    # Perform training
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print("Done training.")
    return

  # If just playing (no training)
  policy = runner.get_inference_policy(device=device)

  # Example: run a single rollout
  eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(_SEED.value)
  state = jit_reset(rng)
  rollout = [state]

  # We’ll assume your environment’s observation is in state.obs["state"].
  obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])

  for _ in range(env_cfg.episode_length):
    with torch.no_grad():
      actions = policy(obs_torch)
    # Step environment
    state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
    rollout.append(state)
    obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
    if state.done:
      break

  # Render
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_every = 2
  # If your environment is wrapped multiple times, adjust as needed:
  base_env = eval_env  # or brax_env.env.env.env
  fps = 1.0 / base_env.dt / render_every
  traj = rollout[::render_every]
  frames = eval_env.render(
      traj,
      camera=_CAMERA.value,
      height=480,
      width=640,
      scene_option=scene_option,
  )
  media.write_video("rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'.")


if __name__ == "__main__":
  app.run(main)
