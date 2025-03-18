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
# pylint: skip-file
import os

os.environ[
    "XLA_PYTHON_CLIENT_PREALLOCATE"
] = (  # Ensure that Madrona gets the chance to pre-allocate memory before Jax
    "false"
)


from datetime import datetime
import functools
import json

from brax.io import model
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.bc import train as bc_fast
from etils import epath
from flax import linen
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from orbax import checkpoint as ocp
import typer

from mujoco_playground import manipulation
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aloha.s2r import distillation

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    seed: int = typer.Option(0, help="Random seed"),
    print_reward: bool = typer.Option(
        False, help="Prints the per-step reward in the data collection phase"
    ),
    print_loss: bool = typer.Option(
        False, help="Prints the actor loss in the student-fitting phase"
    ),
    domain_randomization: bool = typer.Option(
        False, help="Use domain randomization"
    ),
    vision: bool = typer.Option(True, help="Use vision"),
    policy_save_path: str = typer.Option(None, help="Path to save the policy"),
    num_envs: int = typer.Option(1024, help="Number of parallel environments"),
    episode_length: int = typer.Option(160, help="Length of each episode"),
    dagger_iterations: int = typer.Option(
        400, help="Number of DAgger iterations"
    ),
    num_evals: int = typer.Option(5, help="Number of evaluation episodes"),
    demo_length: int = typer.Option(6, help="Length of demonstrations"),
):
  env_name = "AlohaS2RPegInsertionDistill"
  env_cfg = manipulation.get_default_config(env_name)

  config_overrides = {
      "episode_length": episode_length,
      "vision": vision,
      "vision_config.enabled_geom_groups": [
          1,
          2,
          5,
      ],  # Disable mocaps on group 0.
      "vision_config.use_rasterizer": False,
      "vision_config.render_batch_size": num_envs,
      "vision_config.render_width": 32,
      "vision_config.render_height": 32,
  }

  env = manipulation.load(
      env_name, config=env_cfg, config_overrides=config_overrides
  )

  randomizer = None
  if domain_randomization:
    randomizer = registry.get_domain_randomizer(env_name)
    # Randomizer expected to only require mjx model input.
    key_rng = jax.random.PRNGKey(seed)
    randomizer = functools.partial(
        randomizer, rng=jax.random.split(key_rng, num_envs)
    )

  env = wrapper.wrap_for_brax_training(
      env,
      vision=vision,
      num_vision_envs=num_envs,
      episode_length=episode_length,
      action_repeat=1,
      randomization_fn=randomizer,
  )

  network_factory = functools.partial(  # Student network factory.
      bc_networks.make_bc_networks,
      policy_hidden_layer_sizes=(256,) * 3,
      activation=linen.relu,
      policy_obs_key=("proprio" if vision else "state_with_time"),
      vision=vision,
  )

  teacher_inference_fn = distillation.make_teacher_policy()

  # Generate unique experiment name.
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{env_name}-{timestamp}"

  ckpt_path = epath.Path("logs").resolve() / exp_name
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_json(), fp, indent=4)

  epochs = 4
  augment_pixels = True
  dagger_beta_fn = lambda iter: jp.where(iter == 0, 1.0, 0.0)
  restore_checkpoint_path = None

  def get_num_dagger_iters(num_evals, target):
    """Round down to the nearest multiple of num_evals - 1."""
    dagger_iters = target // (num_evals - 1) * (num_evals - 1)
    print("Dagger iters:", dagger_iters)
    print(
        "Episodes per environment:",
        (dagger_iters * demo_length / episode_length),
    )
    print(
        "Total episodes across all environments:",
        num_envs * (dagger_iters * demo_length / episode_length),
    )
    print(
        "Total steps across all environments:",
        (dagger_iters * demo_length * num_envs),
    )
    return dagger_iters

  dagger_iterations = get_num_dagger_iters(num_evals, dagger_iterations)

  def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

  def progress(epoch, metrics: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if epoch == 0 and num_evals > 0:
      print(
          f"""[{timestamp}] Dagger Iteration {epoch}:
            Eval Reward: {metrics['eval/episode_reward']:.4f} ± {metrics['eval/episode_reward_std']:.4f} \n\n"""
      )
      return

    actor_loss = jp.mean(metrics["actor_loss"], axis=(-1, -2))

    if print_loss:
      actor_loss = jp.ravel(actor_loss)
      for loss in actor_loss:
        print(f"SGD Actor Loss: {loss:.4f}")

    if print_reward:
      r_means = metrics["reward_mean"].ravel()
      # Ensure divisibility by 30.
      r_means = r_means[: len(r_means) // 30 * 30]
      r_means = r_means.reshape(-1, 30).mean(axis=1)  # Downsample.
      for r_mean in r_means:
        print(f"Rewards: {r_mean:.4f}")

    print(
        f"""[{timestamp}] Dagger Iteration {epoch}:
        Actor Loss: {jp.mean(actor_loss):.4f}, SPS:  {metrics['SPS']:.4f}, Walltime:  {metrics['walltime']:.4f} s"""
    )
    suffix = (
        "\n\n"
        if num_evals == 0
        else (
            f"\t\tEval Reward: {metrics['eval/episode_reward']:.4f} ±"
            f" {metrics['eval/episode_reward_std']:.4f}\n\n"
        )
    )
    print(suffix)

  train_fn = bc_fast.train
  train_fn = functools.partial(
      train_fn,
      dagger_iterations=dagger_iterations,
      demo_length=demo_length,
      tanh_squash=True,
      teacher_inference_fn=teacher_inference_fn,
      normalize_observations=True,
      epochs=epochs,
      scramble_time=episode_length,
      dagger_beta_fn=dagger_beta_fn,
      batch_size=256,
      env=env,
      num_envs=num_envs,
      num_eval_envs=num_envs,
      num_evals=num_evals,
      eval_length=episode_length * 1.15,
      network_factory=network_factory,
      progress_fn=progress,
      madrona_backend=True,
      seed=seed,
      learning_rate=4e-4,
      augment_pixels=augment_pixels,
      restore_checkpoint_path=restore_checkpoint_path,
      policy_params_fn=policy_params_fn,
  )
  print(f"Training start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  _, params, _ = train_fn()
  print(f"Training done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

  if policy_save_path is not None:
    model.save_params(policy_save_path, params)


if __name__ == "__main__":
  app()
