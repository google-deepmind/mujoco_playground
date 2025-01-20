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
"""Benchmark Madrona MJX."""

import argparse
import csv
import enum
import functools
import os
import time

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground import registry
from mujoco_playground import wrapper


class MeasurementMode(enum.Enum):
  """
  Each step includes all components of previous steps.
  """

  STATE = 0
  STATE_VISION = 1
  STATE_VISION_INF = 2


def unvmap(x, ind):
  return jax.tree.map(lambda y: y[ind], x)


if __name__ == "__main__":
  np.set_printoptions(precision=3, suppress=True, linewidth=100)

  parser = argparse.ArgumentParser(description="Benchmark Madrona MJX")
  parser.add_argument(
      "--num-envs",
      type=int,
      default=None,
      help="Number of environments to simulate.",
  )
  parser.add_argument(
      "--img-size",
      type=int,
      default=0,
      help="Width of render. Only used by benchmark environments",
  )
  parser.add_argument(
      "--env-name",
      type=str,
      default="CartpoleBalance",
      help="Name of the environment.",
  )
  parser.add_argument(
      "--measurement-mode",
      type=int,
      default=-1,
      help="See class MeasurementMode.",
  )
  parser.add_argument(
      "--bottleneck-mode",
      action="store_true",
      help="Are we measuring the bottlenecks for paper table 6?",
  )
  args_cli, _ = parser.parse_known_args()

  # Ensure env-name is valid.
  assert args_cli.env_name in [
      "CartpoleBalance",
      "PandaPickCubeCartesian",
  ], f"Invalid env-name: {args_cli.env_name}"
  assert args_cli.measurement_mode in [-1, 0, 1, 2]
  mode = args_cli.measurement_mode
  img_size = args_cli.img_size
  num_envs = args_cli.num_envs
  env_name = args_cli.env_name
  bottleneck_mode = args_cli.bottleneck_mode

  vision = True
  if mode == MeasurementMode.STATE.value:
    args_cli.img_size = 0  # default value.
    vision = False

  N = 1000

  if vision:
    # Load the compiled rendering backend to save time!
    # os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = (
    # "<YOUR_PATH>/madrona_mjx/build/cache"
    # )

    # Coordinate between Jax and the Madrona rendering backend
    def limit_jax_mem(limit):
      os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"

    # Sufficient settings for running get_data.sh on an RTX4090.
    limit_jax_mem(0.55)
    heap_size = 2**31
    os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = str(heap_size)

  use_rasterizer = False
  randomization_fn = None
  env_specific = {}
  if env_name == "CartpoleBalance":
    if bottleneck_mode:
      # Match settings from our trainable pixels cartpole env.
      ctrl_dt = 0.04
      sim_dt = 0.01
    else:
      # Match physics settings from other benchmarking work.
      # https://github.com/haosulab/ManiSkill/blob/beb585b1b78dc4bfdf074926975992c7d8b4f81b/mani_skill/examples/benchmarking/envs/isaaclab/cartpole_visual.py#L30
      ctrl_dt = 1 / 60
      sim_dt = 1 / 120

    episode_length = int(3 / ctrl_dt)
    if img_size > 400:
      # Memory saving mode.
      # Should not affect benchmarking results, as the same rendering calls are
      #  made.
      env_specific = {"vision_config.history": 1}

  else:
    assert (
        bottleneck_mode
    ), "Nothing to benchmark against for FrankaPickCubeCartesian."
    ctrl_dt = 0.05  # Freeze to values at time of paper-writing.
    sim_dt = 0.005
    use_rasterizer = True  # Match provided FrankaPickCubeCartesian notebook.
    episode_length = int(4 / ctrl_dt)
    from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize

    randomization_fn = functools.partial(
        randomize.domain_randomize, num_worlds=num_envs
    )
    env_specific = {"obs_noise.brightness": [0.25, 1.1]}

  config_overrides = {
      "vision_config.render_batch_size": num_envs,
      "vision_config.render_height": img_size,
      "vision_config.render_width": img_size,
      "vision_config.use_rasterizer": use_rasterizer,
      "action_repeat": 1,
      "ctrl_dt": ctrl_dt,
      "sim_dt": sim_dt,
      "vision": vision,
      "episode_length": episode_length,
  } | env_specific

  env = registry.load(env_name, config_overrides=config_overrides)
  env = wrapper.wrap_for_brax_training(
      env,
      vision=vision,
      num_vision_envs=num_envs,
      action_repeat=1,
      episode_length=episode_length,
      randomization_fn=randomization_fn,
  )

  jit_reset = jax.jit(env.reset)
  state_outer = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))

  # Random noise inference function.
  if mode in [MeasurementMode.STATE.value, MeasurementMode.STATE_VISION.value]:

    def inference_fn(_, key):
      return (
          jax.random.uniform(
              key, (num_envs, env.action_size), minval=-1.0, maxval=1.0
          ),
          None,
      )

    jit_inference_fn = jax.jit(inference_fn)
  else:
    # Randomly initialized Brax inference function.
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import losses as ppo_losses
    from brax.training.agents.ppo.networks import make_inference_fn
    from brax.training.agents.ppo.networks_vision import make_ppo_networks_vision
    from brax.training.agents.ppo.train import _remove_pixels

    network_factory = make_ppo_networks_vision

    env_state = unvmap(state_outer, 0)
    preprocess_fn = running_statistics.normalize
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )

    k1, k2 = jax.random.split(jax.random.key(0))
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(k1),
        value=ppo_network.value_network.init(k2),
    )
    normalizer_init = running_statistics.init_state(
        _remove_pixels(env.observation_size)
    )

    pars = (normalizer_init, init_params.policy)  # stores both value and policy

    make_policy = make_inference_fn(ppo_network)
    policy = make_policy(
        pars, deterministic=False
    )  # Match inference in training
    jit_inference_fn = jax.jit(policy)  # batched obs, batched rng

    # Test: how many parameters in policy?
    policy_params = pars[1]["params"]
    from jax.flatten_util import ravel_pytree

    flat, unflat = ravel_pytree(policy_params)
    print(f"Policy has {len(flat)} parameters!")

  @jax.jit
  def rollout(state, seed):
    """
    Main benchmarking component.
    The "token" system ensures proper timing, as it depends on all of the
    final output dimensions. Naively returning the final state adds several GB
    of additional memory requirement for high res.
    """

    def env_step(c, _):
      state, key = c
      key_act, key = jax.random.split(key)
      act, _ = jit_inference_fn(state.obs, key_act)
      token = jp.sum(state.obs["pixels/view_0"])
      assert token.shape == (), token.shape
      return (env.step(state, act), key), token

    key_act = jax.random.PRNGKey(seed)
    (state, _), tokens = jax.lax.scan(env_step, (state, key_act), length=N)
    return jp.sum(tokens)

  jit_rollout = rollout.lower(state_outer, 0).compile()

  t0 = time.time()
  output = jit_rollout(state_outer, 1)
  jax.tree_util.tree_map(
      lambda x: x.block_until_ready(), output
  )  # Await device completion
  dt = time.time() - t0

  fps = int(N * num_envs / dt)

  #### Configure Outputs ####
  cur_row = {
      "num_envs": num_envs,
      "measurement_mode": mode,
      "env_name": env_name,
      "img_size": img_size,
      "bottleneck_mode": bottleneck_mode,
      "fps": fps,
  }

  print(cur_row)

  #### Write CSV ####
  fname = "madrona_mjx"

  from pathlib import Path

  base_path = Path(__file__).parent
  csv_file_path = base_path / f"data/{fname}.csv"

  # Check if the file already exists to write the header only if needed
  try:
    with open(csv_file_path, "r", encoding="utf-8") as f:
      existing_headers = f.readline().strip().split(",")
  except FileNotFoundError:
    existing_headers = []

  # Open the CSV file in append mode
  with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cur_row.keys())

    # Write the header if the file is new or doesn't have matching headers
    if not existing_headers or set(existing_headers) != set(cur_row.keys()):
      writer.writeheader()

    # Write the current row
    writer.writerow(cur_row)
