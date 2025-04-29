import functools
import logging
import os
from typing import Any

import hydra
import jax
import jax.numpy as jp
from numpy import typing as npt
import numpy as np
import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from mujoco_playground import registry
from mujoco_playground import wrapper

_LOG = logging.getLogger(__name__)


class WeightAndBiasesWriter:

  def __init__(self, config: DictConfig):
    import wandb

    try:
      name = config.wandb.name
    except InterpolationKeyError:
      name = None
    config.wandb.name = name
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    assert isinstance(config_dict, dict)
    wandb.init(project="mjp", resume=True, config=config_dict, **config.wandb)
    self._handle = wandb

  def log(self, summary: dict[str, float], step: int):
    self._handle.log(summary, step=step)

  def log_video(
      self,
      images: npt.ArrayLike,
      step: int,
      name: str = "policy",
      fps: int | float = 30,
  ):
    self._handle.log(
        {
            name: self._handle.Video(
                np.array(images, copy=False),
                fps=int(fps),
                caption=name,
            )
        },
        step=step,
    )

  def log_artifact(
      self,
      path: str,
      type: str,
      name: str | None = None,
      description: str | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    if name is None:
      name = self._handle.run.id
    if metadata is None:
      metadata = dict(self._handle.config)
    artifact = self._handle.Artifact(name, type, description, metadata)
    artifact.add_file(path)
    self._handle.log_artifact(artifact, aliases=[self._handle.run.id])


def get_state_path() -> str:
  log_path = os.getcwd()
  return log_path


def get_ppo_train_fn(env_name):
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.ppo import train as ppo

  from mujoco_playground.config import dm_control_suite_params

  ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
  train_fn = functools.partial(
      ppo.train,
      **dict(ppo_training_params),
      network_factory=network_factory,
  )
  return train_fn


def get_sac_train_fn(env_name):
  from brax.training.agents.sac import networks as sac_networks
  from brax.training.agents.sac import train as sac

  from mujoco_playground.config import dm_control_suite_params

  sac_params = dm_control_suite_params.brax_sac_config(env_name)
  sac_training_params = dict(sac_params)
  network_factory = sac_networks.make_sac_networks
  if "network_factory" in sac_params:
    del sac_training_params["network_factory"]
    network_factory = functools.partial(
        sac_networks.make_sac_networks, **sac_params.network_factory
    )
  train_fn = functools.partial(
      sac.train,
      **dict(sac_training_params),
      network_factory=network_factory,
  )
  return train_fn


class Counter:

  def __init__(self):
    self.count = 0


def report(logger, step, num_steps, metrics):
  metrics = {k: float(v) for k, v in metrics.items()}
  logger.log(metrics, num_steps)
  step.count = num_steps


@functools.partial(jax.jit, static_argnames=("env", "policy", "steps"))
def rollout(
    env,
    policy,
    steps,
    rng,
    state,
):
  def f(carry, _):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    action, _ = policy(state.obs, current_key)
    nstate = env.step(
        state,
        action,
    )
    return (nstate, next_key), nstate

  (final_state, _), data = jax.lax.scan(f, (state, rng), (), length=steps)
  return final_state, data


def pytrees_unstack(pytree):
  leaves, treedef = jax.tree.flatten(pytree)
  n_trees = leaves[0].shape[0]
  new_leaves = [[] for _ in range(n_trees)]
  for leaf in leaves:
    for i in range(n_trees):
      new_leaves[i].append(leaf[i])
  new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
  return new_trees


def render(env, policy, steps, rng, camera=None):
  state = env.reset(rng)
  _, trajectory = rollout(env, policy, steps, rng[0], state)
  videos = []
  ep_trajectory = pytrees_unstack(trajectory)
  video = env.render(ep_trajectory, camera=camera)
  videos.append(video)
  return np.asarray(videos).transpose(0, 1, 4, 2, 3)


@hydra.main(version_base=None, config_path="config", config_name="train_brax")
def main(cfg):
  _LOG.info(
      "Setting up experiment with the following configuration: "
      f"\n{OmegaConf.to_yaml(cfg)}"
  )
  logger = WeightAndBiasesWriter(cfg)
  if cfg.training.agent_name == "SAC":
    train_fn = get_sac_train_fn(cfg.training.task_name)
  elif cfg.training.agent_name == "PPO":
    train_fn = get_ppo_train_fn(cfg.training.task_name)
  else:
    raise NotImplementedError
  rng = jax.random.PRNGKey(cfg.training.seed)
  steps = Counter()
  env = registry.load(cfg.training.task_name)
  env_cfg = registry.get_default_config(cfg.training.task_name)
  eval_env = registry.load(cfg.training.task_name, config=env_cfg)
  with jax.disable_jit(not cfg.jit):
    make_policy, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=functools.partial(report, logger, steps),
    )
  if cfg.training.render:
    rng = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 128)
    video = render(
        eval_env,
        make_policy(params, deterministic=True),
        1000,
        rng,
    )
    logger.log_video(video, steps.count, "eval/video")
  _LOG.info("Done training.")


if __name__ == "__main__":
  main()
