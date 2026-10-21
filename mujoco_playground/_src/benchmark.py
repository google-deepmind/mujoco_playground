# Copyright 2026 Google LLC
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
"""Core library functions for benchmarking suite."""

from collections.abc import Iterable
import dataclasses
import gc
import json
import logging
import multiprocessing as mp
import re
import tempfile
import time
import traceback
from typing import Any

from etils import epath
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl import rng
import torch
import warp as wp

logger = logging.getLogger(__name__)
BenchmarkSpec = dict[str, Any]


def clear_gpu_memory() -> None:
  """Clears GPU memory cache across PyTorch and Warp."""
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  try:
    wp.synchronize()
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def parse_tfevents(logdir: epath.Path) -> dict[str, Any]:
  """Extract metrics from TensorBoard event logs."""
  try:
    from mujoco_playground._src.internal_imports.tensorboard import event_accumulator  # pyrefly: ignore[missing-import]
  except ImportError:
    from tensorboard.backend.event_processing import event_accumulator  # pyrefly: ignore[missing-import]

  metrics: dict[str, Any] = {}
  ea = event_accumulator.EventAccumulator(
      str(logdir), size_guidance={event_accumulator.SCALARS: 0}
  )
  ea.Reload()

  scalars = ea.Tags().get("scalars")
  for tag in scalars if isinstance(scalars, list) else []:
    events = ea.Scalars(tag)
    if not events:
      continue
    metrics[tag] = float(events[-1].value)
    if "iteration" not in metrics:
      metrics["iteration"] = events[-1].step

  return metrics


def discover(
    env_filter: str = ".*",
    benchmarks: Iterable[BenchmarkSpec] | None = None,
) -> Iterable[BenchmarkSpec]:
  """Discover benchmarks matching the filter.

  Args:
    env_filter: Regular expression filter matched against benchmark name.
    benchmarks: Optional collection of benchmark specifications to filter. If
      None, discovers all environments registered in env_registry.

  Yields:
    BenchmarkSpec matching the filter.
  """
  if benchmarks is not None:
    for bm in benchmarks:
      if re.search(env_filter, str(bm["name"]), re.IGNORECASE):
        yield bm
    return

  for env_id in env_registry._REGISTRY.keys():
    if re.search(env_filter, env_id, re.IGNORECASE):
      yield {
          "name": env_id,
          "env_id": env_id,
          "description": f"RL training benchmark for {env_id}",
      }


def testspeed(
    env_id: str,
    num_envs: int | None = None,
    num_steps: int = 200,
    warmup_steps: int = 50,
    device: str = "cuda:0",
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, float | str | int]:
  """Measures raw physics and env throughput."""
  entry = env_registry.get(env_id)
  n_envs = num_envs or entry.runner_cfg.nworld
  env_kwargs = {"nworld": n_envs, "device": device}
  if env_overrides:
    env_kwargs.update(env_overrides)

  env = entry.env_builder(**env_kwargs)
  env.reset()

  action_spec: Any = env.action_spec()
  if isinstance(action_spec, dict):
    action = {
        k: torch.zeros(v.shape, dtype=torch.float32, device=device)
        for k, v in action_spec.items()
    }
  else:
    action = torch.zeros(action_spec.shape, dtype=torch.float32, device=device)

  env_cfg = entry.env_cfg_fn()
  decimation = int(
      round(env_cfg.sim.control_timestep / env_cfg.sim.simulation_timestep)
  )

  has_direct_sim = hasattr(env.device_layer, "sim")
  if has_direct_sim:
    sim = env.device_layer.sim
  else:
    sim = env.device_layer.device_coordinator.simulation_device
  total_physics_steps = num_steps * decimation

  # Warm up environment.
  for _ in range(warmup_steps):
    env.step(action)
  env.reset()
  wp.synchronize()

  # Physics SPS: Pure physics stepping normalized by decimation for direct scale
  # alignment with Env SPS.
  wp.synchronize()
  t0 = time.perf_counter()
  if has_direct_sim:
    for _ in range(num_steps):
      for _ in range(decimation):
        sim.step()
  else:
    for _ in range(num_steps):
      sim.step(decimation)
  wp.synchronize()
  t_physics = time.perf_counter() - t0
  raw_physics_sps = (
      (n_envs * total_physics_steps) / t_physics if t_physics > 0 else 0.0
  )
  physics_sps = (n_envs * num_steps) / t_physics if t_physics > 0 else 0.0

  # Env Stepping SPS
  env.reset()
  for _ in range(10):
    env.step(action)
  wp.synchronize()

  t0 = time.perf_counter()
  for _ in range(num_steps):
    env.step(action)
  wp.synchronize()
  t_env = time.perf_counter() - t0
  env_sps = (n_envs * num_steps) / t_env if t_env > 0 else 0.0

  # Framework Overhead %: 100 * (1 - env_sps / physics_sps)
  overhead_pct = (
      100.0 * (1.0 - env_sps / physics_sps) if physics_sps > 0 else 0.0
  )
  gpu_name = (
      torch.cuda.get_device_name(device)
      if torch.cuda.is_available() and "cuda" in str(device)
      else "CPU"
  )

  del env
  clear_gpu_memory()

  return {
      "env_id": env_id,
      "num_envs": n_envs,
      "num_steps": num_steps,
      "decimation": decimation,
      "physics_sps": float(physics_sps),
      "raw_physics_sps": float(raw_physics_sps),
      "env_sps": float(env_sps),
      "overhead_pct": float(overhead_pct),
      "device": str(device),
      "gpu": str(gpu_name),
  }


def evaluate(
    env_id: str,
    ckpt_path: epath.Path | str | None = None,
    num_episodes: int = 10,
    device: str = "cuda:0",
    env_overrides: dict[str, Any] | None = None,
    num_envs: int | None = None,
) -> dict[str, float]:
  """Evaluates a trained policy from a saved model checkpoint."""
  from mujoco_playground._src.rl.rsl_rl import checkpoint
  from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
  from rsl_rl import runners as rsl_rl_runners

  if ckpt_path is None:
    raise ValueError("ckpt_path must be provided.")
  resolved_path = epath.Path(ckpt_path)
  if resolved_path.is_dir():
    latest = checkpoint.get_latest_checkpoint(resolved_path)
    if latest:
      resolved_path = epath.Path(latest)

  n_eval = num_envs if num_envs is not None else num_episodes
  entry = env_registry.get(env_id)
  env_kwargs = {"nworld": n_eval, "device": device}
  if env_overrides:
    env_kwargs.update(env_overrides)

  raw_env = entry.env_builder(**env_kwargs)
  agent_cfg = dataclasses.replace(
      entry.runner_cfg,
      nworld=n_eval,
      num_steps_per_env=1,
  )
  env = vecenv_wrapper.RslRlVecEnvWrapper(
      raw_env,
      clip_actions=agent_cfg.clip_actions,
      cfg=entry.runner_cfg.vecenv_wrapper_cfg,
  )

  try:
    runner = rsl_rl_runners.OnPolicyRunner(
        env, dataclasses.asdict(agent_cfg), log_dir=None, device=device
    )
    runner.load(str(resolved_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    done_envs = torch.zeros(n_eval, dtype=torch.bool, device=device)
    cumulative_rewards = torch.zeros(n_eval, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(n_eval, dtype=torch.int32, device=device)

    max_steps = 1000
    for _ in range(max_steps):
      with torch.inference_mode():
        actions = policy(obs)
      obs, rewards, dones, _ = env.step(actions)

      active = ~done_envs
      cumulative_rewards += torch.where(active, rewards.squeeze(-1), 0.0)
      episode_lengths += active.to(torch.int32)
      newly_done = dones.squeeze(-1).bool() & active
      done_envs = done_envs | newly_done
      if done_envs.all():
        break

    return {
        "eval_mean_reward": float(cumulative_rewards.mean().item()),
        "eval_mean_episode_length": float(
            episode_lengths.float().mean().item()
        ),
    }
  finally:
    if hasattr(raw_env, "close"):
      raw_env.close()
    del env, raw_env
    clear_gpu_memory()


def _train_with_time_budget(
    runner: Any,
    max_iterations: int,
    timeout_sec: int,
) -> None:
  """Run training for approximately timeout_sec seconds."""
  runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
  if max_iterations <= 1:
    return

  t0 = time.perf_counter()
  runner.current_learning_iteration += 1
  runner.learn(num_learning_iterations=1)
  time_per_iter = time.perf_counter() - t0

  remaining_iters = min(
      max_iterations - 2, max(0, int(timeout_sec / time_per_iter))
  )

  if remaining_iters > 0:
    runner.current_learning_iteration += 1
    runner.learn(num_learning_iterations=remaining_iters)


def train(
    env_id: str,
    timeout_sec: int | None = None,
    device: str = "cuda:0",
    seed: int | None = 0,
    logdir: epath.Path | str | None = None,
    env_overrides: dict[str, Any] | None = None,
    num_iterations: int | None = None,
    num_envs: int | None = None,
) -> tuple[dict[str, Any], epath.Path]:
  """Runs RL training for the specified env_id with an optional time limit."""
  from mujoco_playground._src.rl.rsl_rl import checkpoint
  from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
  from rsl_rl import runners as rsl_rl_runners

  resolved_logdir = (
      epath.Path(logdir)
      if logdir is not None
      else epath.Path(tempfile.gettempdir()) / "mujoco_playground_benchmarks"
  )
  resolved_logdir.mkdir(parents=True, exist_ok=True)
  run_logdir = resolved_logdir / env_id / f"run_{int(time.time())}"
  run_logdir.mkdir(parents=True, exist_ok=True)

  if seed is not None:
    rng.seed(seed)

  entry = env_registry.get(env_id)
  runner_cfg = entry.runner_cfg
  n_envs = num_envs or runner_cfg.nworld
  env_kwargs = {"nworld": n_envs, "device": device}
  if env_overrides:
    env_kwargs.update(env_overrides)

  raw_env = entry.env_builder(**env_kwargs)
  runner_cfg = entry.runner_cfg
  runner_cfg = dataclasses.replace(
      runner_cfg,
      seed=seed if seed is not None else runner_cfg.seed,
      run_name=run_logdir.name,
      logger="tensorboard",
  )
  env = vecenv_wrapper.RslRlVecEnvWrapper(
      raw_env,
      clip_actions=runner_cfg.clip_actions,
      cfg=runner_cfg.vecenv_wrapper_cfg,
  )

  runner = rsl_rl_runners.OnPolicyRunner(
      env=env,
      train_cfg=dataclasses.asdict(runner_cfg),
      log_dir=str(run_logdir),
      device=device,
  )

  start_time = time.time()
  exit_code = 0
  try:
    if timeout_sec is not None:
      _train_with_time_budget(
          runner=runner,
          max_iterations=runner_cfg.max_iterations,
          timeout_sec=timeout_sec,
      )
    else:
      runner.learn(
          num_learning_iterations=num_iterations or runner_cfg.max_iterations,
          init_at_random_ep_len=True,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.exception("Training failed for %s: %s", env_id, e)
    exit_code = 1
  finally:
    if hasattr(raw_env, "close"):
      raw_env.close()
    del env, raw_env
    clear_gpu_memory()

  elapsed_time = time.time() - start_time
  metrics = parse_tfevents(run_logdir)
  if "Perf/total_fps" in metrics and "steps_per_second" not in metrics:
    metrics["steps_per_second"] = metrics["Perf/total_fps"]
  metrics["timeout_sec"] = timeout_sec
  metrics["elapsed_time"] = elapsed_time
  metrics["exit_code"] = exit_code

  if exit_code != 0:
    metrics["status"] = "failed"
  else:
    metrics["status"] = "completed"
    metrics["iterations"] = getattr(runner, "current_learning_iteration", None)

  metrics["latest_checkpoint"] = checkpoint.get_latest_checkpoint(run_logdir)

  return metrics, run_logdir


def _run_worker_stage(
    queue: Any,
    fn_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
  """Worker executing a single stage inside an isolated child process."""
  try:
    if fn_name == "testspeed":
      res = testspeed(*args, **kwargs)
    elif fn_name == "train":
      metrics, _ = train(*args, **kwargs)
      res = metrics
    elif fn_name == "evaluate":
      res = evaluate(*args, **kwargs)
    else:
      raise ValueError(f"Unknown stage function: {fn_name}")
    queue.put({"success": True, "result": res})
  except Exception as e:  # pylint: disable=broad-exception-caught
    queue.put({
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc(),
    })


def _execute_in_process(
    fn_name: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
  """Executes a benchmark function in a fresh isolated process."""
  ctx = mp.get_context("spawn")
  queue = ctx.Queue()
  process = ctx.Process(
      target=_run_worker_stage,
      args=(queue, fn_name, args, kwargs),
  )
  process.start()
  process.join()

  if not queue.empty():
    res = queue.get()
    if res["success"]:
      return res["result"]
    raise RuntimeError(
        f"Stage {fn_name} failed in isolated process:\n{res['traceback']}"
    )

  if process.exitcode != 0:
    raise RuntimeError(
        f"Stage {fn_name} process crashed with exit code {process.exitcode}"
    )

  raise RuntimeError(
      f"Stage {fn_name} process exited without returning results."
  )


def run_all(
    bm: BenchmarkSpec,
    device: str | int = "0",
    seed: int | None = 0,
    timeout_sec: int | None = None,
    logdir: epath.Path | str | None = None,
    skip_eval: bool = False,
    num_envs: int | None = None,
    num_steps: int = 200,
    warmup_steps: int = 20,
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Runs the complete benchmark suite with per-stage process isolation."""
  env_id = str(bm["env_id"])
  device_str = (
      f"cuda:{device}"
      if isinstance(device, int) or str(device).isdigit()
      else str(device)
  )
  timeout = timeout_sec or int(bm.get("timeout_sec", 1800))
  logdir = logdir or tempfile.mkdtemp(prefix="mujoco_playground_benchmarks")

  results: dict[str, Any] = {
      "name": bm.get("name", env_id),
      "env_id": env_id,
      "device": device_str,
      "seed": seed,
  }

  overrides = (
      env_overrides if env_overrides is not None else bm.get("env_overrides")
  )

  # Throughput
  logger.info("[1/3] Measuring raw physics and env throughput: %s...", env_id)
  speed_metrics = _execute_in_process(
      "testspeed",
      env_id=env_id,
      num_envs=num_envs,
      num_steps=num_steps,
      warmup_steps=warmup_steps,
      device=device_str,
      env_overrides=overrides,
  )
  results.update(speed_metrics)

  # Policy Training
  logger.info(
      "[2/3] Training policy for %s (seed=%s, timeout=%ds)...",
      env_id,
      seed,
      timeout,
  )
  train_metrics = _execute_in_process(
      "train",
      env_id=env_id,
      timeout_sec=timeout,
      device=device_str,
      seed=seed,
      logdir=logdir,
      env_overrides=overrides,
      num_envs=num_envs,
  )
  results.update(train_metrics)

  # Policy Evaluation
  ckpt_path = train_metrics.get("latest_checkpoint")
  if not skip_eval and ckpt_path and epath.Path(ckpt_path).exists():
    logger.info("[3/3] Evaluating policy from %s...", ckpt_path)
    eval_metrics = _execute_in_process(
        "evaluate",
        env_id=env_id,
        ckpt_path=epath.Path(ckpt_path),
        device=device_str,
        num_envs=num_envs,
        env_overrides=overrides,
    )
    results.update(eval_metrics)
  else:
    logger.info(
        "[3/3] Skipping evaluation (no valid checkpoint or skip_eval=True)."
    )

  return results
