# MuJoCo Playground RL Profiling & Benchmarks

Benchmarking suite for RL policy training and simulation throughput across registered environments in `mujoco_playground._src.rl.env_registry`.

## Overview

The benchmark suite:
1. Discovers registered RL environments defined in [`profiling/__init__.py`](__init__.py).
2. Runs single-GPU RL training with enforced wall-clock cutoffs (30-minute default cutoff, 1,800s).
3. Measures raw physics simulation SPS and full environment stepping SPS using `mujoco_warp` device synchronization.
4. Executes a post-training **deterministic evaluation pass** (1024 parallel environments, clean observation rollout, zero action noise) to calculate test return.
5. Extracts training & evaluation metrics (Eval Return, Training Reward, Physics SPS, Env SPS, Framework Overhead %, and Value/Surrogate Losses).

---

## Running Benchmarks

### 1. Single-Commit Benchmark (`profiling/benchmark.py`)

Run benchmarks for all registered environments in the repository:

```bash
uv run --no-sync python -m profiling.benchmark
```

#### Filtering Environments
Filter by regex using `-f` or `--filter`:

```bash
# Run only Cartpole environments
uv run --no-sync python -m profiling.benchmark -f Cartpole

# Run only SO101 cube picking
uv run --no-sync python -m profiling.benchmark -f SO101PickCube

# Run all Vision-based environments
uv run --no-sync python -m profiling.benchmark -f Vision
```

#### Skip Evaluation Pass
```bash
uv run --no-sync python -m profiling.benchmark --skip_eval
```

#### JSON Output
```bash
uv run --no-sync python -m profiling.benchmark --json
```

---

### 2. Standalone Throughput Benchmark (`profiling/testspeed.py`)

Measure raw physics simulation SPS, environment stepping SPS, and framework overhead percentage:

```bash
# Measure CartpoleBalance throughput with default batch size (4096)
uv run --no-sync python -m profiling.testspeed --env=CartpoleBalance

# Benchmark SO101PickCube with custom batch size
uv run --no-sync python -m profiling.testspeed --env=SO101PickCube --num_envs=4096
```

---

## Checkpoints & TensorBoard Logs

Training checkpoints and TensorBoard event files from `profiling/benchmark.py` are saved to `/tmp/playground_benchmark_logs/<env_id>/run_<timestamp>/` (configurable via `--logdir`).
