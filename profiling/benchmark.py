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
"""benchmark.py: runs MuJoCo Playground RL training benchmarks on a single GPU.

Usage:
  uv run --no-sync python -m profiling.benchmark [flags]

Examples:
  uv run --no-sync python -m profiling.benchmark -f CartpoleBalance
  uv run --no-sync python -m profiling.benchmark -f SO101PickCube
  uv run --no-sync python -m profiling.benchmark --timeout_sec 600
"""

import json
import logging

from absl import app
from absl import flags
import profiling  # isort: skip_file
from mujoco_playground._src import benchmark

logger = logging.getLogger(__name__)

_FILTER = flags.DEFINE_string(
    "filter",
    ".*",
    "filter benchmark envs by name (regex)",
    short_name="f",
)
_DEVICE = flags.DEFINE_string(
    "device",
    "0",
    "CUDA GPU device ID to use (e.g. 0)",
)
_TIMEOUT_SEC = flags.DEFINE_integer(
    "timeout_sec",
    None,
    "override timeout limit in seconds for all envs",
)
_SKIP_EVAL = flags.DEFINE_boolean(
    "skip_eval",
    False,
    "skip post-training evaluation pass",
)
_SEED = flags.DEFINE_integer(
    "seed",
    0,
    "random seed for policy training and evaluation",
)
_JSON = flags.DEFINE_boolean(
    "json",
    False,
    "output results in JSON format",
)
_LOGDIR = flags.DEFINE_string(
    "logdir",
    "/tmp/playground_benchmark_logs",
    "directory to save training checkpoints and tensorboard logs",
)


def main(argv: list[str]) -> None:
  del argv  # Unused.

  benchmarks_list = list(
      benchmark.discover(
          env_filter=_FILTER.value, benchmarks=profiling.BENCHMARKS
      )
  )

  logger.info(
      "Discovered %d benchmarks matching '%s': [%s]",
      len(benchmarks_list),
      _FILTER.value,
      ", ".join(str(bm["name"]) for bm in benchmarks_list),
  )

  all_results = {}
  for bm in benchmarks_list:
    res = benchmark.run_all(
        bm=bm,
        device=_DEVICE.value,
        seed=_SEED.value,
        timeout_sec=_TIMEOUT_SEC.value,
        logdir=_LOGDIR.value,
        skip_eval=_SKIP_EVAL.value,
    )
    all_results[bm["name"]] = res
    if not _JSON.value:
      for k, v in res.items():
        print(f"{bm['name']}.{k} {v}")

  if _JSON.value:
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
  app.run(main)
