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

"""testspeed: benchmark MuJoCo Playground simulation & environment throughput.

Measures raw physics simulation SPS (using mujoco_warp methodology), full
environment stepping SPS, and framework overhead percentage.

Usage:
  uv run --no-sync python -m profiling.testspeed --env=CartpoleBalance
  uv run --no-sync python -m profiling.testspeed --env=SO101PickCube
  --num_envs=4096
"""

import json

from absl import app
from absl import flags
from mujoco_playground._src import benchmark as core

_ENV = flags.DEFINE_string(
    "env",
    "CartpoleBalance",
    "Environment ID from registry (e.g. CartpoleBalance, SO101PickCube)",
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs",
    4096,
    "Number of parallel environments for throughput measurement",
)
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps",
    200,
    "Number of control steps to benchmark",
)
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    50,
    "Number of warmup steps prior to timing",
)
_DEVICE = flags.DEFINE_string(
    "device",
    "cuda:0",
    "CUDA GPU device ID to use (e.g. cuda:0)",
)
_ENV_OVERRIDES = flags.DEFINE_string(
    "env_overrides",
    None,
    "JSON encoded dictionary of environment overrides",
)


def main(argv: list[str]) -> None:
  del argv  # Unused.
  num_envs = _NUM_ENVS.value
  num_steps = _NUM_STEPS.value
  warmup_steps = _WARMUP_STEPS.value

  env_overrides = None
  if _ENV_OVERRIDES.value:
    env_overrides = json.loads(_ENV_OVERRIDES.value)

  res = core.testspeed(
      env_id=_ENV.value,
      num_envs=num_envs,
      num_steps=num_steps,
      warmup_steps=warmup_steps,
      device=_DEVICE.value,
      env_overrides=env_overrides,
  )
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  app.run(main)
