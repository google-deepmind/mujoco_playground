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

from typing import Any

"""Profiling and throughput benchmarks for MuJoCo Playground RL policies."""

DEFAULT_STANDARD_TIMEOUT_SEC: int = 1800  # 30 minutes

# Registered benchmarks.
BENCHMARKS: list[dict[str, Any]] = [
    {
        "name": "CartpoleBalance",
        "env_id": "CartpoleBalance",
        "timeout_sec": DEFAULT_STANDARD_TIMEOUT_SEC,
        "description": "Cartpole balance control (state-based)",
    },
    {
        "name": "CartpoleBalanceVision",
        "env_id": "CartpoleBalanceVision",
        "timeout_sec": DEFAULT_STANDARD_TIMEOUT_SEC,
        "description": "Cartpole balance control with pixel observations (CNN)",
    },
    {
        "name": "SO101PickCube",
        "env_id": "SO101PickCube",
        "timeout_sec": DEFAULT_STANDARD_TIMEOUT_SEC,
        "description": "SO101 robotic arm cube picking (state-based)",
    },
]
