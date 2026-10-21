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
"""Runs system identification on SO101 teleop demonstrations (MCAP).

Usage:
  uv run --extra so101 python -m mujoco_playground.tasks.sysid.so101.run --mcap_files=path/to/*.mcap
"""

from absl import app
from absl import flags
from absl import logging
from etils import epath
from mujoco_playground import paths
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.tasks.sysid.so101 import data_loader
from mujoco_playground.tasks.sysid.so101 import sysid
import numpy as np

_MCAP_FILES = flags.DEFINE_list(
    "mcap_files",
    None,
    "MCAP file paths, directories, or glob patterns (default: local"
    " <robot>_teleop_logs).",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for sys-id report and results (default:"
    " .local/so101_sysid_results).",
)
_MAX_ITERS = flags.DEFINE_integer(
    "max_iters",
    150,
    "Maximum iterations for optimization.",
)
_OPTIMIZER = flags.DEFINE_enum(
    "optimizer",
    "mujoco",
    ["mujoco", "scipy", "scipy_parallel_fd"],
    "Least-squares optimizer backend.",
)
_GENERATE_VIDEOS = flags.DEFINE_boolean(
    "generate_videos",
    False,
    "Whether to render rollout comparison videos in the report.",
)


def main(argv):
  del argv  # Unused.

  mcap_inputs = _MCAP_FILES.value or [
      str(paths.local_dir() / f"{constant.ROBOT}_teleop_logs")
  ]
  files = []
  for item in mcap_inputs:
    files.extend(data_loader.find_mcap_files(item))
  if not files:
    raise RuntimeError(f"No .mcap files found matching: {mcap_inputs}")

  logging.info("Loading %d MCAP file(s)...", len(files))
  trajectories = data_loader.load_trajectories(files)
  if not trajectories:
    raise RuntimeError(
        "No valid trajectory steps found in the provided MCAP files."
    )

  total_steps = sum(len(t.timestamps) for t in trajectories)
  delays_ms = np.concatenate([
      (t.follower_timestamps_ns - t.leader_timestamps_ns) * 1e-6
      for t in trajectories
  ])
  mean_delay_ms = float(np.mean(delays_ms))
  std_delay_ms = float(np.std(delays_ms))

  logging.info(
      "Loaded %d trajectory(ies) (%d total timesteps).",
      len(trajectories),
      total_steps,
  )
  print(
      f"\nEstimated hardware delay (t1 - t0): {mean_delay_ms:.2f} ±"
      f" {std_delay_ms:.2f} ms (min: {np.min(delays_ms):.2f} ms, max:"
      f" {np.max(delays_ms):.2f} ms)"
  )

  save_dir = (
      epath.Path(_OUTPUT_DIR.value)
      if _OUTPUT_DIR.value
      else paths.local_dir() / f"{constant.ROBOT}_sysid_results"
  )

  sysid.run_sysid(
      trajectories=trajectories,
      save_path=save_dir,
      max_iters=_MAX_ITERS.value,
      optimizer=_OPTIMIZER.value,  # pyrefly: ignore[bad-argument-type]
      generate_videos=_GENERATE_VIDEOS.value,
  )

  report_html = save_dir / "report.html"
  print(f"\nSystem identification finished successfully.")
  print(f"Interactive HTML Report: file://{report_html.resolve()}")


if __name__ == "__main__":
  app.run(main)
