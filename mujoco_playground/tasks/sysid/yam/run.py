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
"""Runs system identification on YAM excitation logs (from `collect`).

Usage:
  uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.run
"""

from collections.abc import Sequence
import json

from absl import app
from absl import flags
from etils import epath
from mujoco_playground import paths
from mujoco_playground.tasks.sysid.yam import data_loader
from mujoco_playground.tasks.sysid.yam import excitation
from mujoco_playground.tasks.sysid.yam import sysid

_INPUT_DIR = flags.DEFINE_string(
    "input_dir", None, "ep*.npz logs (default: <local dir>/yam_sysid_logs)."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Report directory (default: <local dir>/yam_sysid_results).",
)
_VALIDATION = flags.DEFINE_list(
    "validation",
    [excitation.EPISODES["validation"]],
    "Episodes held out of the fit and only evaluated.",
)
_MAX_ITERS = flags.DEFINE_integer("max_iters", 150, "Optimizer iterations.")
_OPTIMIZER = flags.DEFINE_enum(
    "optimizer",
    "mujoco",
    ["mujoco", "scipy", "scipy_parallel_fd"],
    "Least-squares optimizer backend.",
)
_FREEZE_KP = flags.DEFINE_list(
    "freeze_kp",
    ["joint1"],
    "Actuators whose kp stays at the I2RT SDK's gain.",
)
_FREEZE_JOINTS = flags.DEFINE_list(
    "freeze_joints",
    [],
    "Actuators whose four parameters stay at their nominals.",
)
_FREEZE_PARAMS = flags.DEFINE_list(
    "freeze_params",
    [],
    "Individual parameters that stay at their nominals, e.g. the ones the"
    " previous run listed as unidentified.",
)
_WRIST_CAMERA = flags.DEFINE_boolean(
    "wrist_camera", False, "Also fit the wrist camera's mass and CoM."
)
_GENERATE_VIDEOS = flags.DEFINE_boolean(
    "generate_videos", False, "Render rollout comparison videos."
)


def main(argv: Sequence[str]) -> None:
  del argv
  input_dir = _INPUT_DIR.value or paths.local_dir() / "yam_sysid_logs"
  files = data_loader.find_episode_files(input_dir)
  if not files:
    raise RuntimeError(
        f"No ep*.npz in {input_dir}. Record them with"
        " `python -m mujoco_playground.tasks.sysid.yam.collect`."
    )
  trajectories = [data_loader.load_trajectory(f) for f in files]
  train = [t for t in trajectories if t.name not in _VALIDATION.value]
  validation = [t for t in trajectories if t.name in _VALIDATION.value]
  save_dir = (
      epath.Path(_OUTPUT_DIR.value)
      if _OUTPUT_DIR.value
      else paths.local_dir() / "yam_sysid_results"
  )
  result = sysid.run_sysid(
      train=train,
      validation=validation,
      save_path=save_dir,
      max_iters=_MAX_ITERS.value,
      optimizer=_OPTIMIZER.value,  # pyrefly: ignore[bad-argument-type]
      generate_videos=_GENERATE_VIDEOS.value,
      initial_params=sysid.build_parameters(
          freeze_kp=_FREEZE_KP.value,
          freeze_joints=_FREEZE_JOINTS.value,
          freeze_params=_FREEZE_PARAMS.value,
          wrist_camera=_WRIST_CAMERA.value,
      ),
  )
  print(json.dumps(result.summary["rmse"], indent=1))
  unidentified = sysid.unidentified_parameters(result.summary)
  if unidentified:
    print(f"\nUnidentified parameters: {json.dumps(unidentified, indent=1)}")
    frozen = [*_FREEZE_PARAMS.value, *unidentified]
    print(f"Suggested flag: --freeze_params={','.join(frozen)}")
  print(f"\nInteractive HTML Report: file://{save_dir / 'report.html'}")


if __name__ == "__main__":
  app.run(main)
