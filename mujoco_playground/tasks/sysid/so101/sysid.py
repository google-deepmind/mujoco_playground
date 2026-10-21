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
"""System identification for SO101 using mujoco.sysid."""

from typing import Literal
from typing import Sequence

from absl import logging
from etils import epath
from mujoco import sysid
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.robots.so101.sim import scene
from mujoco_playground.tasks.sysid.so101 import data_loader
import numpy as np


def build_parameters() -> sysid.ParameterDict:
  """Builds the ParameterDict for SO101 sysid."""
  params = sysid.ParameterDict()

  for joint in constant.JOINTS:
    params.add(
        sysid.Parameter(
            f"{joint}_armature",
            nominal=sim_constant.ARMATURE[joint],
            min_value=1e-4,
            max_value=0.5,
            modifier=lambda s, p, name=joint: setattr(
                s.joint(name), "armature", float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"{joint}_frictionloss",
            nominal=sim_constant.FRICTIONLOSS[joint],
            min_value=0.0,
            max_value=2.0,
            modifier=lambda s, p, name=joint: setattr(
                s.joint(name), "frictionloss", float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"{joint}_stiffness",
            nominal=sim_constant.STIFFNESS[joint],
            min_value=1.0,
            max_value=2500.0,
            modifier=lambda s, p, name=joint: sysid.apply_pgain(
                s, name, float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"{joint}_damping",
            nominal=sim_constant.DAMPING[joint],
            min_value=0.01,
            max_value=100.0,
            modifier=lambda s, p, name=joint: sysid.apply_dgain(
                s, name, float(p.value[0])
            ),
        )
    )

  return params


def run_sysid(
    trajectories: Sequence[data_loader.Trajectory],
    save_path: epath.PathLike | None = None,
    max_iters: int = 150,
    optimizer: Literal["mujoco", "scipy", "scipy_parallel_fd"] = "mujoco",
    generate_videos: bool = False,
):
  """Runs system identification on recorded SO101 trajectories.

  Args:
    trajectories: Sequence of loaded Trajectory instances.
    save_path: Directory path where results and report.html will be saved.
    max_iters: Maximum iterations for the nonlinear least squares optimizer.
    optimizer: Backend optimizer ('mujoco' or 'scipy').
    generate_videos: Whether to render rollout comparison videos in the report.

  Returns:
    Tuple of (opt_params, opt_result, report_builder).
  """
  if not trajectories:
    raise ValueError("No trajectories provided for system identification.")

  spec = scene.so101()
  model_sequences = data_loader.to_model_sequences(trajectories, spec)
  initial_params = build_parameters()

  delays_ms = np.concatenate([
      (t.follower_timestamps_ns - t.leader_timestamps_ns) * 1e-6
      for t in trajectories
  ])
  mean_delay_ms = float(np.mean(delays_ms))
  std_delay_ms = float(np.std(delays_ms))
  min_delay_ms = float(np.min(delays_ms))
  max_delay_ms = float(np.max(delays_ms))

  logging.info(
      "Estimated hardware command delay (t1 - t0): %.2f ± %.2f ms (min: %.2f"
      " ms, max: %.2f ms)",
      mean_delay_ms,
      std_delay_ms,
      min_delay_ms,
      max_delay_ms,
  )

  logging.info(
      "Building residual function for %d trajectory sequences...",
      len(trajectories),
  )
  residual_fn = sysid.build_residual_fn(models_sequences=[model_sequences])

  logging.info("Starting sys-id optimization using %s backend...", optimizer)
  opt_params, opt_result = sysid.optimize(
      initial_params=initial_params,
      residual_fn=residual_fn,
      optimizer=optimizer,
      max_iters=max_iters,
  )

  save_dir = epath.Path(save_path) if save_path is not None else None
  if save_dir is not None:
    save_dir.mkdir(parents=True, exist_ok=True)

  logging.info("Generating system identification HTML report...")
  report_builder = sysid.default_report(
      models_sequences=[model_sequences],
      initial_params=initial_params,
      opt_params=opt_params,
      residual_fn=residual_fn,
      opt_result=opt_result,
      title="SO101 System Identification Report",
      save_path=save_dir,
      generate_videos=generate_videos,
  )

  return opt_params, opt_result, report_builder
