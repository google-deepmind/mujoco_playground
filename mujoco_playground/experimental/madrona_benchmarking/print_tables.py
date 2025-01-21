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
# pylint: skip-file
"""Print tables to analyze bottlenecks for the Cartpole and Franka Pixel
environments."""

from pathlib import Path

import pandas as pd

from mujoco_playground.experimental.madrona_benchmarking.benchmark import MeasurementMode

fname = "madrona_mjx.csv"
fpath = Path(__file__).parent / "data" / fname
df = pd.read_csv(fpath)
# Select the correct mode and image size.
df = df[(df["bottleneck_mode"] is True) & (df["img_size"] == 64)]

fname_train = "madrona.csv"
fpath_train = Path(__file__).parent.parent / "data" / fname_train
df_train = pd.read_csv(fpath_train)


def calculate_average_fps(df, env):
  """
  Calculate the average FPS for a given env across all seeds.

  Parameters:
  df (pd.DataFrame): The input dataframe with columns 'step', 'seed',
    'training/walltime', 'env'.
  env (str): The environment to filter by.

  Returns:
  float: The average FPS for the given environment.
  """
  # Filter by the specified environment
  env_df = df[df["env"] == env]

  # Group by seed and calculate FPS for each seed
  fps_values = []
  for _, seed_df in env_df.groupby("seed"):
    # Sort by step to ensure chronological order
    seed_df = seed_df.sort_values("step")

    # Get the last two entries
    if len(seed_df) >= 2:
      last_two = seed_df.iloc[-2:]
      step_diff = last_two["step"].iloc[-1] - last_two["step"].iloc[0]
      time_diff = (
          last_two["training/walltime"].iloc[-1]
          - last_two["training/walltime"].iloc[0]
      )

      # Avoid division by zero
      if time_diff > 0:
        fps = step_diff / time_diff
        fps_values.append(fps)

  # Calculate and return the average FPS
  assert len(fps_values) == 5  # num seeds
  return sum(fps_values) / len(fps_values) if fps_values else 0


for env_name in ["CartpoleBalance", "PandaPickCubeCartesian"]:
  print(f"########## {env_name.upper()} ##########")
  fps_train = calculate_average_fps(df_train, env_name)
  tm_train = 1 / fps_train
  df_env = df[df["env_name"] == env_name]
  # tm_winf =  1/df[df["bottleneck_mode"] == \
  #   MeasurementMode.STEP_WINFERENCE.value]["fps"].mean()
  # tm_wvis =  1/df[df["bottleneck_mode"] == \
  #   MeasurementMode.STEP_WVISION.value]["fps"].mean()
  # tm_state = 1/df[df["bottleneck_mode"] == \
  #   MeasurementMode.STEP_STATE.value]["fps"].mean()
  tm_winf = (
      1
      / df_env[
          df_env["measurement_mode"] == MeasurementMode.STATE_VISION_INF.value
      ]["fps"].mean()
  )
  tm_wvis = (
      1
      / df_env[
          df_env["measurement_mode"] == MeasurementMode.STATE_VISION.value
      ]["fps"].mean()
  )
  tm_state = (
      1
      / df_env[df_env["measurement_mode"] == MeasurementMode.STATE.value][
          "fps"
      ].mean()
  )

  t_train = tm_train - tm_winf
  p_train = t_train / tm_train

  t_inf = tm_winf - tm_wvis
  p_inf = t_inf / tm_train

  t_vis = tm_wvis - tm_state
  p_vis = t_vis / tm_train

  t_state = tm_state
  p_state = t_state / tm_train

  # Assert the fractions sum to 1
  assert (
      abs((p_train + p_inf + p_vis + p_state) - 1.0) < 1e-9
  ), "Fractions do not sum to 1!"

  # Show mean FPS for each 'bottleneck_mode' as extra info
  df_modes = pd.DataFrame([
      {"Mode": "TRAIN (train+inf+vis+state)", "Mean FPS": 1 / tm_train},
      {"Mode": "STEP_WINFERENCE (inf+vis+state)", "Mean FPS": 1 / tm_winf},
      {"Mode": "STEP_WVISION (vis+state)", "Mean FPS": 1 / tm_wvis},
      {"Mode": "STEP_STATE (state)", "Mean FPS": 1 / tm_state},
  ])
  # print("Mean FPS per bottleneck mode:")
  # print(df_modes.to_string(index=False))
  # print()

  # Also show their time per step.
  df_modes["Time per step"] = [tm_train, tm_winf, tm_wvis, tm_state]
  df_modes["Time per step"] = df_modes["Time per step"].apply(
      lambda x: f"{x:.6e}"
  )
  print("Time per step for each bottleneck mode:")
  print(df_modes.to_string(index=False))
  print()

  # Show the breakdown of total training time
  df_breakdown = pd.DataFrame({
      "Component": ["Training", "Inference", "Vision", "State"],
      "Time per step": [t_train, t_inf, t_vis, t_state],
      "Fraction of Total Train": [p_train, p_inf, p_vis, p_state],
  })

  print("Breakdown of total training time by component:")
  print(df_breakdown.to_string(index=False))
  print()
