# Copyright 2024 DeepMind Technologies Limited
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
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display


class TrainingPlotter:
  def __init__(
    self,
    max_timesteps: int = 50_000_000,
    figsize: Tuple[int, int] = (12, 8),
    max_cols: int = 3,
  ):
    self.max_timesteps = max_timesteps
    self.max_cols = max_cols

    # Default main metrics that we always want to plot.
    self.default_metrics = [
      "eval/episode_reward",
      "eval/avg_episode_length",
      "steps_per_second",
    ]

    self.metrics = []
    self.metrics_std = []
    self.metric_labels = []
    self.error_metrics = []
    self.error_metrics_std = []
    self.error_metric_labels = []
    self.reward_detail_metrics = []
    self.reward_detail_metrics_std = []
    self.reward_detail_metric_labels = []
    self.termination_metrics = []
    self.termination_metrics_std = []
    self.termination_metric_labels = []

    self.x_data = []
    self.times = [datetime.now()]
    self.metrics_data = {}
    self.metrics_std_data = {}

    self.fps_data = []  # Store calculated steps per second

    # Use default matplotlib style
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.edgecolor"] = "#888888"
    plt.rcParams["axes.linewidth"] = 0.8

    # Create initial figure and axes - we'll resize this later
    n_cols = min(self.max_cols, 1)  # Start with at least 1 column, but respect max_cols
    n_rows = 1
    self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    self.axes = np.array([[self.axes]])

    # Set figure background to white for clean look
    self.fig.patch.set_facecolor("white")

    # Set up the layout with reasonable spacing
    plt.tight_layout(pad=2.5, h_pad=1.5, w_pad=1.0)

  def _get_label_from_metric(self, metric: str) -> str:
    parts = metric.split("/")
    if len(parts) > 1:
      label = parts[-1]
    else:
      label = metric
    if label == "episode_reward":
      return "reward_per_episode"
    elif label == "avg_episode_length":
      return "episode_length"
    elif label == "steps_per_second":
      return "steps_per_second"
    else:
      return label

  def _initialize_metrics(self, metrics: Dict[str, Any]) -> None:
    """Initialize all metrics from the first metrics dictionary."""
    # Start with default metrics
    self.metrics = self.default_metrics.copy()
    self.metrics_std = [f"{m}_std" for m in self.metrics]
    self.metric_labels = [self._get_label_from_metric(m) for m in self.metrics]

    # Initialize data storage for default metrics
    for metric in self.metrics:
      self.metrics_data[metric] = []
    for metric_std in self.metrics_std:
      self.metrics_std_data[metric_std] = []

    # Find all reward detail metrics (eval/episode_reward/*)
    reward_prefix = "eval/episode_reward/"
    for key in metrics:
      if (
        key.startswith(reward_prefix)
        and not key.endswith("_std")
        and key != "eval/episode_reward"
      ):
        self.reward_detail_metrics.append(key)
        self.reward_detail_metrics_std.append(f"{key}_std")
        label = key[len(reward_prefix) :]
        self.reward_detail_metric_labels.append(label)  # Keep underscores

        # Initialize data storage
        self.metrics_data[key] = []
        self.metrics_std_data[f"{key}_std"] = []

    # Find all error metrics (eval/episode_error/*)
    error_prefix = "eval/episode_error/"
    for key in metrics:
      if key.startswith(error_prefix) and not key.endswith("_std"):
        self.error_metrics.append(key)
        self.error_metrics_std.append(f"{key}_std")
        label = key[len(error_prefix) :]
        self.error_metric_labels.append(label)  # Keep underscores

        # Initialize data storage
        self.metrics_data[key] = []
        self.metrics_std_data[f"{key}_std"] = []

    # Find all termination metrics (eval/episode_termination/*)
    termination_prefix = "eval/episode_termination/"
    for key in metrics:
      if key.startswith(termination_prefix) and not key.endswith("_std"):
        self.termination_metrics.append(key)
        self.termination_metrics_std.append(f"{key}_std")
        label = key[len(termination_prefix) :]
        self.termination_metric_labels.append(label)  # Keep underscores

        # Initialize data storage
        self.metrics_data[key] = []
        self.metrics_std_data[f"{key}_std"] = []

  def update(self, num_steps: int, metrics: Dict[str, float]) -> None:
    self.x_data.append(num_steps)
    current_time = datetime.now()
    self.times.append(current_time)

    # Calculate steps per second if we have at least two data points
    if len(self.x_data) > 1:
      time_diff = (current_time - self.times[-2]).total_seconds()
      steps_diff = self.x_data[-1] - self.x_data[-2]
      if time_diff > 0:
        fps = steps_diff / time_diff
      else:
        fps = 0
      self.fps_data.append(fps)
    else:
      self.fps_data.append(0)  # First point has no previous data to compare

    # Initialize metrics if this is the first update.
    if len(self.x_data) == 1:
      self._initialize_metrics(metrics)
      # Add fps to metrics data structure
      self.metrics_data["steps_per_second"] = []
      self.metrics_std_data["steps_per_second_std"] = []

    # Update all metrics data.
    all_metrics = (
      self.metrics
      + self.reward_detail_metrics
      + self.error_metrics
      + self.termination_metrics
    )
    all_metrics_std = (
      self.metrics_std
      + self.reward_detail_metrics_std
      + self.error_metrics_std
      + self.termination_metrics_std
    )

    for metric in all_metrics:
      if metric == "steps_per_second":
        self.metrics_data[metric].append(self.fps_data[-1])
      elif metric in metrics:
        self.metrics_data[metric].append(metrics[metric])
      else:
        last_value = self.metrics_data[metric][-1] if self.metrics_data[metric] else 0
        self.metrics_data[metric].append(last_value)

    for metric_std in all_metrics_std:
      if metric_std in metrics:
        self.metrics_std_data[metric_std].append(metrics[metric_std])
      else:
        last_value = (
          self.metrics_std_data[metric_std][-1]
          if self.metrics_std_data[metric_std]
          else 0
        )
        self.metrics_std_data[metric_std].append(last_value)

    clear_output(wait=True)

    # Combine all metrics for plotting.
    all_metrics = (
      self.metrics
      + self.reward_detail_metrics
      + self.error_metrics
      + self.termination_metrics
    )
    all_metrics_std = (
      self.metrics_std
      + self.reward_detail_metrics_std
      + self.error_metrics_std
      + self.termination_metrics_std
    )
    all_labels = (
      self.metric_labels
      + self.reward_detail_metric_labels
      + self.error_metric_labels
      + self.termination_metric_labels
    )

    # Calculate grid dimensions using max_cols.
    total_plots = len(all_metrics)
    n_cols = min(self.max_cols, total_plots)  # Use max_cols parameter.
    n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division.

    # Check if we need to resize the axes grid
    if n_rows > self.axes.shape[0] or n_cols > self.axes.shape[1]:
      plt.close(self.fig)
      # Calculate a better figure size based on the number of plots and columns
      width = max(12, n_cols * 3.5)  # 3.5 inches per column
      height = max(8, n_rows * 2.5)  # 2.5 inches per row
      self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=(width, height))

      # Handle case where there's only one plot.
      if n_rows == 1 and n_cols == 1:
        self.axes = np.array([[self.axes]])
      elif n_rows == 1:
        self.axes = np.array([self.axes])
      elif n_cols == 1:
        self.axes = np.array([[ax] for ax in self.axes])

    # Plot all metrics
    self._plot_metrics(all_metrics, all_metrics_std, all_labels, self.axes)

    # Add a single x-axis label at the bottom of the figure.
    self.fig.text(
      0.5, 0.01, "# environment steps", ha="center", fontsize=12, fontweight="bold"
    )

    # Update layout and display.
    self.fig.tight_layout(pad=2.5, h_pad=1.5, w_pad=1.0)
    self.fig.subplots_adjust(bottom=0.08)
    display(self.fig)

  def _plot_metrics(
    self,
    metrics_list: List[str],
    metrics_std_list: List[str],
    labels_list: List[str],
    axes_grid: np.ndarray,
  ) -> None:
    """Plot a set of metrics on the given axes grid."""
    for i, (metric, metric_std, label) in enumerate(
      zip(metrics_list, metrics_std_list, labels_list)
    ):
      row, col = i // axes_grid.shape[1], i % axes_grid.shape[1]
      if row < axes_grid.shape[0] and col < axes_grid.shape[1]:
        ax = axes_grid[row][col]
        ax.clear()
        ax.set_xlim([0, self.max_timesteps * 1.25])

        # Remove x-axis labels from all subplots
        ax.set_xlabel("")

        # Make tick labels smaller to save space
        ax.tick_params(axis="both", which="major", labelsize=9)

        # Add subtle grid for better readability
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.2)

        # Clean background
        ax.set_facecolor("white")

        # Format y-axis with fewer decimal places for cleaner look
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        y_values = self.metrics_data[metric]
        yerr_values = (
          self.metrics_std_data[metric_std]
          if metric_std in self.metrics_std_data
          else None
        )

        if y_values:
          # Add prefix based on metric type
          prefix = ""
          if "eval/episode_error/" in metric:
            prefix = "error/"
          elif "eval/episode_reward/" in metric:
            prefix = "reward/"
          elif "eval/episode_termination/" in metric:
            prefix = "termination/"

          # Use smaller font for title to save space
          ax.set_title(
            f"{prefix}{label}: {y_values[-1]:.3f}", fontsize=10, fontweight="bold"
          )

          # Plot the line with improved styling
          line = ax.errorbar(
            self.x_data,
            y_values,
            yerr=yerr_values,
            color="black",
            linewidth=1.5,
            elinewidth=0.7,
            capsize=2,
          )

          # Add very subtle shading under the curve for better visibility
          ax.fill_between(self.x_data, 0, y_values, alpha=0.05, color="black")

    # Hide unused subplots
    for i in range(len(metrics_list), axes_grid.shape[0] * axes_grid.shape[1]):
      row, col = i // axes_grid.shape[1], i % axes_grid.shape[1]
      if row < axes_grid.shape[0] and col < axes_grid.shape[1]:
        axes_grid[row][col].set_visible(False)

  def save_figure(self, filename: str) -> None:
    self.fig.savefig(filename, dpi=300, bbox_inches="tight")
