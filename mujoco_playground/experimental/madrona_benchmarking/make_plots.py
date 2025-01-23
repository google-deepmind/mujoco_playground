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
# =================================================================================
# pylint: skip-file
"""
Generate plots comparing Madrona MJX with ManiSkill3 and Isaac Lab for the
CartpoleBalance environment.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sn


# Custom function for scientific notation formatting
def scientific_notation_formatter(x, _):
  if x == 0:
    return "0"
  else:
    return f"{x:.1e}"


# Format the y-axis as powers of 2
def log2_to_exp_formatter(x, _):
  return f"${{2^{{{int(x)}}}}}$"


# Custom formatter function
def format_power_of_10(num, _):
  # num = 2**num
  """Format a number into scientific notation as, for example, 2.3 x 10^4."""
  if num == 0:
    return "0"
  exponent = int(np.floor(np.log10(abs(num))))
  base = num / 10**exponent

  # Decide whether to show the base separately or just 10^exponent
  if abs(base - 1) < 1e-10:
    # If base is very close to 1, show only 10^exponent
    return r"$10^{%d}$" % exponent
  else:
    # Otherwise, show something like 2.3 x 10^4
    return f"${base:.1f}\times 10^{{{exponent}}}$"


def load_maniskill_result(name, state=False):
  file_name = name
  if state:
    file_name += "_state"
  file_path = f"data/maniskill3/{file_name}.csv"
  df = pd.read_csv(file_path)
  df_filtered = df[
      (df["gpu_type"] == "NVIDIA GeForce RTX 4090")
      & (df["env_id"] == "CartpoleBalanceBenchmark-v1")
  ]
  if not state:
    df_filtered = df_filtered[
        (df_filtered["num_cameras"] == 1) & (df_filtered["obs_mode"] == "rgb")
    ]

  df_dict = {
      "num_envs": df_filtered["num_envs"],
      "fps": df_filtered["env.step/fps"],
      "source_file": name,
  }
  if not state:
    df_dict["camera_size"] = df_filtered["camera_width"]
  return pd.DataFrame(df_dict)


def load_madrona_mjx_result(name, state=False):
  file_path = f"data/{name}.csv"
  df = pd.read_csv(file_path)
  df = df[df["bottleneck_mode"] == False]  # benchmarking only.
  if state:
    df = df[df["img_size"] == 0]
  else:
    df = df[df["img_size"] > 0]
  df_dict = {
      "num_envs": df["num_envs"],
      "fps": df["fps"],
      "source_file": name,
      "camera_size": df["img_size"],
  }
  return pd.DataFrame(df_dict)


# --- Vision ---
madrona_mjx_file = "data/madrona_mjx.csv"
df1 = load_maniskill_result("isaac_lab")
df2 = load_maniskill_result("maniskill")
df3 = load_madrona_mjx_result("madrona_mjx")
df3 = (
    df3.groupby(["num_envs", "camera_size", "source_file"])["fps"]
    .mean()
    .reset_index()
)

# Concatenate everything
df_pixels = pd.concat([df1, df2, df3], ignore_index=True)


# --- State ---
df1 = load_maniskill_result("isaac_lab", state=True)
df2 = load_maniskill_result("maniskill", state=True)
df3 = load_madrona_mjx_result("madrona_mjx", state=True)
df3 = df3.groupby(["num_envs", "source_file"])["fps"].mean().reset_index()

df_state = pd.concat([df1, df2, df3], ignore_index=True)

#### SET STYLE ####
# Tableau 10 color palette
tableau10 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]
tableau10 = [(r / 255, g / 255, b / 255) for r, g, b in tableau10]


def configure_plotting_sn_params(
    sn, SCALE=8, HEIGHT_SCALE=0.8, use_autolayout=True, font_size=22
):
  pd.set_option("mode.chained_assignment", None)
  sn.set(
      rc={
          "figure.figsize": (SCALE, int(HEIGHT_SCALE * SCALE)),
          "figure.autolayout": use_autolayout,
      }
  )
  sn.set(font_scale=1.5)
  sn.set_style(
      "whitegrid",
      {
          "font.family": "serif",
          "font.serif": "Times New Roman",
          "pdf.fonttype": 42,
          "ps.fonttype": 42,
          "font.size": font_size,
          "grid.linewidth": 0.5,
          "grid.color": "#EEEEEE",
      },
  )
  sn.color_palette("colorblind")
  return sn


sn = configure_plotting_sn_params(sn, font_size=24)

#### COMBINED STATE AND PIXELS FPS ####

# ----- Common Settings ----- #
sim_colors = tableau10[:3]
simulators = ["maniskill", "isaac_lab", "madrona_mjx"]

# ---------------------------
# 1) Prepare the STATE-BASED data
# ---------------------------
num_envs_to_plot_state = [1024, 2048, 4096, 8192, 16384]
df_state_filtered = df_state[
    df_state["num_envs"].isin(num_envs_to_plot_state)
].copy()

# Create a dummy camera_size = 0
df_state_filtered["camera_size"] = 0

df_seaborn_state = df_state_filtered.melt(
    id_vars=["num_envs", "camera_size", "source_file"],
    value_vars=["fps"],
    var_name="metric",
    value_name="value",
)

# Optional log2 scaling
# df_seaborn_state["value"] = np.log2(df_seaborn_state["value"])

# ---------------------------
# 2) Prepare the PIXELS (Render) data
# ---------------------------
camera_sizes = [80, 128, 256]  # Drop 224
num_envs_to_plot_pixels = [128, 256, 512, 1024]  # Or whatever you prefer
df_filtered = df_pixels[
    df_pixels["num_envs"].isin(num_envs_to_plot_pixels)
].copy()
df_filtered = df_filtered[df_filtered["camera_size"].isin(camera_sizes)].copy()

df_filtered["camera_size"] = df_filtered["camera_size"].astype(int)

df_seaborn_pixels = df_filtered.melt(
    id_vars=["num_envs", "camera_size", "source_file"],
    value_vars=["fps"],
    var_name="metric",
    value_name="value",
)

# ---------------------------
# 3) Create a single 2×2 figure
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

#####################################
# Subplot 1: State-based (top-left) #
#####################################
ax0 = axes[0]
sn.barplot(
    data=df_seaborn_state,
    x="num_envs",
    y="value",
    hue="source_file",
    hue_order=simulators,
    palette=sim_colors,
    errorbar=None,
    ax=ax0,
)
ax0.set_title("State-based")
ax0.set_xlabel("Batch Size")
ax0.set_ylabel("FPS")

ax0.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

#######################################################
# Subplots 2, 3, 4: 80x80, 128x128, and 256x256 Render #
#######################################################
for i, cs in enumerate(camera_sizes):
  ax = axes[i + 1]  # i=0 -> axes[1], i=1 -> axes[2], i=2 -> axes[3]
  data_subset = df_seaborn_pixels[df_seaborn_pixels["camera_size"] == cs]

  sn.barplot(
      data=data_subset,
      x="num_envs",
      y="value",
      hue="source_file",
      hue_order=simulators,
      palette=sim_colors,
      errorbar=None,
      ax=ax,
  )
  ax.set_title(f"{cs}×{cs} Render")
  ax.set_xlabel("Batch Size")
  ax.set_ylabel("FPS")
  ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
  # If you only want a single legend overall, remove the per-axes legend and
  #  place one globally at the end
  ax.legend().remove()

# Optionally add a single legend for the entire figure
handles, labels = ax0.get_legend_handles_labels()

# Set the title of the ax0 legend.
ax0.legend().set_title("")

fig.tight_layout()
plt.savefig(
    "figures/cartpole_benchmark_combined.png", dpi=600, bbox_inches="tight"
)

#### PIXELS FPS ####

for full_plot in [True, False]:
  camera_sizes_to_plot = [80, 128, 224, 256] if full_plot else [80, 256]
  num_envs_to_plot = [128, 256, 512, 1024]

  # Suppose df_pixels is your original DataFrame
  # Filter df_pixels to keep only the desired num_envs
  df_filtered = df_pixels[df_pixels["num_envs"].isin(num_envs_to_plot)].copy()
  df_filtered = df_filtered[
      df_filtered["camera_size"].isin(camera_sizes_to_plot)
  ]
  # Convert the camera_size column to ints.
  df_filtered["camera_size"] = df_filtered["camera_size"].astype(int)

  df_seaborn = df_filtered.melt(
      id_vars=["num_envs", "camera_size", "source_file"],
      value_vars=["fps"],
      var_name="metric",
      value_name="value",
  )

  # df_seaborn = df_filtered
  sim_colors = tableau10[:3]

  # Use seaborn for figure creation
  g = sn.FacetGrid(
      df_seaborn,
      col="camera_size",
      col_wrap=2,
      height=5,
      sharex=False,
      sharey=False,
      margin_titles=True,
  )

  simulators = ["maniskill", "isaac_lab", "madrona_mjx"]

  def plot_bars(data, **kwargs):
    sn.barplot(
        data=data,
        dodge=True,
        x="num_envs",
        y="value",
        hue="source_file",
        hue_order=simulators,
        palette=sim_colors,
        errorbar=None,
        **kwargs,
    )

  g.map_dataframe(plot_bars)

  # Format the y axes of each subplot.
  for i, ax in enumerate(g.axes.flat):
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
    if i == 0:
      # Set legend.
      ax.legend(title="")

  # Customize titles
  g.set_titles("{col_name}x{col_name} Render")
  g.set_axis_labels("Batch Size", "FPS")
  # g.add_legend()
  fname = "cartpole_benchmark"
  if full_plot:
    fname += "_full"
  plt.savefig(f"figures/{fname}.png", dpi=600)

#### STATE FPS ####
for _ in [None]:
  sim_colors = tableau10[:3]
  simulators = ["maniskill", "isaac_lab", "madrona_mjx"]

  num_envs_to_plot = [64, 256, 1024, 4096, 16384]
  df_state_filtered = df_state[
      df_state["num_envs"].isin(num_envs_to_plot)
  ].copy()

  # Dummy column for camera size.
  df_state_filtered["camera_size"] = 0
  df_seaborn = df_state_filtered.melt(
      id_vars=["num_envs", "camera_size", "source_file"],
      value_vars=["fps"],
      var_name="metric",
      value_name="value",
  )

  # Scale for better visualization

  df_seaborn["value"] = np.log2(df_seaborn["value"])

  # Use seaborn for figure creation
  g = sn.FacetGrid(
      df_seaborn,
      col="camera_size",
      height=5,
      sharex=False,
      sharey=False,
      margin_titles=False,
  )

  def plot_bars(data, **kwargs):
    sn.barplot(
        data=data,
        dodge=True,
        x="num_envs",
        y="value",
        hue="source_file",
        hue_order=simulators,
        palette=sim_colors,
        errorbar=None,
        **kwargs,
    )

  # set ylimits
  g.set(ylim=(14, 25))
  g.map_dataframe(plot_bars)

  # Custom function for formatting y-axis ticks
  def log2_to_exp_formatter(x, pos):
    return f"${{{2**int(x)}}}$"

  # Apply custom y-axis formatting for all subplots
  for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(FuncFormatter(log2_to_exp_formatter))
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right"
    )  # Slant x-axis labels

  # Customize titles
  g.set_titles("")
  g.set_axis_labels("Batch Size", "FPS")
  g.add_legend()

  plt.savefig(
      f"figures/cartpole_benchmark_state.png", dpi=600, bbox_inches="tight"
  )

#### Stacked Plots ####

# ----------------------------
# Data Preparation for Seaborn
# ----------------------------


for full_plot in [True, False]:
  camera_sizes_to_plot = [80, 128, 224, 256] if full_plot else [80, 256]
  num_envs_to_plot = [128, 256, 512, 1024]

  df_merged = df_pixels.merge(
      df_state[["num_envs", "source_file", "fps"]].rename(
          columns={"fps": "fps_state"}
      ),
      on=["num_envs", "source_file"],
      how="left",
  )

  # Compute time-per-step (tps) for total, state, and render
  df_merged["tps_total"] = 1.0 / df_merged["fps"]
  df_merged["tps_state"] = 1.0 / df_merged["fps_state"]
  # df_merged["tps_render"] = df_merged["tps_total"] - df_merged["tps_state"]

  # Filter to only the desired environments and melt data for Seaborn
  df_seaborn = df_merged[df_merged["num_envs"].isin(num_envs_to_plot)].copy()
  df_seaborn = df_seaborn[df_seaborn["camera_size"].isin(camera_sizes_to_plot)]
  # Melt into long format for Seaborn
  df_seaborn = df_seaborn.melt(
      id_vars=["num_envs", "camera_size", "source_file"],
      value_vars=["tps_state", "tps_total"],
      var_name="time_component",
      value_name="time_per_step",
  )

  # Convert the camera_size column to ints.
  df_seaborn["camera_size"] = df_seaborn["camera_size"].astype(int)

  # Add a column for bar stack order (state on bottom, render on top).
  # Seaborn doesn't support having bars on top of eachother so we plot total
  #  in back and state on top of it.
  df_seaborn["stack_order"] = df_seaborn["time_component"].map(
      {"tps_state": 0, "tps_total": 1}
  )

  def plot_stacked_bars_seaborn(data, camera_sizes, simulators, sim_colors):
    """
    Generate a FacetGrid of stacked bar charts comparing rendering and state
     times.
    """
    # Configure Seaborn
    global sn
    # sn.set_theme(style="whitegrid", font_scale=1.2)

    # Initialize a FacetGrid to create a subplot for each camera_size
    g = sn.FacetGrid(
        data,
        col="camera_size",
        col_wrap=2,
        height=5,
        sharex=False,
        sharey=True,
        margin_titles=True,
    )

    # Plot stacked bars
    def plot_bars(data, **kwargs):
      # Plot the "state" bars
      sn.barplot(
          data=data[data["stack_order"] == 0],
          x="num_envs",
          y="time_per_step",
          hue="source_file",
          hue_order=simulators,
          palette=sim_colors,
          errorbar=None,
          dodge=True,
          **kwargs,
      )
      # Overlay the "render" bars on top of "state"
      sn.barplot(
          data=data[data["stack_order"] == 1],
          x="num_envs",
          y="time_per_step",
          hue="source_file",
          hue_order=simulators,
          palette=sim_colors,
          errorbar=None,
          dodge=True,
          alpha=0.6,
          **kwargs,
      )

    g.map_dataframe(plot_bars)

    # Add titles and adjust layout
    g.set_titles("{col_name}x{col_name} Render")
    g.set_axis_labels("Batch Size", "Time per Step (s)")
    # g.add_legend()

    return g

  # ------------------------
  # Define Parameters & Plot
  # ------------------------

  # Simulators and their colors
  ordered_sources = ["maniskill", "isaac_lab", "madrona_mjx"]
  # sim_colors = sns.color_palette("tab10", n_colors=len(ordered_sources))
  sim_colors = tableau10[:3]

  # Call the plotting function
  g = plot_stacked_bars_seaborn(
      df_seaborn, camera_sizes_to_plot, ordered_sources, sim_colors
  )

  # Apply custom formatter to y-axis
  for i, ax in enumerate(g.axes.flat):
    # Add a legend to ax 0
    if i == 0:
      # ax.legend(title="")
      # Make a legend but only keep the first 3 elements of it
      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles[:3], labels[:3])
    # Set the x axis name
    ax.set_xlabel("Batch Size")
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

  # Save or show the plot
  fname = "cartpole_benchmark_stacked"
  if full_plot:
    fname += "_full"
  plt.savefig(f"figures/{fname}.png", dpi=600, bbox_inches="tight")

##########################
#### Main Figure Plot ####
##########################

font_params = {
    "font.size": 18,  # Sets the font size for all elements
    "axes.titlesize": 20,  # Sets the font size for subplot titles
    "axes.labelsize": 18,  # Sets the font size for axis labels
    "xtick.labelsize": 16,  # Sets the font size for x-axis tick labels
    "ytick.labelsize": 16,  # Sets the font size for y-axis tick labels
    "legend.fontsize": 16,  # Sets the font size for the legend
    "figure.titlesize": 22,  # Sets the font size for the figure title
}
# Increase by factor of 2.
font_params = {k: v * 1.45 for k, v in font_params.items()}
plt.rcParams.update(font_params)

# Read CSV (adjust the path if needed)
df = pd.read_csv("data/madrona_mjx.csv")

# Filter: only bottleneck_mode = True, measurement_mode = 1
df = df[(df["bottleneck_mode"] == True) & (df["measurement_mode"] == 1)]

# Filter: only num_envs = 1024
df = df[df["num_envs"] == 1024]

# Rename columns for convenience
df = pd.DataFrame({
    "num_envs": df["num_envs"],
    "fps": df["fps"],
    "env_name": df["env_name"],
    "camera_size": df["img_size"],
})

# Keep only the environment names of interest
envs = ["CartpoleBalance", "PandaPickCubeCartesian"]
df = df[df["env_name"].isin(envs)]

# Keep only the camera sizes of interest
camera_sizes_to_plot = [64, 128, 256, 512]
df = df[df["camera_size"].isin(camera_sizes_to_plot)]

# Convert camera_size to int just in case
df["camera_size"] = df["camera_size"].astype(int)

# Melt into Seaborn’s long-form requirement
df_seaborn = df.melt(
    id_vars=["num_envs", "camera_size", "env_name"],
    value_vars=["fps"],  # which columns to “unpivot”
    var_name="metric",  # name for the variable column
    value_name="value",  # name for the value column
)

# Create a FacetGrid: 1 simulator, but 2 different RL envs across columns
g = sn.FacetGrid(
    df_seaborn,
    col="env_name",
    col_wrap=2,
    height=5,
    sharex=False,
    sharey=False,
    margin_titles=True,
)


def float_to_sci_str(value: float, precision: int = 1) -> str:
  """
  Convert a float to scientific notation with a trimmed exponent.
  Example: 44000.0 -> '4.4e4'

  :param value: The float to format
  :param precision: Number of digits after the decimal point (default=1)
  :return: A string representation in scientific notation
  """
  # Format with the given precision: e.g. "4.4e+04"
  sci = f"{value:.{precision}e}"
  # Split into base and exponent: ["4.4", "+04"]
  base, exp_str = sci.split("e")
  # Convert exponent to integer to remove '+0...' or '-0...'
  exponent = int(exp_str)
  # Rebuild the string with the simplified exponent
  return f"{base}e{exponent}"


def plot_bars(data, **kwargs):
  sn.barplot(
      data=data,
      x="camera_size",
      y="value",
      order=camera_sizes_to_plot,  # ensure order of x-axis
      hue="env_name",
      errorbar=None,
      palette=tableau10[2:],
      **kwargs,
  )


# Map our function to each facet
g = sn.FacetGrid(
    df_seaborn,
    col="env_name",
    col_wrap=2,
    height=5,
    sharex=False,
    sharey=False,
    margin_titles=True,
)

g.map_dataframe(plot_bars)


def scientific_notation_formatter(x, pos):
  if x == 0:
    return "0"
  else:
    return f"{x:.0e}"


# Format each subplot’s axis
ticks = [
    [1e5, 2e5, 3e5, 4e5],
    [1e4, 2e4, 3e4, 4e4],
]
ylims = [4.1e5, 4.1e4]
for i, ax in enumerate(g.axes.flat):
  ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
  # Set the y range.
  ax.set_ylim(0, ylims[i])
  ax.set_yticks(ticks[i])
  ax.grid(color="gray", linestyle="--", linewidth=2, axis="y")

ax = g.axes[0]

# Heights you want to annotate:
height_third_bar = 4.4e4
height_fourth_bar = 1.9e4

# X-coordinates corresponding to camera_size = [64, 128, 256, 512].
# By default, Seaborn places bars at x=0,1,2,3.
x_third_bar = 2
x_fourth_bar = 3

# Label for the third bar
ax.text(
    x_third_bar - 0.5,
    height_third_bar,
    "4.4e4",
    ha="left",
    va="bottom",
    fontsize=18,
)

# Label for the fourth bar
ax.text(
    x_fourth_bar + 0.5,
    height_fourth_bar,
    "1.9e4",
    ha="right",
    va="bottom",
    fontsize=18,
)

g.set_titles(col_template="{col_name}", y=1.1)
g.set_axis_labels("Image Resolution", "FPS")

plt.tight_layout()
plt.savefig("figures/madrona_mjx_main.png", dpi=600, bbox_inches="tight")
