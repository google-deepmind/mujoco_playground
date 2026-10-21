# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for creating and managing line plots within the MuJoCo viewer.

This module provides classes to facilitate the creation of both single and
multi-line plots that can be rendered as MuJoCo figures (`mjvFigure`). These
plots
are useful for visualizing data streams in real-time during a simulation.
"""

import collections
import colorsys
import copy

import mujoco
import numpy as np

# Defines the figure's background color and transparency (RGBA).
_FIGURE_RGBA = (0.0, 0.0, 0.0, 0.4)

# Sets the grid dimensions for the plot (vertical, horizontal lines).
_GRID_SIZE = (5, 5)

# The maximum number of data points to store and display per line.
_MAX_NUM_POINTS = 100


class LinePlot:
  """Manages a single-line plot for the MuJoCo viewer.

  This class wraps a `mujoco.MjvFigure` to provide a simple interface for
  creating and updating a 2D line plot.

  Attributes:
    figure: The underlying `mujoco.MjvFigure` object.
  """

  def __init__(self, title: str, xlabel: str):
    """Initializes a new LinePlot instance.

    Args:
      title: The title of the plot.
      xlabel: The label for the x-axis.
    """
    self._fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(self._fig)

    self._fig.figurergba[:] = _FIGURE_RGBA
    self._fig.gridsize[:] = _GRID_SIZE
    self._fig.linewidth = 1
    self._fig.linepnt[1:] = 0
    self._fig.title = title
    self._fig.xlabel = xlabel
    self._data = collections.deque(maxlen=_MAX_NUM_POINTS)
    self._init_fig = copy.deepcopy(self._fig)

  def clear(self):
    """Clears all data points from the plot."""
    self._data.clear()
    self._fig = copy.deepcopy(self._init_fig)

  def add_point(self, x: float, y: float):
    """Adds a new data point to the plot.

    If the number of points exceeds `_MAX_NUM_POINTS`, the oldest point is
    removed to maintain a fixed-size buffer.

    Args:
      x: The x-coordinate of the data point.
      y: The y-coordinate of the data point.
    """
    self._data.append([x, y])

    self._fig.linepnt[0] = len(self._data)
    # Flatten the deque of points into the figure's linedata array.
    self._fig.linedata[0, 0 : 2 * len(self._data)] = np.concatenate(
        self._data, axis=None
    )

    # Get the ranges.
    self._fig.range[0][0] = self._data[0][0]
    self._fig.range[0][1] = self._data[-1][0]

  @property
  def figure(self) -> mujoco.MjvFigure:
    """Returns the underlying `mujoco.MjvFigure` instance."""
    return self._fig


class MultiLinePlot:
  """Manages a multi-line plot for the MuJoCo viewer.

  This class allows for plotting multiple data series on the same figure,
  each with its own legend entry and color.

  Attributes:
    figure: The underlying `mujoco.MjvFigure` object.
  """

  def __init__(
      self,
      title: str,
      xlabel: str,
      legends: list[str],
      colors: dict[str, tuple[float, float, float]] | None = None,
  ):
    """Initializes a new MultiLinePlot instance.

    Args:
      title: The title of the plot.
      xlabel: The label for the x-axis.
      legends: A list of strings, where each string is the name of a line to be
        plotted. The order of names determines their plot index.
      colors: An optional dictionary mapping legend names to RGB color tuples.
        If not provided, colors will be generated automatically by evenly
        spacing hues in the HSV color space.
    """
    n_plots = len(legends)

    self._fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(self._fig)

    self._fig.flg_legend = True
    self._fig.figurergba[:] = _FIGURE_RGBA
    self._fig.gridsize[:] = _GRID_SIZE
    self._fig.linewidth = 1
    self._fig.linepnt[n_plots:] = 0
    self._fig.title = title
    self._fig.xlabel = xlabel

    self._data_dicts = {
        name: collections.deque(maxlen=_MAX_NUM_POINTS) for name in legends
    }
    color_fraction = 1.0 / n_plots
    for i, name in enumerate(legends):
      if colors is None:
        # Automatically assign a color if none is provided.
        rgb = colorsys.hsv_to_rgb(color_fraction * i, 1.0, 1.0)
      else:
        rgb = colors[name]
      self._fig.linergb[i][:] = rgb
      self._fig.linename[i] = name

    self._init_fig = copy.deepcopy(self._fig)

  def clear(self):
    """Clears all data points from all lines in the plot."""
    for data_deque in self._data_dicts.values():
      data_deque.clear()
    self._fig = copy.deepcopy(self._init_fig)

  def add_points(self, x: float, y: dict[str, float]):
    """Adds a new set of data points to the plot at a common x-coordinate.

    Each key in the `y` dictionary must correspond to a line's legend name
    defined during initialization. If the number of points in any line exceeds
    `_MAX_NUM_POINTS`, the oldest point for that line is removed.

    Args:
      x: The common x-coordinate for the new data points.
      y: A dictionary mapping all legend names to their corresponding y-values.

    Raises:
      ValueError: If a key corresponding to a legend name is not found in
        the `y` dictionary.
    """
    for i, name in enumerate(self._data_dicts.keys()):
      if name not in y:
        raise ValueError(f'{name} not found in `y`.')

      data = self._data_dicts[name]
      data.append([x, y[name]])

      self._fig.linepnt[i] = len(data)
      # Flatten the deque of points into the figure's linedata array.
      self._fig.linedata[i, 0 : 2 * len(data)] = np.concatenate(data, axis=None)

    # Get the ranges.
    range_data = next(iter(self._data_dicts.values()))
    self._fig.range[0][0] = range_data[0][0]
    self._fig.range[0][1] = range_data[-1][0]

  @property
  def figure(self) -> mujoco.MjvFigure:
    """Returns the underlying `mujoco.MjvFigure` instance."""
    return self._fig
