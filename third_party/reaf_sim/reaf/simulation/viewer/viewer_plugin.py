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
"""Viewer-side plugin for REAF Studio Viewer."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import logging
from typing import Any

from mujoco.experimental.studio import messages
from mujoco.experimental.studio import plugin
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
import numpy as np
from reaf.simulation.viewer import line_plots
from reaf.simulation.viewer import messages as reaf_messages
import tree

from mujoco.experimental.dear_imgui import dear_imgui as imgui  # isort: skip
from mujoco.experimental.implot import implot  # isort: skip

_module_logger = logging.getLogger(__name__)

# ImPlot flags for the reward plot.
_REWARD_PLOT_FLAGS = (
    implot.Flags.NoBoxSelect.value  # Disable drag-to-select regions.
)

_REWARD_AXIS_FLAGS = (
    implot.AxisFlags.NoGridLines.value  # Hide background grid lines.
    | implot.AxisFlags.NoTickMarks.value  # Hide small tick marks on the axis.
)

# TODO(taylorhowell): Add CameraDepth.


class Image2D:
  """Manages 2D image textures and Studio sidebar GUI state."""

  def __init__(self, title: str = 'Camera RGB', scale: int = 1) -> None:
    self._title = title
    self._scale = max(1, int(scale))
    self._images: dict[str, np.ndarray] = {}
    self._textures: dict[str, int] = {}
    self._dirty: set[str] = set()
    self._shown: set[str] | None = None
    self._dock_id: int = 0
    self._focused: bool = False

  def set_images(self, images: Mapping[str, Any]) -> None:
    """Converts and stores 2D+ image streams for rendering."""
    self._images = {}
    for key, img in images.items():
      try:
        while len(img.shape) > 3:
          img = img[0]
      except AttributeError:
        continue
      try:
        img = img.detach().cpu().numpy()
      except AttributeError:
        img = np.asarray(img)
      if img.ndim == 2:
        img = img[..., None]
      elif img.ndim == 3 and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
      if img.ndim != 3 or img.size == 0:
        continue
      if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
      elif img.shape[-1] > 4:
        img = img[..., :3]
      if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
      self._images[key] = img
    if not self._images:
      self._shown = None
    elif self._shown is None:
      self._shown = set(self._images)
    self._dirty = set(self._images)

  def update(self, viewer: Any) -> None:
    """Uploads dirty textures for visible streams."""
    if not self._images or viewer is None or not self._shown:
      return
    for name, img in self._images.items():
      if name in self._dirty and name in self._shown:
        scaled = (
            np.repeat(np.repeat(img, self._scale, axis=0), self._scale, axis=1)
            if self._scale > 1
            else img
        )
        self._textures[name] = viewer.upload_image(
            self._textures.get(name, 0),
            np.ascontiguousarray(scaled).tobytes(),
            scaled.shape[1],
            scaled.shape[0],
            bpp=scaled.shape[2],
        )
        self._dirty.discard(name)

  def build_gui(self) -> None:
    """Builds the sidebar tab with raw-pixel images for each stream."""
    if not self._images or not imgui.GetCurrentContext():
      return
    if not self._dock_id:
      imgui.Begin('Inspector')
      self._dock_id = imgui.GetWindowDockID()
      imgui.End()
      if self._dock_id:
        imgui.SetNextWindowDockID(self._dock_id, imgui.Cond.Once)
        imgui.SetNextWindowFocus()
    elif not self._focused:
      imgui.SetNextWindowFocus()
      self._focused = True
    node_flags = (
        int(imgui.TreeNodeFlags.SpanAvailWidth)
        | int(imgui.TreeNodeFlags.Framed)
        | int(imgui.TreeNodeFlags.DefaultOpen)
    )
    shown: set[str] = set()
    opened, _ = imgui.Begin(self._title)
    if opened:
      changed, self._scale = imgui.SliderInt('Scale', self._scale, 1, 16, '%dx')
      if changed:
        self._dirty = set(self._images)
      for name, img in self._images.items():
        if imgui.TreeNodeEx(name, node_flags):
          shown.add(name)
          tex_id = self._textures.get(name, 0)
          if tex_id:
            imgui.Image(
                tex_id,
                imgui.Vec2(
                    img.shape[1] * self._scale, img.shape[0] * self._scale
                ),
            )
          imgui.TreePop()
    imgui.End()
    self._shown = shown


def _flatten_reward_tree(reward_tree: Any) -> list[tuple[str, Any]]:
  """Returns (leaf_name, leaf) pairs for a possibly nested reward or spec.

  Nested dict keys are joined with '/', e.g. ``terms/weighted/height``. A
  top-level leaf has the empty name.

  Args:
    reward_tree: A reward or reward spec, as a leaf or nested dicts of leaves.
  """
  return [
      ('/'.join(str(key) for key in key_path), leaf)
      for key_path, leaf in tree.flatten_with_path(reward_tree)
  ]


def _reward_leaf_legends(leaf_name: str, num_elements: int) -> list[str]:
  """Returns the plot legends for a scalar or vector reward leaf."""
  if num_elements == 1:
    return [leaf_name or 'reward']
  return [f'{leaf_name}[{i}]' for i in range(num_elements)]


def reward_spec_legends(
    reward_spec: Any, keys: Iterable[str] | None = None
) -> list[str]:
  """Returns the reward plot legends for a reward spec.

  Every scalar or 1-D leaf of the spec is plotted; other leaves are skipped.

  Args:
    reward_spec: The environment's reward spec: an array spec or nested dicts of
      array specs.
    keys: Optional subset of the legends to plot, in order.

  Raises:
    ValueError: If a key is not a legend of the spec.
  """
  legends = []
  for leaf_name, leaf_spec in _flatten_reward_tree(reward_spec):
    shape = tuple(leaf_spec.shape)
    if len(shape) > 1:
      _module_logger.warning(
          'Not plotting reward %r with shape %s.', leaf_name or 'reward', shape
      )
      continue
    legends.extend(_reward_leaf_legends(leaf_name, int(np.prod(shape))))
  if keys is None:
    return legends
  keys = list(keys)
  missing = set(keys) - set(legends)
  if missing:
    raise ValueError(
        f'Reward keys {sorted(missing)} not found in the reward spec; expected'
        f' a subset of {legends}.'
    )
  return keys


def _reward_values(reward: Any) -> dict[str, float]:
  """Returns a reward's values, named like the legends of its spec."""
  values = {}
  for leaf_name, leaf_value in _flatten_reward_tree(reward):
    array = np.asarray(leaf_value).ravel()
    values.update(
        zip(_reward_leaf_legends(leaf_name, array.size), array.tolist())
    )
  return values


class ReafViewerPlugin(plugin.ViewerPlugin):
  """Viewer-side REAF plugin.

  Derives the episode count, step count, and reward plot from the stream of
  ``TimestepEvent``s sent by the sim-side ``SimPlugin``.
  """

  def __init__(
      self,
      *,
      reward_legends: Sequence[str] = (),
      instruction: str = '',
  ) -> None:
    """Initializes the plugin.

    Args:
      reward_legends: The reward plot legends, typically from
        ``reward_spec_legends(environment.reward_spec())``. No reward plot is
        shown if empty.
      instruction: The initial task instruction shown in the prompt.
    """
    self._app: viewer_app.ViewerApp | None = None

    self.show_reward: bool = True
    self.episode_count: int = 0
    self.n_steps: int = 0
    self.reward_plot: line_plots.MultiLinePlot | None = None
    if reward_legends:
      self.reward_plot = line_plots.MultiLinePlot(
          title='Reward', xlabel='Env Step', legends=list(reward_legends)
      )

    # Instruction prompt state.
    self._instruction: str = instruction
    self._sent_instruction: str = instruction
    self._show_instruction_prompt: bool = False
    self._focus_instruction_input: bool = False

    # 2D image stream state.
    self._timestep: Any = None
    self._image2d = Image2D()

  @messages.handler
  def on_viewer_app_init(self, event: viewer_app.ViewerAppInitEvent) -> None:
    """Caches the ViewerApp reference on startup."""
    assert isinstance(event.viewer_app, viewer_app.ViewerApp)
    self._app = event.viewer_app

  @messages.handler
  def on_timestep(self, event: reaf_messages.TimestepEvent) -> None:
    """Updates episode progress and the reward plot from a timestep."""
    timestep = event.timestep
    self._timestep = timestep
    if timestep.first():
      self.episode_count += 1
      self.n_steps = 0
      if self.reward_plot is not None:
        self.reward_plot.clear()
      return

    self.n_steps += 1
    if self.reward_plot is not None:
      self.reward_plot.add_points(
          x=self.n_steps - 1, y=_reward_values(timestep.reward)
      )

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    """Handles REAF keyboard shortcuts and updates the status bar."""
    assert self._app is not None
    pause_state = self._app.step_control_state.get_pause_state()
    pause_str = (
        'Running' if pause_state == sim.PauseState.UNPAUSED else 'Paused'
    )
    self._app.status = (
        f'REAF | Episode: {self.episode_count} |'
        f' Step: {self.n_steps} | {pause_str}'
    )

    if self._timestep is not None:
      obs = self._timestep.observation
      self._timestep = None
      if isinstance(obs, Mapping):
        images = {}
        for k, v in obs.items():
          try:
            if len(v.shape) >= 4:
              images[k] = v
          except AttributeError:
            continue
        self._image2d.set_images(images)
    self._image2d.update(self._app.viewer)

    if not imgui.GetCurrentContext() or imgui.GetIO().WantCaptureKeyboard:
      return
    pressed = imgui.IsKeyChordPressed
    if pressed(imgui.Key.R):
      self.show_reward = not self.show_reward
    elif pressed(imgui.Key.P):
      self._show_instruction_prompt = not self._show_instruction_prompt
      if self._show_instruction_prompt:
        self._focus_instruction_input = True

  @messages.handler
  def build_reaf_gui(self, _: messages.BuildGuiEvent) -> None:
    """Renders the REAF instruction prompt and reward plot in ImGui."""
    assert self._app is not None

    # -- Instruction prompt ---------------------------------------------------
    if self._show_instruction_prompt:
      io = imgui.GetIO()
      imgui.SetNextWindowPos(
          imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y * 0.5),
          imgui.Cond.FirstUseEver,
          imgui.Vec2(0.5, 0.5),
      )
      imgui.SetNextWindowSize(imgui.Vec2(800, 400), imgui.Cond.FirstUseEver)
      opened, self._show_instruction_prompt = imgui.Begin(
          'Instruction###InstructionPrompt', True
      )
      if opened:
        # Show the current (sent) instruction.
        if self._sent_instruction:
          imgui.Text('Current:')
          imgui.SameLine()
          avail_width = imgui.GetContentRegionAvail().x
          text_size = imgui.CalcTextSize(self._sent_instruction)
          if text_size.x > avail_width:
            imgui.TextDisabled(self._sent_instruction[:80] + '...')
          else:
            imgui.TextDisabled(self._sent_instruction)
          if imgui.IsItemHovered():
            imgui.SetTooltip(self._sent_instruction)

        # Editable input box.
        input_flags = (
            imgui.InputTextFlags.EnterReturnsTrue.value
            | imgui.InputTextFlags.CtrlEnterForNewLine.value
        )
        if self._focus_instruction_input:
          imgui.SetKeyboardFocusHere()
          self._focus_instruction_input = False
        submitted, self._instruction = imgui.InputTextMultiline(
            '##Instruction',
            self._instruction,
            imgui.Vec2(-1, imgui.GetTextLineHeight() * 3.5),
            input_flags,
        )
        if imgui.IsItemHovered():
          imgui.SetTooltip(
              'Press Enter to send instruction.\nCtrl+Enter for newlines.'
          )
        if submitted and self._instruction.strip():
          self._sent_instruction = self._instruction
          self._app.viewer.send_to_sim(
              reaf_messages.InstructionEvent(instruction=self._instruction)
          )
      imgui.End()

    if self.show_reward and self.reward_plot is not None:
      fig = self.reward_plot.figure

      io = imgui.GetIO()
      imgui.SetNextWindowPos(
          imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y * 0.5),
          imgui.Cond.FirstUseEver,
          imgui.Vec2(0.5, 0.5),
      )
      imgui.SetNextWindowSize(imgui.Vec2(900, 450), imgui.Cond.FirstUseEver)
      if imgui.Begin('Reward###RewardPlot'):
        avail = imgui.GetContentRegionAvail()
        plot_size = imgui.Vec2(avail.x, avail.y)

        if implot.BeginPlot(
            fig.title or 'Reward', plot_size, flags=_REWARD_PLOT_FLAGS
        ):
          implot.SetupAxis(
              implot.Axis.X1, fig.xlabel or 'Step', _REWARD_AXIS_FLAGS
          )
          implot.SetupAxisLimits(
              implot.Axis.X1,
              float(fig.range[0][0]),
              float(fig.range[0][1]),
              cond=implot.Cond.Always,
          )
          implot.SetupAxis(
              implot.Axis.Y1,
              '',
              _REWARD_AXIS_FLAGS | implot.AxisFlags.AutoFit.value,
          )

          for i in range(len(fig.linepnt)):
            n_pts = int(fig.linepnt[i])
            if n_pts == 0:
              continue

            raw = fig.linedata[i, : 2 * n_pts]
            xs = raw[0::2].tolist()
            ys = raw[1::2].tolist()

            name = fig.linename[i] if fig.linename[i] else f'Line {i}'
            implot.PlotLine(name, xs, ys)

          implot.EndPlot()
      imgui.End()

    # -- 2D image streams -----------------------------------------------------
    self._image2d.build_gui()
