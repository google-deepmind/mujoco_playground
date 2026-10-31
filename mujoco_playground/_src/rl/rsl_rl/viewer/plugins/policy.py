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

"""Studio viewer plugin for RL policy controls and checkpoint loading."""

from collections.abc import Callable
import os
from typing import Any

from absl import logging
from etils import epath
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco_playground._src.rl.rsl_rl import checkpoint

try:
  from mujoco_playground._src.internal_imports.dear_imgui import imgui  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
except ImportError:
  from mujoco.studio.dear_imgui import dear_imgui as imgui  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top


class PolicyPlugin(plugin.ViewerPlugin):
  """Studio viewer plugin managing policy stochasticity and checkpoints."""

  def __init__(
      self,
      policy: Any = None,
      *,
      initial_stochastic: bool = False,
      resume_path: str | os.PathLike[str] | None = None,
      load_checkpoint_fn: Callable[[str], Any] | None = None,
  ) -> None:
    self.stochastic = initial_stochastic
    self.load_checkpoint_fn = load_checkpoint_fn
    self.checkpoint_names: list[str] = []
    self.current_checkpoint_idx: int = -1
    self.current_checkpoint_name: str | None = None
    self._options_dock_id: int | None = None

    if resume_path:
      path = epath.Path(resume_path)
      if path.suffix == ".pt" or path.is_file():
        self.checkpoint_dir = str(path.parent)
        self.current_checkpoint_name = path.name
      else:
        self.checkpoint_dir = str(path)
    else:
      self.checkpoint_dir = ""
    self._checkpoint_dir_input = self.checkpoint_dir

    if policy is not None:
      original_forward = policy.forward

      def custom_forward(*args, **kwargs):
        kwargs["stochastic_output"] = self.stochastic
        return original_forward(*args, **kwargs)

      policy.forward = custom_forward

    self.update_checkpoints()

  def update_checkpoints(self) -> None:
    """Scans the checkpoint directory and updates available checkpoint names."""
    self.checkpoint_names = [
        f.name for f in checkpoint.get_checkpoints(self.checkpoint_dir)
    ]
    if (
        self.current_checkpoint_name is not None
        and self.current_checkpoint_name in self.checkpoint_names
    ):
      self.current_checkpoint_idx = self.checkpoint_names.index(
          self.current_checkpoint_name
      )
    else:
      self.current_checkpoint_idx = -1

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    """Builds the Policy window in Dear ImGui."""
    if not imgui.GetCurrentContext():
      return

    if self._options_dock_id is None:
      imgui.Begin("Options")
      self._options_dock_id = imgui.GetWindowDockID()
      imgui.End()
      if self._options_dock_id:
        imgui.SetNextWindowDockID(
            self._options_dock_id, imgui.Cond.FirstUseEver
        )

    expanded, _ = imgui.Begin(
        "Policy", flags=int(imgui.WindowFlags.NoFocusOnAppearing)
    )
    if expanded:
      self._draw_policy_window()
    imgui.End()

  def _draw_policy_window(self) -> None:
    """Renders the contents of the Policy window."""
    changed_stoch, stoch_val = imgui.Checkbox("Stochastic", self.stochastic)
    if changed_stoch:
      self.stochastic = stoch_val

    imgui.Separator()

    changed_dir, new_dir = imgui.InputText(
        "##Dir Path", self._checkpoint_dir_input
    )
    if changed_dir:
      self._checkpoint_dir_input = new_dir

    if imgui.Button("Set Directory"):
      self.checkpoint_dir = self._checkpoint_dir_input
      self.current_checkpoint_name = None
      self.update_checkpoints()

    if not self.checkpoint_dir:
      return

    imgui.Separator()

    if not self.checkpoint_names:
      imgui.Text("No checkpoints found in directory.")
      return

    changed, selected_idx = imgui.Combo(
        "##Select Checkpoint",
        self.current_checkpoint_idx,
        self.checkpoint_names,
    )
    if not changed or selected_idx < 0:
      return

    selected_name = self.checkpoint_names[selected_idx]
    if self.load_checkpoint_fn is not None:
      full_path = str(epath.Path(self.checkpoint_dir) / selected_name)
      try:
        self.load_checkpoint_fn(full_path)
        logging.info("Loaded checkpoint: %s", full_path)
      except Exception as e:  # pylint: disable=broad-except
        logging.error("Failed to load checkpoint: %s", e)
        return

    self.current_checkpoint_idx = selected_idx
    self.current_checkpoint_name = selected_name
