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

"""Tests for Studio PolicyPlugin."""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from mujoco.experimental.studio import messages
from mujoco_playground._src.rl.rsl_rl.viewer.plugins import policy


class _DummyPolicy:

  def __init__(self):
    self.last_stochastic_output = None

  def forward(self, obs, stochastic_output=False):
    self.last_stochastic_output = stochastic_output
    return obs


class PolicyPluginTest(absltest.TestCase):

  def _temp_dir(self) -> epath.Path:
    return epath.Path(self.enterContext(tempfile.TemporaryDirectory()))

  def test_stochasticity_wrapper(self):
    dummy_policy = _DummyPolicy()
    plugin = policy.PolicyPlugin(policy=dummy_policy, initial_stochastic=False)

    dummy_policy.forward("obs")
    self.assertFalse(dummy_policy.last_stochastic_output)

    plugin.stochastic = True
    dummy_policy.forward("obs")
    self.assertTrue(dummy_policy.last_stochastic_output)

  def test_update_checkpoints(self):
    temp_dir = self._temp_dir()
    (temp_dir / "model_100.pt").write_text("")
    (temp_dir / "model_500.pt").write_text("")
    (temp_dir / "other.txt").write_text("")

    plugin = policy.PolicyPlugin(
        policy=_DummyPolicy(),
        resume_path=str(temp_dir / "model_500.pt"),
    )

    self.assertEqual(plugin.checkpoint_dir, str(temp_dir))
    self.assertEqual(plugin.checkpoint_names, ["model_500.pt", "model_100.pt"])
    self.assertEqual(plugin.current_checkpoint_name, "model_500.pt")
    self.assertEqual(plugin.current_checkpoint_idx, 0)

  def test_directory_resume_path_and_default_state(self):
    empty_plugin = policy.PolicyPlugin()
    self.assertEqual(empty_plugin.checkpoint_dir, "")
    self.assertEqual(empty_plugin.checkpoint_names, [])
    self.assertIsNone(empty_plugin.current_checkpoint_name)
    self.assertEqual(empty_plugin.current_checkpoint_idx, -1)

    temp_dir = self._temp_dir()
    (temp_dir / "model_10.pt").write_text("")
    dir_plugin = policy.PolicyPlugin(resume_path=str(temp_dir))
    self.assertEqual(dir_plugin.checkpoint_dir, str(temp_dir))
    self.assertEqual(dir_plugin.checkpoint_names, ["model_10.pt"])
    self.assertIsNone(dir_plugin.current_checkpoint_name)
    self.assertEqual(dir_plugin.current_checkpoint_idx, -1)

    missing_dir = temp_dir / "missing_dir"
    missing_dir_plugin = policy.PolicyPlugin(resume_path=str(missing_dir))
    self.assertEqual(missing_dir_plugin.checkpoint_dir, str(missing_dir))
    self.assertEqual(missing_dir_plugin.checkpoint_names, [])
    self.assertIsNone(missing_dir_plugin.current_checkpoint_name)
    self.assertEqual(missing_dir_plugin.current_checkpoint_idx, -1)

  def test_build_gui_headless_no_context_is_noop(self):
    plugin = policy.PolicyPlugin()
    plugin.on_build_gui(messages.BuildGuiEvent())

  def test_build_gui_queries_options_dock_id_once_and_loads_checkpoint(self):
    temp_dir = self._temp_dir()
    (temp_dir / "model_100.pt").write_text("")
    (temp_dir / "model_500.pt").write_text("")

    loaded_paths = []
    plugin = policy.PolicyPlugin(
        resume_path=str(temp_dir),
        load_checkpoint_fn=loaded_paths.append,
    )

    with mock.patch.object(policy, "imgui") as mock_imgui:
      mock_imgui.GetCurrentContext.return_value = True
      mock_imgui.GetWindowDockID.return_value = 42
      mock_imgui.WindowFlags.NoFocusOnAppearing = 4096
      mock_imgui.Begin.side_effect = lambda name, **kwargs: (
          name == "Policy",
          True,
      )
      mock_imgui.Checkbox.return_value = (True, True)
      mock_imgui.InputText.return_value = (False, str(temp_dir))
      mock_imgui.Button.return_value = False
      mock_imgui.Combo.return_value = (True, 1)

      plugin.on_build_gui(messages.BuildGuiEvent())
      self.assertTrue(plugin.stochastic)
      self.assertEqual(plugin.current_checkpoint_idx, 1)
      self.assertEqual(plugin.current_checkpoint_name, "model_100.pt")
      self.assertEqual(loaded_paths, [str(temp_dir / "model_100.pt")])
      self.assertEqual(mock_imgui.GetWindowDockID.call_count, 1)

      # Second frame should not re-open Options or re-query GetWindowDockID.
      mock_imgui.Combo.return_value = (False, 1)
      plugin.on_build_gui(messages.BuildGuiEvent())
      self.assertEqual(mock_imgui.GetWindowDockID.call_count, 1)

  def test_build_gui_set_directory_and_load_failure(self):
    dir_a = self._temp_dir()
    (dir_a / "model_100.pt").write_text("")
    (dir_a / "model_200.pt").write_text("")
    dir_b = self._temp_dir()
    (dir_b / "model_100.pt").write_text("")
    empty_dir = self._temp_dir()

    def fail_loader(unused_path: str) -> None:
      raise RuntimeError("corrupt checkpoint")

    plugin = policy.PolicyPlugin(
        resume_path=str(dir_a / "model_100.pt"),
        load_checkpoint_fn=fail_loader,
    )
    self.assertEqual(plugin.current_checkpoint_name, "model_100.pt")
    self.assertEqual(plugin.current_checkpoint_idx, 1)

    with mock.patch.object(policy, "imgui") as mock_imgui:
      mock_imgui.GetCurrentContext.return_value = True
      mock_imgui.GetWindowDockID.return_value = 42
      mock_imgui.WindowFlags.NoFocusOnAppearing = 4096
      mock_imgui.Begin.side_effect = lambda name, **kwargs: (
          name == "Policy",
          True,
      )
      mock_imgui.Checkbox.return_value = (False, False)
      mock_imgui.InputText.return_value = (False, str(dir_a))
      mock_imgui.Button.return_value = False
      mock_imgui.Combo.return_value = (True, 0)

      # Selecting a checkpoint when load_checkpoint_fn fails keeps the previous
      # active selection.
      plugin.on_build_gui(messages.BuildGuiEvent())
      self.assertEqual(plugin.current_checkpoint_name, "model_100.pt")
      self.assertEqual(plugin.current_checkpoint_idx, 1)

      # Switching directories resets selection even if model_100.pt exists.
      mock_imgui.InputText.return_value = (True, str(dir_b))
      mock_imgui.Button.return_value = True
      mock_imgui.Combo.return_value = (False, -1)
      plugin.on_build_gui(messages.BuildGuiEvent())
      self.assertEqual(plugin.checkpoint_dir, str(dir_b))
      self.assertIsNone(plugin.current_checkpoint_name)
      self.assertEqual(plugin.current_checkpoint_idx, -1)
      self.assertEqual(plugin.checkpoint_names, ["model_100.pt"])

      # Switching to an empty directory shows empty directory message.
      mock_imgui.InputText.return_value = (True, str(empty_dir))
      mock_imgui.Button.return_value = True
      plugin.on_build_gui(messages.BuildGuiEvent())
      self.assertEqual(plugin.checkpoint_names, [])
      mock_imgui.Text.assert_called_once_with(
          "No checkpoints found in directory."
      )


if __name__ == "__main__":
  absltest.main()
