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

"""Tests for the `playground` CLI dispatcher."""

import os
import sys
from unittest import mock

from absl.testing import absltest
from mujoco_playground._src import cli


class CliTest(absltest.TestCase):

  def test_help_prints_usage(self) -> None:
    with mock.patch.object(sys, "argv", ["playground", "--help"]):
      with mock.patch("builtins.print") as mock_print:
        cli.main()
    mock_print.assert_called_once_with(cli._USAGE)

  def test_subcommand_help_and_missing_target(self) -> None:
    with mock.patch.object(sys, "argv", ["playground", "sysid", "--help"]):
      with mock.patch("builtins.print") as mock_print:
        cli.main()
    mock_print.assert_called_once_with(
        "Usage: playground sysid <so101|yam> [flags...]"
    )
    with mock.patch.object(sys, "argv", ["playground", "sysid"]):
      with self.assertRaisesRegex(
          SystemExit, "Usage: playground sysid <so101|yam>"
      ):
        cli.main()

  def test_dispatch_flat_command(self) -> None:
    argv = ["playground", "view_log", "--file=ep.mcap"]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch.object(cli.runpy, "run_module") as mock_run:
        cli.main()
      self.assertEqual(sys.argv, ["playground view_log", "--file=ep.mcap"])
    mock_run.assert_called_once_with(
        "mujoco_playground.real.view_log",
        run_name="__main__",
        alter_sys=True,
    )

  def test_dispatch_targeted_command(self) -> None:
    argv = ["playground", "train", "rsl_rl", "--env=SO101PickCube"]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch.object(cli.runpy, "run_module") as mock_run:
        cli.main()
      self.assertEqual(
          sys.argv, ["playground train rsl_rl", "--env=SO101PickCube"]
      )
    mock_run.assert_called_once_with(
        "mujoco_playground.rl.rsl_rl.train",
        run_name="__main__",
        alter_sys=True,
    )

  def test_banner_only_on_tty_command_not_help(self) -> None:
    with mock.patch.dict(os.environ, {"COLORTERM": "truecolor"}, clear=True):
      with mock.patch.object(sys.stderr, "isatty", return_value=True):
        with mock.patch.object(
            sys,
            "argv",
            ["playground", "train", "rsl_rl", "--env=SO101PickCube"],
        ):
          with mock.patch.object(cli.runpy, "run_module"):
            with mock.patch("builtins.print") as mock_print:
              cli.main()
        mock_print.assert_called_once_with(
            cli._format_banner(), file=sys.stderr
        )
        self.assertIn("\033[38;2;", mock_print.call_args.args[0])

        for argv in (
            ["playground", "--help"],
            ["playground", "sysid", "--help"],
            ["playground", "train", "rsl_rl", "--help"],
        ):
          with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(cli.runpy, "run_module"):
              with mock.patch("builtins.print") as mock_print:
                cli.main()
          for call in mock_print.call_args_list:
            self.assertNotIn("\033[38;2;", call.args[0])

  def test_banner_color_modes(self) -> None:
    with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
      banner = cli._format_banner()
      self.assertIn("\033[38;5;", banner)
      self.assertNotIn("\033[38;2;", banner)

    with mock.patch.dict(os.environ, {"WT_SESSION": "1"}, clear=True):
      self.assertIn("\033[38;2;", cli._format_banner())

    for env in (
        {"COLORTERM": "truecolor", "NO_COLOR": "1"},
        {"TERM": "dumb"},
        {},
    ):
      with mock.patch.dict(os.environ, env, clear=True):
        self.assertEqual(cli._format_banner(), cli._BANNER)

  def test_unknown_command_and_target_exit(self) -> None:
    with mock.patch.object(sys, "argv", ["playground", "bogus"]):
      with self.assertRaisesRegex(SystemExit, "Unknown command 'bogus'"):
        cli.main()
    with mock.patch.object(sys, "argv", ["playground", "sysid", "bogus"]):
      with self.assertRaisesRegex(SystemExit, "Unknown sysid target 'bogus'"):
        cli.main()

  def test_missing_external_dependency_suggests_extra(self) -> None:
    err = ModuleNotFoundError("No module named 'rerun'", name="rerun")
    with mock.patch.object(sys, "argv", ["playground", "view_log"]):
      with mock.patch.object(cli.runpy, "run_module", side_effect=err):
        with self.assertRaisesRegex(SystemExit, "uv sync --extra real"):
          cli.main()

  def test_missing_internal_module_reraises(self) -> None:
    err = ModuleNotFoundError(
        "No module named 'mujoco_playground.broken'",
        name="mujoco_playground.broken",
    )
    with mock.patch.object(sys, "argv", ["playground", "view_log"]):
      with mock.patch.object(cli.runpy, "run_module", side_effect=err):
        with self.assertRaises(ModuleNotFoundError):
          cli.main()


if __name__ == "__main__":
  absltest.main()
