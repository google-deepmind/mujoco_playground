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

"""Tests for playground paths."""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from mujoco_playground._src import paths


class LocalDirTest(absltest.TestCase):
  """Covers which directory local_dir() picks, and why."""

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self._tmp = epath.Path(tmp.name)
    # clear os environ
    self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
    # mock the pathlib.Path home directory
    self._home = self._tmp / "home"
    self.enterContext(
        mock.patch.object(pathlib.Path, "home", lambda: self._home)
    )

  def _fake_source_checkout(self) -> epath.Path:
    """Runs the remaining assertions as if imported from a git checkout."""
    checkout = self._tmp / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").touch()  # What marks a checkout.
    self.enterContext(mock.patch.object(paths, "_REPO_ROOT", checkout))
    return checkout

  def _fake_installed_package(self) -> None:
    """Runs the remaining assertions as if pip-installed (no pyproject.toml)."""
    site_packages = self._tmp / "site-packages"
    site_packages.mkdir()
    self.enterContext(mock.patch.object(paths, "_REPO_ROOT", site_packages))

  def test_env_var_overrides_source_checkout(self):
    override = self._tmp / "elsewhere"
    self._fake_source_checkout()
    os.environ[paths.LOCAL_DIR_ENV_VAR] = str(override)
    self.assertEqual(paths.local_dir(), override)

  def test_source_checkout_uses_gitignored_dot_local(self):
    checkout = self._fake_source_checkout()
    self.assertEqual(paths.local_dir(), checkout / ".local")

  def test_installed_package_uses_persistent_user_config_dir(self):
    config_home = self._tmp / "config"
    self._fake_installed_package()
    os.environ["XDG_CONFIG_HOME"] = str(config_home)
    self.assertEqual(paths.local_dir(), config_home / "mujoco_playground")

  def test_relative_xdg_config_home_is_ignored(self):
    self._fake_installed_package()
    os.environ["XDG_CONFIG_HOME"] = "relative/not/absolute"
    self.assertEqual(
        paths.local_dir(), self._home / ".config" / "mujoco_playground"
    )

  def test_missing_parent_directories_are_created(self):
    override = self._tmp / "created" / "nested"
    os.environ[paths.LOCAL_DIR_ENV_VAR] = str(override)
    self.assertTrue(paths.local_dir().is_dir())


if __name__ == "__main__":
  absltest.main()
