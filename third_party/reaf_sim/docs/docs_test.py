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
"""Sphinx build verification test for reaf-sim documentation."""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
from unittest import mock

from absl.testing import absltest

_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_DOCS_DIR)
_EXCLUDED_NAMES = ("_build", "internal", "playground", "__pycache__")


class DocsTest(absltest.TestCase):

  def test_sphinx_build(self) -> None:
    if importlib.util.find_spec("sphinx") is None:
      self.skipTest("sphinx is not installed")
    from sphinx.cmd import build  # pylint: disable=g-import-not-at-top

    temp_root = self.enterContext(tempfile.TemporaryDirectory())
    # Exported repo layout (<root>/reaf) vs. monorepo layout (<parent>/reaf).
    reaf_src = os.path.join(_PKG_ROOT, "reaf")
    reaf_dst = os.path.join(temp_root, "reaf")
    if os.path.isdir(reaf_src):
      os.symlink(reaf_src, reaf_dst)
    else:
      os.makedirs(reaf_dst)
      os.symlink(_PKG_ROOT, os.path.join(reaf_dst, "simulation"))

    docs_dir = os.path.join(temp_root, "docs")
    shutil.copytree(
        _DOCS_DIR, docs_dir, ignore=shutil.ignore_patterns(*_EXCLUDED_NAMES)
    )
    out_dir = os.path.join(temp_root, "_build")

    stderr = io.StringIO()
    with (
        mock.patch.object(sys, "path", sys.path.copy()),
        mock.patch.dict(sys.modules),
        contextlib.redirect_stderr(stderr),
    ):
      exit_code = build.main(
          ["-b", "dummy", "-W", "--keep-going", "-T", "-q", docs_dir, out_dir]
      )
    self.assertEqual(exit_code, 0, msg=stderr.getvalue())


if __name__ == "__main__":
  absltest.main()
