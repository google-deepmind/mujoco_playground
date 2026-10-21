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

"""Playground-wide paths and constants.

Central location for paths to shared assets and third-party resources.
"""

import os
import pathlib
from typing import Optional
import zipfile

from etils import epath

PROJECT_ROOT = epath.Path(__file__).parent.parent
LOCAL_DIR_ENV_VAR = "MUJOCO_PLAYGROUND_LOCAL_DIR"
_REPO_ROOT = PROJECT_ROOT.parent


def _user_config_dir() -> epath.Path:
  """Returns the per-user config directory (XDG convention)."""
  xdg = os.environ.get("XDG_CONFIG_HOME")
  if xdg and os.path.isabs(xdg):
    return epath.Path(xdg)
  return epath.Path(pathlib.Path.home()) / ".config"


def local_dir() -> epath.Path:
  """Returns the directory for machine-local, generated config."""
  override = os.environ.get(LOCAL_DIR_ENV_VAR)
  if override:
    root = epath.Path(override).expanduser()
  elif (_REPO_ROOT / "pyproject.toml").is_file():
    root = _REPO_ROOT / ".local"
  else:
    root = _user_config_dir() / "mujoco_playground"
  root.mkdir(parents=True, exist_ok=True)
  return root


def _exists_or_in_archive(path: epath.Path) -> bool:
  """Checks if path exists on the filesystem or inside an enclosing archive."""
  if path.exists():
    return True
  curr = path
  while curr != curr.parent:
    if curr.is_file() and zipfile.is_zipfile(str(curr)):
      rel = str(path)[len(str(curr)) :].lstrip("/")
      try:
        with zipfile.ZipFile(str(curr), "r") as zf:
          try:
            zf.getinfo(rel)
            return True
          except KeyError:
            pass
      except (OSError, zipfile.BadZipFile):
        pass
    curr = curr.parent
  return False


# TODO(btaba): clean up the odd menagerie and aloha sim dir paths. These
# are internal impl details.
_MENAGERIE_DIR = PROJECT_ROOT.parent.parent / "mujoco_menagerie"
_ALOHA_SIM_DIR = PROJECT_ROOT.parent / "aloha_sim"
# Map URIs used in asset.py to local directories.
URI_PREFIX_TO_LOCAL_DIR = {
    "github:google-deepmind/mujoco_menagerie": _MENAGERIE_DIR,
    "github:google-deepmind/aloha_sim/aloha_sim": _ALOHA_SIM_DIR,
}


def resolve_local_path(uri: str) -> Optional[epath.Path]:
  """Finds the local path for a URI based on prefix mapping."""
  for prefix in sorted(URI_PREFIX_TO_LOCAL_DIR.keys(), key=len, reverse=True):
    if not uri.startswith(prefix):
      continue
    local_root = URI_PREFIX_TO_LOCAL_DIR[prefix]
    if local_root is None:
      continue
    relative_path = uri[len(prefix) :].lstrip("/")
    candidate = epath.Path(local_root / relative_path)
    if _exists_or_in_archive(candidate):
      return candidate
  return None
