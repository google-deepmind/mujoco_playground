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

"""Utilities for loading assets, including sparse checkout from GitHub."""

import concurrent.futures
import fcntl
import logging
import shutil
import subprocess
from urllib import parse

from etils import epath
from mujoco_playground._src import paths

_log = logging.getLogger(__name__)


class AssetFuture:
  """Future-like object for async asset downloads."""

  def __init__(
      self,
      future: concurrent.futures.Future[None] | None,
      cache_paths: list[epath.Path],
      single: bool,
  ):
    self._future = future
    self._cache_paths = cache_paths
    self._single = single

  @property
  def paths(self) -> epath.Path | list[epath.Path]:
    """Get local path(s) immediately (does not block)."""
    return self._cache_paths[0] if self._single else self._cache_paths

  def wait(self) -> None:
    """Block until download completes."""
    if self._future:
      self._future.result()

  def __enter__(self) -> "AssetFuture":
    return self

  def __exit__(self, *args) -> None:
    self.wait()


def _sparse_checkout(
    repo_url: str,
    commit: str,
    paths: list[str],
    target_dir: epath.Path,
) -> None:
  """Clone repo with sparse checkout for specific paths, or add paths to existing clone."""
  target_dir.parent.mkdir(parents=True, exist_ok=True)
  lock_path = target_dir.parent / f"{target_dir.name}.lock"
  with open(lock_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    missing_paths = [p for p in paths if not (target_dir / p).exists()]
    if not missing_paths:
      return

    target = str(target_dir)
    if not (target_dir / ".git" / "info" / "sparse-checkout").exists():
      # Clean up stale or partially-initialized cache dir if it exists.
      if target_dir.exists():
        shutil.rmtree(target)
      # Initial clone
      _log.info("Cloning %s (sparse)...", repo_url)
      subprocess.run(
          [
              "git",
              "clone",
              "--no-checkout",
              "--filter=blob:none",
              repo_url,
              target,
          ],
          check=True,
      )
      # Configure sparse-checkout
      _log.info("Setting sparse-checkout paths: %s", missing_paths)
      subprocess.run(
          ["git", "-C", target, "sparse-checkout", "set"] + missing_paths,
          check=True,
      )
      # Checkout the specific commit
      _log.info("Checking out %s", commit[:12])
      subprocess.run(
          ["git", "-C", target, "checkout", commit],
          check=True,
      )
    else:
      # Add paths to existing clone
      _log.info("Adding sparse-checkout paths: %s", missing_paths)
      subprocess.run(
          ["git", "-C", target, "sparse-checkout", "add"] + missing_paths,
          check=True,
      )

  _log.info("Sparse checkout complete")


def get_from_git(
    repo_url: str,
    path: str | list[str],
    commit: str,
) -> AssetFuture:
  """Get asset(s) from a source using git sparse checkout."""
  paths_list = [path] if isinstance(path, str) else path
  single = isinstance(path, str)

  # Build cache paths
  p = parse.urlparse(repo_url)
  safe_repo_name = (
      f"{p.netloc}{p.path}".replace("/", "_")
      .replace(":", "_")
      .removesuffix(".git")
  )
  cache_dir = paths.local_dir() / safe_repo_name / commit[:12]
  cache_paths = [cache_dir / p for p in paths_list]

  executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
  future = executor.submit(
      _sparse_checkout, repo_url, commit, paths_list, cache_dir
  )
  executor.shutdown(wait=False)

  return AssetFuture(future, cache_paths, single)
