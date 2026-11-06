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

"""API for asset retrieval from disparate sources.

Assets are identified by URIs of the form `source:path` where the source
selects the loader and the path (plus optional query params) carries all
source-specific information:

  local:/tmp/robot.xml
  mjz:/tmp/robot.mjz
  runfile:path/to/model.xml
  git:google-deepmind/mujoco_menagerie/aloha/scene.xml
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol
from urllib import parse as urlparse
import zipfile

from etils import epath
import mujoco
from mujoco_playground._src import paths
from mujoco_playground._src.sim import asset_loader


class Asset(Protocol):
  """A protocol for simulation assets."""

  def spec(self) -> mujoco.MjSpec:
    """Returns the MuJoCo spec for the asset."""

  @property
  def identifier(self) -> str:
    """Returns the identifier for the asset."""

  @property
  def path(self) -> epath.Path:
    """Returns the filesystem path to the asset."""


class AssetLoader(Protocol):
  """A protocol for asset loaders.

  All loaders identify assets from a single URI string. The loader receives
  the full URI (including source) so it can parse path and query params.
  """

  def __call__(self, uri: str) -> Asset:
    """Returns the asset for the given URI."""


def get(uri: str) -> Asset:
  """Returns the asset for the given URI.

  The URI source selects the loader, e.g. ``local:/tmp/foo.xml`` dispatches
  to the ``local`` loader with path ``/tmp/foo.xml``.

  Args:
    uri: A URI string of the form ``source:path[?params]``.

  Raises:
    ValueError: If the URI has no source.
    KeyError: If the source is not registered.
  """
  source, _ = _parse_source(uri)
  if source not in _SOURCE_REGISTRY:
    raise KeyError(f'Source {source!r} not registered. Available: {sources()}')
  return _SOURCE_REGISTRY[source](uri)


def query(uri: str) -> Iterator[Asset]:
  """Queries for assets matching a URI pattern.

  The URI source selects the loader, e.g. ``db:bowl?origin=ORIGIN_MESHY``
  dispatches to the ``db`` loader.

  Args:
    uri: A URI string whose source selects the source, and whose path/params are
      forwarded to the loader's ``query`` method.

  Raises:
    ValueError: If the URI has no source.
    KeyError: If the source is not registered.
    NotImplementedError: If the source does not support querying.

  Returns:
    An iterator over assets matching the query.
  """
  source, _ = _parse_source(uri)
  if source not in _SOURCE_REGISTRY:
    raise KeyError(f'Source {source!r} not registered. Available: {sources()}')
  loader = _SOURCE_REGISTRY[source]
  if not hasattr(loader, 'query'):
    raise NotImplementedError(f'Source {source!r} does not support querying.')
  return loader.query(uri)


_SOURCE_REGISTRY: dict[str, AssetLoader] = {}


def register_source(source: str, loader: AssetLoader) -> None:
  """Registers an asset loader for the given URI source."""
  if source in _SOURCE_REGISTRY:
    raise ValueError(f'Source {source!r} already registered.')
  _SOURCE_REGISTRY[source] = loader


def sources() -> list[str]:
  """Returns the registered URI sources."""
  return list(_SOURCE_REGISTRY.keys())


def _parse_source(uri: str) -> tuple[str, str]:
  """Splits 'source:query' and returns (source, query)."""
  if ':' not in uri:
    raise ValueError(
        f'URI must have a source (e.g. "local:query"), got: {uri!r}'
    )
  source, _, query_ = uri.partition(':')
  return source, query_


def _normalize_asset_path(path: str | epath.Path) -> str:
  """Normalizes a path, converting to runfile URI if inside an enclosing archive."""
  path_str = str(path)
  was_runfile = path_str.startswith('runfile:')
  if was_runfile:
    path_str = path_str[len('runfile:') :]

  curr = epath.Path(path_str)
  while curr != curr.parent:
    if curr.is_file() and zipfile.is_zipfile(str(curr)):
      rel = path_str[len(str(curr)) :].lstrip('/')
      return f'runfile:{rel}'
    curr = curr.parent

  if was_runfile and not epath.Path(path_str).exists():
    return f'runfile:{path_str}'

  return path_str


# Asset Loaders


class LocalAsset:
  """An asset loaded from a local filesystem path.

  URI format: ``local:path/to/model.xml``
  """

  _path: str

  def __init__(self, uri: str):
    _, self._path = _parse_source(uri)

  def spec(self) -> mujoco.MjSpec:
    norm_path = _normalize_asset_path(self._path)
    return mujoco.MjSpec.from_file(norm_path)

  @property
  def identifier(self) -> str:
    return f'local:{self._path}'

  @property
  def path(self) -> epath.Path:
    norm_path = _normalize_asset_path(self._path)
    return epath.Path(norm_path)


class MjzAsset:
  """An asset loaded from a zipped MJCF (.mjz) file.

  URI format: ``mjz:path/to/model.mjz``
  """

  _path: str

  def __init__(self, uri: str):
    _, self._path = _parse_source(uri)

  def spec(self) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_zip(self._path)

  @property
  def identifier(self) -> str:
    return f'mjz:{self._path}'

  @property
  def path(self) -> epath.Path:
    return epath.Path(self._path)


class RunfileAsset:
  """An asset loaded from a runfile path.

  URI format: ``runfile:path/to/model.xml``
  """

  _path: str

  def __init__(self, uri: str):
    _, self._path = _parse_source(uri)

  def spec(self) -> mujoco.MjSpec:
    norm_path = _normalize_asset_path(f'runfile:{self._path}')
    return mujoco.MjSpec.from_file(norm_path)

  @property
  def identifier(self) -> str:
    return f'runfile:{self._path}'

  @property
  def path(self) -> epath.Path:
    norm_path = _normalize_asset_path(f'runfile:{self._path}')
    return epath.Path(norm_path)


register_source('local', LocalAsset)
register_source('mjz', MjzAsset)
register_source('runfile', RunfileAsset)


# Git repos, with local runfile fallback if available.


class GitLoader:
  """Base class for loading assets from Git repositories."""

  def __init__(self, scheme: str):
    self.scheme = scheme

  def get_repo_url(self, repo: str) -> str:
    raise NotImplementedError

  def __call__(self, uri: str) -> Asset:
    _, full_path = _parse_source(uri)
    repo, path, params = _parse_github_uri(full_path)
    commit = params.get('commit', 'HEAD')

    local_path = paths.resolve_local_path(uri)
    if local_path is not None:
      return LocalAsset(f'local:{local_path}')

    # Sparse checkout the parent directory of the file.
    parent_dir = str(epath.Path(path).parent)
    repo_url = self.get_repo_url(repo)
    future = asset_loader.get_from_git(repo_url, parent_dir, commit)
    future.wait()
    paths_result = future.paths
    assert not isinstance(paths_result, list)
    file_path = paths_result / epath.Path(path).name
    return LocalAsset(f'local:{file_path}')


class GithubLoader(GitLoader):
  """Loads assets from GitHub repos, with local runfile fallback.

  URI format: ``github:org/repo/path/to/model.xml[?commit=SHA]``

  Examples::

    github:google-deepmind/mujoco_menagerie/aloha/scene.xml
    github:google-deepmind/aloha_sim/aloha_sim/assets/edr/pen/model.xml
    github:google-deepmind/aloha_sim/aloha_sim/assets/edr/pen/model.xml?commit=abc123
  """

  def __init__(self):
    super().__init__('github')

  def get_repo_url(self, repo: str) -> str:
    return f'https://github.com/{repo}.git'


def _parse_github_uri(rest: str) -> tuple[str, str, dict[str, str]]:
  """Parse ``org/repo/path[?params]`` into (repo, path, params)."""
  query_str = ''
  if '?' in rest:
    rest, query_str = rest.split('?', 1)

  parts = rest.split('/', 2)
  if len(parts) < 3:
    raise ValueError(
        f'github URI must be github:org/repo/path, got github:{rest}'
    )
  repo = f'{parts[0]}/{parts[1]}'
  path = parts[2]
  params = dict(urlparse.parse_qsl(query_str))
  return repo, path, params


register_source('github', GithubLoader())
