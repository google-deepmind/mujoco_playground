# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Checkpoint utilities for RSL-RL."""

import io
import os
import re
from typing import Union

from etils import epath

_CHECKPOINT_REGEX = re.compile(r"model_(\d+)\.pt")


def get_checkpoints(
    directory: Union[str, os.PathLike[str], epath.Path],
) -> list[epath.Path]:
  """Returns model_*.pt checkpoints in a directory sorted newest-first."""
  if not directory:
    return []
  dir_path = epath.Path(directory)
  if not dir_path.exists() or not dir_path.is_dir():
    return []

  checkpoints = []
  for f in dir_path.glob("model_*.pt"):
    if match := _CHECKPOINT_REGEX.fullmatch(f.name):
      checkpoints.append((int(match.group(1)), f))

  checkpoints.sort(key=lambda x: x[0], reverse=True)
  return [f for _, f in checkpoints]


def get_latest_checkpoint(
    directory: Union[str, os.PathLike, epath.Path],
) -> str | None:
  """Returns the latest model_*.pt checkpoint path in a directory, or None."""
  checkpoints = get_checkpoints(directory)
  return str(checkpoints[0]) if checkpoints else None


def read_checkpoint(
    path: Union[str, os.PathLike, epath.Path],
) -> io.BytesIO:
  """Reads a checkpoint into memory from any `etils.epath` location.

  `torch.load` only understands local files; reading through `epath` also
  supports `gs://` and other remote filesystems.

  Args:
    path: A `model_*.pt` file, or a directory whose latest `model_*.pt` is used.

  Returns:
    An in-memory buffer that can be passed to `torch.load` / `runner.load`.

  Raises:
    FileNotFoundError: If `path` is a directory with no `model_*.pt` files.
  """
  ckpt_path = epath.Path(path)
  if ckpt_path.is_dir():
    latest = get_latest_checkpoint(ckpt_path)
    if latest is None:
      raise FileNotFoundError(f"No model_*.pt checkpoints found in {path}")
    ckpt_path = epath.Path(latest)
  return io.BytesIO(ckpt_path.read_bytes())
