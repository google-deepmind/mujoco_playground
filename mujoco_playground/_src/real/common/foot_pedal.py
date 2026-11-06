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

"""Foot pedal devices."""

import dataclasses
import select
import sys
import termios
import threading
import tty
import types
from typing import Mapping

from absl import logging

_ROLES = ("left", "middle", "right")
_DEFAULT_KEYS = {"left": "a", "middle": "s", "right": "d"}


@dataclasses.dataclass(frozen=True, kw_only=True)
class FootPedalState:
  """One reading of the pedal: which buttons were pressed since the last."""

  left_pressed: bool
  middle_pressed: bool
  right_pressed: bool


class KeyboardFootPedal:
  """Reads pedal presses from terminal keys (left/middle/right = a/s/d)."""

  def __init__(self, keys: Mapping[str, str] | None = None):
    """Initializes the pedal."""
    self._keys = {
        role: key.lower()
        for role, key in (_DEFAULT_KEYS if keys is None else keys).items()
    }
    if set(self._keys) != set(_ROLES):
      raise ValueError(
          f"Expected a key for each of {_ROLES}, got {sorted(self._keys)}."
      )
    if len(set(self._keys.values())) != len(_ROLES):
      raise ValueError(f"Each pedal needs a distinct key, got {self._keys}.")

    self._pending: set[str] = set()
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._saved_termios = None

  @property
  def keys(self) -> Mapping[str, str]:
    """The key each pedal emits, by role, lowercased."""
    return types.MappingProxyType(self._keys)

  def press(self, key: str) -> None:
    """Records a press (called by the reader thread; useful in tests)."""
    with self._lock:
      self._pending.add(key)

  @property
  def state(self) -> FootPedalState:
    """Returns and clears the presses latched since the previous read."""
    with self._lock:
      pending, self._pending = self._pending, set()
    return FootPedalState(
        left_pressed=self._keys["left"] in pending,
        middle_pressed=self._keys["middle"] in pending,
        right_pressed=self._keys["right"] in pending,
    )

  def start(self) -> None:
    """Puts the terminal in cbreak mode and starts the reader thread.

    cbreak keeps Ctrl-C working; key echo is disabled so pedal
    presses do not type into the terminal.
    """
    if self._thread is not None:
      # A second start would save the already-cbreak attributes as the ones to
      # restore, so `stop` would leave the terminal without echo.
      raise RuntimeError("KeyboardFootPedal is already started.")
    if not sys.stdin.isatty():
      raise RuntimeError(
          "KeyboardFootPedal requires an interactive terminal (stdin is not"
          " a tty)."
      )
    fd = sys.stdin.fileno()
    self._saved_termios = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    self._stop.clear()  # A pedal stopped once can be started again.
    self._thread = threading.Thread(target=self._read_loop, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    """Stops the reader thread and restores the terminal."""
    self._stop.set()
    if self._thread is not None:
      # The reader polls `_stop` every 50ms, so this only expires if it is
      # wedged — in which case it keeps consuming this terminal's keystrokes.
      self._thread.join(timeout=1.0)
      if self._thread.is_alive():
        logging.warning(
            "Foot pedal reader thread did not exit; it may still consume"
            " keystrokes from this terminal."
        )
      self._thread = None
    if self._saved_termios is not None:
      termios.tcsetattr(
          sys.stdin.fileno(), termios.TCSADRAIN, self._saved_termios
      )
      self._saved_termios = None

  def __enter__(self) -> "KeyboardFootPedal":
    self.start()
    return self

  def __exit__(self, *exc_info) -> None:
    self.stop()

  def _read_loop(self) -> None:
    while not self._stop.is_set():
      readable, _, _ = select.select([sys.stdin], [], [], 0.05)
      if readable:
        self.press(sys.stdin.read(1).lower())
