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
"""Profiling utilities that wrap the Tracy profiler.

This module provides decorators and context managers for profiling REAF code
with Tracy. When Tracy is available, the profiling zones are active. When Tracy
is unavailable the decorators and context managers are no-ops.

Usage as a decorator:

  from reaf.simulation.torch._src.core import profile

  @profile.trace
  def my_function():
    ...

  @profile.trace("custom zone name")
  def my_other_function():
    ...

Usage as a context manager:

  with profile.scope("my_zone"):
    ...
"""

import contextlib
import enum
import functools
from typing import Any
from typing import Callable
from typing import Optional

try:
  from tracy_client import tracy as _tracy  # pylint: disable=g-import-not-at-top

  _TRACY_AVAILABLE = True
except ImportError:
  _tracy = None
  _TRACY_AVAILABLE = False


def is_available() -> bool:
  """Returns True if Tracy profiling is available."""
  return _TRACY_AVAILABLE


def trace(
    name: Optional[str] = None,
    color: int = 0,
    depth: Optional[int] = None,
    active: bool = True,
) -> Callable[..., Any]:
  """Decorator that wraps a function in a Tracy profiling zone.

  Can be used with or without arguments:

    @profile.trace
    def my_function():
      ...

    @profile.trace("custom name")
    def my_other_function():
      ...

    @profile.trace(color=profile.Color.Red)
    def my_red_function():
      ...

  Args:
    name: Optional name for the profiling zone. If not provided, the function
      name is used automatically by Tracy.
    color: Optional color for the zone (Color enum or raw integer).
    depth: Optional callstack depth to capture.
    active: Whether the zone is active.

  Returns:
    A decorator that wraps the function in a Tracy profiling zone.
  """

  def make_decorator(zone_name: Optional[str]) -> Callable[..., Any]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
      if not _TRACY_AVAILABLE:
        return fn
      zname = zone_name or fn.__name__

      @functools.wraps(fn)
      def wrapper(*args: Any, **kwargs: Any) -> Any:
        assert _tracy is not None
        zone = _tracy.ScopedZone(
            name=zname, color=color, depth=depth, active=active
        )
        zone.enter()
        try:
          return fn(*args, **kwargs)
        finally:
          zone.exit()

      return wrapper

    return decorator

  # Handle @profile.trace (no parentheses, function passed directly).
  if callable(name):
    fn = name
    return make_decorator(None)(fn)

  # Handle @profile.trace() or @profile.trace("name") etc.
  return make_decorator(name)


@contextlib.contextmanager
def scope(
    name: str,
    color: int = 0,
    depth: Optional[int] = None,
    active: bool = True,
):
  """Context manager that creates a Tracy profiling zone.

  Usage:
    with profile.scope("my_zone"):
      ...

  Args:
    name: Name for the profiling zone.
    color: Optional color for the zone (Tracy color integer).
    depth: Optional callstack depth to capture.
    active: Whether the zone is active.

  Yields:
    The Tracy ScopedZone if available, otherwise None.
  """
  if _TRACY_AVAILABLE:
    assert _tracy is not None
    zone = _tracy.ScopedZone(name=name, color=color, depth=depth, active=active)
    zone.enter()
    try:
      yield zone
    finally:
      zone.exit()
  else:
    yield None


def frame_mark() -> None:
  """Marks the start of a frame for Tracy frame timing."""
  if _TRACY_AVAILABLE:
    assert _tracy is not None
    _tracy.frame_mark()


def message(text: str) -> None:
  """Sends a message to the Tracy profiler.

  Args:
    text: The message text.
  """
  if _TRACY_AVAILABLE:
    assert _tracy is not None
    _tracy.message(text)


class Color(enum.IntEnum):
  """Tracy profiling zone colors.

  These mirror the named color constants from Tracy's TracyColor.hpp. They can
  be passed directly to `trace()` and `scope()` as the `color` argument.

  Usage:
    @profile.trace(color=profile.Color.RED)
    def my_function():
      ...

    with profile.scope("physics", color=profile.Color.DODGER_BLUE):
      ...
  """

  # Whites & pastels
  SNOW = 0xFFFAFA
  GHOST_WHITE = 0xF8F8FF
  WHITE_SMOKE = 0xF5F5F5
  GAINSBORO = 0xDCDCDC
  FLORAL_WHITE = 0xFFFAF0
  OLD_LACE = 0xFDF5E6
  LINEN = 0xFAF0E6
  ANTIQUE_WHITE = 0xFAEBD7
  PAPAYA_WHIP = 0xFFEFD5
  BLANCHED_ALMOND = 0xFFEBCD
  BISQUE = 0xFFE4C4
  PEACH_PUFF = 0xFFDAB9
  NAVAJO_WHITE = 0xFFDEAD
  MOCCASIN = 0xFFE4B5
  CORNSILK = 0xFFF8DC
  IVORY = 0xFFFFF0
  LEMON_CHIFFON = 0xFFFACD
  SEASHELL = 0xFFF5EE
  HONEYDEW = 0xF0FFF0
  MINT_CREAM = 0xF5FFFA
  AZURE = 0xF0FFFF
  ALICE_BLUE = 0xF0F8FF
  LAVENDER = 0xE6E6FA
  LAVENDER_BLUSH = 0xFFF0F5
  MISTY_ROSE = 0xFFE4E1
  WHITE = 0xFFFFFF
  BLACK = 0x000000

  # Grays
  DARK_SLATE_GRAY = 0x2F4F4F
  DIM_GRAY = 0x696969
  SLATE_GRAY = 0x708090
  LIGHT_SLATE_GRAY = 0x778899
  GRAY = 0xBEBEBE
  LIGHT_GRAY = 0xD3D3D3
  DARK_GRAY = 0xA9A9A9
  SILVER = 0xC0C0C0

  # Blues
  MIDNIGHT_BLUE = 0x191970
  NAVY = 0x000080
  CORNFLOWER_BLUE = 0x6495ED
  DARK_SLATE_BLUE = 0x483D8B
  SLATE_BLUE = 0x6A5ACD
  MEDIUM_SLATE_BLUE = 0x7B68EE
  MEDIUM_BLUE = 0x0000CD
  ROYAL_BLUE = 0x4169E1
  BLUE = 0x0000FF
  DODGER_BLUE = 0x1E90FF
  DEEP_SKY_BLUE = 0x00BFFF
  SKY_BLUE = 0x87CEEB
  LIGHT_SKY_BLUE = 0x87CEFA
  STEEL_BLUE = 0x4682B4
  LIGHT_STEEL_BLUE = 0xB0C4DE
  LIGHT_BLUE = 0xADD8E6
  POWDER_BLUE = 0xB0E0E6
  DARK_BLUE = 0x00008B

  # Cyans & teals
  PALE_TURQUOISE = 0xAFEEEE
  DARK_TURQUOISE = 0x00CED1
  MEDIUM_TURQUOISE = 0x48D1CC
  TURQUOISE = 0x40E0D0
  CYAN = 0x00FFFF
  LIGHT_CYAN = 0xE0FFFF
  CADET_BLUE = 0x5F9EA0
  DARK_CYAN = 0x008B8B
  TEAL = 0x008080

  # Greens
  MEDIUM_AQUAMARINE = 0x66CDAA
  AQUAMARINE = 0x7FFFD4
  DARK_GREEN = 0x006400
  DARK_OLIVE_GREEN = 0x556B2F
  DARK_SEA_GREEN = 0x8FBC8F
  SEA_GREEN = 0x2E8B57
  MEDIUM_SEA_GREEN = 0x3CB371
  LIGHT_SEA_GREEN = 0x20B2AA
  PALE_GREEN = 0x98FB98
  SPRING_GREEN = 0x00FF7F
  LAWN_GREEN = 0x7CFC00
  GREEN = 0x00FF00
  CHARTREUSE = 0x7FFF00
  MEDIUM_SPRING_GREEN = 0x00FA9A
  GREEN_YELLOW = 0xADFF2F
  LIME_GREEN = 0x32CD32
  YELLOW_GREEN = 0x9ACD32
  FOREST_GREEN = 0x228B22
  OLIVE_DRAB = 0x6B8E23
  LIGHT_GREEN = 0x90EE90
  OLIVE = 0x808000
  WEB_GREEN = 0x008000

  # Yellows & golds
  DARK_KHAKI = 0xBDB76B
  KHAKI = 0xF0E68C
  PALE_GOLDENROD = 0xEEE8AA
  LIGHT_GOLDENROD_YELLOW = 0xFAFAD2
  LIGHT_YELLOW = 0xFFFFE0
  YELLOW = 0xFFFF00
  GOLD = 0xFFD700
  LIGHT_GOLDENROD = 0xEEDD82
  GOLDENROD = 0xDAA520
  DARK_GOLDENROD = 0xB8860B

  # Browns & tans
  ROSY_BROWN = 0xBC8F8F
  INDIAN_RED = 0xCD5C5C
  SADDLE_BROWN = 0x8B4513
  SIENNA = 0xA0522D
  PERU = 0xCD853F
  BURLYWOOD = 0xDEB887
  BEIGE = 0xF5F5DC
  WHEAT = 0xF5DEB3
  SANDY_BROWN = 0xF4A460
  TAN = 0xD2B48C
  CHOCOLATE = 0xD2691E

  # Reds & oranges
  FIREBRICK = 0xB22222
  BROWN = 0xA52A2A
  DARK_SALMON = 0xE9967A
  SALMON = 0xFA8072
  LIGHT_SALMON = 0xFFA07A
  ORANGE = 0xFFA500
  DARK_ORANGE = 0xFF8C00
  CORAL = 0xFF7F50
  LIGHT_CORAL = 0xF08080
  TOMATO = 0xFF6347
  ORANGE_RED = 0xFF4500
  RED = 0xFF0000
  CRIMSON = 0xDC143C
  DARK_RED = 0x8B0000

  # Pinks
  HOT_PINK = 0xFF69B4
  DEEP_PINK = 0xFF1493
  PINK = 0xFFC0CB
  LIGHT_PINK = 0xFFB6C1
  PALE_VIOLET_RED = 0xDB7093

  # Purples & violets
  MAROON = 0xB03060
  MEDIUM_VIOLET_RED = 0xC71585
  MAGENTA = 0xFF00FF
  VIOLET = 0xEE82EE
  PLUM = 0xDDA0DD
  ORCHID = 0xDA70D6
  MEDIUM_ORCHID = 0xBA55D3
  DARK_ORCHID = 0x9932CC
  DARK_VIOLET = 0x9400D3
  BLUE_VIOLET = 0x8A2BE2
  PURPLE = 0xA020F0
  MEDIUM_PURPLE = 0x9370DB
  THISTLE = 0xD8BFD8
  DARK_MAGENTA = 0x8B008B
  INDIGO = 0x4B0082
  REBECCA_PURPLE = 0x663399


class Section(enum.IntEnum):
  """Predefined colors for common REAF profiling sections.

  Use these to give consistent, distinguishable colors to profiling zones
  across the codebase. Each section maps to a visually distinct color so
  that different parts of the pipeline are easy to identify in Tracy.

  Usage:
    @profile.trace(color=profile.Section.PHYSICS)
    def substep(self):
      ...

    with profile.scope("render_cameras", color=profile.Section.RENDERING):
      ...
  """

  PHYSICS = Color.FOREST_GREEN
  COMMANDS = Color.STEEL_BLUE
  OBSERVATIONS = Color.GOLD
  ENVIRONMENT = Color.MEDIUM_PURPLE
  RENDERING = Color.CORAL
