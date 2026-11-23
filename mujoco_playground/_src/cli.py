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

"""CLI dispatcher for MuJoCo Playground."""

import os
import runpy
import sys

_COMMANDS: dict[str, tuple[str, str] | dict[str, tuple[str, str]]] = {
    "train": {
        "rsl_rl": ("mujoco_playground.rl.rsl_rl.train", "--extra rl"),
    },
    "play": {
        "rsl_rl": ("mujoco_playground.rl.rsl_rl.play", "--extra rl"),
    },
    "deploy": {
        "so101_pick_cube": (
            "mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run",
            "--extra rl --extra so101",
        ),
    },
    "teleop": {
        "so101": (
            "mujoco_playground.tasks.teleop.so101.teleop",
            "--extra so101",
        ),
    },
    "sysid": {
        "so101": ("mujoco_playground.tasks.sysid.so101.run", "--extra so101"),
        "yam": ("mujoco_playground.tasks.sysid.yam.run", "--extra yam"),
    },
    "view_log": ("mujoco_playground.real.view_log", "--extra real"),
}

_USAGE = """Usage: playground <command> [target] [flags...]

Commands:
  train <target>          Train a policy (targets: rsl_rl)
  play <target>           Play a policy in MuJoCo Studio (targets: rsl_rl)
  deploy <target>         Deploy a policy (targets: so101_pick_cube)
  teleop <target>         Run leader-follower teleop (targets: so101)
  sysid <target>          Run system identification (targets: so101, yam)
  view_log                View a logged MCAP episode in Rerun"""

_BANNER = r"""
  __  __        _        ____          ____  _                                             _
 |  \/  |_   _ | | ___  / ___|___     |  _ \| | __ _ _   _  __ _ _ __ ___  _   _ _ __   __| |
 | |\/| | | | || |/ _ \| |   / _ \    | |_) | |/ _` | | | |/ _` | '__/ _ \| | | | '_ \ / _` |
 | |  | | |_| || | (_) | |__| (_) |   |  __/| | (_| | |_| | (_| | | | (_) | |_| | | | | (_| |
 |_|  |_|\__,_|/ |\___/ \____\___/    |_|   |_|\__,_|\__, |\__, |_|  \___/ \__,_|_| |_|\__,_|
             |__/                                    |___/ |___/
"""

# Coral-orange -> teal-cyan -> violet-purple gradient matching the logo banner.
_BANNER_STOPS = (
    (230, 110, 70),
    (25, 185, 185),
    (150, 60, 175),
)

_HELP_FLAGS = frozenset({"-h", "--help", "--helpshort", "--helpfull"})


def _color_mode() -> str:
  """Returns 'truecolor', '256', or 'none' for the current terminal."""
  if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
    return "none"
  colorterm = os.environ.get("COLORTERM", "").lower()
  if colorterm in ("truecolor", "24bit") or os.environ.get("WT_SESSION"):
    return "truecolor"
  if "256color" in os.environ.get("TERM", ""):
    return "256"
  return "none"


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
  """Maps an RGB triple to the nearest xterm 256-color 6x6x6 cube index."""
  return (
      16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)
  )


def _format_banner() -> str:
  """Returns `_BANNER` colored for the terminal's supported color depth."""
  mode = _color_mode()
  if mode == "none":
    return _BANNER
  lines = _BANNER.splitlines()
  width = max(len(line) for line in lines) or 1
  n_seg = len(_BANNER_STOPS) - 1
  colored: list[str] = []
  for line in lines:
    if not line:
      colored.append("")
      continue
    parts: list[str] = []
    for col, ch in enumerate(line):
      if ch == " ":
        parts.append(ch)
        continue
      t = col / max(width - 1, 1) * n_seg
      idx = min(int(t), n_seg - 1)
      frac = t - idx
      c0, c1 = _BANNER_STOPS[idx], _BANNER_STOPS[idx + 1]
      r = round(c0[0] + (c1[0] - c0[0]) * frac)
      g = round(c0[1] + (c1[1] - c0[1]) * frac)
      b = round(c0[2] + (c1[2] - c0[2]) * frac)
      if mode == "truecolor":
        parts.append(f"\033[38;2;{r};{g};{b}m{ch}")
      else:
        parts.append(f"\033[38;5;{_rgb_to_ansi256(r, g, b)}m{ch}")
    colored.append("".join(parts) + "\033[0m")
  return "\n".join(colored)


def main() -> None:
  """Dispatches `playground` subcommands lazily via `runpy`."""
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
    print(_USAGE)
    return

  cmd = sys.argv[1]
  entry = _COMMANDS.get(cmd)
  if entry is None:
    sys.exit(f"Unknown command {cmd!r}.\n\n{_USAGE}")

  if isinstance(entry, dict):
    targets = "|".join(entry)
    usage = f"Usage: playground {cmd} <{targets}> [flags...]"
    if len(sys.argv) < 3:
      sys.exit(usage)
    if sys.argv[2] in ("-h", "--help"):
      print(usage)
      return
    target = sys.argv[2]
    if target not in entry:
      sys.exit(
          f"Unknown {cmd} target {target!r}. Available: {', '.join(entry)}"
      )
    module_path, extra = entry[target]
    label = f"{sys.argv[0]} {cmd} {target}"
    sys.argv = [label, *sys.argv[3:]]
  else:
    module_path, extra = entry
    label = f"{sys.argv[0]} {cmd}"
    sys.argv = [label, *sys.argv[2:]]

  if sys.stderr.isatty() and not any(
      arg in _HELP_FLAGS for arg in sys.argv[1:]
  ):
    print(_format_banner(), file=sys.stderr)

  try:
    runpy.run_module(module_path, run_name="__main__", alter_sys=True)
  except ModuleNotFoundError as e:
    if e.name is None or e.name.split(".")[0] == "mujoco_playground":
      raise
    sys.exit(
        f"Missing dependency {e.name!r} for {label!r}. "
        f"Install with: uv sync {extra}"
    )
