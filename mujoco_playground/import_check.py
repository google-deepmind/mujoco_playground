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

"""Imports every module in the package to show that optional extras are not required."""

import importlib
import pathlib
import pkgutil
import sys
import traceback
from typing import Sequence

from absl import flags
import mujoco_playground

# Subpackages whose dependency on an extra is structural rather than lazy, so
# the convention cannot apply to them.
_EXCLUDED = (
    # Excluded from formatting, linting and the test targets too; see AGENTS.md.
    "mujoco_playground.rss2025",
    # rl extras
    "mujoco_playground.rl",
    "mujoco_playground._src.rl.rsl_rl",
)


def module_names() -> Sequence[str]:
  """Returns every importable module name in the package."""
  package_dir = pathlib.Path(mujoco_playground.__file__).parent
  return sorted(
      module.name
      for module in pkgutil.walk_packages(
          [str(package_dir)], prefix=f"{mujoco_playground.__name__}."
      )
      if not module.name.startswith(_EXCLUDED)
      and not module.name.endswith("_test")
  )


def _import(module_name: str) -> str | None:
  """Imports one module; returns a formatted traceback, or None on success."""
  # CLI modules define absl flags at module scope and several reuse a name, so
  # importing every module into one process collides. Nothing does that in
  # normal use, so drop each module's flags again after importing it.
  known_flags = set(flags.FLAGS)
  try:
    importlib.import_module(module_name)
    return None
  except Exception:  # pylint: disable=broad-except
    return traceback.format_exc()
  finally:
    for name in set(flags.FLAGS) - known_flags:
      delattr(flags.FLAGS, name)


def main() -> int:
  names = module_names()
  if not names:
    print("No modules found — the package layout or prefix is wrong.")
    return 1

  failures = {
      name: error
      for name in names
      if (error := _import(name)) is not None  # pylint: disable=superfluous-parens
  }
  for name, error in failures.items():
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n{error}", file=sys.stderr)

  checked = len(names)
  if failures:
    print(
        f"\n{len(failures)} of {checked} modules need an optional extra to"
        " import. Move the import into the function that uses it.",
        file=sys.stderr,
    )
    return 1

  print(f"All {checked} modules import with no extras installed.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
