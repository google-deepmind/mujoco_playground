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
"""MuJoCo object type mappings, name pattern resolution, and joint helpers."""

from collections.abc import Mapping
from collections.abc import Sequence
import re

import mujoco
import numpy as np

SPEC_TYPE_INFO: dict[type, tuple[mujoco.mjtObj, str, str]] = {
    mujoco.MjsBody: (mujoco.mjtObj.mjOBJ_BODY, "body_", "body"),
    mujoco.MjsJoint: (mujoco.mjtObj.mjOBJ_JOINT, "jnt_", "joint"),
    mujoco.MjsGeom: (mujoco.mjtObj.mjOBJ_GEOM, "geom_", "geom"),
    mujoco.MjsSite: (mujoco.mjtObj.mjOBJ_SITE, "site_", "site"),
    mujoco.MjsLight: (mujoco.mjtObj.mjOBJ_LIGHT, "light_", "light"),
    mujoco.MjsCamera: (mujoco.mjtObj.mjOBJ_CAMERA, "cam_", "camera"),
    mujoco.MjsActuator: (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_", "actuator"),
    mujoco.MjsSensor: (mujoco.mjtObj.mjOBJ_SENSOR, "sensor_", "sensor"),
    mujoco.MjsTendon: (mujoco.mjtObj.mjOBJ_TENDON, "tendon_", "tendon"),
    mujoco.MjsEquality: (mujoco.mjtObj.mjOBJ_EQUALITY, "eq_", "equality"),
}

OBJ_TYPE_TO_ACCESSOR: dict[mujoco.mjtObj, str] = {
    info[0]: info[2] for info in SPEC_TYPE_INFO.values()
}
OBJ_TYPE_TO_ACCESSOR[mujoco.mjtObj.mjOBJ_XBODY] = "body"
PREFIX_TO_OBJ_TYPE: dict[str, mujoco.mjtObj] = {
    info[1]: info[0] for info in SPEC_TYPE_INFO.values()
}
PREFIX_TO_OBJ_TYPE["dof_"] = mujoco.mjtObj.mjOBJ_JOINT


def get_spec_by_obj_type(
    spec: mujoco.MjSpec,
    obj_type: mujoco.mjtObj,
    name: str,
) -> mujoco.MjStruct:
  """Maps obj_type to the corresponding MjSpec accessor."""
  accessor_name = OBJ_TYPE_TO_ACCESSOR.get(obj_type)
  if accessor_name is None:
    raise ValueError(f"Unsupported obj_type: {obj_type}")
  return getattr(spec, accessor_name)(name)


def resolve_names(
    spec: mujoco.MjSpec,
    objtype: mujoco.mjtObj,
    pattern: str | tuple[str, ...],
) -> list[str]:
  """Resolve names from spec via pattern matching.

  Args:
    spec: MjSpec to search.
    objtype: mujoco.mjtObj (mjOBJ_GEOM, mjOBJ_BODY, mjOBJ_XBODY,
      mjOBJ_ACTUATOR).
    pattern: Name(s) to match (exact, glob, or regex). Tuple for multiple.

  Returns:
    List of matched names.

  Raises:
    ValueError: If no names match the pattern.
  """
  patterns = [pattern] if isinstance(pattern, str) else list(pattern)

  if objtype == mujoco.mjtObj.mjOBJ_GEOM:
    all_names = [g.name for g in spec.geoms if g.name]
  elif objtype in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_XBODY):
    all_names = [b.name for b in spec.bodies if b.name]
  elif objtype == mujoco.mjtObj.mjOBJ_ACTUATOR:
    all_names = [a.name for a in spec.actuators if a.name]
  elif objtype == mujoco.mjtObj.mjOBJ_JOINT:
    all_names = [j.name for j in spec.joints if j.name]
  elif objtype == mujoco.mjtObj.mjOBJ_SENSOR:
    all_names = [s.name for s in spec.sensors if s.name]
  elif objtype == mujoco.mjtObj.mjOBJ_CAMERA:
    all_names = [c.name for c in spec.cameras if c.name]
  elif objtype == mujoco.mjtObj.mjOBJ_LIGHT:
    all_names = [l.name for l in spec.lights if l.name]
  elif objtype == mujoco.mjtObj.mjOBJ_SITE:
    all_names = [s.name for s in spec.sites if s.name]
  elif objtype == mujoco.mjtObj.mjOBJ_TENDON:
    all_names = [t.name for t in spec.tendons if t.name]
  else:
    raise ValueError(f"Unsupported objtype: {objtype}")

  matched = []
  for p in patterns:
    is_regex = any(c in p for c in r".*+?[]{}()\|^$")
    if is_regex:
      regex = re.compile(p)
      matched.extend(n for n in all_names if regex.search(n))
      continue

    if p in all_names:
      matched.append(p)

  if not matched:
    raise ValueError(f"Pattern {pattern} matched no names for {objtype}")

  return matched


def resolve_values(
    patterns: Mapping[str, float],
    names: Sequence[str],
    default: float = 0.0,
) -> list[float]:
  """Resolves a `{regex: value}` mapping to a per-name value list.

  Each regex must fully match a name. A name may match at most one pattern;
  names matching none get `default`.

  Args:
    patterns: Map from name regex to the value for matching names.
    names: Names (e.g. joints or actuators), in the order of the returned list.
    default: Value for names that match no pattern.

  Returns:
    One value per entry in `names`.

  Raises:
    ValueError: If a name matches more than one pattern.
  """
  values = []
  for name in names:
    matches = [
        v for pat, v in patterns.items() if re.fullmatch(pat, name) is not None
    ]
    if len(matches) > 1:
      raise ValueError(f"{name!r} matches multiple patterns: {patterns}")
    values.append(matches[0] if matches else default)
  return values


def match_names(patterns: Sequence[str], names: Sequence[str]) -> list[str]:
  """Returns the names that fully match any regex in `patterns`.

  Args:
    patterns: Name regexes.
    names: Candidate names (e.g. joints or actuators).

  Returns:
    The matching names, in `names` order.
  """
  return [
      name
      for name in names
      if any(re.fullmatch(p, name) is not None for p in patterns)
  ]


def soft_joint_limits(
    model: mujoco.MjModel, joint_names: Sequence[str], factor: float
) -> np.ndarray:
  """Returns joint ranges scaled by `factor` about their midpoints.

  Args:
    model: Compiled model containing the joints.
    joint_names: Joints to read, in the order of the returned rows.
    factor: Fraction of each joint's full range to keep (e.g. 0.9).

  Returns:
    Array of shape `(len(joint_names), 2)` holding `[low, high]` per joint.
  """
  jnt_range = np.stack([model.joint(n).range for n in joint_names])
  mid = jnt_range.mean(axis=-1)
  half = 0.5 * (jnt_range[:, 1] - jnt_range[:, 0]) * factor
  return np.stack([mid - half, mid + half], axis=-1)
