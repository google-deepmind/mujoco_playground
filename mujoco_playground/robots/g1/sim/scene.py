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

"""Builds the G1 robot."""

import re

import mujoco
from mujoco_playground import asset
from mujoco_playground.robots.g1.sim import device
from reaf.simulation.torch import mujoco_types


def _apply_full_collision(spec: mujoco.MjSpec) -> None:
  """Configure collision geoms and disable non-collision geoms."""
  foot_pat = re.compile(device.FOOT_GEOM_REGEX)
  collision_pat = re.compile(device.COLLISION_GEOM_REGEX)
  for geom in spec.geoms:
    name = geom.name
    if collision_pat.match(name):
      is_foot = foot_pat.match(name) is not None
      geom.contype = 1
      geom.conaffinity = 1
      geom.condim = device.FOOT_CONDIM if is_foot else device.OTHER_CONDIM
      geom.priority = device.FOOT_PRIORITY if is_foot else 0
      if is_foot:
        geom.friction[0] = device.FOOT_FRICTION
    else:
      geom.contype = 0
      geom.conaffinity = 0


def _add_position_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    group: device.PositionActuatorGroup,
) -> None:
  """Add a PD position actuator and joint armature for `joint_name`."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)
  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  actuator.gainprm[0] = group.stiffness
  actuator.biasprm[1] = -group.stiffness
  actuator.biasprm[2] = -group.damping

  # Position setpoints must be allowed beyond joint limits; force is
  # bounded by forcerange instead.
  actuator.inheritrange = 0.0
  actuator.ctrllimited = False
  actuator.forcelimited = True
  actuator.forcerange[0] = -group.effort_limit
  actuator.forcerange[1] = group.effort_limit
  # Informational ctrlrange (not enforced since ctrllimited=False).
  target_range = spec.joint(joint_name).range
  delta = group.effort_limit / group.stiffness
  actuator.ctrlrange[0] = target_range[0] - delta
  actuator.ctrlrange[1] = target_range[1] + delta

  spec.joint(joint_name).armature = group.armature


def _add_actuators(spec: mujoco.MjSpec) -> None:
  """Create actuators group by group."""
  for group in device.G1_ACTUATOR_GROUPS:
    for joint_name in mujoco_types.match_names(
        group.target_names_expr, device.JOINT_NAMES
    ):
      _add_position_actuator(spec, joint_name, group)


def _add_init_state_keyframe(spec: mujoco.MjSpec) -> None:
  """Add the `init_state` keyframe with the default bent-knees pose."""
  joint_pos = {
      name: value
      for name, value in zip(device.JOINT_NAMES, device.KNEES_BENT_JOINT_POS)
  }
  qpos = (
      list(device.KNEES_BENT_ROOT_POS)
      + list(device.KNEES_BENT_ROOT_QUAT)
      + [joint_pos[n] for n in device.JOINT_NAMES]
  )
  key = spec.add_key(name="init_state", qpos=qpos)
  ctrl = [joint_pos.get(act.target, 0.0) for act in spec.actuators]
  key.ctrl = ctrl


def _apply_tracking_mujoco_options(spec: mujoco.MjSpec) -> None:
  """Configure MuJoCo solver and integrator options."""
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
  spec.option.tolerance = 1e-8
  spec.option.ls_tolerance = 0.01
  spec.option.impratio = 1.0


def g1_robot_spec(xml_uri: str | None = None) -> mujoco.MjSpec:
  """Builds the G1 robot MjSpec with collision, actuators, and keyframe."""
  spec = asset.get(xml_uri or device.G1_XML_URI).spec()
  _apply_full_collision(spec)
  _add_actuators(spec)
  _add_init_state_keyframe(spec)
  return spec


def g1_scene_spec(xml_uri: str | None = None) -> mujoco.MjSpec:
  """Builds G1 robot + plane terrain + light, ready to use as base_scene."""
  spec = g1_robot_spec(xml_uri)
  _apply_tracking_mujoco_options(spec)

  spec.worldbody.add_body(name="terrain").add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[0, 0, 0.01],
  )
  spec.worldbody.add_light(
      pos=[0.0, 0.0, 1.5],
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  )
  return spec
