# Copyright 2025 DeepMind Technologies Limited
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
"""Domain randomization for the K1 environment."""

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
ANKLE_JOINT_IDS = np.array([[20, 21, 26, 27]])


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.3, 0.6).
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.3, maxval=0.6)
    )

    # Shift trunk CoM: x +U(-0.025, 0.025), y/z +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    dcom = jax.random.uniform(
        key,
        shape=(3,),
        minval=jp.array([-0.025, -0.05, -0.05]),
        maxval=jp.array([0.025, 0.05, 0.05]),
    )
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dcom
    )

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(22,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.98, 1.02).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.98, maxval=1.02
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.01, 0.01).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(22,), minval=-0.01, maxval=0.01)
    )

    # Joint stiffness: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.9, maxval=1.1
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Higher range on the ankles.
    rng, key = jax.random.split(rng)
    kd = model.dof_damping[ANKLE_JOINT_IDS] * jax.random.uniform(
        key, (4,), minval=0.5, maxval=2.0
    )
    dof_damping = model.dof_damping.at[ANKLE_JOINT_IDS].set(kd)

    return (
        geom_friction,
        body_ipos,
        dof_armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    )

  (
      friction,
      body_ipos,
      armature,
      body_mass,
      qpos0,
      actuator_gainprm,
      actuator_biasprm,
      dof_damping,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
  })

  return model, in_axes
