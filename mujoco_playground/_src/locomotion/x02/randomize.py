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
"""Domain randomization for the X02 environment."""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(10,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(10,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(10,), minval=-0.05, maxval=0.05)
    )

    # Scale damping: *U(0.95, 1.05).
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[6:] * jax.random.uniform(
        key, shape=(10,), minval=0.95, maxval=1.05
    )
    dof_damping = model.dof_damping.at[6:].set(damping)

    # Scale actuator kp: *U(0.5, 5.0).
    rng, key = jax.random.split(rng)
    noise = jax.random.uniform(
        key, shape=(10,), minval=0.5, maxval=5.0
    )
    actuator_gain = model.actuator_gainprm.at[:, 0].set(model.actuator_gainprm[:, 0] * noise)
    actuator_bias = model.actuator_biasprm.at[:, 1].set(model.actuator_biasprm[:, 1] * noise)

    # Scale actuator kv: *U(0.5, 2.0).
    rng, key = jax.random.split(rng)
    noise = jax.random.uniform(
        key, shape=(10,), minval=0.5, maxval=2.0
    )
    actuator_bias = actuator_bias.at[:, 2].set(actuator_bias[:,2] * noise)

    # Randomize com of torso: +U(-0.07, 0.07).
    rng, key = jax.random.split(rng)
    com_offset = jax.random.uniform(
        key, shape=(3,), minval=-0.07, maxval=0.07
    )
    body_com = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + com_offset
    )

    # Randomize all coms: +U(-0.01, 0.01).
    rng, key = jax.random.split(rng)
    com_offset = jax.random.uniform(
        key, shape=(model.nbody, 3), minval=-0.01, maxval=0.01
    )
    body_com = body_com.at[:].set(
        body_com + com_offset
    )

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gain,
        actuator_bias,
        body_mass,
        body_com,
        qpos0,
    )

  (
      friction,
      frictionloss,
      armature,
      damping,
      actuator_gain,
      actuator_bias,
      body_mass,
      body_com,
      qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "body_mass": 0,
      "body_ipos": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "dof_damping": damping,
      "actuator_gainprm": actuator_gain,
      "actuator_biasprm": actuator_bias,
      "body_mass": body_mass,
      "body_ipos": body_com,
      "qpos0": qpos0,
  })

  return model, in_axes
