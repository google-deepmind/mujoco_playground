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

"""OP3 dynamics randomization for sim-to-real locomotion training."""

import jax
from mujoco import mjx

# The scene declares the ground before the robot, and the robot body first.
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomize physical parameters for a batch of OP3 MJX models.

  The input rng has one PRNG key per environment. Only modified model
  properties are batched; every other model field remains shared.
  """

  @jax.vmap
  def randomize_one(key):
    # Floor sliding friction.
    key, sample = jax.random.split(key)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(sample, minval=0.4, maxval=1.0)
    )

    # Actuated joint friction and armature (OP3 has a floating base).
    key, sample = jax.random.split(key)
    dof_frictionloss = model.dof_frictionloss.at[6:].set(
        model.dof_frictionloss[6:]
        * jax.random.uniform(sample, (model.nv - 6,), minval=0.9, maxval=1.1)
    )

    key, sample = jax.random.split(key)
    dof_armature = model.dof_armature.at[6:].set(
        model.dof_armature[6:]
        * jax.random.uniform(sample, (model.nv - 6,), minval=1.0, maxval=1.05)
    )

    # Scale link masses and perturb the body link by at most 0.3 kg.
    key, sample = jax.random.split(key)
    body_mass = model.body_mass * jax.random.uniform(
        sample, (model.nbody,), minval=0.9, maxval=1.1
    )
    key, sample = jax.random.split(key)
    body_mass = body_mass.at[TORSO_BODY_ID].add(
        jax.random.uniform(sample, minval=-0.3, maxval=0.3)
    )

    # Preserve the MuJoCo position actuators' required negative gain bias.
    key, sample = jax.random.split(key)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        sample, (model.nu,), minval=0.8, maxval=1.2
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Scale the joint PD derivative term separately from position gains.
    key, sample = jax.random.split(key)
    dof_damping = model.dof_damping.at[6:].set(
        model.dof_damping[6:]
        * jax.random.uniform(sample, (model.nv - 6,), minval=0.8, maxval=1.2)
    )

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    )

  (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      actuator_gainprm,
      actuator_biasprm,
      dof_damping,
  ) = randomize_one(rng)

  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })

  model = model.tree_replace({  # pyrefly: ignore[bad-assignment]
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
  })
  return model, in_axes
