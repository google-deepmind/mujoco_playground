"""Domain randomization for the ZBot environment."""

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: PRNGKeyArray) -> tuple[mjx.Model, jnp.ndarray]:
    """Applies domain randomization to the model."""

    @jax.vmap
    def rand_dynamics(rng: PRNGKeyArray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(jax.random.uniform(key, minval=0.4, maxval=1.0))

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(key, shape=(10,), minval=0.9, maxval=1.1)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(key, shape=(10,), minval=1.0, maxval=1.05)
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-1.0, 1.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(qpos0[7:] + jax.random.uniform(key, shape=(10,), minval=-0.05, maxval=0.05))

        return (
            geom_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
        )

    friction, frictionloss, armature, body_mass, qpos0 = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": friction,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
        }
    )

    return model, in_axes
