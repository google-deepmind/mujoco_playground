"""Domain randomization for Solo8 Stage 2 DRL training.

Paper Section 2.3: "We randomize the ground position up and down in the range
of [-5 cm, 5 cm] (approximately 20% of the robot leg length). We also randomize
the ground surface friction coefficient in the range [0.5, 1.0]."

These randomizations are applied per-environment when training with
--domain_randomization flag:
  python learning/train_jax_ppo.py \
    --env_name Solo8TrottingDemonstrationStage2 \
    --domain_randomization \
    --load_checkpoint_path logs/Solo8TrottingDemonstrationStage1-<ts>/checkpoints
"""

import jax
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomize ground height and friction for robust trotting.

  Randomized parameters:
    - Ground height: floor geom z-position offset in [-0.05, 0.05] m.
      This simulates uneven terrain (~20% of leg length 0.32 m).
    - Ground friction: contact pair friction coefficient in [0.5, 1.0].
      All 4 foot-ground contact pairs are set to the same value per env.

  Args:
    model: The MJX model to randomize.
    rng: Per-environment RNG keys, shape (num_envs,).

  Returns:
    model: Randomized model (batched along axis 0 for randomized fields).
    in_axes: Tree of axis specifications for vmap.
  """
  floor_geom_id = 0  # "floor" geom is first in the scene XML.
  num_contact_pairs = 4  # FL, FR, HL, HR foot-ground pairs.

  @jax.vmap
  def rand_dynamics(rng):
    # Ground height: offset floor z-position by U(-0.05, 0.05) m.
    rng, key = jax.random.split(rng)
    ground_offset = jax.random.uniform(key, minval=-0.05, maxval=0.05)
    geom_pos = model.geom_pos.at[floor_geom_id, 2].set(ground_offset)

    # Ground friction: set all contact pair sliding friction to U(0.5, 1.0).
    # Solo8 uses <pair> contacts, so we randomize pair_friction, not
    # geom_friction. pair_friction shape: (npair, 5) where [:, 0] is
    # the primary sliding friction coefficient.
    rng, key = jax.random.split(rng)
    friction_val = jax.random.uniform(key, minval=0.5, maxval=1.0)
    pair_friction = model.pair_friction
    for i in range(num_contact_pairs):
      pair_friction = pair_friction.at[i, 0].set(friction_val)

    return geom_pos, pair_friction

  geom_pos, pair_friction = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_pos": 0,
      "pair_friction": 0,
  })

  model = model.tree_replace({
      "geom_pos": geom_pos,
      "pair_friction": pair_friction,
  })

  return model, in_axes
