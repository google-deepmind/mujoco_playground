Environment reset
=================

:class:`~reaf.simulation.torch.EnvironmentReset` governs state initialization, domain randomization, and reset lifecycle
events when an environment starts up or an episode terminates. It manages both simulation-wide resets and partial resets
across arbitrary subsets of parallel worlds.

Reset pipeline
--------------

When an environment is reset (either completely or partially via a boolean ``env_mask`` of shape ``(nworld,)``), the
reset pipeline executes the following stages in strict sequence:

1. **Model parameter restoration**: If model parameters were perturbed by previous domain randomizations, `reaf-sim`
   restores the default un-randomized model fields (e.g. mass, friction, damping, joint ranges) in-place for the masked
   worlds:

   .. code-block:: python

       sim.model.restore_defaults(env_mask)

   This prevents cumulative parameter drift across successive episodes.

2. **Device layer reset**: Calls ``device_layer.reset(env_mask)``, which resets the underlying
   :class:`~reaf.simulation.torch.Simulator` state (calling ``mj_resetData`` or ``mjw.reset_data`` to reset ``qpos`` to
   ``qpos0`` and zero ``qvel``, ``ctrl``, ``qfrc``), clears device queues and delay lines, and resets substep
   processors.
3. **Resetter callbacks**: Executes registered resetters (such as
   :class:`~reaf.simulation.torch.FunctionalEnvironmentReset`,
   :class:`~reaf.simulation.torch.DomainRandomizationResetter`, or task-specific hooks) to apply initial joint
   positions, velocities, and newly sampled domain randomizations.
4. **Model const update**: If any model fields were modified during the reset callback,
   :class:`~reaf.simulation.torch.TorchWrapper` flushes minimal ``set_const`` stages to update derived physics
   quantities:

   .. code-block:: python

       sim.set_const()

5. **Forward kinematics & sensor evaluation**: Calls ``sim.forward()`` and ``device_layer.forward()`` to ensure that all
   derived kinematics (Cartesian poses, matrices, site positions, camera images) are fresh and valid for the initial
   ``FIRST`` timestep observation.

:class:`~reaf.simulation.torch.DomainRandomizationResetter`
-----------------------------------------------------------

:class:`~reaf.simulation.torch.DomainRandomizationResetter` evaluates declarative randomization terms. Terms can be
triggered either once at environment startup (``trigger="startup"``) or at the start of every episode
(``trigger="reset"``).

Simulation-only terminations
----------------------------

Certain termination conditions represent simulation failures or physical violations (such as objects falling below a
floor plane, excessive contact forces, or simulation instability) that should terminate an episode during training in
simulation, but have no place in real robot task logic.

:class:`~reaf.simulation.torch.TerminationResetter` checks these sim-only criteria at each control step, keeping task
definitions clean, portable, and invariant across simulation and hardware.

For full class signatures and details, see :class:`reaf.simulation.torch.EnvironmentReset` in the :doc:`API reference
<../api>`.
