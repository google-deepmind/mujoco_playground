Changelog
=========

MuJoCo Playground follows `MuJoCo's semantic versioning
<https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`_
(`SUPERMAJOR.MAJOR.MINOR_OR_PATCH`).

Version 1.0.1 (November 8, 2026)
--------------------------------

- ``SO101Follower``, ``YamFollower``, and ``BodyTrackingDevice`` now implement ``get_measurements`` instead of
  ``get_substep_measurements``.
- ``RslRlVecEnvWrappercfg.is_finite_horizon`` now defaults to ``False`` to enable timeout bootstrapping.
- ``MotionReferenceProducer`` and ``SmoothCartesianIKProcessor`` advance state in ``on_step`` / ``on_begin_stepping``
  for partial resets.
- Restored Linux CUDA 13 PyTorch runtime dependencies in ``uv.lock``
  (`#382 <https://github.com/google-deepmind/mujoco_playground/issues/382>`__).

Version 1.0.0 (November 6, 2026)
--------------------------------

Rebuilt on `reaf-sim` (`reaf.simulation.torch`) and MuJoCo Warp with declarative
`flat_scene` environments, paired `sim/` and `real/` robot packages (SO-101,
YAM, Trossen WXAI, Unitree G1), RSL-RL training, teleop, `mujoco.sysid`, and
sim-to-real deployment.
