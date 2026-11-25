Changelog
=========

MuJoCo Playground follows `MuJoCo's semantic versioning
<https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`_
(`SUPERMAJOR.MAJOR.MINOR_OR_PATCH`).

Upcoming Version (not yet released)
-----------------------------------

- Fixed ``impl="cpp"`` builds for vision environments: ``env_registry`` now forwards ``impl`` to environment
  config functions that accept it, and the Cartpole and Trossen vision tasks use the native MuJoCo camera on
  ``cpp``.

Version 1.1.0 (November 23, 2026)
---------------------------------

.. admonition:: Breaking changes
   :class: attention

   - Replaced ``safari-sdk`` with plain JSON-encoded ``mcap`` in ``real`` episodic logging; legacy
     ``tf.train.Example`` MCAP logs are no longer readable.
   - Replaced ``lerobot[feetech]`` with ``rustypot`` for SO-101 serial control.

- Added ``playground`` CLI entrypoint.
- Added Python 3.14 and PyTorch 2.15 support (`#386 <https://github.com/google-deepmind/mujoco_playground/issues/386>`__, `#393 <https://github.com/google-deepmind/mujoco_playground/issues/393>`__).
- Migrated SO-101 teleop and ``tabletop_v0`` viewers to MuJoCo Studio.

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

Rebuilt on `reaf-sim` (`reaf.simulation.torch`) and MuJoCo Warp (CUDA for GPU
training; CPU is also supported) with declarative `flat_scene` environments,
paired `sim/` and `real/` robot packages (SO-101, YAM, Trossen WXAI, Unitree
G1), RSL-RL training, teleop, `mujoco.sysid`, and sim-to-real deployment. The
original RSS 2025 JAX/MJX/Brax environments ended at `0.2.0` on PyPI and remain
available under `rss2025/`.
