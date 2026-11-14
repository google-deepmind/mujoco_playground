Changelog
=========

MuJoCo Playground follows `MuJoCo's semantic versioning
<https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`_
(`SUPERMAJOR.MAJOR.MINOR_OR_PATCH`).

Unreleased
----------

- The MCAP log reader and ``view_log`` decode messages with ``mcap_protobuf`` instead of TensorFlow, so reading an
  episode no longer loads TensorFlow. On Linux, TensorFlow 2.21 crashes a process that loads Triton 3.7 or newer
  after it. The ``real`` extra now declares ``mcap`` and ``mcap-protobuf-support``, which came only through
  safari-sdk before.
- Breaking: ``iter_features`` yields the same ``(topic, features)`` pairs, but the feature values are now protobuf
  messages built from the schema stored in the file, not TensorFlow ``Feature`` objects. They cannot be pickled or
  passed to TensorFlow; convert them to NumPy first.
- The ``feetech`` and ``so101`` extras now need lerobot 0.7 (``>=0.7.0,<0.8``), which drives the Feetech servos
  through ``rustypot`` instead of ``feetech-servo-sdk``. It brings huggingface-hub 2, which needs click 8.4.2 or
  newer, so ``click`` is overridden past i2rt's ``click<8.2.0`` pin.
- lerobot 0.7 fails a bus transfer on one stray byte, so SO-101 joint reads now retry twice, like lerobot's own SO-101
  robot. Calibration retries its position reads, its ``Operating_Mode`` writes and the half-turn homing five times,
  and builds lerobot's ``SerialMotorsBus``, because ``FeetechMotorsBus`` is deprecated in 0.7.

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
