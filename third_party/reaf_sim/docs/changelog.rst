Changelog
=========

Version 1.0.0 (November 6, 2026)
--------------------------------

Device Layer
^^^^^^^^^^^^
.. admonition:: Breaking changes
   :class: attention

   - Refactored lifecycle hooks in :class:`~reaf.simulation.torch.Device` and
     :class:`~reaf.simulation.torch.SubstepMeasurementsProcessor` to separate state updates from measurement reads:

     **Migration:** Move state updates into ``on_substep`` / ``on_step`` / ``on_begin_stepping`` and replace
     ``get_substep_measurements`` / ``process_substep_measurements`` with
     :meth:`~reaf.simulation.torch.Device.get_measurements` /
     :meth:`~reaf.simulation.torch.SubstepMeasurementsProcessor.process_measurements`.

- Refactored lifecycle hooks in :class:`~reaf.simulation.torch.Device`:
    * ``on_substep`` is triggered after every physics substep
    * ``on_step`` is triggered on the final physics substep, after ``on_substep`` and before ``get_measurements``.
    * ``on_begin_stepping`` is triggered after resets, once the state is final.
    * ``get_measurements`` replaces the now deprecated ``get_substep_measurements``.

- Refactored lifecycle hooks in :class:`~reaf.simulation.torch.SubstepMeasurementsProcessor`:
    * ``on_substep`` is triggered after every physics substep.
    * ``on_begin_stepping`` is triggered after resets, once the state is final.
    * ``process_measurements`` replaces the now deprecated ``process_substep_measurements``.

Task Layer
^^^^^^^^^^

- Added :class:`~reaf.simulation.torch.MetricProvider`, :class:`~reaf.simulation.torch.FunctionalMetricProvider`, and
  :class:`~reaf.simulation.torch.flat_scene.Metric` for per-episode metric reductions (``mean``, ``last``, ``max``,
  ``sum``).

Common Devices
^^^^^^^^^^^^^^
.. admonition:: Breaking changes
   :class: attention

   - Removed ``ActuatorDevice.configure_spec()`` and its ``transmission`` parameter.

     **Migration:** Define actuators in the MJCF or ``MjSpec`` before device initialization;
     :class:`~reaf.simulation.torch.ActuatorDevice` now infers transmission types from the compiled ``MjModel``.

- Added :class:`~reaf.simulation.torch.TactileDevice` (`mjSENS_TACTILE`) with optional Laplacian smoothing and 2D
  spatial observation plots in the :mod:`mujoco.studio` viewer.

- Added ``<dcmotor>`` and MIMO actuator support to :class:`~reaf.simulation.torch.TorchWrapper` and
  :class:`~reaf.simulation.torch.ActuatorDevice`.

- After a reset, every history slot of :class:`~reaf.simulation.torch.ContactSensorDevice` holds the reset reading
  instead of zeros, and its history also advances while the device is dark.

- The built-in devices implement ``get_measurements`` instead of ``get_substep_measurements``. Subclasses that override
  ``get_substep_measurements`` must override ``get_measurements`` instead.

- Camera devices render in ``on_substep`` and ``on_begin_stepping``. Their ``get_measurements`` returns the latest frame
  without rendering, and raises a ``RuntimeError`` if nothing was rendered since the last reset.

- After a reset, delayed :class:`~reaf.simulation.torch.StochasticSubstepMeasurements` keys read the reset reading until
  the delay has passed. Keys without delay apply their bias and latest noise sample to the current reading, and readings
  pass through until the processor has seen a substep or reset.

Task Layer
^^^^^^^^^^

- Added lifecycle hooks to :class:`~reaf.simulation.torch.FeatureProducer` and
  :class:`~reaf.simulation.torch.ObservationSpaceAdapter`:

  * ``on_step`` is triggered once per ``Environment.step()``, after the physics step and before the reads. For producers,
    ``TaskLayer.compute_all_features`` triggers it.
  * ``on_begin_stepping`` is triggered after resets, for the reset environments only.
  * ``produce_features`` and ``observations_from_features`` must not advance state: they are called again for all
    environments after a partial reset.

- :class:`~reaf.simulation.torch.TaskLayer` now requires feature producers to derive from
  :class:`~reaf.simulation.torch.FeatureProducer`, and :class:`~reaf.simulation.torch.Environment` requires its
  observation adapter to derive from :class:`~reaf.simulation.torch.ObservationSpaceAdapter`.

- :class:`~reaf.simulation.torch.FunctionalObservationAdapter` advances its history in ``on_step`` and fills it with the
  reset observation in ``on_begin_stepping``; reading a history term returns the stored observations. History terms
  read as zeros until the first ``on_begin_stepping``, and ``reset()`` no longer clears the history, so code that drives
  the adapter outside of :class:`~reaf.simulation.torch.Environment` must call ``on_begin_stepping`` after ``reset()``
  and ``on_step`` on every step.

Bugs Fixed
^^^^^^^^^^

- After a partial reset, :class:`~reaf.simulation.torch.DeviceLayer` only refreshes the measurements of the reset
  environments.

- :class:`~reaf.simulation.torch.ContactSensorDevice` and
  :class:`~reaf.simulation.torch.StochasticSubstepMeasurements` no longer advance their history and delay buffers for
  environments that are not reset.

- :class:`~reaf.simulation.torch.FunctionalObservationAdapter` no longer pushes a duplicate observation into the
  history of environments that are not reset, which happened on every step with any partial reset (and with
  ``AutoResetMode.NO_SYNC``). Without noise, ``AutoResetMode.NO_SYNC`` and ``AutoResetMode.SYNC`` now produce the same
  timesteps.

- Features observers and loggers no longer run twice per step when any environment resets.

0.0.1
-----

Initial open-source release of `reaf-sim` (:mod:`reaf.simulation.torch`).
