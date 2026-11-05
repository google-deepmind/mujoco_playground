Changelog
=========

Unreleased (0.0.2)
------------------

Device Layer
^^^^^^^^^^^^

- Refactored lifecycle hooks in :class:`~reaf.simulation.torch.Device`:
    * ``on_substep`` is triggered after every physics substep
    * ``on_step`` is triggered on the final physics substep, after ``on_substep`` and before ``get_measurements``.
    * ``on_begin_stepping`` is triggered after resets, once the state is final.
    * ``get_measurements`` replaces the now deprecated ``get_substep_measurements``.

- Refactored lifecycle hooks in :class:`~reaf.simulation.torch.SubstepMeasurementsProcessor`:
    * ``on_substep`` is triggered after every physics substep.
    * ``on_begin_stepping`` is triggered after resets, once the state is final.
    * ``process_measurements`` replaces the now deprecated ``process_substep_measurements``.

Common Devices
^^^^^^^^^^^^^^

- After a reset, every history slot of :class:`~reaf.simulation.torch.ContactSensorDevice` holds the reset reading
  instead of zeros, and its history also advances while the device is dark.

- After a reset, delayed :class:`~reaf.simulation.torch.StochasticSubstepMeasurements` keys read the reset reading until
  the delay has passed. Keys without delay apply their bias and latest noise sample to the current reading, and readings
  pass through until the processor has seen a substep or reset.

Bugs Fixed
^^^^^^^^^^

- After a partial reset, :class:`~reaf.simulation.torch.DeviceLayer` only refreshes the measurements of the reset
  environments.

- :class:`~reaf.simulation.torch.ContactSensorDevice` and
  :class:`~reaf.simulation.torch.StochasticSubstepMeasurements` no longer advance their history and delay buffers for
  environments that are not reset.

- Features observers and loggers no longer run twice per step when any environment resets.

0.0.1
-----

Initial open-source release of `reaf-sim` (:mod:`reaf.simulation.torch`).
