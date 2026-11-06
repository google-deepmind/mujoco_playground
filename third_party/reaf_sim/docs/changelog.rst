Changelog
=========

Unreleased (1.0.0)
------------------

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
