Device layer
============

:class:`~reaf.simulation.torch.DeviceLayer` is the simulation and hardware abstraction boundary in `REAF`. In
`reaf-sim`, :class:`~reaf.simulation.torch.DeviceLayer` manages a collection of
:class:`~reaf.simulation.torch.Device`\ s, coordinates the underlying :class:`~reaf.simulation.torch.Simulator`, and
orchestrates the physics substepping loop.

Architecture
------------

In `reaf-sim`, the :class:`~reaf.simulation.torch.TaskLayer` interacts with the
:class:`~reaf.simulation.torch.DeviceLayer` exclusively via standardized commands and measurements:

- **Command ingestion**: Maps commands (e.g. ``"arm/ctrl"``) into actuator inputs, mocap targets, or applied forces.
- **Physics advancement**: Steps the underlying :class:`~reaf.simulation.torch.Simulator` through intermediate physics
  substeps.
- **Measurement extraction**: Extracts physical states (joint positions, velocities, contact forces, camera pixels) into
  batched PyTorch tensors for the task layer.

Device lifecycle
----------------

Each :class:`~reaf.simulation.torch.Device` follows a standardized lifecycle:

1. **Spec configuration** (``configure_spec``): At scene composition time, before compilation, devices attach required
   bodies, joints, actuators, sensors, sites, or cameras to the uncompiled ``mujoco.MjSpec``.
2. **Initialization** (``initialize``): After the :class:`~reaf.simulation.torch.Simulator` compiles the model, devices
   cache indices or allocate runtime buffers.
3. **Substep commands** (``set_substep_commands``): Before each physics substep, devices write control inputs into the
   simulation.
4. **Substep update** (``on_substep``): After each physics substep, devices advance per-substep state (e.g. history
   buffers or captures at a given substep).
5. **Step update** (``on_step``): On the final physics substep, right after ``on_substep`` and before
   ``get_measurements`` and substep processors, devices advance per-control-step state (holding the prior step's value
   across earlier substeps).
6. **Measurements** (``get_measurements``): Returns the current measurements in a dictionary of tensors. By convention,
   it does not advance any state, since :class:`~reaf.simulation.torch.DeviceLayer` may read a device more than once at
   any point in the device layer lifecycle.
7. **Reset** (``reset``): Clears per-world buffers and sets the initial simulation state of the specified environments
   (via ``env_mask``).
8. **Begin stepping** (``on_begin_stepping``): Records the reset state of the reset environments (e.g. fills history
   buffers with the reset reading). ``on_step`` does not run after resets, so devices that latch per-step state set it
   for the reset environments here.

.. note::

   ``on_begin_stepping`` is separate from ``reset`` because ``reset`` runs before the environment resetter,
   ``sim.set_const()``, and ``sim.forward()``, so ``sim.data`` (e.g. ``sensordata`` or ``xpos``) does not hold the reset
   state yet. ``on_begin_stepping`` runs after them and before the reset state is read, which lets stateful devices seed
   their buffers for the reset environments in a hook rather than in ``get_measurements``.

Only measurements of the reset environments are refreshed after a partial reset; the other environments keep the
measurements of their last control step. Measurements are committed into persistent contiguous output buffers, so
returned tensors are isolated from subsequent ``sim.data`` updates and keep stable memory addresses across steps and
resets.

Substep loop (simulation)
-------------------------

:class:`~reaf.simulation.torch.DeviceLayer` runs ``n_substeps = round(control_timestep / simulation_timestep)`` physics
steps per control step. Commands and measurements are processed at the physics frequency:

- :class:`~reaf.simulation.torch.SubstepCommandsProcessor` (e.g.
  :class:`~reaf.simulation.torch.StochasticSubstepCommands`): Transforms commands before they reach devices (actuation
  latency, dropout, jitter).
- :class:`~reaf.simulation.torch.SubstepMeasurementsProcessor` (e.g.
  :class:`~reaf.simulation.torch.StochasticSubstepMeasurements`): Post-processes raw device measurements before they
  reach the task layer (sensor noise, latency, bias). Like devices, processors should advance state in ``on_substep``
  and ``on_begin_stepping`` and, by convention, not in ``process_measurements``. Substep measurements are zero-copy
  views of simulator or device state, so processors must not modify them in place, and must copy them into their own
  buffers to keep them (e.g. as a history).

Built-in devices
----------------

:mod:`reaf.simulation.torch` provides built-in devices organized by functionality:

- **Actuation**:

  - :class:`~reaf.simulation.torch.ActuatorDevice`: Controls existing MJCF joint, tendon, or site actuators and exposes
    their state and force measurements.
  - :class:`~reaf.simulation.torch.ArticulatedDevice`: Manages articulated robot mechanisms, exposing joint positions,
    velocities, effort limits, and home pose resets.

- **Bodies**:

  - :class:`~reaf.simulation.torch.FreeDevice`: Manages 6-DoF free-floating bodies (positions, orientations, and
    linear/angular velocities).
  - :class:`~reaf.simulation.torch.MocapDevice`: Controls kinematic motion targets and mocap bodies for Cartesian
    end-effector tracking.

- **Sites & sensors**:

  - :class:`~reaf.simulation.torch.SiteDevice`: Measures global Cartesian positions, orientations, and coordinate
    transform matrices of MuJoCo sites.
  - :class:`~reaf.simulation.torch.InsideSiteDevice`: Performs spatial containment queries to test whether points or
    bodies reside inside volumetric site bounds.
  - :class:`~reaf.simulation.torch.ContactSensorDevice`: Captures contact forces, normal pressures, and collision
    impulses between bodies or geom pairs.
  - :class:`~reaf.simulation.torch.TactileDevice`: Attaches a ``mjSENS_TACTILE`` sensor with optional post-processing.
  - :class:`~reaf.simulation.torch.MeasurementDevice`: Evaluates custom callable or :class:`~reaf.simulation.torch.Term`
    functions over ``(model, data)`` for simulation-only state measurements.

- **Cameras & visual rendering**:

  - :class:`~reaf.simulation.torch.CameraDevice`: Batched GPU camera rendering RGB images and metric depth maps via
    MuJoCo Warp.
  - :class:`~reaf.simulation.torch.MujocoCameraDevice`: Headless CPU camera rendering RGB, depth, and segmentation masks
    via ``mujoco.Renderer``.
  - :class:`~reaf.simulation.torch.BatchedMujocoCameraDevice`: Multi-camera CPU device rendering RGB, depth, and
    segmentation masks across multiple cameras via ``mujoco.Renderer``.

  Both ``mujoco.Renderer`` devices also offer ``render_now(...)`` for single-world ``Impl.CPP`` simulations, which
  renders the current state (optionally a subset of cameras and render modes) without touching cached measurements.
  Pass ``on_demand_only=True`` (with ``render_rate_hz=None``) for cameras that are only read through ``render_now``;
  such devices publish no measurements and skip the render after each reset.

- **Heterogeneous**:

  - :class:`~reaf.simulation.torch.MultiAssetDevice`: Assigns and resamples different ``MultiAsset`` geoms and meshes
    for a target body across parallel worlds.

Adding a device
---------------

Custom devices subclass :class:`~reaf.simulation.torch.Device` and implement its abstract methods.

For full signature details of all methods, see :class:`reaf.simulation.torch.DeviceLayer` and
:class:`reaf.simulation.torch.Device` in the :doc:`API reference <../api>`.

Deprecated methods
------------------

``Device.get_substep_measurements`` and ``SubstepMeasurementsProcessor.process_substep_measurements`` are deprecated in
favor of the hooks above. Components that override them (or that override none of the replacement methods and declare
measurements) keep their previous per-substep and post-reset behavior; :class:`~reaf.simulation.torch.DeviceLayer` emits
a ``DeprecationWarning`` for legacy processors, and raises a ``TypeError`` if a component overrides both a deprecated
method and one of its replacements.
