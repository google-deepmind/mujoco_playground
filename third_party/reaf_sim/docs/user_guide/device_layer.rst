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
4. **Substep measurements** (``get_substep_measurements``): After each physics substep, devices read fresh state from
   ``sim.data`` (or cameras/sensors) and return this information in a dictionary of tensors.
5. **Reset** (``reset``): Resets internal buffers, delay queues, or hardware caches for specified environments (via
   ``env_mask``).

Substep loop (simulation)
-------------------------

:class:`~reaf.simulation.torch.DeviceLayer` runs ``n_substeps = round(control_timestep / simulation_timestep)`` physics
steps per control step, with optional processors transforming commands and measurements at the physics frequency.

- :class:`~reaf.simulation.torch.SubstepCommandsProcessor` (e.g.
  :class:`~reaf.simulation.torch.StochasticSubstepCommands`): Transforms commands before they reach devices (actuation
  latency, dropout, jitter).
- :class:`~reaf.simulation.torch.SubstepMeasurementsProcessor` (e.g.
  :class:`~reaf.simulation.torch.StochasticSubstepMeasurements`): Post-processes raw device measurements before they
  reach the task layer (sensor noise, latency, bias).

Built-in devices
----------------

:mod:`reaf.simulation.torch` provides built-in devices organized by functionality:

- **Actuation**:

  - :class:`~reaf.simulation.torch.ActuatorDevice`: Configures joint, tendon, or slider actuators with position,
    velocity, or torque (motor) control modes.
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

- **Cameras & visual rendering**:

  - :class:`~reaf.simulation.torch.CameraDevice`: Batched GPU camera rendering RGB images and metric depth maps via
    MuJoCo Warp.
  - :class:`~reaf.simulation.torch.MujocoCameraDevice`: Headless CPU camera rendering RGB, depth, and segmentation masks
    via ``mujoco.Renderer``.
  - :class:`~reaf.simulation.torch.BatchedMujocoCameraDevice`: Multi-camera CPU entity rendering multiple cameras in a
    single device via ``mujoco.Renderer``.

- **Heterogeneous**:

  - :class:`~reaf.simulation.torch.MultiAssetDevice`: Assigns and resamples different ``MultiAsset`` geoms and meshes
    for a target body across parallel worlds.

Adding a device
---------------

Custom devices subclass :class:`~reaf.simulation.torch.Device` and implement its abstract methods.

For full signature details of all methods, see :class:`reaf.simulation.torch.DeviceLayer` and
:class:`reaf.simulation.torch.Device` in the :doc:`API reference <../api>`.
