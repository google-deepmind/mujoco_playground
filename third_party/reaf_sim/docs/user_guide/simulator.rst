Simulator
=========

:class:`~reaf.simulation.torch.Simulator` manages the physics engine and exposes a unified PyTorch tensor interface on
CPU and GPU.

Physics backends
----------------

`reaf-sim` supports two backends via :class:`~reaf.simulation.torch.Impl`:

- ``Impl.CPP``: `MuJoCo <https://github.com/google-deepmind/mujoco>`_ (CPU, ``nworld=1``) for evaluation and debugging.
- ``Impl.WARP``: `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ (GPU) for batched simulation and RL
  training.

Usage
-----

Initialized with an `MjSpec <https://mujoco.readthedocs.io/en/stable/python.html#model-editing>`_,
:class:`~reaf.simulation.torch.Simulator` compiles the model, allocates buffers, and can be stepped standalone without
an :class:`~reaf.simulation.torch.Environment`:

.. code-block:: python

    import mujoco
    from reaf.simulation.torch import Impl, Simulator

    spec = mujoco.MjSpec.from_string(...)
    sim = Simulator(mj_spec=spec, nworld=16, xpu_device="cuda:0", impl=Impl.WARP)

    sim.reset()      # Reset state (mj_resetData / mjw.reset_data)
    sim.step()       # Advance 1 physics substep
    sim.step_n(5)    # Advance 5 physics substeps
    sim.forward()    # Recompute kinematics and dynamics without advancing time

State and model access
----------------------

``sim.model`` and ``sim.data`` are wrapped in :class:`~reaf.simulation.torch.TorchWrapper`, providing zero-copy PyTorch
tensor views of MuJoCo buffers (sharing memory with ``wp.array`` on GPU and NumPy arrays on CPU) with a leading
``nworld`` dimension:

.. code-block:: python

    qpos = sim.data.qpos  # (nworld, nq)
    sim.data.ctrl[:] = torch.randn_like(sim.data.ctrl)
    sim.data.qvel[env_mask] = 0.0

Named ``MjSpec`` elements (bodies, joints, geoms, actuators, sensors, sites) can be accessed via ``bind`` or
``bind_pattern``:

.. code-block:: python

    joint_data = sim.data.bind(sim.model, spec.joint("slider"))
    joint_data.qpos  # (nworld,)
    joint_data.set("qpos", torch.zeros(sim.nworld, device=sim.xpu_device))

    geom_model = sim.model.bind(spec.geom("pole_1"))
    geom_model.geom_size  # (nworld, 3)

Modifying model parameters
--------------------------

When model fields (e.g., ``body_mass``, ``geom_size``, ``geom_friction``) are modified in-place through ``sim.model``,
:class:`~reaf.simulation.torch.TorchWrapper` automatically:

- Lazily clones initial field values on first write so ``sim.model.restore_defaults(env_mask)`` can restore baseline
  values on reset without compounding changes across episodes.
- Tracks the minimal required :class:`~reaf.simulation.torch.SetConstLevel`
  (``NONE``, ``FIXED``, ``KIN0``, ``INERT0``, ``FULL``) flushed by ``sim.set_const()`` to recompute derived constants.

.. code-block:: python

    sim.model.geom_size[env_mask] = new_sizes
    sim.set_const()  # Called automatically during Environment startup and resets

Write tracking relies on :class:`~reaf.simulation.torch.TorchTensor` ``__setitem__``:

- :badge-green:`USE` :green:`Tracked writes:` ``sim.model.<field>[idx] = val`` or ``bound.set("<field>", val)``.
- :badge-red:`AVOID` :red:`Untracked writes:` Augmented assignments (``sim.model.geom_size[idx] += delta``), in-place
  Tensor methods (``.copy_()``, ``.add_()``, ``out=...``), or direct writes to ``_tensor``, ``warp_model``, or
  ``mj_model``.

CUDA graphs
-----------

On CUDA devices (``Impl.WARP``), :class:`~reaf.simulation.torch.Simulator` captures ``step()``, ``forward()``, and
``reset()`` as CUDA graphs via ``wp.ScopedCapture`` and replays them with ``wp.capture_launch``. Tensor attributes on
``sim.model`` and ``sim.data`` are read-only so device pointers remain fixed across graph replays.

For full signature details, see :class:`reaf.simulation.torch.Simulator` in the :doc:`API reference <../api>`.
