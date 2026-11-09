Task layer
==========

:class:`~reaf.simulation.torch.TaskLayer` defines a Markov Decision Process (MDP): processing actions into device
commands, extracting task features from device measurements, and computing rewards, terminations, metrics, and
observations.

Importantly, :class:`~reaf.simulation.torch.TaskLayer` has **no direct access to privileged simulation information** and
operates strictly on information provided via the layer interface. This information boundary enables unmodified
:class:`~reaf.simulation.torch.TaskLayer` code to directly interface with different simulation backends or physical
robot hardware.

.. figure:: ../_static/images/task_layer_architecture_light.svg
    :class: only-light
    :alt: Task Layer Architecture
    :align: center
    :width: 100%

    **Figure 1:** Information flow in :class:`~reaf.simulation.torch.TaskLayer`. Actions map to device commands; device
    measurements expand into features, which construct rewards, terminations, and observations.

.. figure:: ../_static/images/task_layer_architecture_dark.svg
    :class: only-dark
    :alt: Task Layer Architecture
    :align: center
    :width: 100%

    **Figure 1:** Information flow in :class:`~reaf.simulation.torch.TaskLayer`. Actions map to device commands; device
    measurements expand into features, which construct rewards, terminations, and observations.

The `REAF documentation <https://reaf.readthedocs.io/en/latest/core_concepts/#task-layer>`_ provides additional details
about the task layer design.

Commands
--------

Before policy actions reach devices, ``CommandsProcessor`` components (such as
:class:`~reaf.simulation.torch.DeltaActions` and :class:`~reaf.simulation.torch.TransformActions`) can intercept,
transform, scale, or buffer them.

Features
--------

:class:`~reaf.simulation.torch.FeatureProducer` components compute derived physical quantities from raw device
measurements. These are collected into a consolidated features bus, ensuring geometric calculations (such as
end-effector frames, relative distances, or filtered velocities) are performed exactly once per step.

Lifecycle
---------

Feature producers and observation adapters follow one contract: **hooks advance state, reads are pure, and resets only
touch the reset environments.**

- ``on_step(features)`` runs once per ``Environment.step()``, after the physics step and right before the read. Advance
  per-step state here (e.g. previous velocities, history buffers, per-step noise).
- ``on_begin_stepping(features, env_mask)`` runs after a reset, for the ``env_mask`` environments only. Seed post-reset
  state here (e.g. fill history buffers with the reset value).
- ``produce_features(features)`` and ``observations_from_features(features)`` are pure reads. After a partial reset,
  the environment calls them again for all environments, and the recomputed features of every environment are the ones
  observers and loggers record. Since the reads are pure, these are the same values for the environments that were not
  reset.

Components that wrap another producer or adapter must forward both hooks to it.

Commands processors need no hooks: the task layer runs them exactly once per ``Environment.step()``, and never again
after a reset, so they can advance their state in ``process_commands``.

Features observers and loggers record once per timestep returned by ``step()``. After an auto-reset, reset environments
are recorded with their reset (``FIRST``) features.

Rewards
-------

Rewards are configured declaratively using :class:`~reaf.simulation.torch.flat_scene.Reward`.

By default, :class:`~reaf.simulation.torch.FunctionalRewardProvider` computes the scalar reward as the weighted sum of
terms scaled by the control timestep:

.. code-block:: text

    r_t = control_timestep * sum(weight_i * term_func_i(features))

Metrics
-------

Diagnostic metrics are configured declaratively using :class:`~reaf.simulation.torch.flat_scene.Metric` with an explicit
per-episode reduction mode (``reduce="mean" | "last" | "max" | "sum"``):

.. code-block:: python

    metrics = {
        "tracking_error": flat_scene.Metric(pose_error, reduce="mean"),
        "peak_height": flat_scene.Metric(object_height, reduce="max"),
        "contact_events": flat_scene.Metric(new_contact_event, reduce="sum"),
        "success": flat_scene.Metric(goal_reached, threshold=0.02, reduce="last"),
    }

- **Per-episode metrics**: ``env.task_layer.get_episode_metrics(env_mask)`` returns ``dict[str, torch.Tensor]`` of 0-D
  scalar means over the masked worlds without host-device synchronization.

Terminations and discounts
--------------------------

Episode endings distinguish between task failure/completion (termination) and time limits (truncation).

- **Task terminations**: Evaluated by :class:`~reaf.simulation.torch.FunctionalTerminationChecker` based on criteria
  from features.
- **Timeouts**: Handled automatically by :class:`~reaf.simulation.torch.TimeoutTerminationChecker` based on
  ``episode_length_secs`` in the environment configuration.
- **Discounts**: Computed by :class:`~reaf.simulation.torch.DefaultDiscountProvider`, emitting ``gamma = 0.0`` on
  termination (task failure/success) and ``gamma = 1.0`` on truncation (timeout) or continuation.

Observation adapters
--------------------

:class:`~reaf.simulation.torch.FunctionalObservationAdapter` organizes features into named
:class:`~reaf.simulation.torch.flat_scene.ObsGroup` dictionaries (e.g., ``"policy"``, ``"critic"``,
``"proprioception"``). Each :class:`~reaf.simulation.torch.flat_scene.Obs` term supports declarative data
transformations evaluated in order: ``func -> bias -> noise -> scale -> offset -> clip -> history``. When
``history_length > 0``, past and current frames are stacked oldest-to-newest (``[t - H, ..., t - 1, t]``). Per-step
noise is drawn and the history advances in ``on_step``; after a reset, every history slot holds the reset observation.

Task layer assembly
-------------------

Task components assemble inside :class:`~reaf.simulation.torch.flat_scene.Environment`:

.. code-block:: python

    from reaf.simulation.torch import flat_scene

    config = flat_scene.Environment(
        base_scene=spec,
        device_layer=device_layer_config,
        task_layer=flat_scene.TaskLayer(
            command_processors=command_processors,
            feature_producers=feature_producers,
            command_feature_producers=command_feature_producers,
            rewards=rewards_dict,
            metrics=metrics_dict,
            terminations=terminations_dict,
        ),
        observation_adapter=observation_adapter,
        episode_length_secs=10.0,
    )

    env = flat_scene.build_env(config)

For full class signatures and details, see :class:`reaf.simulation.torch.TaskLayer` in the :doc:`API reference
<../api>`.
