Flat scene config
=================

:mod:`reaf.simulation.torch.flat_scene` provides a declarative dataclass schema and modular builders that pull together
all of the `reaf-sim` components. This is the primary mechanism to author `reaf-sim` environments.

Configuration schema
--------------------

Environments are defined using :class:`~reaf.simulation.torch.flat_scene.Environment`:

.. code-block:: python

    import mujoco
    from reaf.simulation.torch import ActuatorDevice, CameraDevice, flat_scene

    config = flat_scene.Environment(
        # Base physics scene
        base_scene=mujoco.MjSpec.from_string(...),

        # Simulator backend settings
        sim=flat_scene.Sim(
            nworld=1024,
            xpu_device="cuda:0",
            impl="warp",
            simulation_timestep=0.005,
            control_timestep=0.02,
        ),

        # Devices participating in the simulation
        device_layer=flat_scene.DeviceLayer(
            devices={
                "arm": flat_scene.Component(
                    ActuatorDevice,
                    targets=["joint1", "joint2"],
                    measurements=["joint_pos"],
                ),
                "camera": flat_scene.Component(
                    CameraDevice,
                    cam_name="wrist",
                    cam_resolution=(128, 128),
                    render_rate_hz=20.0,
                    output_name_prefix="camera/",
                ),
            }
        ),

        # Task terms (rewards, metrics, & terminations)
        task_layer=flat_scene.TaskLayer(
            rewards={"distance": flat_scene.Reward(reach_goal_reward, weight=1.0)},
            metrics={"success": flat_scene.Metric(reach_goal_metric, reduce="last")},
            terminations={"bounds": flat_scene.Term(workspace_bounds_checker)},
        ),

        # Observation groups for policy and value heads
        observation_adapter={
            "policy": flat_scene.ObsGroup(
                terms={
                    "joint_pos": flat_scene.Obs("arm/joint_pos").configure(output_shape=(2,)),
                    "wrist_rgb": flat_scene.Obs("camera/wrist_rgb").configure(output_shape=(128, 128, 3)),
                },
                enable_noise=True,
            )
        },

        episode_length_secs=10.0,
    )

    env = flat_scene.build_env(config)

Attaching sub-models with :class:`~reaf.simulation.torch.flat_scene.Component`
------------------------------------------------------------------------------

When a device represents an external model or robot embodiment (such as a robotic arm or gripper),
:class:`~reaf.simulation.torch.flat_scene.Component` supports procedural attachment to the base arena:

.. code-block:: python

    flat_scene.Component(
        ArticulatedDevice,
        measurements=["joint_pos", "joint_vel"],
    ).attach(
        load_robot_arm_spec,
        prefix="robot/",
        site="table_mount_site",
        pos=(0.0, 0.0, 0.0),
    )

At build time, :func:`~reaf.simulation.torch.flat_scene.build_scene` copies the base arena, attaches child specs to
target sites with proper namespace prefixes, and applies simulation options.

Building an Environment
-----------------------

:mod:`~reaf.simulation.torch.flat_scene` provides modular builders for assembling environments:

- :func:`~reaf.simulation.torch.flat_scene.build_env`: The main entry point. Compiles the scene, builds the layers, and
  returns a runnable :class:`~reaf.simulation.torch.Environment`.
- :func:`~reaf.simulation.torch.flat_scene.build_scene`: Convenience function to extract the scene after devices attach
  and configure their specs.
- :func:`~reaf.simulation.torch.flat_scene.make_deployment_config`: Generates a deployment configuration for hardware or
  single-world simulation:

  - Drops training-only rewards, metrics, and terminations.
  - Disables observation noise across observation groups.
  - Sets ``sim.nworld = 1``.

For full class signatures and details, see :doc:`API reference <../api>`.
