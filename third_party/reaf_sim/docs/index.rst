reaf-sim
========

`reaf-sim` is a `PyTorch <https://pytorch.org/>`_ simulation framework built by the MuJoCo team on top of `MuJoCo
<https://github.com/google-deepmind/mujoco>`_ and `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_, and using the
`Robotics Environment Authoring Framework (REAF) <https://github.com/google-deepmind/reaf>`_.

`reaf-sim` enables creation of environments in simulation that can be zero-shot deployed onto hardware and vice versa.
Workflows include large-batch RL on GPU, sim evaluation, sim and hardware teleoperation, system identification, agentic
workflows, and sim2real deployment.

Quickstart
----------

Try it in your browser with the `Quickstart Colab
<https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/notebooks/quickstart.ipynb>`_, or
install from PyPI with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    uv pip install "reaf-sim>=1.0"

Or clone `reaf-sim` from GitHub for development:

.. code-block:: bash

    git clone https://github.com/google-deepmind/reaf-sim.git && cd reaf-sim

    # Then install with one of:
    make install  # For GPU, good for RL training

    make install-cu12  # For CUDA 12 on Linux

    make install-cpu  # Or for CPU only, (i.e. on macOS)

    make test  # Run all tests, verify everything works

Cartpole Example
~~~~~~~~~~~~~~~~

For a quick preview, let's create a `reaf-sim` environment controlling a cartpole system alongside a camera rendering
RGB observations across 16 parallel environments on GPU (``impl="warp"``). Step the environments with control commands
and capture visual rollouts:

.. code-block:: python

    import mujoco
    from reaf.simulation.torch import ActuatorDevice, CameraDevice, flat_scene
    from reaf.simulation.torch.examples.cartpole import CARTPOLE_XML
    import torch

    spec = mujoco.MjSpec.from_string(CARTPOLE_XML)

    # configure the environment
    config = flat_scene.Environment(
        base_scene=spec,
        sim=flat_scene.Sim(
            nworld=16,
            xpu_device="cuda:0",
            impl="warp",
            simulation_timestep=0.01,
            control_timestep=0.05,
        ),
        device_layer=flat_scene.DeviceLayer(
            devices={
                "slider": flat_scene.Component(
                    ActuatorDevice, targets=["slider"]
                ),
                "camera": flat_scene.Component(
                    CameraDevice,
                    cam_name="fixed",
                    cam_resolution=(160, 120),
                    render_rate_hz=20.0,
                    output_name_prefix="camera/",
                ),
            }
        ),
        observation_adapter={
            "camera/fixed_rgb": flat_scene.ObsGroup(
                terms={
                    "rgb": flat_scene.Obs("camera/fixed_rgb").configure(
                        output_shape=(120, 160, 3), dtype="uint8"
                    ),
                }
            ),
        },
    )

    # build the environment
    env = flat_scene.build_env(config)

    # step the environment with batched actions and extract camera frames
    timestep = env.reset()
    for _ in range(100):
      commands = torch.rand((env.nworld, 1), device="cuda:0") * 2.0 - 1.0
      timestep = env.step({"slider/ctrl": commands})
      camera_frames = timestep.observation["camera/fixed_rgb"]  # (16, 120, 160, 3)

.. note::

    For the complete, runnable script that loads the Cartpole model and generates the tiled grid video below, see
    `reaf/simulation/torch/examples/cartpole.py
    <https://github.com/google-deepmind/reaf-sim/blob/main/reaf/simulation/torch/examples/cartpole.py>`_.

.. image:: https://www.gstatic.com/reaf-sim/cartpole/0/cartpole_grid_rollout.gif
    :align: center
    :alt: 16-environment parallel Cartpole rollout on GPU
    :width: 100%

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: reaf-sim

    User Guide <user_guide/index>
    API Reference <api>
    Contributing <contributing>
    Changelog <changelog>
    Versioning <versioning>

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: MuJoCo Playground

    Overview <playground/index>
    Robots <playground/robots>
    Teleoperation <playground/teleop>
    System Identification <playground/sysid>
    RL Training <playground/rl_training>
    Sim-to-Real <playground/sim2real>
    Agentic <playground/agentic>
    Contributing <playground/contributing>
    Changelog <playground/changelog>

.. toctree::
    :hidden:
    :caption: Project Links

    reaf-sim <https://github.com/google-deepmind/reaf-sim>
    MuJoCo Playground <https://github.com/google-deepmind/mujoco_playground>
    REAF <https://github.com/google-deepmind/reaf>
    MuJoCo <https://github.com/google-deepmind/mujoco>
    MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>
