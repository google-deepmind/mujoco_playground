Overview
========

MuJoCo Playground is a suite of robot environments, reinforcement learning tasks, and hardware workflows built on
`reaf-sim <https://github.com/google-deepmind/reaf-sim>`_.

Quickstart
----------

Clone the repository and sync dependencies with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground
    make sync   # GPU (CUDA 12.4) + RSL-RL (or `make sync-cpu` for CPU-only)
    make test   # Run tests

Train a PPO policy on GPU using the environment's pre-configured observations, rewards, and domain randomization, then
play back the checkpoint in the interactive viewer:

.. code-block:: bash

    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
      --env=SO101PickCube \
      --logdir=/tmp/rslrl_logs/so101_pick_cube

    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.play \
      --env=SO101PickCube \
      --resume_path=/tmp/rslrl_logs/so101_pick_cube

Highlights
----------

.. list-table::
    :widths: 50 50

    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/g1_lafan_dance/2/g1_lafan_dance.webp
            :alt: Whole body motion tracking
            :width: 100%

            :doc:`Whole-body motion tracking <rl_training>`
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_sim_teleop/2/so101_sim_teleop.webp
            :alt: SO-101 Simulation Teleoperation Demo
            :width: 100%

            :doc:`Sim teleoperation <teleop>`
    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_real_teleop/2/so101_real_teleop.webp
            :alt: SO-101 Real Teleoperation Demo
            :width: 100%

            :doc:`Real teleoperation <teleop>`
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_sysid/2/so101_sysid.webp
            :alt: SO-101 System Identification Rollout
            :width: 100%

            :doc:`System identification <sysid>`
    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_pick_cube_sim2real/3/so101_pick_cube_sim2real.webp
            :alt: Sim2Real and interactive viewer
            :width: 100%

            :doc:`Sim-to-real deployment <sim2real>`
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/yam_mug_real/2/yam_mug_real.webp
            :alt: YAM Agentic Tabletop Manipulation
            :width: 100%

            :doc:`Agentic manipulation <agentic>`
