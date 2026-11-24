Overview
========

MuJoCo Playground is a suite of robot environments, reinforcement learning tasks, and hardware workflows built on
`reaf-sim <https://github.com/google-deepmind/reaf-sim>`_.

Quickstart
----------

Run directly from PyPI with `uvx <https://docs.astral.sh/uv/>`_ (no clone required) to train a PPO policy and play back
the checkpoint in MuJoCo Studio:

.. code-block:: bash

    uvx --default-index https://pypi.org/simple --python 3.14 "playground[rl]>=1.1" train rsl_rl \
      --env=SO101PickCube \
      --logdir=/tmp/rslrl_logs/so101_pick_cube
      # --overrides='{"nworld": 64}'  # For CPU

    uvx --default-index https://pypi.org/simple --python 3.14 "playground[rl]>=1.1" play rsl_rl \
      --env=SO101PickCube \
      --resume_path=/tmp/rslrl_logs/so101_pick_cube

.. note::

    On Linux with CUDA 12 drivers, add
    ``--index https://download.pytorch.org/whl/cu129 -f https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0 --with "warp-lang==1.18.0+cu12"``
    before ``"playground[rl]>=1.1"``.

For local development from ``HEAD`` (or robots requiring git extras like YAM), clone the repository and sync:

.. code-block:: bash

    git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground

    # Install with one of:
    make sync  # GPU (CUDA 13)
    make sync-cu12  # GPU (CUDA 12 on Linux)
    make sync-cpu  # CPU only (e.g. macOS)

    make test  # Run tests

Highlights
----------

.. list-table::
    :widths: 50 50

    * - .. figure:: https://www.gstatic.com/reaf-sim/demos/g1_lafan_dance/0/g1_lafan_dance.webp
            :alt: Whole body motion tracking
            :width: 100%

            :doc:`Whole-body motion tracking <rl_training>`
      - .. figure:: https://www.gstatic.com/reaf-sim/demos/so101_sim_teleop/0/so101_sim_teleop.webp
            :alt: SO-101 Simulation Teleoperation Demo
            :width: 100%

            :doc:`Sim teleoperation <teleop>`
    * - .. figure:: https://www.gstatic.com/reaf-sim/demos/so101_real_teleop/0/so101_real_teleop.webp
            :alt: SO-101 Real Teleoperation Demo
            :width: 100%

            :doc:`Real teleoperation <teleop>`
      - .. figure:: https://www.gstatic.com/reaf-sim/demos/so101_sysid/0/so101_sysid.webp
            :alt: SO-101 System Identification Rollout
            :width: 100%

            :doc:`System identification <sysid>`
    * - .. figure:: https://www.gstatic.com/reaf-sim/demos/so101_pick_cube_sim2real/1/so101_pick_cube_sim2real.webp
            :alt: Sim2Real and interactive viewer
            :width: 100%

            :doc:`Sim-to-real deployment <sim2real>`
      - .. figure:: https://www.gstatic.com/reaf-sim/demos/yam_mug_real/0/yam_mug_real.webp
            :alt: YAM Agentic Tabletop Manipulation
            :width: 100%

            :doc:`Agentic manipulation <agentic>`
