Sim-to-Real
===========

.. TODO(btaba): Describe how reaf-sim supports sim2real natively.

``tasks/manipulation/pick_cube/so101/deploy`` deploys the vision-based ``SO101PickCubeVision`` policy in simulation
(``--mode=sim``) and on the physical SO-101 follower arm (``--mode=real``).

.. list-table::
    :widths: 50 50

    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_pick_cube_sim2real/3/so101_pick_cube_sim2real.webp
            :alt: SO-101 Pick-Cube Sim-to-Real Deployment
            :width: 100%

            Sim-to-real deployment & interactive viewer
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_pick_cube_sim2real2/0/so101_pick_cube_sim2real2.webp
            :alt: SO-101 Pick-Cube Sim-to-Real Deployment on a separately verified robot
            :width: 100%

            Vision policy deployment on hardware

Workflow
--------

Complete :doc:`robots` first before deploying on hardware, then sync the ``so101`` and ``rl`` extras:

.. code-block:: bash

    uv sync --extra so101 --extra rl --group dev

    # 1. Train vision policy on GPU
    uv run --no-sync --extra rl python -m mujoco_playground.rl.rsl_rl.train \
      --env=SO101PickCubeVision \
      --logdir=/tmp/rslrl_logs/so101_pick_cube_vision

    # 2. Test in simulation
    uv run --no-sync --extra so101 --extra rl python -m \
      mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
      --mode=sim \
      --impl=cpp \
      --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision

    # 3. Deploy on hardware
    uv run --no-sync --extra so101 --extra rl python -m \
      mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
      --mode=real \
      --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision

    # 4. Replay logged deployment episodes in Rerun
    uv run --no-sync --extra real python -m mujoco_playground.real.view_log

Useful Flags
------------

- ``--homing_duration_secs=2.0``: Duration to cosine-interpolate the arm to the initial hover pose at episode start.
- ``--gfx=web``: Opens the Studio viewer in the browser (pass ``""`` for native window).
- ``--episode_length_secs=20.0``: Episode duration before automatic reset.
