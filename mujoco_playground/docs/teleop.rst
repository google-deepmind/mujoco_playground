Teleoperation
=============

``tasks/teleop/so101`` collects demonstrations by driving an SO-101 follower arm from a physical SO-101 leader arm in
simulation or on hardware (complete :doc:`robots` first to identify and calibrate the arms).

.. list-table::
    :widths: 50 50

    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_sim_teleop/2/so101_sim_teleop.webp
            :alt: SO-101 Simulation Teleoperation Demo
            :width: 100%

            Sim Teleop (``--sim``)
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/so101_real_teleop/2/so101_real_teleop.webp
            :alt: SO-101 Real Teleoperation Demo
            :width: 100%

            Real Teleop

Running Teleoperation
---------------------

.. code-block:: bash

    # Simulation (on macOS, replace `python` with `mjpython`)
    uv run --no-sync --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop --sim

    # Hardware
    uv run --no-sync --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop

Episodes are gated by a three-button foot pedal or terminal keys:

- ``a`` **(left)**: Start the next episode; during an episode, end it as **failure**.
- ``s`` **(middle)**: End the episode as **invalid** (discard).
- ``d`` **(right)**: End the episode as **success**.

Episode Logs
------------

Episodes are logged as MCAP files under ``.local/so101_teleop_logs/YYYY/MM/DD/`` and can be replayed in
`Rerun <https://rerun.io>`_:

.. code-block:: bash

    uv run --no-sync --extra real python -m mujoco_playground.real.view_log
    uv run --no-sync --extra real python -m mujoco_playground.real.view_log --file=/path/to/episode.mcap
