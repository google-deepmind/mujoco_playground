System Identification
=====================

System identification closes the sim-to-real gap by tuning the simulated robot's joint dynamics to match trajectories
recorded on hardware. MuJoCo Playground uses the `MuJoCo sysid toolbox
<https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sysid>`_ end-to-end to fit joint dynamics parameters
from logged trajectories and generate diagnostic comparison reports.

SO-101 Example
--------------

In ``tasks/sysid/so101``, ``mujoco.sysid`` fits ``armature``, ``frictionloss``, ``stiffness``, and ``damping`` across
all 6 SO-101 joints from teleoperation MCAP logs.

.. image:: https://www.gstatic.com/mujoco/playground/demos/so101_sysid/2/so101_sysid.webp
    :align: center
    :alt: SO-101 System Identification Rollout Comparison
    :width: 80%

|

.. code-block:: bash

    # 1. Record an excitation trajectory on hardware (press `a` to start, sweep all joints, press `d` to save)
    uv run --no-sync --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop \
      --instruction="sysid_excitation" \
      --num_episodes=1

    # 2. Fit joint dynamics parameters from `.local/so101_teleop_logs`
    uv run --no-sync --extra so101 python -m mujoco_playground.tasks.sysid.so101.run

    # 3. View the generated HTML report
    open .local/so101_sysid_results/report.html

Pre-identified dynamics constants for the SO-101 are checked into ``robots/so101/sim/constant.py``. To use newly fitted
parameters from your own hardware, copy the identified values from ``report.html`` into
``robots/so101/sim/constant.py``.

i2rt YAM Example
----------------

In ``tasks/sysid/yam``, ``collect`` verifies three scripted 50 Hz excitation trajectories in simulation (checking
contacts, desk clearance, and joint limits) before replaying and logging them on hardware through the I2RT SDK. ``run``
then fits ``armature``, ``frictionloss``, ``kp``, and ``kv`` with ``mujoco.sysid``, reports tracking RMSE on training
and held-out validation episodes, and prints a ``--freeze_params`` flag for any parameters the data does not identify.

.. image:: https://www.gstatic.com/mujoco/playground/demos/yam_sysid/0/yam_sysid.webp
    :align: center
    :alt: YAM System Identification Rollout Comparison
    :width: 80%

|

.. code-block:: bash

    # 1. Verify excitation trajectories in simulation and record them on hardware
    uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.collect

    # 2. Fit joint dynamics parameters from `.local/yam_sysid_logs`
    uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.run

    # 3. View the generated HTML report and `identified_params.json`
    open .local/yam_sysid_results/report.html

Pre-identified dynamics constants for the YAM are checked into ``robots/yam/sim/constant.py``. To use newly fitted
parameters from your own hardware, copy the identified values into ``robots/yam/sim/constant.py``.
