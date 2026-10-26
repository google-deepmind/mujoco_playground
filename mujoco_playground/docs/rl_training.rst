RL Training
===========

MuJoCo Playground provides reinforcement learning environments for manipulation, control, and humanoid motion tracking.
Policies train in parallel GPU simulation and can then be deployed onto physical hardware using the
:doc:`Robots <robots>`, :doc:`Teleoperation <teleop>`, :doc:`System Identification <sysid>`, and
:doc:`Sim-to-Real <sim2real>` workflows.

Example Environments
--------------------

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Environment ID
      - Observations
      - Description
    * - ``SO101PickCube``
      - State
      - SO-101 cube pick-and-lift manipulation task.
    * - ``SO101PickCubeVision``
      - Vision + State
      - SO-101 pick-and-lift with wrist camera observations and domain randomization for sim-to-real transfer.
    * - ``G1WholeBodyTracking``
      - State
      - Unitree G1 humanoid whole-body motion tracking (BeyondMimic-style).

Training and Evaluation
-----------------------

Train a PPO policy across batched GPU worlds with ``mujoco_playground.rl.rsl_rl.train`` and evaluate checkpoints in the
interactive viewer with ``mujoco_playground.rl.rsl_rl.play`` (while your policy trains, you can proceed to
:doc:`Robots <robots>` to connect and calibrate your hardware):

.. code-block:: bash

    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
      --env=SO101PickCube \
      --logdir=/tmp/rslrl_logs/so101_pick_cube

    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.play \
      --env=SO101PickCube \
      --resume_path=/tmp/rslrl_logs/so101_pick_cube

Pass JSON dicts to ``--overrides`` (runner settings) or ``--env_overrides`` (environment config kwargs), or use
``torchrun`` for multi-GPU training:

.. code-block:: bash

    # Customize runner settings
    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
      --env=SO101PickCube \
      --overrides='{"max_iterations": 2000, "nworld": 2048}'

    # Multi-GPU training
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      -m mujoco_playground.rl.rsl_rl.train --env=G1WholeBodyTracking --logdir=/tmp/g1_tracking

Monitoring with TensorBoard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training writes TensorBoard event files to ``--logdir``. Launch TensorBoard and open ``http://localhost:6006`` in your
browser to monitor training rewards, episode length, policy entropy, and losses:

.. code-block:: bash

    uv run --no-sync tensorboard --logdir=/tmp/rslrl_logs/so101_pick_cube

Unitree G1 Whole-Body Tracking
------------------------------

.. image:: https://www.gstatic.com/mujoco/playground/demos/g1_lafan_dance/2/g1_lafan_dance.webp
    :align: center
    :alt: Unitree G1 Whole-Body Motion Tracking
    :width: 80%

|

Convert a retargeted CSV clip (e.g. from the `LAFAN1 Retargeting Dataset for G1
<https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1>`_) to ``.npz``, verify it in the
viewer, and train a tracking policy:

.. code-block:: bash

    # 1. Convert CSV trajectory to .npz and preview in viewer
    uv run --no-sync python -m mujoco_playground.tasks.whole_body.tracking.csv_to_npz \
      --input_csv=/path/to/input.csv --output_npz=/path/to/output.npz
    uv run --no-sync python -m mujoco_playground.tasks.whole_body.tracking.replay_npz \
      --trajectory_path=/path/to/output.npz

    # 2. Train tracking policy
    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
      --env=G1WholeBodyTracking \
      --env_overrides='{"trajectory_path": "/path/to/output.npz"}' \
      --logdir=/tmp/g1_tracking

    # 3. Play checkpoint with reference ghost enabled
    uv run --no-sync python -m mujoco_playground.rl.rsl_rl.play \
      --env=G1WholeBodyTracking \
      --env_overrides='{"trajectory_path": "/path/to/output.npz", "play": true, "enable_ghost": true}' \
      --resume_path=/tmp/g1_tracking \
      --nworld=1
