Robots
======

Each robot in ``mujoco_playground/robots/<robot>/`` includes a simulation model (``sim/``) and, for supported hardware,
drivers and setup utilities (``real/``) that share the same joint ordering and control interface.

Supported Robots
----------------

.. list-table::
    :header-rows: 1
    :widths: 20 15 20 45

    * - Robot
      - Modes
      - Hardware Extra
      - Description
    * - **SO-101** (``robots/so101``)
      - ``sim``, ``real``
      - ``--extra so101``
      - 5-DoF + gripper follower and leader arms with wrist camera.
    * - **I2RT YAM Pro** (``robots/yam``)
      - ``sim``, ``real``
      - ``--extra yam``
      - 6-DoF + gripper arm with wrist camera.
    * - **Trossen WXAI** (``robots/trossen_wxai``)
      - ``sim``
      - —
      - 6-DoF + gripper tabletop manipulation arm.
    * - **Unitree G1** (``robots/unitree_g1``)
      - ``sim``
      - —
      - 29-DoF humanoid robot for whole-body motion tracking.

SO-101 Setup
------------

All generated hardware configs are machine-local and stored under ``.local/``.

1. Install & Identify Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sync the ``so101`` extra, then identify the follower/leader USB ports (unplugging one arm at a time when prompted) and
the wrist camera:

.. code-block:: bash

    uv sync --extra cpu --extra so101 --group dev
    uv run --no-sync --extra so101 python -m mujoco_playground.robots.so101.real.get_ports --identify
    uv run --no-sync --extra so101 python -m mujoco_playground.robots.so101.real.get_cameras --identify

2. Calibrate Follower & Leader Arms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibrate one arm at a time (writes to ``.local/so101_calibration.json``):

.. code-block:: bash

    uv run --no-sync --extra so101 python -m mujoco_playground.robots.so101.real.get_calibration --roles=follower
    uv run --no-sync --extra so101 python -m mujoco_playground.robots.so101.real.get_calibration --roles=leader

1. Hold the arm in the **middle pose** (pan centered, upper arm vertical, elbow bent 90° with forearm level, wrist
   straight, gripper vertical with camera on the left) and press ``ENTER``.
2. Sweep every joint except ``wrist_roll`` through its full range in both directions, then press ``ENTER``.

.. note::

    If motors are missing from the bus or a joint is seized, check for **swapped power supplies**—the leader and
    follower use different-voltage bricks with identical barrel plugs.

I2RT YAM Pro Setup
------------------

Install the ``yam`` extra, bring up the ``can0`` interface, and identify the wrist camera:

.. code-block:: bash

    uv sync --extra cu124 --extra yam --group dev
    sudo ip link set can0 up type can bitrate 1000000
    uv run --no-sync --extra yam python -m mujoco_playground.robots.yam.real.get_cameras --identify

.. note::

    On Linux, select the RealSense D405 **ISP RGB stream** (``...-video-index4``, ``640x480`` ``4:3``) rather than the
    raw stereo sensor (``...-video-index2``).
