Agentic
=======

``tasks/agentic/tabletop_v0`` lets a vision-language model (VLM) control a tabletop arm robot in simulation
(``--mode=sim``) or on hardware (``--mode=real``).

Each turn, a local HTTP bridge serves the robot's state (``obs.json``) and an annotated wrist camera image
(``wrist_cam.png``), and the VLM replies with a Python plan (``POST /plan``) using high-level IK primitives
(``look_at``, ``grasp``, ``place``, ``goto``, ``move_by``, ``move_until_contact``, and ``sweep``).

.. warning::

    **Use at your own risk.** The harness executes VLM-generated Python plans (``exec``) that drive the robot directly.

.. list-table::
    :widths: 50 50

    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/yam_mug_real/2/yam_mug_real.webp
            :alt: YAM Agentic Hardware Manipulation
            :width: 100%

            YAM agentic manipulation (Real)
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/yam_pantry_spice/2/yam_pantry_spice.webp
            :alt: YAM Agentic Pantry Spices
            :width: 100%

            YAM agentic pantry spices (Sim)
    * - .. figure:: https://www.gstatic.com/mujoco/playground/demos/yam_chess/2/yam_chess.webp
            :alt: YAM Agentic Chess Manipulation
            :width: 100%

            YAM agentic chess (Sim)
      - .. figure:: https://www.gstatic.com/mujoco/playground/demos/yam_rock_cairn/2/yam_rock_cairn.webp
            :alt: YAM Agentic Rock Balancing
            :width: 100%

            YAM agentic rock balancing (Sim)

Running Tabletop V0
-------------------

Start the HTTP bridge in simulation or on hardware (use ``--extra so101 --robot=so101`` for SO-101), then prompt your
VLM agent to follow ``mujoco_playground/tasks/agentic/tabletop_v0/AGENT_PROMPT.md`` to complete a task:

.. code-block:: bash

    # Start VLM bridge in simulation
    uv run --no-sync --extra yam python -m mujoco_playground.tasks.agentic.tabletop_v0.run \
      --mode=sim --robot=yam --agent=bridge --viewer

    # Start VLM bridge on hardware
    uv run --no-sync --extra yam python -m mujoco_playground.tasks.agentic.tabletop_v0.run \
      --mode=real --robot=yam --agent=bridge
