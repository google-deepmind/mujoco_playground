Robots
======

Each robot in `mujoco_playground/robots/<robot>/` separates `sim/` (MJCF scene,
identified constants, and simulated devices) from `real/` (shared joint/key
constants and hardware bus/camera devices).

Supported robots
----------------

Playground includes SO-101 (`sim` and `real`), I2RT YAM Pro (`sim` and `real`),
Trossen WXAI (`sim`), and Unitree G1 (`sim`).

Hardware setup utilities
------------------------

Robots with a `real/` package provide CLI scripts (`get_cameras`, plus
`get_ports` and `get_calibration` in `so101/real/`) to discover ports,
calibrate joints, and record machine-local hardware configs under `.local/`.
