# OneRobotics A1 asset attribution

The A1 right-arm MJCF model (`xmls/a1.xml`) and eight STL mesh files
(`xmls/assets/`) are copyright © 2026 OneRobotics and licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The complete license text is included in `LICENSE-CC-BY-4.0.txt`.

Original assets: https://github.com/katazen/onerobot_h1

Source revision referenced by the asset license:
`ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70`

Adapted from the OneRobotics mjlab model:
https://github.com/T1Amoo/onerobotics-a1-mjlab

## Changes for MuJoCo Playground

- Replaced the mocap base wrapper with a fixed base body.
- Added a non-colliding mocap target marker for the Reach task.
- Replaced STL collision geometry with capsule and box approximations, while
  retaining the original STL meshes for visualization.
- Added a base collision box and excluded collision between the base and Link1.

The seven arm joints, their limits and inertial parameters, actuator settings,
link transforms, and home configuration are retained from the mjlab model.
The collision geometry is approximate and differs from the source model.

New Python integration code is licensed under Apache-2.0, separately from the
CC BY 4.0 robot assets.
