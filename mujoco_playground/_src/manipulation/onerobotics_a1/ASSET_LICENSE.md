# OneRobotics A1 robot asset attribution

The model `xmls/a1.xml` and eight STL meshes in `xmls/assets/` are
copyright © 2026 OneRobotics, under Creative Commons Attribution 4.0
International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/.
The full license text is retained as `LICENSE-CC-BY-4.0.txt`.

Source: https://github.com/katazen/onerobot_h1
Source asset revision used by the mjlab distribution:
`ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70`.

Model imported from https://github.com/T1Amoo/onerobotics-a1-mjlab,
`src/onerobotics_a1_mjlab/a1/xmls/a1.xml`. That project documents
its provenance and modifications in `ASSET_LICENSE.md`.

Changes made for MuJoCo Playground and MJX compatibility:

- A non-colliding, visual-only mocap sphere named `mocap_target` shows
  the reaching goal.
- The source `mocap_base` wrapper is replaced by a regular fixed body,
  so the robot base is fixed to the world by construction.
- The seven high-resolution STL collision geoms are replaced by capsule
  approximations (links 1--6) and a box approximation (link 7), fitted
  to the source mesh bounds. The source STL meshes remain unchanged and
  are still used for rendering. This avoids degenerate/coplanar mesh
  faces in MJX collision processing while retaining link collision
  envelopes. The primitives are fitted in the original STL/body coordinates:
  links 1, 3 and 5 run along Z; links 2, 4 and 6 run along Y. Capsule radii
  enclose the radial vertex bounds with 0.5 mm padding; segment endpoints
  enclose all vertices including the hemispherical ends. Link 7 is a box
  with its thin dimension along Z. Rounded coordinates allow sub-micrometre
  fitting tolerance.
- The original base meshes are retained, including the mesh used to infer
  the fixed base inertia. A separate collision box is added,
  enclosing the base STL with 0.5 mm padding and zero density. The base/Link1
  pair is excluded because the revolute joint sits inside the base housing;
  all other contact masks and the default parent-child filtering are retained.

These fits correct earlier primitives that used centered mesh bounds as body
coordinates, leaving substantial portions of the visible links unprotected.
They remain conservative envelopes, not exact mesh collision surfaces.
`tools/check_a1_collision.py` checks vertex containment, task-range clearance,
and actual contact generation using an independent sphere probe.

The seven arm joints, inertial parameters, actuator gains, mesh bytes,
joint ranges, body transforms, and original `home` keyframe are unchanged.

New Python task code is independently authored under Apache-2.0 (see
file headers and the upstream project's root LICENSE). The CC BY 4.0
asset licensing is not replaced by the code license.
