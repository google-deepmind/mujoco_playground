Robotic Workflows
=================

Every workflow in MuJoCo Playground runs on the same `reaf-sim` `Environment` by
varying the device layer, the agent, and the batch size.

RL training at scale
--------------------

Thousands of worlds step in lockstep on GPU (`impl="warp"`) with CUDA graph
capture, per-world masked resets, and domain randomization.

Evaluation and visualization
----------------------------

`reaf.simulation.torch.viewer.launch` and `rl/rsl_rl/play.py` run any
environment in an interactive MuJoCo viewer with world ghosts and camera
overlays.

Teleoperation and demonstration collection
------------------------------------------

A leader arm drives a simulated or real follower arm through the same device
protocol while an episodic logger records trajectories to MCAP (`tasks/teleop/`).

System identification
---------------------

`tasks/sysid/` feeds logged teleop MCAP trajectories into `mujoco.sysid` to fit
joint armature, frictionloss, stiffness, and damping for the simulated robot
model.

Sim-to-real deployment
----------------------

`flat_scene.make_deployment_config` derives a single-world deployment config
that runs unchanged in CPU simulation pre-flight (`--mode=sim`) and on the
physical robot (`--mode=real`).

Vision policies
---------------

`CameraDevice` (batched GPU rendering on Warp) and `MujocoCameraDevice`
(single-world rendering via `mujoco.Renderer`) emit identical image measurement
keys so pixel policies train on GPU and deploy on CPU or hardware.
