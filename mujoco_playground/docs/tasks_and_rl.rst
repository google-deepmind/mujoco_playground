RL
==

Every RL environment in MuJoCo Playground is a function returning a
`flat_scene.Environment` config paired with an RSL-RL runner config in the
environment registry.

Registered environments
-----------------------

The registry includes DM Control Cartpole (`CartpoleBalance`,
`CartpoleSwingUp`, `CartpoleBalanceVision`), manipulation pick-and-lift
(`SO101PickCube`, `SO101PickCubeVision`, `TrossenPickCube`,
`TrossenPickCubeVision`), and humanoid tracking (`G1WholeBodyTracking`).

How tasks are structured
------------------------

Multi-robot tasks factor robot-agnostic MDP terms (`mdp.py`, `reset.py`,
`task.py`) from per-robot scene and device bindings (`so101/task.py`,
`trossen_wxai/task.py`).

Training and evaluation
-----------------------

`mujoco_playground.rl.rsl_rl.train` trains PPO policies across batched GPU
worlds via `RslRlVecEnvWrapper`, and `mujoco_playground.rl.rsl_rl.play`
evaluates checkpoints in the interactive viewer.

Benchmarking and profiling
--------------------------

`profiling/benchmark.py` runs standardized single-GPU throughput benchmarks and
integrates Tracy profiling zones across physics substeps, MDP evaluation, and
PPO updates.
