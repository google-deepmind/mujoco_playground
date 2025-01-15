# MuJoCo Playground

<h1>
  <a href="#"><img alt="MuJoCo Playground" src="assets/banner.png" width="100%"></a>
</h1>

A comprehensive suite of environments for robot learning research, accelerated by [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

Features include:

- Classic control environments from `dm_control` reimplemented in MJX.
- Quadruped and bipedal locomotion environments.
- Non-prehensile and dexterous manipulation environments.
- Vision-based support via Madrona.

## Installation

> [!IMPORTANT]
> Requires Python 3.9 or later.

1. `pip install -U "jax[cuda12]"`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print `gpu`
2. Clone this repository
3. Install submodules: `git submodule update --init --recursive`
4. `pip install -e ".[all]"`

## Common Gotchas

- Version mismatch between mjx and mujoco can cause issues. If encountered:
    ```bash
    pip uninstall mujoco mujoco-mjx
    pip install --upgrade mujoco mujoco-mjx
    ```

## Playground environments

### Locomotion Suite

| Environment | Visualization
|------------|---------------|
| `Env1`     | [hopper.gif]  |
| `Env2`     | [walker.gif]  |
| `Env3`     | [humanoid.gif]|

### Manipulation Suite

| Environment | Description | Simulation | Real Robot |
|------------|-------------|------------|------------|
| `ReachEnv` | Reaching task with robot arm | [reach.gif] | [real_reach.gif] |
| `PushEnv` | Object pushing with robot arm | [push.gif] | [real_push.gif] |
| `PickPlaceEnv` | Pick and place objects | [pick.gif] | N/A |

### DM Control Suite

| Environment | Description | Simulation |
|------------|-------------|------------|
| `Cartpole` | Classic cartpole balancing | [cartpole.gif] |
| `Pendulum` | Inverted pendulum control | [pendulum.gif] |
| `Cheetah` | 2D cheetah running | [cheetah.gif] |
| `Finger` | Finger spinning task | [finger.gif] |

## Frequently Asked Questions

* Q1
* Q2
* Q3

## How can I contribute?

Install the library and use it! Report bugs in the issue tracker. If you are a developer with some robotics experience looking to hack on open source, check out the [contribution guidelines](CONTRIBUTING).

## Citation

If you use Playground in your scientific works, please cite it as follows:

```bibtex
@misc{mujoco_playground2025,
  author = {[Your Name]},
  title = {MuJoCo Playground},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco-playground}
}
```
