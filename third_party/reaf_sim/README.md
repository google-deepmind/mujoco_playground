# `reaf-sim`

[![CI](https://github.com/google-deepmind/reaf-sim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/google-deepmind/reaf-sim/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/reaf-sim)](https://pypi.org/project/reaf-sim/)
[![Documentation Status](https://img.shields.io/readthedocs/reaf-sim/latest)](https://reaf-sim.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`reaf-sim` is a simulation environment framework written in [PyTorch](https://pytorch.org/) and built on [MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) with the [Robotics Environment Authoring Framework (REAF)](https://github.com/google-deepmind/reaf).

`reaf-sim` is designed explicitly for sim-to-real robotics workflows. To achieve this, we decouple hardware and sensor I/O from task logic. The hardware layer is called the [`DeviceLayer`](https://reaf-sim.readthedocs.io/en/latest/user_guide/device_layer.html), which wraps sim in an interface that is consistent with physical hardware. The task logic is implemented in the [`TaskLayer`](https://reaf-sim.readthedocs.io/en/latest/user_guide/task_layer.html), which handles MDP processing of actions, observations, terminations, and rewards. Critically, the `TaskLayer` has no access to privileged simulation state, which makes sim workflows portable to real hardware.

> [!NOTE]
> For a full suite of sim and real robotic examples, check out [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), which showcases `reaf-sim` across a variety of tasks!

This project adheres to the [GDM Robotics Environment interface](https://github.com/google-deepmind/gdm_robotics/blob/main/src/gdm_robotics/interfaces/environment.py).

## Cartpole Example

For a quick preview into `reaf-sim`, let's create an environment controlling a cartpole alongside a camera rendering RGB observations across 16 parallel environments on GPU (`impl="warp"`). Step the environments with control commands and capture visual rollouts:

```python
import mujoco
from reaf.simulation.torch import ActuatorDevice, CameraDevice, flat_scene
import torch

spec = mujoco.MjSpec.from_file("cartpole.xml")

# configure the environment
config = flat_scene.Environment(
    base_scene=spec,
    sim=flat_scene.Sim(nworld=16, xpu_device="cuda:0", impl="warp"),
    device_layer=flat_scene.DeviceLayer(
        devices={
            "slider": flat_scene.Component(
                ActuatorDevice, targets=["slider"]
            ),
            "camera": flat_scene.Component(
                CameraDevice,
                cam_name="fixed",
                cam_resolution=(160, 120),
                render_rate_hz=20.0,
                output_name_prefix="camera/",
            ),
        }
    ),
)

# build the environment
env = flat_scene.build_env(config)

# step the environment with batched actions and extract camera frames
timestep = env.reset()
for _ in range(100):
  commands = torch.rand((env.nworld, 1), device="cuda:0") * 2.0 - 1.0
  timestep = env.step({"slider/ctrl": commands})
  camera_frames = timestep.observation["camera/fixed_rgb"]  # (16, 120, 160, 3)
```

For the complete, runnable script that loads the Cartpole model and generates the tiled grid video below, see [`reaf/simulation/torch/examples/cartpole.py`](reaf/simulation/torch/examples/cartpole.py).

<p align="center">
  <img src="https://www.gstatic.com/reaf-sim/cartpole/0/cartpole_grid_rollout.gif" alt="16-environment parallel Cartpole rollout on GPU" width="100%">
</p>

## Documentation & Contributing

- **Documentation:** See the [reaf-sim documentation](https://reaf-sim.readthedocs.io/en/latest/) for the [user guide](https://reaf-sim.readthedocs.io/en/latest/user_guide/index.html) and [API reference](https://reaf-sim.readthedocs.io/en/latest/api.html).
- **Contributing:** See the [contributing guide](https://reaf-sim.readthedocs.io/en/latest/contributing.html) for contribution guidelines.
- **Changelog:** See the [changelog](https://reaf-sim.readthedocs.io/en/latest/changelog.html).

## Citation

If you use `reaf-sim` in your work, please cite it as follows (see the [GitHub contributors page](https://github.com/google-deepmind/reaf-sim/graphs/contributors?all=1) for the full list of contributors):

```bibtex
@software{reaf_sim_2026,
  author = {{reaf-sim Contributors}},
  title = {{reaf-sim: Robotics Environment Authoring Framework - Simulation}},
  url = {https://github.com/google-deepmind/reaf-sim},
  year = {2026},
}
```

## Acknowledgements

We are grateful to all members of the MuJoCo team for their support in developing `reaf-sim`. Additionally, we thank the authors of REAF, mjlab, and Isaac Lab for inspiring `reaf-sim`.

## License & Disclaimer

`reaf-sim` is released under the Apache 2.0 license. See [LICENSE](LICENSE) for details.

This is not an official Google product.
