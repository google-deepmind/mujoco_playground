# MuJoCo Playground

MuJoCo Playground contains a suite of classic control and robotic environments built on top of the MuJoCo physics engine. Environments contain recipes for training classic control, locomotion, and manipulation behaviors via RL.

## Installation

> [!IMPORTANT]
> Requires Python 3.9 or later.

1. `pip install -U "jax[cuda12]"`
    * `python -c "import jax; print(jax.default_backend())"` should print `gpu`.
2. Clone this repository.
3. `pip install -e ".[all]"`
