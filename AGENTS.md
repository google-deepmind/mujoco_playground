# Development Workflow

- Always use `uv run`, not bare `python`.
- Always use `absl` (`from absl import app, flags, logging`) for scripts, tools, and entrypoints. **NEVER use `argparse`**.
- Dependencies are managed with `uv`. Sync before doing anything else — the test targets pass `--no-sync` and will run against a stale environment otherwise.
- `reaf` builds from git (PyPI lags the API the sim stack uses) and its CMake does not pin the C++ >= 17 standard abseil requires, so any `uv` invocation that may build it needs `CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17"`.
- Run pyink and isort against **only the specific paths you changed**, not a parent directory:
  `uv run --no-sync pyink <path>` and `uv run --no-sync isort <path>`.
- For headless rendering / tests on Linux, set `MUJOCO_GL=osmesa`. Omit on macOS.
- Prefer running individual tests rather than the full test suite to improve iteration speed:
  `MUJOCO_GL=osmesa uv run --no-sync --group dev pytest <path/to/test.py> -v`.

## Quick Commands

```bash
make sync          # GPU (CUDA 12.4) — the default
make sync-cpu      # CPU-only
make format        # pyink + isort over the repo
make typecheck     # pyrefly type check
make test          # pytest over mujoco_playground/ and third_party/reaf_sim/
make docs          # build HTML docs (or `make serve-docs` to preview locally)
uv run --no-sync --group dev pre-commit run -a
```

## Commits and PRs

- PR body should be plain, concise prose. Describe the problem, what the change does, and any non-obvious tradeoffs. Bullet points listing changes are fine, but avoid section headers, structured templates, and emojis. A good PR description reads like a short paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 88 columns. Let each sentence flow on one line.
- Push branches to your own fork, not to the main repo directly.
- Do not add AI assistants (e.g. Claude, Antigravity) as commit co-authors. The CLA check requires all commit authors to have signed the CLA, and AI tools cannot sign it.
- Amending commits is fine before a PR has reviewers looking at it. Once a PR is under review, use new commits so reviewers can see what changed.
- When responding to PR review comments:
  - Reply to each comment individually confirming what you did (or why you didn't).
  - Resolve comment threads that are addressed.
  - Add a summary comment on the PR after responding, covering what was applied and what was intentionally skipped.

# Code Style

- **CLI / Entrypoints**: Always use `absl.flags` and `absl.app.run(main)`. Never use `argparse`.
- **Path Handling**: Prefer `etils` (`from etils import epath`) over `pathlib` for file and directory paths (note that recursive `**` globs are not supported by `epath.Path.glob()`, so use `glob.glob` when matching glob patterns).
- **Formatting**: `pyink` with **2-space indentation** and an 80-column limit; `isort` with `force_single_line = true`. Run pyink and isort against modified files.
- **Docstrings & Comments**: Preserve all existing docstrings, TODOs, and comments when editing.
- **Type Annotations**: Use modern union syntax (`T | None`), standard collections (`list[str]`, `dict[str, Any]`), and explicit return types.
- **Error Handling**: Fail gracefully with informative error messages.

# Excluded Directories

`third_party/` and `mujoco_playground/rss2025/` are excluded from formatting
(pyink), import sorting (isort), and pre-commit. Do not reformat them.

# Architecture

The environment stack is `reaf` and `reaf-sim` (`reaf.simulation.torch`,
vendored under `third_party/reaf_sim`).

### `_src/` is the implementation; the top level is the public API

Every public module is a thin re-export shim over `_src/`. In
`reaf.simulation.torch`, `reaf/simulation/torch/__init__.py` re-exports core,
common, builders, and viewer simulation components from
`reaf/simulation/torch/_src/{core,common,builders,viewer}/`.

Consequence: **code under `tasks/` and `robots/` should import from the public
paths** (`from reaf.simulation import torch as reaf_sim`,
`from reaf.simulation.torch import flat_scene`), while
code inside `_src/` imports its siblings directly.

### Layered environment (`reaf.simulation.torch`)

`environment.py` composes the layers, from hardware upward:

- `device_layer.py` — devices bound to the physics model (joints, actuators, cameras, sites, contact sensors)
- `task_layer.py` — features, rewards, terminations, discounts
- `torch_wrapper.py` — tensor/device handling

Supporting protocols (`feature_producer.py`, `reward_provider.py`, `termination_checker.py`, `discount_provider.py`) define what a task plugs in. `EnvironmentReset` in `environment.py` is the privileged-access escape hatch: `on_startup` / `on_reset` / `on_step` / `check_termination`, with direct model and data access, used for domain randomization and NaN/limit detection.

### Environments are built from config, not subclassed

`reaf.simulation.torch.flat_scene.build_env` takes a
`reaf.simulation.torch.flat_scene.Environment` dataclass and
assembles the layered `Environment`. A config declares `device_layer.devices`,
feature producers, rewards, terminations, and resetters as `flat_scene.Component`
entries (`Component(Cls, **kwargs).configure_spec(...).attach(...).mark_privileged()`), plus
`flat_scene.Obs`, `flat_scene.ObsGroup`, `flat_scene.Reward`, and `flat_scene.Term`.

So adding a task means **writing a function that returns a config**, not implementing an env class. See `tasks/manipulation/pick_cube/so101/task.py` for the shape: a shared `pick_cube/task.py` base config, mutated by a robot-specific `_configure_*_robot` function.

`reaf.simulation.torch.flat_scene.Sim` holds the simulation knobs —
`nworld` (batch size), `xpu_device`, `impl` (`"warp"` on CUDA, `"cpp"` on CPU),
and the two timesteps.

### `robots/` vs `tasks/`

- `robots/<name>/sim/scene.py` — loads the `MjSpec` for the robot
- `robots/<name>/sim/device.py` — joint names, home qpos/ctrl, stiffness, damping, force ranges, armature
- `tasks/<task>/` — the MDP (`mdp.py`), reset logic (`reset.py`), and config assembly (`task.py`)

A task/robot pair lives at `tasks/<task>/<robot>/task.py` and combines the two.

### Real hardware (`robots/<name>/real/`)

Mirrors `sim/` alongside it. **The real robot is the source of truth: `real/constant.py` defines the canonical joint ordering, home pose, and hardware constants, and `sim/device.py` re-exports them** — sim depends on real, not the other way around. Sim and real silently disagreeing on joint order corrupts everything downstream, so verify the correspondence for a new robot rather than assuming it. For the SO101 the order matches lerobot's motor order.

Three conventions hold here:

1. **Hardware SDKs are optional extras, imported lazily.** Never import a driver at module scope — put a plain `import` inside the function that uses it, with a comment naming the extra that installs it.
2. **Machine-local config is generated, never checked in.** Values that identify one person's hardware (USB serial numbers, calibration) live under `paths.local_dir()`.
3. **Missing config raises with the command that generates it.** See `get_ports.load_port_config`.

### RL registry

`_src/rl/env_registry.py` maps a string env id to `(env_builder, runner_cfg, env_cfg_fn)`. Registrations are at the bottom of that file — `SO101PickCube`, `TrossenPickCube`, `CartpoleBalance`, `G1WholeBodyTracking`, and their `*Vision` variants. `env_cfg_fn` is called **lazily**, deliberately: config functions touch the filesystem to load assets, and that must not happen at import time.

Adding an env to the registry means importing its task module there, so a broken task config breaks registry import for everything.

Training entry points are `rl/rsl_rl/train.py` (`--env`, `--seed`, `--overrides`, `--env_overrides` as JSON dicts with dotted keys) and `rl/rsl_rl/play.py` (`--env`, `--env_overrides`, `--resume_path`, `--nworld`). RSL-RL is an optional dependency under the `rl` extra.

### Assets

`asset.get("github:google-deepmind/mujoco_menagerie/<path>.xml")` resolves URIs through `_src/paths.py`, which maps URI prefixes to local checkout directories. Robot scenes call this rather than hardcoding paths.

# Bringing up an SO101

Two READMEs take a user from an unopened box to logged teleop episodes, in this order:

1. `mujoco_playground/robots/so101/real/README.md` — sync the `so101` driver extra, identify the two arms by USB serial and the wrist camera by sight (`real.get_ports --identify`, `real.get_cameras --identify`), then calibrate both arms (`real.get_calibration`; `--roles` re-does just one).
2. `mujoco_playground/tasks/teleop/so101/README.md` — teleop in sim first, then on hardware; foot-pedal gating, MCAP episode logs, and `view_log` to replay an episode in rerun.
