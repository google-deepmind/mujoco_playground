# Changelog

All notable changes to this project will be documented in this file.

## [0.0.5] - 2025-06-23

- Change `light_directional` to `light_type` following MuJoCo API change from version 3.3.2 to 3.3.3. Fixes https://github.com/google-deepmind/mujoco_playground/issues/142.
- Fix bug in `get_qpos_ids`.
- Implement `render` in Wrapper.
- Fix https://github.com/google-deepmind/mujoco_playground/issues/123.
- Fix https://github.com/google-deepmind/mujoco_playground/issues/126.
- Fix https://github.com/google-deepmind/mujoco_playground/issues/41.

## [0.0.4] - 2025-02-07

### Added

- Added ALOHA handover task (thanks to @Andrew-Luo1).
- Added Booster T1 joystick task.

### Changed

- Fixed foot friction randomization for G1 tasks.
- Fix various bugs in `train_jax_ppo.py` (thanks to @vincentzhang).
- Fixed a small bug in the privileged state of the go1 joystick task.

## [0.0.3] - 2025-01-18

### Changed

- Updated supported Python versions to 3.10-3.12.

## [0.0.2] - 2025-01-16

Initial release.
