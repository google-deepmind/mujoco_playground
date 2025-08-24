# Changelog

All notable changes to this project will be documented in this file.

## Next release

- Pass through the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
  (MjWarp) implementation to MJX, so that MuJoCo Playground environments can
  train with MuJoCo Warp! You can pass through the implementation via the config
  override
  `registry.load('CartpoleBalance', config_overrides={'impl': 'warp'})`.
- Update environments to utilize contact sensors and remove `collision.py`.
- Remove `mjx_env.init` in favor of `mjx_env.make_data` since `make_data`
  now requires an `MjModel` argument rather than an `mjx.Model` argument.
- Add device to `mjx_env.make_data`, fixes #174.
- Update AutoResetWrapper to allow full resets on done. Fixes #179. Also
  provides a means for doing curriculum learning via
  `state.info['AutoResetWrapper_done_count']`, see #140.

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
