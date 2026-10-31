# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""System identification for the YAM using mujoco.sysid."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import json
import types
from typing import Literal
from typing import TypedDict

from absl import logging
from etils import epath
import mujoco
from mujoco import sysid
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
from mujoco_playground.robots.yam.sim import scene
from mujoco_playground.tasks.sysid.yam import data_loader
import numpy as np
from scipy import optimize as scipy_optimize

ACTUATORS = constant.ARM_JOINTS + (sim_constant.GRIPPER_ACTUATOR,)
# Bounds per parameter, by motor type (`constant.MOTOR_TYPES`), and for the
# gripper's slide joints.
_BOUNDS = types.MappingProxyType({
    "DM4340": dict(
        armature=(1e-4, 0.5),
        frictionloss=(0.0, 3.0),
        kp=(5.0, 400.0),
        kv=(0.05, 50.0),
    ),
    "DM4310": dict(
        armature=(1e-5, 0.2),
        frictionloss=(0.0, 2.0),
        kp=(1.0, 250.0),
        kv=(0.01, 30.0),
    ),
    sim_constant.GRIPPER_ACTUATOR: dict(
        armature=(1e-3, 5.0),
        frictionloss=(0.0, 20.0),
        kp=(50.0, 20000.0),
        kv=(1.0, 2000.0),
    ),
})


def _bounds(name: str) -> dict[str, tuple[float, float]]:
  if name == sim_constant.GRIPPER_ACTUATOR:
    return _BOUNDS[name]
  return _BOUNDS[constant.MOTOR_TYPES[name]]


# The wrist camera (D405 and bracket) is the arm's only body without gravity
# compensation, in the MJCF as on the real arm (I2RT's model leaves it out).
WRIST_CAMERA_BODY = "wrist_camera"
WRIST_CAMERA_IPOS_RANGE = 0.04  # m, around the CAD center of mass.


def _joints(name: str) -> tuple[str, ...]:
  if name == sim_constant.GRIPPER_ACTUATOR:
    return sim_constant.FINGER_JOINTS
  return (name,)


def _set_joint_attr(
    spec: mujoco.MjSpec, name: str, attr: str, value: float
) -> None:
  for joint in _joints(name):
    setattr(spec.joint(joint), attr, value)


def nominal_values(spec: mujoco.MjSpec) -> dict[str, float]:
  """Armature, frictionloss, kp and kv of each actuator in the MJCF."""
  model = spec.compile()
  values = {}
  for name in ACTUATORS:
    dof = model.joint(_joints(name)[0]).dofadr[0]
    act = model.actuator(name)
    values[f"{name}_armature"] = float(model.dof_armature[dof])
    values[f"{name}_frictionloss"] = float(model.dof_frictionloss[dof])
    values[f"{name}_kp"] = float(act.gainprm[0])
    values[f"{name}_kv"] = float(-act.biasprm[2])
  return values


def build_parameters(
    spec: mujoco.MjSpec | None = None,
    freeze_kp: Sequence[str] = ("joint1",),
    freeze_gripper: bool = True,
    freeze_joints: Sequence[str] = (),
    freeze_params: Sequence[str] = (),
    wrist_camera: bool = False,
) -> sysid.ParameterDict:
  """Armature, frictionloss, kp and kv per actuator, from the MJCF nominals.

  Args:
    spec: The YAM spec whose values are the nominals (default: Menagerie's,
      `scene.yam_menagerie()`, so a refit doesn't start from a previous one).
    freeze_kp: Actuators whose kp stays at its nominal, the I2RT SDK's gain.
      Joint 1 carries no gravity load, so its tracking only fixes ratios between
      kp, kv, friction and armature, not their scale.
    freeze_gripper: Keep the gripper at its nominals. Its tracking barely
      depends on them, so they are not identifiable from these episodes.
    freeze_joints: Actuators whose four parameters all stay at their nominals.
    freeze_params: Individual parameters (e.g. "joint3_armature") that stay at
      their nominals, typically the ones a previous fit flagged with
      `unidentified_parameters`.
    wrist_camera: Also fit the wrist camera's mass and center of mass. Off by
      default: the camera's real load on the wrist (likely including its cable)
        is not a rigid payload, and the fitted center of mass runs to its
        bounds.

  Returns:
    The parameters, with the frozen ones marked.
  """
  nominal = nominal_values(spec or scene.yam_menagerie())
  params = sysid.ParameterDict()
  for name in ACTUATORS:
    bounds = _bounds(name)
    for attr in ("armature", "frictionloss"):
      params.add(
          sysid.Parameter(
              f"{name}_{attr}",
              nominal=nominal[f"{name}_{attr}"],
              min_value=bounds[attr][0],
              max_value=bounds[attr][1],
              modifier=lambda s, p, n=name, a=attr: _set_joint_attr(
                  s, n, a, float(p.value[0])
              ),
          )
      )
    params.add(
        sysid.Parameter(
            f"{name}_kp",
            nominal=nominal[f"{name}_kp"],
            min_value=bounds["kp"][0],
            max_value=bounds["kp"][1],
            modifier=lambda s, p, n=name: sysid.apply_pgain(
                s, n, float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"{name}_kv",
            nominal=nominal[f"{name}_kv"],
            min_value=bounds["kv"][0],
            max_value=bounds["kv"][1],
            modifier=lambda s, p, n=name: sysid.apply_dgain(
                s, n, float(p.value[0])
            ),
        )
    )
  if wrist_camera:
    body = (spec or scene.yam_menagerie()).body(WRIST_CAMERA_BODY)
    ipos = np.asarray(body.ipos, dtype=np.float64)
    params.add(
        sysid.Parameter(
            "wrist_camera_mass",
            nominal=float(body.mass),
            min_value=0.03,
            max_value=0.5,
            modifier=lambda s, p: setattr(
                s.body(WRIST_CAMERA_BODY), "mass", float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            "wrist_camera_ipos",
            nominal=ipos,
            min_value=ipos - WRIST_CAMERA_IPOS_RANGE,
            max_value=ipos + WRIST_CAMERA_IPOS_RANGE,
            modifier=lambda s, p: setattr(
                s.body(WRIST_CAMERA_BODY), "ipos", p.value.copy()
            ),
        )
    )
  for name in freeze_kp:
    params[f"{name}_kp"].frozen = True
  for name in freeze_joints:
    for attr in ("armature", "frictionloss", "kp", "kv"):
      params[f"{name}_{attr}"].frozen = True
  for name in freeze_params:
    if name not in params:
      raise ValueError(f"Unknown parameter {name!r}.")
    params[name].frozen = True
  if freeze_gripper:
    for attr in ("armature", "frictionloss", "kp", "kv"):
      params[f"{sim_constant.GRIPPER_ACTUATOR}_{attr}"].frozen = True
  return params


def tracking_rmse(
    params: sysid.ParameterDict, model_sequences: sysid.ModelSequences
) -> dict[str, list[float]]:
  """Per-episode RMSE per sensor: mrad for the arm, mm for the finger."""
  p = params.copy()
  residual_fn = sysid.build_residual_fn(models_sequences=[model_sequences])
  _, preds, records = residual_fn(p.as_vector(), p, return_pred_all=True)
  return {
      name: (
          (
              1000 * np.sqrt(np.mean((pred[0].data - rec[0].data) ** 2, axis=0))
          ).tolist()
      )
      for name, pred, rec in zip(
          model_sequences.sequence_name, preds, records, strict=True
      )
  }


# A parameter whose 95% confidence half-width is wider than this fraction of
# its value counts as poorly identified.
WIDE_INTERVAL = 0.5


def unidentified_reason(
    value: float,
    nominal: float,
    bounds: tuple[float, float],
    half_width: float | None,
) -> str | None:
  """Why the data doesn't identify a fitted value, or None if it does.

  Args:
    value: The identified value.
    nominal: The MJCF's value.
    bounds: The parameter's (min, max).
    half_width: Its 95% confidence half-width, if known.

  Returns:
    "at bound", "interval contains nominal", "wide interval" or None.
  """
  lo, hi = bounds
  if min(value - lo, hi - value) <= 1e-3 * (hi - lo):
    return "at bound"
  if half_width is None or not np.isfinite(half_width):
    return "wide interval"
  if abs(value - nominal) <= half_width:
    return "interval contains nominal"
  if half_width > WIDE_INTERVAL * abs(value):
    return "wide interval"
  return None


class ParameterSummary(TypedDict, total=False):
  nominal: list[float]
  identified: list[float]
  frozen: bool
  ci95_half_width: float | None
  unidentified: str | None


class Summary(TypedDict):
  sensors: list[str]
  rmse_units: str
  rmse: dict[str, dict[str, dict[str, list[float]]]]
  command_delay_ms: dict[str, float]
  parameters: dict[str, ParameterSummary]
  correlations_abs_ge_0p8: list[tuple[str, str, float]]


def _diagnostics(
    opt_params: sysid.ParameterDict,
    opt_result: scipy_optimize.OptimizeResult,
    residual_fn: Callable[
        ...,
        tuple[
            list[np.ndarray],
            list[tuple[sysid.TimeSeries, ...]],
            list[tuple[sysid.TimeSeries, ...]],
        ],
    ],
) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
  """95% half-widths and strong correlations of the fitted parameters.

  The intervals come from the Jacobian at the optimum and assume uncorrelated
  residuals; tracking residuals are correlated in time, so they are optimistic.

  Args:
    opt_params: The identified parameters.
    opt_result: The optimizer's result, with the Jacobian at the optimum.
    residual_fn: The residual function the fit minimized.

  Returns:
    (half-width per fitted parameter, pairs with |correlation| >= 0.8).
  """
  residuals, _, _ = residual_fn(
      opt_params.as_vector(), opt_params.copy(), return_pred_all=True
  )
  cov, half_widths = sysid.calculate_intervals(residuals, opt_result.jac)
  names = opt_params.get_non_frozen_parameter_names()
  if len(half_widths) != len(names):
    return {}, []
  widths = {n: float(w) for n, w in zip(names, half_widths, strict=True)}
  sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
  corr = cov / np.outer(sd, sd)
  pairs = []
  for i, j in zip(*np.triu_indices(len(names), k=1)):
    if abs(corr[i, j]) >= 0.8:
      pairs.append((names[i], names[j], float(corr[i, j])))
  return widths, sorted(pairs, key=lambda p: -abs(p[2]))


def unidentified_parameters(summary: Summary) -> dict[str, str]:
  """Fitted parameters the data doesn't identify, with the reason."""
  return {
      k: reason
      for k, v in summary["parameters"].items()
      if (reason := v.get("unidentified")) is not None
  }


@dataclasses.dataclass(frozen=True)
class Result:
  initial_params: sysid.ParameterDict
  opt_params: sysid.ParameterDict
  summary: Summary


def run_sysid(
    train: Sequence[data_loader.Trajectory],
    validation: Sequence[data_loader.Trajectory] = (),
    save_path: epath.PathLike | None = None,
    max_iters: int = 150,
    optimizer: Literal["mujoco", "scipy", "scipy_parallel_fd"] = "mujoco",
    generate_videos: bool = False,
    initial_params: sysid.ParameterDict | None = None,
) -> Result:
  """Fits on `train`, then reports tracking RMSE on `train` and `validation`.

  Args:
    train: Episodes to fit on.
    validation: Held-out episodes, only evaluated.
    save_path: Where to write report.html and identified_params.json.
    max_iters: Optimizer iterations.
    optimizer: mujoco.sysid least-squares backend.
    generate_videos: Render rollout videos into the report.
    initial_params: Parameters to start from (default: `build_parameters`).

  Returns:
    The initial and identified parameters and a summary.
  """
  if not train:
    raise ValueError("No training trajectories.")

  def spec() -> mujoco.MjSpec:
    s = scene.yam_menagerie()
    s.option.timestep = 0.002
    return s

  train_ms = data_loader.to_model_sequences(train, spec(), "yam_train")
  initial_params = initial_params or build_parameters()
  residual_fn = sysid.build_residual_fn(models_sequences=[train_ms])
  opt_params, opt_result = sysid.optimize(
      initial_params=initial_params,
      residual_fn=residual_fn,
      optimizer=optimizer,
      max_iters=max_iters,
  )

  rmse: dict[str, dict[str, dict[str, list[float]]]] = {
      "train": {},
      "validation": {},
  }
  sequences = {"train": train_ms}
  if validation:
    sequences["validation"] = data_loader.to_model_sequences(
        validation, spec(), "yam_validation"
    )
  for split, ms in sequences.items():
    for label, p in (("nominal", initial_params), ("identified", opt_params)):
      rmse[split][label] = tracking_rmse(p, ms)
  delays = np.concatenate([t.command_delay_ms for t in [*train, *validation]])
  widths, correlations = _diagnostics(opt_params, opt_result, residual_fn)
  parameters: dict[str, ParameterSummary] = {}
  for k in initial_params.keys():
    p = initial_params[k]
    entry: ParameterSummary = {
        "nominal": p.nominal.tolist(),
        "identified": opt_params[k].value.tolist(),
        "frozen": bool(p.frozen),
    }
    if not p.frozen and p.nominal.size == 1:
      entry["ci95_half_width"] = widths.get(k)
      entry["unidentified"] = unidentified_reason(
          float(opt_params[k].value[0]),
          float(p.nominal[0]),
          (float(p.min_value[0]), float(p.max_value[0])),
          widths.get(k),
      )
    parameters[k] = entry
  summary: Summary = {
      "sensors": list(data_loader.SENSOR_JOINTS),
      "rmse_units": "mrad for arm joints, mm for left_finger",
      "rmse": rmse,
      "command_delay_ms": {
          "mean": float(delays.mean()),
          "std": float(delays.std()),
      },
      "parameters": parameters,
      "correlations_abs_ge_0p8": correlations,
  }

  if save_path is not None:
    save_dir = epath.Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "identified_params.json").write_text(
        json.dumps(summary, indent=1)
    )
    sysid.default_report(
        models_sequences=[train_ms],
        initial_params=initial_params,
        opt_params=opt_params,
        residual_fn=residual_fn,
        opt_result=opt_result,
        title="YAM System Identification Report",
        save_path=save_dir,
        generate_videos=generate_videos,
    )
  logging.info("Identified: %s", parameters)
  return Result(initial_params, opt_params, summary)
