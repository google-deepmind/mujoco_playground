# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core classes for MuJoCo Playground."""

import abc
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from etils import epath
from flax import struct
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
import tqdm

# Root path is used for loading XML strings directly using etils.epath.
ROOT_PATH = epath.Path(__file__).parent
# Base directory for external dependencies.
EXTERNAL_DEPS_PATH = epath.Path(__file__).parent.parent / "external_deps"
# The menagerie path is used to load robot assets.
# Resource paths do not have glob implemented, so we use a bare epath.Path.
MENAGERIE_PATH = EXTERNAL_DEPS_PATH / "mujoco_menagerie"
# Commit SHA of the menagerie repo.
MENAGERIE_COMMIT_SHA = "14ceccf557cc47240202f2354d684eca58ff8de4"


def _clone_with_progress(
    repo_url: str, target_path: str, commit_sha: str
) -> None:
  """Clone a git repo with progress bar."""
  process = subprocess.Popen(
      ["git", "clone", "--progress", repo_url, target_path],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
  )

  with tqdm.tqdm(
      desc="Cloning mujoco_menagerie",
      bar_format="{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
  ) as pbar:
    pbar.total = 100  # Set to 100 for percentage-based progress.
    current = 0
    while True:
      # Read output line by line.
      output = process.stderr.readline()  # pytype: disable=attribute-error
      if not output and process.poll() is not None:
        break
      if output:
        if "Receiving objects:" in output:
          try:
            percent = int(output.split("%")[0].split(":")[-1].strip())
            if percent > current:
              pbar.update(percent - current)
              current = percent
          except (ValueError, IndexError):
            pass

    # Ensure the progress bar reaches 100%.
    if current < 100:
      pbar.update(100 - current)

  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, ["git", "clone"])

  # Checkout specific commit.
  print(f"Checking out commit {commit_sha}")
  subprocess.run(
      ["git", "-C", target_path, "checkout", commit_sha],
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )


def ensure_menagerie_exists() -> None:
  """Ensure mujoco_menagerie exists, downloading it if necessary."""
  if not MENAGERIE_PATH.exists():
    print("mujoco_menagerie not found. Downloading...")

    # Create external deps directory if it doesn't exist
    EXTERNAL_DEPS_PATH.mkdir(exist_ok=True, parents=True)

    try:
      _clone_with_progress(
          "https://github.com/deepmind/mujoco_menagerie.git",
          str(MENAGERIE_PATH),
          MENAGERIE_COMMIT_SHA,
      )
      print("Successfully downloaded mujoco_menagerie")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading mujoco_menagerie: {e}", file=sys.stderr)
      raise


Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


def update_assets(
    assets: Dict[str, Any],
    path: Union[str, epath.Path],
    glob: str = "*",
    recursive: bool = False,
):
  for f in epath.Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)


def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  data = mjx.make_data(
      model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
  )
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  if act is not None:
    data = data.replace(act=act)
  if mocap_pos is not None:
    data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
  if mocap_quat is not None:
    data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
  return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  def single_step(data, _):
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]


@struct.dataclass
class State:
  """Environment state for training and inference."""

  data: mjx.Data
  obs: Observation
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array]
  info: Dict[str, Any]

  def tree_replace(
      self, params: Dict[str, Optional[jax.typing.ArrayLike]]
  ) -> "State":
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split("."), v)
    return new


def _tree_replace(
    base: Any,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Any:
  """Sets attributes in a struct.dataclass with values."""
  if not attr:
    return base

  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    raise NotImplementedError("List attributes are not supported.")

  if len(attr) == 1:
    return base.replace(**{attr[0]: val})

  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )


class MjxEnv(abc.ABC):
  """Base class for playground environments."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    self._config = config.lock()
    if config_overrides:
      self._config.update_from_flattened_dict(config_overrides)

    self._ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

  @abc.abstractmethod
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  @abc.abstractmethod
  def xml_path(self) -> str:
    """Path to the xml file for the environment."""

  @property
  @abc.abstractmethod
  def action_size(self) -> int:
    """Size of the action space."""

  @property
  @abc.abstractmethod
  def mj_model(self) -> mujoco.MjModel:
    """Mujoco model for the environment."""

  @property
  @abc.abstractmethod
  def mjx_model(self) -> mjx.Model:
    """Mjx model for the environment."""

  @property
  def dt(self) -> float:
    """Control timestep for the environment."""
    return self._ctrl_dt

  @property
  def sim_dt(self) -> float:
    """Simulation timestep for the environment."""
    return self._sim_dt

  @property
  def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.dt / self.sim_dt))

  @property
  def observation_size(self) -> ObservationSize:
    abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
    obs = abstract_state.obs
    if isinstance(obs, Mapping):
      return jax.tree_util.tree_map(lambda x: x.shape, obs)
    return obs.shape[-1]

  @property
  def model_assets(self) -> Dict[str, Any]:
    """Dictionary of model assets to use with MjModel.from_xml_path."""
    if hasattr(self, "_model_assets"):
      return self._model_assets
    raise NotImplementedError(
        "_model_assets not defined for this environment"
        "see cartpole.py for an example."
    )

  def render(
      self,
      trajectory: List[State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return render_array(
        self.mj_model,
        trajectory,
        height,
        width,
        camera,
        scene_option=scene_option,
        modify_scene_fns=modify_scene_fns,
    )

  @property
  def unwrapped(self) -> "MjxEnv":
    return self


def render_array(
    mj_model: mujoco.MjModel,
    trajectory: Union[List[State], State],
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[
        Sequence[Callable[[mujoco.MjvScene], None]]
    ] = None,
    hfield_data: Optional[jax.Array] = None,
):
  """Renders a trajectory as an array of images."""
  renderer = mujoco.Renderer(mj_model, height=height, width=width)
  camera = camera if camera is not None else -1

  if hfield_data is not None:
    mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
    mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

  def get_image(state, modify_scn_fn=None) -> np.ndarray:
    d = mujoco.MjData(mj_model)
    d.qpos, d.qvel = state.data.qpos, state.data.qvel
    d.mocap_pos, d.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
    d.xfrc_applied = state.data.xfrc_applied
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
    if modify_scn_fn is not None:
      modify_scn_fn(renderer.scene)
    return renderer.render()

  if isinstance(trajectory, list):
    out = []
    for i, state in enumerate(tqdm.tqdm(trajectory)):
      if modify_scene_fns is not None:
        modify_scene_fn = modify_scene_fns[i]
      else:
        modify_scene_fn = None
      out.append(get_image(state, modify_scene_fn))
  else:
    out = get_image(trajectory)

  renderer.close()
  return out


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
  """Gets sensor data given sensor name."""
  sensor_id = model.sensor(sensor_name).id
  sensor_adr = model.sensor_adr[sensor_id]
  sensor_dim = model.sensor_dim[sensor_id]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qvel."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qpos."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> np.ndarray:
  index_list: list[int] = []
  for jnt_name in joint_names:
    jnt = model.joint(jnt_name).id
    jnt_type = model.jnt_type[jnt]
    qadr = model.jnt_qposadr[jnt]
    qdim = qpos_width(jnt_type)
    index_list.extend(range(qadr, qadr + qdim))
  return np.array(index_list)


def get_qvel_ids(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> np.ndarray:
  index_list: list[int] = []
  for jnt_name in joint_names:
    jnt = model.joint(jnt_name).id
    jnt_type = model.jnt_type[jnt]
    vadr = model.jnt_dofadr[jnt]
    vdim = dof_width(jnt_type)
    index_list.extend(range(vadr, vadr + vdim))
  return np.array(index_list)
