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

"""Script to train RL agent with RSL-RL.

Supports single-GPU and multi-GPU distributed training via torchrun.
"""

import dataclasses
import json
import os
import shutil
import sys

from absl import app
from absl import flags
from absl import logging
from etils import epath
from rsl_rl import runners as rls_rl_runners
import torch
import warp as wp

try:
  from mujoco_playground._src.internal_imports import multiproc  # pyrefly: ignore[missing-import]
except ImportError:
  multiproc = None

from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl import rng
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper


def configure_torch_backends(
    allow_tf32: bool = True, deterministic: bool = False
):
  """Configure PyTorch CUDA and cuDNN."""
  precision = "high" if allow_tf32 else "highest"
  torch.set_float32_matmul_precision(precision)
  torch.backends.cudnn.allow_tf32 = allow_tf32
  torch.backends.cudnn.benchmark = not deterministic
  torch.backends.cudnn.deterministic = deterministic


_REMOTE_LOG_DIR = flags.DEFINE_string(
    "remote_logdir", None, "Remote path to save checkpoints and logs."
)
_ENV = flags.DEFINE_string("env", "SO101PickCube", "Environment name to train.")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed.")
_OVERRIDES = flags.DEFINE_string(
    "overrides",
    None,
    "JSON dict string for config overrides with dotted keys for nested params.",
)
_WP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "wp_kernel_cache_dir",
    "/tmp/wp_kernel_cache",
    "Path to the Warp kernel cache directory.",
)
_LOG_DIR = flags.DEFINE_string(
    "logdir",
    "/tmp/rslrl_logs",
    "Local directory to save logs and checkpoints.",
)
_RESUME_PATH = flags.DEFINE_string(
    "resume_path",
    None,
    "Path to the checkpoint to resume training from.",
)
_ENV_OVERRIDES = flags.DEFINE_string(
    "env_overrides",
    None,
    "JSON dict string for environment config overrides.",
)
_USE_BFLOAT16 = flags.DEFINE_bool(
    "use_bfloat16", False, "Use bfloat16 AMP autocast for PPO updates."
)
_TORCH_COMPILE_MODE = flags.DEFINE_string(
    "torch_compile_mode",
    None,
    "Override torch_compile_mode ('default', 'reduce-overhead', etc.).",
)


def update_from_dotted_keys(dc_instance, overrides_dict):
  for key, value in overrides_dict.items():
    *path, leaf = key.split(".")
    obj = dc_instance
    for part in path:
      obj = getattr(obj, part)
    setattr(obj, leaf, value)


class OnPolicyRunnerTB(rls_rl_runners.OnPolicyRunner):
  """OnPolicyRunner with custom logger and mixed precision update support."""

  def __init__(self, *args, **kwargs):
    self.remote_log_dir = kwargs.pop("remote_log_dir", None)
    self.use_bfloat16 = kwargs.pop("use_bfloat16", True)
    self.writer = None
    super().__init__(*args, **kwargs)

    if (
        self.use_bfloat16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
      orig_update = self.alg.update

      def bf16_update(*u_args, **u_kwargs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          return orig_update(*u_args, **u_kwargs)

      self.alg.update = bf16_update

  def _prepare_logging_writer(self):
    """Prepares the logging writers."""
    self.logger_type = self.cfg.get("logger", "tensorboard")
    self.logger_type = self.logger_type.lower()
    if self.logger_type != "tensorboard":
      raise ValueError("Logger type not found. Please choose 'tensorboard'.")
    self.logger.logger_type = self.logger_type
    log_dir = (
        self.remote_log_dir if self.remote_log_dir else self.logger.log_dir
    )
    if (
        log_dir is not None
        and self.logger.writer is None
        and not self.logger.disable_logs
    ):
      try:
        from clu import metric_writers
        import tensorflow as tf  # pylint: disable=g-import-not-at-top

        # Prevent TensorFlow from allocating GPU memory when imported by CLU to
        # avoid GPU memory contention with PyTorch and MuJoCo Warp.
        tf.config.set_visible_devices([], "GPU")

        writer = metric_writers.create_default_writer(
            log_dir,
            asynchronous=True,
        )
        pending_scalars: dict[str, float] = {}

        def add_scalar(name: str, value: float | torch.Tensor, step: int):
          del step
          if not name.endswith("/time"):
            pending_scalars[name] = float(value)

        writer.add_scalar = add_scalar  # pyrefly: ignore[missing-attribute]
        orig_log = self.logger.log

        def batched_log(*args, **kwargs):
          orig_log(*args, **kwargs)
          sys.stdout.flush()
          if pending_scalars:
            step = int(kwargs["it"] if "it" in kwargs else args[0])
            writer.write_scalars(step, dict(pending_scalars))
            pending_scalars.clear()

        self.logger.log = batched_log  # pyrefly: ignore[bad-assignment]
        self.writer = writer
      except ImportError:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.writer.write_hparams = lambda h: self.writer.add_hparams(  # pytype: disable=attribute-error  # pyrefly: ignore[missing-attribute]
            h, {}
        )
      if self.writer is not None:
        self.logger.writer = self.writer  # pyrefly: ignore[bad-assignment]
        self.logger.init_logging_writer = lambda *args, **kwargs: None

  def save(self, path, infos=None):
    super().save(path, infos)
    if self.remote_log_dir and self.logger.log_dir:
      rel = os.path.relpath(path, self.logger.log_dir)
      remote_dest = epath.Path(self.remote_log_dir) / rel
      remote_dest.parent.mkdir(parents=True, exist_ok=True)
      epath.Path(path).copy(remote_dest, overwrite=True)
      logging.info("Copied checkpoint to remote logdir: %s", remote_dest)


def main(argv):
  del argv

  configure_torch_backends()

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  torch.cuda.set_device(local_rank)
  rank = int(os.environ.get("RANK", "0"))

  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  device = f"cuda:{local_rank}"

  seed = _SEED.value + local_rank
  rng.seed(seed)

  if _WP_KERNEL_CACHE_DIR.value:
    wp.config.kernel_cache_dir = _WP_KERNEL_CACHE_DIR.value

  logging.info(
      "Training with: device=%s, seed=%d, local_rank=%d, rank=%d",
      device,
      seed,
      local_rank,
      rank,
  )

  entry = env_registry.get(_ENV.value)
  agent_cfg = entry.runner_cfg
  if rank == 0:
    logging.info(
        "Agent config:\n%s",
        json.dumps(dataclasses.asdict(agent_cfg), indent=2),
    )

  overrides = json.loads(_OVERRIDES.value) if _OVERRIDES.value else {}
  update_from_dotted_keys(agent_cfg, overrides)

  agent_cfg.seed = seed

  env_overrides = (
      json.loads(_ENV_OVERRIDES.value) if _ENV_OVERRIDES.value else {}
  )

  env = entry.env_builder(
      nworld=agent_cfg.nworld,
      device=device,
      **env_overrides,
  )
  env = vecenv_wrapper.RslRlVecEnvWrapper(
      env,
      clip_actions=agent_cfg.clip_actions,
      cfg=agent_cfg.vecenv_wrapper_cfg,
  )

  log_dir = _LOG_DIR.value
  remote_dest_dir = None

  if rank == 0:
    epath.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if _REMOTE_LOG_DIR.value:
      remote_dest_dir = epath.Path(_REMOTE_LOG_DIR.value)
      remote_dest_dir.mkdir(parents=True, exist_ok=True)
      logging.info("Logging remotely to %s", remote_dest_dir)
    logging.info("Logging locally to %s", log_dir)

  if _TORCH_COMPILE_MODE.value is not None:
    agent_cfg.torch_compile_mode = _TORCH_COMPILE_MODE.value

  runner = OnPolicyRunnerTB(
      env,
      dataclasses.asdict(agent_cfg),
      device=device,
      log_dir=log_dir,
      use_bfloat16=_USE_BFLOAT16.value,
      remote_log_dir=_REMOTE_LOG_DIR.value if rank == 0 else None,
  )

  if rank == 0:
    hparams_to_log = {
        "nworld": agent_cfg.nworld,
        "learn_iterations": agent_cfg.max_iterations,
    }
    for k, v in overrides.items():
      if not isinstance(v, (bool, str, float, int, type(None))):
        v = str(v)
      hparams_to_log[k.replace(".", "/")] = v
    runner._prepare_logging_writer()  # pylint: disable=protected-access
    if runner.writer is not None:
      runner.writer.write_hparams(  # pyrefly: ignore[missing-attribute]
          hparams_to_log
      )
      runner.writer.flush()
    logging.info("Logged hparams to TensorBoard dir %s", log_dir)

  if _RESUME_PATH.value and epath.Path(_RESUME_PATH.value).exists():
    logging.info("Loading from checkpoint: %s", _RESUME_PATH.value)
    load_cfg = {
        "actor": True,
        # re-initialize the critic assuming the mdp changed and the old critic
        # is no longer valid.
        "critic": False,
        "optimizer": True,
        "iteration": True,
    }
    runner.load(str(_RESUME_PATH.value), load_cfg=load_cfg, map_location=device)

  runner.learn(
      num_learning_iterations=agent_cfg.max_iterations,
      init_at_random_ep_len=True,
  )
  if runner.writer is not None:
    runner.writer.flush()

  if rank == 0 and _REMOTE_LOG_DIR.value:
    if epath.Path(log_dir).exists():
      try:
        shutil.copytree(  # pyrefly: ignore[bad-specialization]
            log_dir, remote_dest_dir, dirs_exist_ok=True
        )  # pyrefly: ignore[bad-specialization]
        logging.info("Copied logs to %s", remote_dest_dir)
      except Exception as e:
        logging.exception(e)

  env.close()


if __name__ == "__main__":
  if multiproc is not None:
    multiproc.handle_main(main)
  else:
    app.run(main)
