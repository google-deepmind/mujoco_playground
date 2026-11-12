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

"""Tests the SO-101 pick-cube deploy script."""

import copy
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from dm_env import specs
from etils import epath
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env as pick_cube_env
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import run as deploy_run
import numpy as np
import torch


class _Actor(torch.nn.Module):
  """Takes inputs like RSL-RL's export: 1D observations, then images."""

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(8, 6)

  def forward(
      self, obs: torch.Tensor, wrist: torch.Tensor, camera: torch.Tensor
  ) -> torch.Tensor:
    # Channel 0 alone, so pixels read in the wrong layout change the action.
    return self.linear(obs) * wrist.mean() - camera[:, 0].mean()


def _env(
    obs: dict[str, torch.Tensor],
    action_shape: tuple[int, ...] = (1, 6),
    device: str = "cpu",
) -> mock.Mock:
  observation_spec = {
      k: specs.Array(tuple(v.shape), np.float32) for k, v in obs.items()
  }
  return mock.Mock(
      xpu_device=device,
      **{
          "timestep_spec.return_value.observation": observation_spec,
          "action_spec.return_value": {
              "action": specs.Array(action_shape, np.float32)
          },
      },
  )


class MainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enterContext(
        flagsaver.flagsaver(
            checkpoint=None,
            viewer=True,
            mcap_log_dir=self.enterContext(tempfile.TemporaryDirectory()),
        )
    )

  def test_checks_for_executorch_before_the_env(self):
    directory = self.enterContext(tempfile.TemporaryDirectory())
    path = os.path.join(directory, "policy.pte")
    open(path, "wb").close()
    with (
        flagsaver.flagsaver(checkpoint=path),
        mock.patch.object(importlib.util, "find_spec", return_value=None),
        mock.patch.object(deploy_run, "create_env") as create_env,
    ):
      with self.assertRaisesRegex(ImportError, "--with"):
        deploy_run.main([])
    create_env.assert_not_called()

  def test_closes_the_env_when_the_policy_fails_to_load(self):
    env = mock.Mock()
    with (
        mock.patch.object(deploy_run, "create_env", return_value=env),
        mock.patch.object(
            deploy_run,
            "create_policy",
            side_effect=ValueError("bad checkpoint"),
        ),
    ):
      with self.assertRaisesRegex(ValueError, "bad checkpoint"):
        deploy_run.main([])
    env.close.assert_called_once()

  def test_closes_the_env_when_a_logger_fails_to_close(self):
    env = mock.Mock()
    logger = mock.Mock(**{"close.side_effect": OSError("disk full")})
    with (
        mock.patch.object(deploy_run, "create_env", return_value=env),
        mock.patch.object(deploy_run, "create_policy"),
        mock.patch.object(deploy_run, "create_logger", return_value=logger),
        mock.patch.object(deploy_run.deploy_viewer, "launch"),
    ):
      with self.assertRaisesRegex(OSError, "disk full"):
        deploy_run.main([])
    env.close.assert_called_once()


class LoadRunnerTest(absltest.TestCase):

  @absltest.skipUnless(importlib.util.find_spec("rsl_rl"), "Requires rsl_rl.")
  def test_loads_the_actor_from_a_checkpoint(self):
    from rsl_rl import runners  # pylint: disable=g-import-not-at-top
    import tensordict  # pylint: disable=g-import-not-at-top

    env = pick_cube_env.create_sim_env(episode_length_secs=1.0)
    with mock.patch.object(runners.OnPolicyRunner, "load"):
      saved = deploy_run.load_runner(env, "unused.pt")
    directory = self.enterContext(tempfile.TemporaryDirectory())
    saved.save(os.path.join(directory, "model_1.pt"))

    actor = deploy_run.load_actor(env, directory)
    obs = env.reset().observation
    want = saved.get_inference_policy(device=env.xpu_device)(
        tensordict.TensorDict(dict(obs), batch_size=[1])
    )
    torch.testing.assert_close(actor(obs), want)


class MissingExecuTorchTest(absltest.TestCase):

  def test_says_how_to_add_executorch(self):
    with mock.patch.dict(sys.modules, {"executorch.runtime": None}):
      with self.assertRaisesRegex(ImportError, "--with"):
        deploy_run.load_executorch_actor(mock.Mock(), "policy.pte")

  def test_says_how_to_add_executorch_when_it_is_broken(self):
    runtime = types.ModuleType("executorch.runtime")
    with mock.patch.dict(sys.modules, {"executorch.runtime": runtime}):
      with self.assertRaisesRegex(ImportError, "--with"):
        deploy_run.load_executorch_actor(mock.Mock(), "policy.pte")


# `make test-executorch` sets REQUIRE_EXECUTORCH, so there these tests run.
@absltest.skipUnless(
    importlib.util.find_spec("executorch")
    or os.environ.get("REQUIRE_EXECUTORCH"),
    "Requires executorch.",
)
class ExecuTorchActorTest(parameterized.TestCase):

  def _stub_tensorrt_runtime(self, source: str | None = "") -> None:
    """Puts a stand-in torch_tensorrt_executorch_runtime first, None hides it."""
    self.enterContext(mock.patch.dict(sys.modules))
    sys.modules.pop("torch_tensorrt_executorch_runtime", None)
    if source is None:
      self.enterContext(
          mock.patch.dict(
              sys.modules, {"torch_tensorrt_executorch_runtime": None}
          )
      )
      return
    directory = epath.Path(self.enterContext(tempfile.TemporaryDirectory()))
    (directory / "torch_tensorrt_executorch_runtime.py").write_text(source)
    self.enterContext(
        mock.patch.object(sys, "path", [str(directory), *sys.path])
    )

  def setUp(self):
    super().setUp()
    self.obs = {
        "policy": torch.randn(1, 6),
        "camera": torch.rand(1, 3, 8, 8),
        "joints": torch.randn(1, 2),
        "wrist": torch.rand(1, 3, 8, 8),
    }
    self.vector = torch.cat([self.obs["policy"], self.obs["joints"]], dim=-1)
    self.enterContext(
        mock.patch.object(
            deploy_run.pick_cube_env,
            "OBSERVATION_GROUPS",
            ("policy", "camera", "joints", "wrist"),
        )
    )

  def _export(
      self,
      module: torch.nn.Module,
      inputs: tuple[torch.Tensor, ...],
      input_names: list[str] | None = None,
      weights_in_ptd: bool = False,
      xnnpack: bool = False,
  ) -> epath.Path:
    from executorch import exir  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
    from executorch.backends.xnnpack.partition import xnnpack_partitioner  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    program = exir.to_edge_transform_and_lower(
        torch.export.export(module, inputs),
        partitioner=[xnnpack_partitioner.XnnpackPartitioner()]
        if xnnpack
        else None,
        constant_methods={"input_names": input_names} if input_names else None,
    ).to_executorch(
        exir.ExecutorchBackendConfig(external_constants=weights_in_ptd)
    )
    directory = epath.Path(self.enterContext(tempfile.TemporaryDirectory()))
    path = directory / "policy.pte"
    path.write_bytes(program.buffer)
    if weights_in_ptd:
      program.write_tensor_data_to_file(str(directory))
      # RSL-RL's CUDA export names the weights file after the program.
      (directory / "_default_external_constant.ptd").rename(
          directory / "policy.ptd"
      )
    return path

  @parameterized.named_parameters(
      ("weights_in_pte", False, False),
      ("weights_in_ptd", True, False),
      ("xnnpack", False, True),
  )
  def test_feeds_inputs_in_the_program_order(self, weights_in_ptd, xnnpack):
    obs = self.obs
    actor = _Actor()
    path = self._export(
        actor,
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
        weights_in_ptd,
        xnnpack,
    )
    self.assertEqual(b"XnnpackBackend" in path.read_bytes(), xnnpack)

    # A .pte has to run without RSL-RL and tensordict.
    with mock.patch.dict(sys.modules, {"rsl_rl": None, "tensordict": None}):
      policy = deploy_run.create_policy(_env(obs), path)
    # Overwriting the files, as a new export does, must not affect the policy.
    path.write_bytes(b"")
    if weights_in_ptd:
      (path.parent / "policy.ptd").write_bytes(b"")
    # A cropped image is a view with gaps, which ExecuTorch only takes copied.
    camera = torch.cat([obs["camera"], obs["camera"]], dim=-1)[..., :8]
    (action, _), _ = policy.step(
        mock.Mock(observation=dict(obs, camera=camera))
    )

    torch.testing.assert_close(
        action["action"], actor(self.vector, obs["wrist"], obs["camera"])
    )

  @parameterized.named_parameters(
      (
          "other_image_size",
          (1, 3, 4, 4),
          ["obs", "wrist", "camera"],
          (1, 6),
          "failed on zero observations",
      ),
      ("no_input_names", (1, 3, 8, 8), None, (1, 6), "no input_names"),
      (
          "other_action_shape",
          (1, 3, 8, 8),
          ["obs", "wrist", "camera"],
          (1, 2),
          r"actions of shape \(1, 6\)",
      ),
  )
  def test_rejects_a_program_for_another_env(
      self, image_shape, input_names, action_shape, error
  ):
    image = torch.rand(image_shape)
    path = self._export(_Actor(), (self.vector, image, image), input_names)

    with self.assertRaisesRegex(ValueError, error):
      deploy_run.load_executorch_actor(_env(self.obs, action_shape), path)

  def test_rejects_a_program_without_forward(self):
    from executorch import exir  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    linear = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    program = exir.to_edge_transform_and_lower(
        {"act": linear}, constant_methods={"input_names": ["obs"]}
    ).to_executorch()
    directory = epath.Path(self.enterContext(tempfile.TemporaryDirectory()))
    path = directory / "policy.pte"
    path.write_bytes(program.buffer)

    with self.assertRaisesRegex(ValueError, "no forward method"):
      deploy_run.load_executorch_actor(_env(self.obs), path)

  @parameterized.named_parameters(
      (
          "tensorrt_runtime_missing",
          None,
          r"policy\.ptd.*instead of `--with executorch==1\.6\.0`",
      ),
      (
          "tensorrt_runtime_fails_to_import",
          "raise ImportError('no TensorRT delegate')",
          r"policy\.ptd.*backend failed to load: no TensorRT delegate",
      ),
      (
          "tensorrt_runtime_fails_to_load_a_library",
          "raise ImportError('libnvinfer.so: cannot open', name=__name__)",
          r"policy\.ptd.*backend failed to load: libnvinfer",
      ),
      (
          "tensorrt_runtime_misses_a_dependency",
          "import _missing_dependency",
          r"policy\.ptd.*backend failed to load: No module named",
      ),
      ("tensorrt_runtime_imports", "", r"policy\.ptd.*same kind of GPU"),
  )
  def test_reports_a_program_that_fails_to_load(self, runtime_source, error):
    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
        weights_in_ptd=True,
    )
    (path.parent / "policy.ptd").unlink()
    self._stub_tensorrt_runtime(runtime_source)

    with self.assertRaisesRegex(ValueError, error):
      deploy_run.load_executorch_actor(_env(obs), path)

  @parameterized.named_parameters(
      ("weights_missing", ["obs", "wrist", "camera"], True, "Could not load"),
      ("no_input_names", None, False, "no input_names"),
  )
  def test_main_loads_the_program_before_the_env(
      self, input_names, weights_in_ptd, error
  ):
    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        input_names,
        weights_in_ptd=weights_in_ptd,
    )
    if weights_in_ptd:
      (path.parent / "policy.ptd").unlink()
    flags.FLAGS.mark_as_parsed()
    with (
        flagsaver.flagsaver(checkpoint=str(path)),
        mock.patch.object(deploy_run, "create_env") as create_env,
    ):
      with self.assertRaisesRegex(ValueError, error):
        deploy_run.main([])
    create_env.assert_not_called()

  def test_uses_the_program_loaded_before_the_env(self):
    obs = self.obs
    actor = _Actor()
    path = self._export(
        actor,
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
    )
    program = deploy_run.load_executorch_program(path)
    path.unlink()

    with mock.patch.dict(sys.modules, {"rsl_rl": None, "tensordict": None}):
      policy = deploy_run.create_policy(_env(obs), path, program=program)
    (action, _), _ = policy.step(mock.Mock(observation=obs))

    torch.testing.assert_close(
        action["action"], actor(self.vector, obs["wrist"], obs["camera"])
    )

  def test_rejects_a_program_with_other_input_types(self):
    obs = self.obs
    path = self._export(
        _Actor().double(),
        (self.vector.double(), obs["wrist"].double(), obs["camera"].double()),
        ["obs", "wrist", "camera"],
    )

    with self.assertRaisesRegex(ValueError, "failed on zero observations"):
      deploy_run.load_executorch_actor(_env(obs), path)

  def test_names_the_weights_file_that_fails_to_load(self):
    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
        weights_in_ptd=True,
    )
    (path.parent / "policy.ptd").write_bytes(b"not a weights file")

    with self.assertRaisesRegex(ValueError, r"policy\.pte with policy\.ptd"):
      deploy_run.load_executorch_actor(_env(obs), path)

  def test_names_the_weights_file_when_the_program_fails_to_load(self):
    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
        weights_in_ptd=True,
    )
    other = self._export(
        torch.nn.Linear(2, 2), (torch.zeros(1, 2),), weights_in_ptd=True
    )
    shutil.copy(other.parent / "policy.ptd", path.parent / "policy.ptd")
    self._stub_tensorrt_runtime()

    with self.assertRaisesRegex(
        ValueError, r"policy\.pte with policy\.ptd"
    ) as raised:
      deploy_run.load_executorch_actor(_env(obs), path)
    self.assertIn("same kind of GPU", str(raised.exception))
    self.assertNotIn("weights file", str(raised.exception))

  def test_imports_the_tensorrt_runtime_before_loading(self):
    from executorch.runtime import Runtime  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
    )
    self._stub_tensorrt_runtime()
    get_runtime = Runtime.get

    def get():
      self.assertIn("torch_tensorrt_executorch_runtime", sys.modules)
      return get_runtime()

    with mock.patch.object(Runtime, "get", side_effect=get) as patched:
      deploy_run.load_executorch_actor(_env(obs), path)

    patched.assert_called()

  @absltest.skipUnless(
      importlib.util.find_spec("torch_tensorrt_executorch_runtime")
      and torch.cuda.is_available(),
      "Requires torch_tensorrt[executorch] and a CUDA GPU.",
  )
  def test_runs_a_tensorrt_program(self):
    import torch_tensorrt  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    obs = self.obs
    actor = _Actor().eval()
    inputs = [x.cuda() for x in (self.vector, obs["wrist"], obs["camera"])]
    compiled = torch_tensorrt.dynamo.compile(
        torch.export.export(copy.deepcopy(actor).cuda(), tuple(inputs)),
        inputs=inputs,
        min_block_size=1,
        disable_tf32=True,
    )
    directory = epath.Path(self.enterContext(tempfile.TemporaryDirectory()))
    torch_tensorrt.save(
        compiled,
        str(directory / "policy.pte"),
        output_format="executorch",
        constant_methods={"input_names": ["obs", "wrist", "camera"]},
    )
    self.assertIn(b"TensorRTBackend", (directory / "policy.pte").read_bytes())

    policy = deploy_run.create_policy(_env(obs), directory / "policy.pte")
    (action, _), _ = policy.step(mock.Mock(observation=obs))

    torch.testing.assert_close(
        action["action"], actor(self.vector, obs["wrist"], obs["camera"])
    )

  def test_rejects_a_program_with_an_image_the_env_lacks(self):
    obs = self.obs
    path = self._export(
        _Actor(),
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
    )
    del obs["wrist"]

    with mock.patch.object(
        deploy_run.pick_cube_env,
        "OBSERVATION_GROUPS",
        ("policy", "camera", "joints"),
    ):
      with self.assertRaisesRegex(ValueError, "failed on zero observations"):
        deploy_run.load_executorch_actor(_env(obs), path)

  @absltest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU.")
  def test_returns_the_action_on_the_env_device(self):
    obs = self.obs
    actor = _Actor()
    path = self._export(
        actor,
        (self.vector, obs["wrist"], obs["camera"]),
        ["obs", "wrist", "camera"],
    )

    network = deploy_run.load_executorch_actor(_env(obs, device="cuda"), path)
    action = network({k: v.cuda() for k, v in obs.items()})

    self.assertEqual(action.device.type, "cuda")
    torch.testing.assert_close(
        action.cpu(), actor(self.vector, obs["wrist"], obs["camera"])
    )


if __name__ == "__main__":
  absltest.main()
