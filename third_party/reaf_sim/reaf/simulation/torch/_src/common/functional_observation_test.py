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

"""Tests for FunctionalObservationAdapter, verifying observations and specs."""

from absl.testing import absltest
from reaf.simulation.torch._src.builders.flat_scene.config import Obs
from reaf.simulation.torch._src.builders.flat_scene.config import ObsGroup
from reaf.simulation.torch._src.common import functional_observation
from reaf.simulation.torch._src.common import math
from reaf.simulation.torch._src.common import noise
import torch


class FunctionalObservationAdapterTest(absltest.TestCase):

  def test_observations_and_spec(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "policy": ObsGroup(
                terms={
                    "a": (
                        Obs(
                            "a"
                        ).configure(output_shape=(3,), dtype="float32",)
                    ),
                    "b": (
                        Obs(
                            "b"
                        ).configure(output_shape=(2,), dtype="float32",)
                    ),
                },
                enable_noise=False,
            ),
            "critic": ObsGroup(
                terms={
                    "c": (
                        Obs(
                            "c"
                        ).configure(output_shape=(5,), dtype="float32",)
                    )
                },
                enable_noise=False,
            ),
        }
    )

    spec = provider.observation_spec()
    self.assertEqual(spec["policy"].shape, (1, 5))  # 3 + 2
    self.assertEqual(spec["critic"].shape, (1, 5))

    features = {
        "a": torch.zeros(4, 3),
        "b": torch.ones(4, 2),
        "c": torch.zeros(4, 5),
    }

    obs = provider.observations_from_features(features)

    self.assertEqual(obs["policy"].shape, (4, 5))
    self.assertEqual(obs["critic"].shape, (4, 5))
    # Verify concatenation: policy should be [a, b] -> [0, 0, 0, 1, 1]
    expected_policy = torch.cat([torch.zeros(4, 3), torch.ones(4, 2)], dim=-1)
    torch.testing.assert_close(obs["policy"], expected_policy)

  def test_observations_with_noise(self):
    """Test that noise is applied when enable_noise=True."""
    term_with_noise = Obs("a").configure(
        noise=noise.Noise(
            dist=noise.Uniform(low=0.1, high=0.2),
            op="add",
        ),
        output_shape=(3,),
        dtype="float32",
    )
    term_without_noise = Obs("a").configure(
        output_shape=(3,),
        dtype="float32",
    )

    # 1. Noise enabled in group, term has noise -> should have noise
    provider_noisy = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={"t": term_with_noise},
                enable_noise=True,
            )
        },
        xpu_device="cpu",
    )
    # 2. Noise disabled in group, term has noise -> should NOT have noise
    provider_clean_override = (
        functional_observation.FunctionalObservationAdapter(
            observations={
                "obs": (
                    ObsGroup(
                        terms={"t": term_with_noise},
                        enable_noise=False,
                    )
                )
            },
            xpu_device="cpu",
        )
    )
    # 3. Noise enabled in group, term has NO noise -> should NOT have noise
    provider_clean_no_config = (
        functional_observation.FunctionalObservationAdapter(
            observations={
                "obs": (
                    ObsGroup(
                        terms={"t": term_without_noise},
                        enable_noise=True,
                    )
                )
            },
            xpu_device="cpu",
        )
    )

    features = {"a": torch.zeros(4, 3)}

    obs_noisy = provider_noisy.observations_from_features(features)["obs"]
    obs_clean_override = provider_clean_override.observations_from_features(
        features
    )["obs"]
    obs_clean_no_config = provider_clean_no_config.observations_from_features(
        features
    )["obs"]

    # Clean versions should be zero
    self.assertTrue(torch.allclose(obs_clean_override, torch.zeros(4, 3)))
    self.assertTrue(torch.allclose(obs_clean_no_config, torch.zeros(4, 3)))

    # Noisy version should be within noise bounds [0.1, 0.2]
    self.assertTrue((obs_noisy >= 0.1).all())
    self.assertTrue((obs_noisy <= 0.2).all())

  def test_scale_offset_clip(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs("a").configure(
                            scale=2.0,
                            offset=-1.0,
                            clip={"min": 0.0, "max": 0.5},
                            output_shape=(1,),
                            dtype="float32",
                        )
                    )
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.ones(2, 1)}
    obs = provider.observations_from_features(features)
    # 1.0 * 2.0 - 1.0 = 1.0 -> clamp(1.0, 0, 0.5) = 0.5
    torch.testing.assert_close(obs["obs"], torch.full((2, 1), 0.5))

  def test_history(self):
    nworld = 2
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs("a").configure(
                            history_length=2,
                            output_shape=(3,),
                            dtype="float32",
                        )
                    )
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
        nworld=nworld,
    )

    # Spec should be (1 + 2) * 3 = 9.
    spec = provider.observation_spec()
    self.assertEqual(spec["obs"].shape, (nworld, 9))

    # Step 0: history filled with current value.
    features = {"a": torch.ones(nworld, 3)}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.ones(nworld, 9))

    # Step 1: current=[2,2,2], history=[1,1,1, 1,1,1]
    features = {"a": torch.full((nworld, 3), 2.0)}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"][:, :3], torch.full((nworld, 3), 2.0))
    torch.testing.assert_close(obs["obs"][:, 3:6], torch.ones(nworld, 3))
    torch.testing.assert_close(obs["obs"][:, 6:], torch.ones(nworld, 3))

  def test_image_history(self):
    nworld = 2
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "camera": ObsGroup(
                terms={
                    "camera": (
                        Obs("rgb").configure(
                            history_length=2,
                            output_shape=(3, 32, 32),
                            dtype="float32",
                        )
                    )
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
        nworld=nworld,
    )

    # Spec should be (1 + 2) * 3 = 9 channels -> (9, 32, 32).
    spec = provider.observation_spec()
    self.assertEqual(spec["camera"].shape, (nworld, 9, 32, 32))

    # Step 0: history filled with current value (all 9 channels are 1.0).
    features = {"rgb": torch.ones(nworld, 3, 32, 32)}
    obs = provider.observations_from_features(features)
    self.assertEqual(obs["camera"].shape, (nworld, 9, 32, 32))
    torch.testing.assert_close(obs["camera"], torch.ones(nworld, 9, 32, 32))

    # Step 1: current=2.0 (channels 0:3), past=1.0 (channels 3:9).
    features = {"rgb": torch.full((nworld, 3, 32, 32), 2.0)}
    obs = provider.observations_from_features(features)
    self.assertEqual(obs["camera"].shape, (nworld, 9, 32, 32))
    torch.testing.assert_close(
        obs["camera"][:, :3], torch.full((nworld, 3, 32, 32), 2.0)
    )
    torch.testing.assert_close(
        obs["camera"][:, 3:], torch.ones(nworld, 6, 32, 32)
    )

  def test_scale_vectorized(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs("a").configure(
                            scale=[1.0, 2.0, 3.0],
                            output_shape=(3,),
                        )
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.ones(2, 3)}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.tensor([[1.0, 2.0, 3.0]] * 2))

  def test_offset(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs(
                            "a"
                        ).configure(offset=0.5, output_shape=(3,),)
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.ones(2, 3)}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.full((2, 3), 1.5))

  def test_clip(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs("a").configure(
                            clip={"min": -1.0, "max": 1.0},
                            output_shape=(3,),
                        )
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.tensor([[5.0, -5.0, 0.5]])}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.tensor([[1.0, -1.0, 0.5]]))

  def test_clip_vectorized(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs("a").configure(
                            clip={
                                "min": [-1.0, -2.0, -3.0],
                                "max": [1.0, 2.0, 3.0],
                            },
                            output_shape=(3,),
                        )
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.tensor([[5.0, -5.0, 0.5]])}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.tensor([[1.0, -2.0, 0.5]]))

  def test_clip_max_only(self):
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs(
                            "a"
                        ).configure(clip={"max": 1.0}, output_shape=(3,),)
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.tensor([[5.0, -5.0, 0.5]])}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.tensor([[1.0, -5.0, 0.5]]))

  def test_history_reset(self):
    """Test that reset() fills history with current obs for reset envs."""
    nworld = 4
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs(
                            "a"
                        ).configure(history_length=1, output_shape=(2,),)
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
        nworld=nworld,
    )

    # Fill history with a step.
    features = {"a": torch.ones(nworld, 2)}
    provider.observations_from_features(features)

    # Reset envs 0 and 2 only.
    provider.reset(torch.tensor([True, False, True, False]))

    # Next step: reset envs get history filled with current obs (5.0);
    # non-reset envs keep their previous history (1.0).
    features = {"a": torch.full((nworld, 2), 5.0)}
    obs = provider.observations_from_features(features)
    # History portion (last 2 values of dim=-1).
    history = obs["obs"][:, 2:]
    torch.testing.assert_close(history[0], torch.full((2,), 5.0))  # filled
    torch.testing.assert_close(history[1], torch.ones(2))  # kept
    torch.testing.assert_close(history[2], torch.full((2,), 5.0))  # filled
    torch.testing.assert_close(history[3], torch.ones(2))  # kept

  def test_history_reset_no_zeros(self):
    """First obs after reset contains no zeros — history filled with current."""
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs(
                            "a"
                        ).configure(history_length=2, output_shape=(2,),)
                    )
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
        nworld=1,
    )
    provider.reset(None)
    obs = provider.observations_from_features({"a": torch.full((1, 2), 3.0)})
    # All 6 values (current + 2 history slots) should be 3.0, not zero.
    torch.testing.assert_close(obs["obs"], torch.full((1, 6), 3.0))

  def test_plain_term_unchanged(self):
    """Plain Obs (no scale/clip/history) works as expected."""
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "obs": ObsGroup(
                terms={
                    "t": (
                        Obs(
                            "a"
                        ).configure(output_shape=(3,),)
                    ),
                },
                enable_noise=False,
            )
        },
        xpu_device="cpu",
    )
    features = {"a": torch.ones(2, 3)}
    obs = provider.observations_from_features(features)
    torch.testing.assert_close(obs["obs"], torch.ones(2, 3))

  def test_episodic_bias_lifecycle_and_validation(self):
    """Per-episode bias supports per-dim bounds, scale ordering, 1D terms, and partial resets."""
    nworld = 4
    low = torch.tensor([0.1, 1.0, 10.0])
    high = torch.tensor([0.2, 2.0, 20.0])
    add_term = Obs(lambda f: f["a"]).configure(
        bias=noise.Noise(dist=noise.Uniform(low=low, high=high)),
        scale=2.0,
        output_shape=(3,),
    )
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "policy": ObsGroup(terms={"t": add_term}, enable_noise=True),
            "critic": ObsGroup(terms={"t": add_term}, enable_noise=False),
            "multi_1d": ObsGroup(
                terms={
                    "scalar": (
                        Obs(lambda f: f["a"][:, 0]).configure(
                            bias=noise.uniform(2.0, 3.0, op="scale"),
                            output_shape=(1,),
                        )
                    ),
                    "vec": (
                        Obs(
                            lambda f: f["a"][:, 1:3]
                        ).configure(output_shape=(2,),)
                    ),
                },
                enable_noise=True,
            ),
        },
        xpu_device="cpu",
        nworld=nworld,
    )
    # Reset before first step works with per-dimension tensor bounds.
    provider.reset(torch.tensor([True, False, True, False]))
    features = {"a": torch.ones(nworld, 3)}
    obs0 = provider.observations_from_features(features)
    obs1 = provider.observations_from_features(features)

    # Critic (enable_noise=False) skips bias, only applies scale=2.0.
    torch.testing.assert_close(obs0["critic"], torch.full((nworld, 3), 2.0))
    # Policy applies bias before scale: (1.0 + bias) * 2.0 in [2*(1+low), 2*(1+high)].
    self.assertTrue(
        (obs0["policy"] >= 2.0 * (1.0 + low)).all()
        and (obs0["policy"] <= 2.0 * (1.0 + high)).all()
    )
    torch.testing.assert_close(obs0["policy"], obs1["policy"])

    # 1D term output (nworld,) with output_shape=(1,) reshapes cleanly without broadcasting to (nworld, nworld).
    self.assertEqual(obs0["multi_1d"].shape, (nworld, 3))
    self.assertTrue(
        (obs0["multi_1d"][:, 0] >= 2.0).all()
        and (obs0["multi_1d"][:, 0] <= 3.0).all()
    )

    # Partial reset resamples only masked worlds; repeated resets do not accumulate.
    provider.reset(torch.tensor([True, False, True, False]))
    obs2 = provider.observations_from_features(features)
    torch.testing.assert_close(obs2["policy"][1], obs0["policy"][1])
    torch.testing.assert_close(obs2["policy"][3], obs0["policy"][3])
    self.assertFalse(torch.allclose(obs2["policy"][0], obs0["policy"][0]))
    for _ in range(50):
      provider.reset()
    obs50 = provider.observations_from_features(features)["policy"]
    self.assertTrue(
        (obs50 >= 2.0 * (1.0 + low)).all()
        and (obs50 <= 2.0 * (1.0 + high)).all()
    )

    # Validation: nworld is required when bias is enabled.
    with self.assertRaisesRegex(ValueError, "nworld must be provided"):
      functional_observation.FunctionalObservationAdapter(
          observations={
              "p": ObsGroup(terms={"t": add_term}, enable_noise=True)
          },
          xpu_device="cpu",
          nworld=None,  # pyrefly: ignore[bad-argument-type]
      )

  def test_episodic_rotate_bias_and_step_noise(self):
    torch.manual_seed(7)
    nworld = 4
    provider = functional_observation.FunctionalObservationAdapter(
        observations={
            "quat": ObsGroup(
                terms={
                    "t": (
                        Obs(lambda f: f["q"]).configure(
                            bias=noise.Noise(
                                dist=noise.SO3Uniform(low=-0.2, high=0.2)
                            ),
                            output_shape=(4,),
                        )
                    )
                },
                enable_noise=True,
            ),
            "rot6d": ObsGroup(
                terms={
                    "t": (
                        Obs(lambda f: f["r6"]).configure(
                            bias=noise.Noise(
                                dist=noise.SO3Gaussian(mean=0.0, std=0.1)
                            ),
                            output_shape=(6,),
                        )
                    )
                },
                enable_noise=True,
            ),
            "quat_bias_plus_noise": ObsGroup(
                terms={
                    "t": (
                        Obs(lambda f: f["q"]).configure(
                            bias=noise.Noise(
                                dist=noise.SO3Uniform(low=-0.2, high=0.2)
                            ),
                            noise=noise.Noise(
                                dist=noise.SO3Uniform(low=-0.02, high=0.02)
                            ),
                            output_shape=(4,),
                        )
                    )
                },
                enable_noise=True,
            ),
        },
        xpu_device="cpu",
        nworld=nworld,
    )
    q_in = noise.sample_so3_uniform(
        (nworld, 4), torch.device("cpu"), low=-1.0, high=1.0
    )
    features = {
        "q": q_in,
        "r6": math.rot_matrix_to_6d(math.quat_to_matrix(q_in)),
    }
    obs0 = provider.observations_from_features(features)
    obs1 = provider.observations_from_features(features)

    torch.testing.assert_close(obs0["quat"], obs1["quat"])
    torch.testing.assert_close(obs0["rot6d"], obs1["rot6d"])
    self.assertFalse(
        torch.allclose(
            obs0["quat_bias_plus_noise"], obs1["quat_bias_plus_noise"]
        )
    )
    torch.testing.assert_close(
        torch.linalg.norm(obs0["quat_bias_plus_noise"], dim=-1),
        torch.ones(nworld),
    )

    provider.reset(torch.tensor([True, False, True, False]))
    obs2 = provider.observations_from_features(features)
    torch.testing.assert_close(obs2["quat"][1], obs0["quat"][1])
    self.assertFalse(torch.allclose(obs2["quat"][0], obs0["quat"][0]))
    torch.testing.assert_close(obs2["rot6d"][1], obs0["rot6d"][1])
    self.assertFalse(torch.allclose(obs2["rot6d"][0], obs0["rot6d"][0]))


if __name__ == "__main__":
  absltest.main()
