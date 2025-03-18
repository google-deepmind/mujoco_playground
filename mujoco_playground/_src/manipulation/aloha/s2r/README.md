### Sim-to-Real Transfer of a Bi-Arm RL Policy via Pixel-Based Behaviour Cloning

https://github.com/user-attachments/assets/205fe8b9-1773-4715-8025-5de13490d0da

---

**Distillation**

In this module, we demonstrate policy distillation—a straightforward method for deploying a simulation-trained reinforcement learning policy that initially uses privileged state observations (such as object positions). The process involves two steps: 

1. **Teacher Policy Training:** A state-based teacher policy is trained using RL.
2. **Student Policy Distillation:** The teacher is then distilled into a student policy via behaviour cloning (BC), where the student learns to map its observations $o_s(x)$ (e.g., exteroceptive RGBD images) to the teacher’s deterministic actions $\pi_t(o_t(x))$. For example, while both policies observe joint angles, the student uses RGBD images, whereas the teacher directly accesses (noisy) object positions.

The distillation process—where the student uses left and right wrist-mounted RGBD cameras for exteroception—takes about **3 minutes** on an RTX4090. This rapid turnaround is due to three factors:

1. [Very fast rendering](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/experimental/madrona_benchmarking/figures/cartpole_benchmark_full.png) provided by Madrona MJX.
2. The sample efficiency of behaviour cloning.
3. The use of low-resolution (32×32) rendering, which is sufficient for precise alignment given the wrist camera placement.

For further details on the teacher policy and RGBD sim-to-real techniques, please refer to the [technical report](https://docs.google.com/presentation/d/1v50Vg-SJdy5HV5JmPHALSwph9mcVI2RSPRdrxYR3Bkg/edit?usp=sharing).

---

**Teacher Policy**

This module includes an alternative state-based peg insertion implementation compared to the [single-file implementation](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/aloha/single_peg_insertion.py). The two main differences are:

1. **Robustness Enhancements:** We add observation noise and random action delays so that the teacher policy does not act with an unrealistic degree of certainty.
2. **Phased Training:** To better control reward shaping, the training is split into two phases:
   - **Phase 1:** Bringing the objects to a pre-insertion position using a single-arm pick policy trained to pick up randomly-sized blocks, which is then deployed on both arms to ensure symmetry.
   - **Phase 2:** The actual peg insertion phase.

---

**A Note on Sample Efficiency**

Behaviour cloning (BC) can be orders of magnitude more sample-efficient than reinforcement learning. In our approach, we use an L2 loss defined as:

$|| \pi_s(o_s(x)) - \pi_t(o_t(x)) ||$

In contrast, the policy gradient in RL generally takes the form:

![Equation](https://latex.codecogs.com/svg.latex?\nabla_\theta%20J(\theta)%20=%20\mathbb{E}_{\tau%20\sim%20\theta}%20\left[\sum_t%20\nabla_\theta%20\log%20\pi_\theta(a_t%20|%20s_t)%20R(\tau)\right])

Two key observations highlight why BC’s direct supervision is more efficient:

- **Explicit Loss Signal:** The BC loss compares against the teacher action, giving explicit feedback on how the action should be adjusted. In contrast, the policy gradient only provides directional guidance, instructing the optimizer to increase or decrease an action’s likelihood based solely on its downstream rewards.
- **Per-Dimension Supervision:** While the policy gradient applies a uniform weighting across all action dimensions, BC supplies per-dimension information, making it easier to scale to high-dimensional action spaces.

---

### Training Instructions

**Pre-requisites**

- **Teacher Phases 1 and 2:** Require only the standard Playground setup.
- **Distillation:** Requires the installation of Madrona MJX.
- **Jax-to-ONNX Conversion:** Requires several additional Python packages. The script `aloha_nets_to_onnx` converts all checkpoints into ONNX policies under `experimental/jax2onnx/onnx_policies`.

**Running the Training**

```bash
cd <PATH_TO_YOUR_CLONE>
export PARAMS_PATH=mujoco_playground/_src/manipulation/aloha/s2r/params

# Teacher phase 1 (Note: Domain randomization disables rendering)
python learning/train_jax_ppo.py --env_name AlohaS2RPick --domain_randomization --norender_final_policy --save_params_path $PARAMS_PATH/AlohaS2RPick.prms
sleep 0.5

# Teacher phase 2
python learning/train_jax_ppo.py --env_name AlohaS2RPegInsertion --save_params_path $PARAMS_PATH/AlohaS2RPegInsertion.prms
sleep 0.5

# Distill to student (skip evaluations to save time)
python mujoco_playground/experimental/train_dagger.py --domain-randomization --num-evals 0 --print-loss

# Bonus: Convert to ONNX for easy deployment
python mujoco_playground/experimental/jax2onnx/aloha_nets_to_onnx.py --checkpoint_path <YOUR_DISTILL_CHECKPOINT_DIR>
```

The expected training times on an RTX4090 are as follows, with a total training time of approximately **23.5 minutes**:

| Phase                     | Time (jit) | Time (run) | Algorithm           | Inputs |
|---------------------------|------------|------------|---------------------|--------|
| **Teacher phase 1**       | 1.5 min   | 8.5 min    | PPO                 | State  |
| **Teacher phase 2**       | 1.5 min   | 8.5 min    | PPO                 | State  |
| **Student Distillation**  | 24 s      | 2 min 55 s | Behaviour Cloning   | Pixels |

---

### Pre-trained Parameters

All pre-trained parameters are stored in `_src/manipulation/aloha/s2r/params`.

| File                         | Description                                                                                                                                           |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **AlohaS2RPick.prms**        | Used during teacher phase 2 training and distillation. A default policy (from `pick.py`) is provided.                                                 |
| **AlohaS2RPegInsertion.prms**| Used during distillation. A default policy (from `peg_insertion.py`) is provided.                                                                    |
| **VisionMLP2ChanCIFAR10.prms** | Based on [NatureCNN](https://github.com/google/brax/blob/241f9bc5bbd003f9cfc9ded7613388e2fe125af6/brax/training/networks.py#L153), also known as AtariCNN. This model is pre-trained on CIFAR10 to achieve over 70% classification accuracy. |

*Note:* The standard CIFAR10 pre-training code is not included. For reference, see [this tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial5/Inception_ResNet_DenseNet.html).

---

### XML Additions

Modifications have been made to the original XML files (located in `_src/manipulation/aloha/xmls/`) to support training and visually match the physical robot. These files are now stored under `_src/manipulation/aloha/xmls/s2r`.

| File                  | Description                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| **mjx_aloha.xml**     | Adjusts lighting, adds additional sites, and hides wrist cameras from the batch renderer.                         |
| **mjx_scene.xml**     | Removes certain lights and adjusts the table color.                                                              |
| **mjx_peg_insertion.xml** | Adds additional sites for improved simulation accuracy.                                                      |
| **mjx_aloha_single.xml** | Removes the right arm from the simulation.                                                                     |
| **mjx_scene_single.xml** | Loads the single-arm configuration defined in `mjx_aloha_single.xml`.                                           |
| **mjx_pick.xml**      | Contains a single block for pick tasks.                                                                            |

---

### Aloha Deployment Setup

For deployment, the ONNX policy is executed on the Aloha robot using a custom fork of [OpenPI](https://github.com/Physical-Intelligence/openpi) along with the Interbotix Aloha ROS packages. Acknowledgements to Kevin Zakka, Laura Smith and the Levine Lab for robot deployment setup!
