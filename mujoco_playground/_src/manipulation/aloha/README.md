### Quickstart


**Pre-requisites**

- *Handover, Pick, Peg Insertion:* The standard Playground setup
- *Behaviour Cloning for Peg Insertion:* Madrona MJX
- *Jax-to-ONNX Conversion:* Onnx, Tensorflow, tf2onnx

```bash
# Train Aloha Handover. Documentation at https://github.com/google-deepmind/mujoco_playground/pull/29
python learning/train_jax_ppo.py --env_name AlohaHandOver
```

```bash
# Plots for pick and peg-insertion at https://github.com/google-deepmind/mujoco_playground/pull/76
cd <PATH_TO_YOUR_CLONE>
export PARAMS_PATH=mujoco_playground/_src/manipulation/aloha/params

# Train a single arm to pick up a cube.
python learning/train_jax_ppo.py --env_name AlohaPick --domain_randomization --norender_final_policy --save_params_path $PARAMS_PATH/AlohaPick.prms
sleep 0.5

# Train a biarm to insert a peg into a socket. Requires above policy.
python learning/train_jax_ppo.py --env_name AlohaPegInsertion --save_params_path $PARAMS_PATH/AlohaPegInsertion.prms
sleep 0.5

# Train a student policy to insert a peg into a socket using *pixel inputs*. Requires above policy.
python mujoco_playground/experimental/bc_peg_insertion.py --domain-randomization --num-evals 0 --print-loss

# Convert checkpoints from the above run to ONNX for easy robot deployment.
# ONNX policies are written to `experimental/jax2onnx/onnx_policies`.
python mujoco_playground/experimental/jax2onnx/aloha_nets_to_onnx.py --checkpoint_path <YOUR_DISTILL_CHECKPOINT_DIR>
```

### Sim-to-Real Transfer of a Bi-Arm RL Policy via Pixel-Based Behaviour Cloning

https://github.com/user-attachments/assets/205fe8b9-1773-4715-8025-5de13490d0da

---

**Distillation**

In this module, we demonstrate policy distillation: a straightforward method for deploying a simulation-trained reinforcement learning policy that initially uses privileged state observations (such as object positions). The process involves two steps: 

1. **Teacher Policy Training:** A state-based teacher policy is trained using RL.
2. **Student Policy Distillation:** The teacher is then distilled into a student policy via behaviour cloning (BC), where the student learns to map its observations $o_s(x)$ (e.g., exteroceptive RGBD images) to the teacher’s deterministic actions $\pi_t(o_t(x))$. For example, while both policies observe joint angles, the student uses RGBD images, whereas the teacher directly accesses (noisy) object positions.

The distillation process—where the student uses left and right wrist-mounted RGBD cameras for exteroception—takes about **3 minutes** on an RTX4090. This rapid turnaround is due to three factors:

1. [Very fast rendering](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/experimental/madrona_benchmarking/figures/cartpole_benchmark_full.png) provided by Madrona MJX.
2. The sample efficiency of behaviour cloning.
3. The use of low-resolution (32×32) rendering, which is sufficient for precise alignment given the wrist camera placement.

For further details on the teacher policy and RGBD sim-to-real techniques, please refer to the [technical report](https://docs.google.com/presentation/d/1v50Vg-SJdy5HV5JmPHALSwph9mcVI2RSPRdrxYR3Bkg/edit?usp=sharing).

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

**Frozen Encoders**

*VisionMLP2ChanCIFAR10_OCP* is an Orbax checkpoint of [NatureCNN](https://github.com/google/brax/blob/241f9bc5bbd003f9cfc9ded7613388e2fe125af6/brax/training/networks.py#L153) (AtariCNN) pre-trained on CIFAR10 to achieve over 70% classification accuracy. We omit the supervised training code, see [this tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial5/Inception_ResNet_DenseNet.html) for reference.

---

**Aloha Deployment Setup**

For deployment, the ONNX policy is executed on the Aloha robot using a custom fork of [OpenPI](https://github.com/Physical-Intelligence/openpi) along with the Interbotix Aloha ROS packages. Acknowledgements to Kevin Zakka, Laura Smith and the Levine Lab for robot deployment setup!
