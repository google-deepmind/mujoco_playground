import os
os.environ['MUJOCO_GL'] = 'egl'  # Set the environment variable for EGL rendering

import numpy as np
import matplotlib.pyplot as plt
import datetime

from datetime import datetime
import functools
import os
import mediapy as media
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
#from brax.training.agents.sac import networks as sac_networks
#from brax.training.agents.sac import train as sac
from etils import epath
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from mujoco_playground.experimental.x02_walking.convert_to_onnx import conv_to_onnx
from mujoco_playground._src.gait import draw_joystick_command

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
import mujoco


def parse_kv(s):
    key, value = s.split('=')
    if '.' in value:
      return key, float(value)
    else:
      return key, int(value)

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
parser = ArgumentParser(description="Train a walking agent with joystick control.")
parser.add_argument('-n', '--run-name', type=str, required=True, help='Name of the run for saving parameters')
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU device to use')
parser.add_argument('-e', '--env', type=str, default='X02JoystickFlatTerrain')
parser.add_argument('-w', '--wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('-c', '--config', nargs="+", type=parse_kv, help='Overwrites for default configuration of environment' )
parser.add_argument('-C', "--rl-config", nargs="+", type=parse_kv, help='Overwrites for default configuration of RL algorithm' )
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config_overrides = dict(args.config) if args.config else {}
rl_config_overrides = dict(args.rl_config) if args.rl_config else {}

ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
ckpt_path.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=3, suppress=True, linewidth=100)

env_name = args.env
env = registry.load(env_name, config_overrides=config_overrides)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)
ppo_params.update(rl_config_overrides)


if args.wandb:
  import wandb
  run = wandb.init(project="mujoco_playground", 
                   entity="bitbots",
                   name=args.run_name,
                   #dir=ckpt_path,
                   config={
                          "env": args.env,
                          "gpu": args.gpu,
                          "run_name": args.run_name,} | dict(ppo_params) | dict(env_cfg) | config_overrides,)
else:
  from mujoco_playground.experimental.utils.plotting import TrainingPlotter
  plotter = TrainingPlotter(max_timesteps=ppo_params.num_timesteps, figsize=(15, 10))

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

pbar = tqdm(total=ppo_params.num_timesteps, desc="Training Progress", unit="steps", dynamic_ncols=True)

def save_params(ckpt_path, params, step=-1):
  normalizer_params, policy_params, value_params = params
  filename = ckpt_path / f"params_{step:012}.pkl" if step >= 0 else ckpt_path / "params.pkl"
  with open(filename, "wb") as f:
    data = {
      "normalizer_params": normalizer_params,
      "policy_params": policy_params,
      "value_params": value_params,
    }
    pickle.dump(data, f)

def progress(num_steps, metrics):
  times.append(datetime.now())
  if args.wandb:
    metrics_name_replaced = {k.replace("eval/episode_reward/", "rewards/"): v for k, v in metrics.items()}
    for k, v in metrics.items():
      if "rewards" in k and "_std" in k:
        new_k = k.replace("rewards", "rewards_std")
        print(f'replace {k} with {new_k}')
        del(metrics_name_replaced[k])
        metrics_name_replaced[new_k] = v
    wandb.log(metrics_name_replaced, step=num_steps)
  else:
    plotter.update(num_steps, metrics)
    plotter.save_figure(ckpt_path / f"ppo_training_progress_plots_{num_steps:012}.png")
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    fig, ax = plt.subplots(1,1)
    ax.set_ylim([0, ppo_params["num_timesteps"] * 1.25])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"y={y_data[-1]:.3f}")
    ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    fig.savefig(ckpt_path / f"ppo_training_progress_{num_steps:012}.png", dpi=300, bbox_inches='tight')
  pbar.update(num_steps - pbar.n)

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)


network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )



def policy_params_fn(current_step, make_policy, params):
  del make_policy  # Unused.
  save_params(ckpt_path, params, current_step)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    #save_checkpoint_path=checkpoint_path,
    policy_params_fn=policy_params_fn,
)
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg, config_overrides=config_overrides),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

print("Rendering Video")

eval_env = registry.load(env_name, config_overrides=config_overrides)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

commands = jp.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [-0.5, 0.0, 0.0],
                     [0.0, 0.4, 0.0],
                     [0.0, -0.4, 0.0],
                     [0.0, 0.0, 2.0],
                     [0.0, 0.0, -2.0],])
phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase = jp.array([0, jp.pi])

for j in range(commands.shape[0]):
  print(f"episode {j}")
  state = jit_reset(rng)
  state.info["phase_dt"] = phase_dt
  state.info["phase"] = phase
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    if state.done:
      break
    state.info["command"] = commands[j]
    rollout.append(state)

    xyz = np.array(state.data.xpos[eval_env.mj_model.body("pelvis_link").id])
    xyz += np.array([0, 0.0, 0])
    x_axis = state.data.xmat[eval_env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    modify_scene_fns.append(
        functools.partial(
            draw_joystick_command,
            cmd=state.info["command"],
            xyz=xyz,
            theta=yaw,
            scl=1.0,
        )
    )

render_every = 2
fps = 1.0 / eval_env.dt / render_every
print(f"fps: {fps}")
traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

frames = eval_env.render(
    traj,
    camera="track",
    scene_option=scene_option,
    width=640,
    height=480*2,
    modify_scene_fns=mod_fns,
)
media.write_video(ckpt_path / f"{args.run_name}_eval.mp4", frames, fps=fps)
if args.wandb:
    wandb.log({"video": wandb.Video(str(ckpt_path / f"{args.run_name}_eval.mp4"), fps=fps, format="mp4")})

save_params(ckpt_path, params)
# conv_to_onnx(ckpt_path / "params.pkl", f"{args.run_name}.onnx", env_name)