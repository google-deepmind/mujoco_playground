import os
def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
limit_jax_mem(0.1)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jp
import tensorflow as tf
from flax import linen as nn
import numpy as np

from brax.envs import training
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from brax.training.acme import running_statistics
from brax.training.acme import specs
from pathlib import Path
from brax.training.agents.ppo import train as ppo_train


from pathlib import Path
from brax.training.agents.ppo import networks_vision as vision_ppo
import functools
from flax import linen

from tensorflow.keras import layers, Model
import numpy as np
from orbax import checkpoint as ocp
from brax.training.agents.ppo.train import _remove_pixels

import tf2onnx
import onnxruntime as rt
import argparse

from brax.training.agents.bc import networks as bc_networks
from brax.training.acme import running_statistics
from brax.training.acme import specs

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha.s2r.distillation import get_frozen_encoder_fn
from mujoco_playground.experimental.jax2onnx.aloha_nets_utils import transfer_jax_params_to_tf, TFVisionMLP

TEST_SCALE = 0.001
action_size = 14

def parse_args():
    parser = argparse.ArgumentParser(description="Script to convert from brax to Onnx network.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
    return parser.parse_args()

args = parse_args()

print(f"Checkpoint path: {args.checkpoint_path}")
# Make sure it doesn't end with 'checkpoints'.
if args.checkpoint_path.endswith("checkpoints"):
    raise ValueError("Don't end with 'checkpoints'")

ckpt_path = args.checkpoint_path
if ckpt_path.endswith("/"):
    ckpt_path = ckpt_path[:-1]
onnx_foldername = ckpt_path.split("/")[-1] # foldername is envname-date-time

onnx_base_path = Path(__file__).parent / "onnx_policies"
# Make sure it exists.
onnx_base_path.mkdir(parents=True, exist_ok=True)
onnx_base_path = onnx_base_path / onnx_foldername

# Define the observation sizes
pix_shape = (32, 32, 3)

""" 
May be different than the obs you get from the env.
The env obs already pre-applies the frozen encoder
But we need to test the frozen encoder, so we give
the obs as an internal state in the env. (only revelant to pixels/)
"""
obs_shape = {
    'has_switched': (1,),
    'pixels/rgb_l': pix_shape,
    'pixels/rgb_r': pix_shape,
    'pixels/latent_0': (64,), # For simplicity, also include the 4 latents as in the obs for checkpoint-related param init.
    'pixels/latent_1': (64,),
    'pixels/latent_2': (64,),
    'pixels/latent_3': (64,),
    'privileged': (110,),
    'proprio': (33,),
    'state': (109,),
    'state_pickup': (106,),
    'socket_hidden': (1,),
    'peg_hidden': (1,),
}

# For tf
tf_pix_shape = (1, 3, 32, 32)
state_dim = (1, 33)

# Trace / compile
class TFPolicyWrapper(tf.keras.Model):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net

    def call(self, inputs):
        # Unpack the dictionary into pixel streams and state vector.
        pixel_streams = [inputs['pixels/view_0'], inputs['pixels/view_1']]
        state = inputs['proprio']
        latents = [inputs['pixels/latent_0'], inputs['pixels/latent_1']]
        return self.policy_net(pixel_streams, states=state, depth_latent_streams=latents)

policy_hidden_layer_sizes = (256,) * 3

network_factory = functools.partial(
    bc_networks.make_bc_networks,
    policy_hidden_layer_sizes=policy_hidden_layer_sizes,
    activation=linen.relu,
    policy_obs_key='proprio',
    vision=True
)

bc_network = network_factory(
    obs_shape, action_size,
    preprocess_observations_fn=running_statistics.normalize)
make_inference_fn = bc_networks.make_inference_fn(bc_network)
encoder_fn = get_frozen_encoder_fn()

# Load encoder params for transfering to TF.
encoder_path = (mjx_env.ROOT_PATH / 
         "manipulation" / 
         "aloha" / 
         "s2r" / 
         "params" / 
         "VisionMLP2ChanCIFAR10.prms")
encoder_params = model.load_params(encoder_path)

# Initialize param structure for loading with orbax checkpointer.
dummy_obs = {k: TEST_SCALE*jp.ones(v) for k, v in obs_shape.items()}
specs_obs_shape = jax.tree_util.tree_map(
    lambda x: specs.Array(x.shape[-1:], jp.dtype('float32')), dummy_obs
)
spec_obs_shape = ppo_train._remove_pixels(specs_obs_shape)

normalizer_params = running_statistics.init_state(spec_obs_shape)

init_params = (
    normalizer_params,
    bc_network.policy_network.init(jax.random.PRNGKey(0))
)

def make_inference_fn_wrapper(params):
    inference_fn = make_inference_fn(params, deterministic=True, tanh_squash=True)
    base_inference_fn = inference_fn
    def inference_fn(jax_input, _):
        # encoder_fn adds a batch dim.
        encoder_inputs = {}
        encoder_inputs['pixels/view_0'] = jax_input['pixels/rgb_l'][0]
        encoder_inputs['pixels/view_1'] = jax_input['pixels/rgb_r'][0]

        latents = encoder_fn(encoder_inputs)
        latents = jp.split(latents[None, ...], 2, axis=-1)

        p_ins = {
            'proprio': jax_input['proprio'],
            'pixels/latent_0': latents[0], # RGB latents
            'pixels/latent_1': latents[1],
            'pixels/latent_2': jax_input['pixels/latent_2'], # Depth latents
            'pixels/latent_3': jax_input['pixels/latent_3']
        }

        return base_inference_fn(p_ins, _)
    return inference_fn


def jax_params_to_onnx(params, output_path):

    onnx_input = {
        'pixels/view_0': TEST_SCALE*np.ones(tf_pix_shape, dtype=np.float32),
        'pixels/view_1': TEST_SCALE*np.ones(tf_pix_shape, dtype=np.float32),
        'proprio': TEST_SCALE*np.ones((state_dim), dtype=np.float32),
        'pixels/latent_0': TEST_SCALE*np.ones((1, 64), dtype=np.float32),
        'pixels/latent_1': TEST_SCALE*np.ones((1, 64), dtype=np.float32)
    }

    mean = params[0].mean["proprio"]
    std  = params[0].std["proprio"]
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
    tf_policy_network = TFVisionMLP(
        layer_sizes=policy_hidden_layer_sizes + (2*action_size,), 
        normalise_channels=False, 
        layer_norm=False, 
        num_pixel_streams=2,
        state_mean_std=mean_std,
        action_size=action_size,
        latent_dense_size=64,)

    tf_policy_network = TFPolicyWrapper(tf_policy_network)
    tf_policy_network(onnx_input)[0] # Initialize.
    transfer_jax_params_to_tf((encoder_params['params'],
        params[1]['params']), tf_policy_network)

    # --- Convert to ONNX ---
    # Define the input signature for conversion.
    input_signature = [{
        'pixels/view_0': tf.TensorSpec(shape=tf_pix_shape, dtype=tf.float32, name='pixels/view_0'),
        'pixels/view_1': tf.TensorSpec(shape=tf_pix_shape, dtype=tf.float32, name='pixels/view_1'),     
        'proprio': tf.TensorSpec(shape=state_dim, dtype=tf.float32, name='proprio'),
        'pixels/latent_0': tf.TensorSpec(shape=(1, 64), dtype=tf.float32, name='pixels/latent_0'),
        'pixels/latent_1': tf.TensorSpec(shape=(1, 64), dtype=tf.float32, name='pixels/latent_1')
    }]

    tf_policy_network.output_names = ['continuous_actions']

    # Convert the model to ONNX
    # Convert using tf2onnx (opset 11 is used in this example).
    tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=input_signature,
        opset=11,
        output_path=output_path
    )

    # Run inference with ONNX Runtime
    output_names = ['continuous_actions']
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_path, providers=providers)

    # Prepare inputs for ONNX Runtime
    onnx_pred = jp.array(m.run(output_names, onnx_input)[0][0])


    jax_input = {key: TEST_SCALE*np.ones((1,) + shape) for key, shape in obs_shape.items()}
    inference_fn = make_inference_fn_wrapper(params)
    jax_pred = inference_fn(jax_input, jax.random.PRNGKey(0))[0][0] # Also returns action extras. Unbatch.

    try:
        # Assert that they're close
        assert np.allclose(jax_pred, onnx_pred, atol=1e-3)
        print("\n\n===============================")
        print("       Predictions match!      ")
        print("===============================\n\n")
    except AssertionError as e:
        print("Predictions do not match:", e)
        print("JAX prediction:", jax_pred)
        print("ONNX prediction:", onnx_pred)
        # exit
        raise e


experiment = Path(ckpt_path)
ckpts = list(experiment.glob("[!c]*"))
ckpts.sort(key=lambda x: int(x.name))
assert ckpts, "No checkpoints found"
orbax_checkpointer = ocp.PyTreeCheckpointer()


for restore_checkpoint_path in ckpts:
    checkpoint_name = restore_checkpoint_path.name
    print("######### CONVERTING CHECKPOINT #########")
    print(f"{' ' * ((40 - len(checkpoint_name)) // 2)}{checkpoint_name}{' ' * ((40 - len(checkpoint_name)) // 2)}") # Print centered.
    print("#########################################")

    params = orbax_checkpointer.restore(
        restore_checkpoint_path, item=init_params)
    jax_params_to_onnx(
        params, 
        onnx_base_path / f"{restore_checkpoint_path.name}.onnx")
