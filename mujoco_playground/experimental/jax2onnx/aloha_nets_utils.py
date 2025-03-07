import jax.numpy as jp
import tensorflow as tf
from flax import linen as nn
import numpy as np

from brax.envs import training
from brax.io import model as brax_loader
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import onnx
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


import tensorflow as tf
from tensorflow.keras import layers


class TFVisionMLP(tf.keras.Model):
    """
    Applies multiple CNN backbones (one per pixel stream) named NatureCNN_i,
    each containing a single sub-block CNN_0 with Conv_0..Conv_2.
    After the CNN output is pooled and projected via Dense_i,
    we combine everything and pass it to MLP_0.
    
    When aux_loss is True, an additional auxiliary branch is computed from
    the concatenation of the raw (pooled) CNN outputs (before the Dense projection)
    via a Dense(6) layer. The aux output is then concatenated with the policy head.
    """
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_initializer="lecun_uniform",
        activate_final=False,
        layer_norm=False,
        normalise_channels=False,
        num_pixel_streams=2,
        state_mean_std=None,  # (mean, std) for normalizing the state vector
        action_size=2,        # for demonstration
        name="tf_vision_mlp",
        latent_dense_size=128,
    ):
        super().__init__(name=name)
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.normalise_channels = normalise_channels
        self.num_pixel_streams = num_pixel_streams
        self.action_size = action_size

        # Store mean and std for state normalization if provided
        if state_mean_std is not None:
            self.mean_state = tf.Variable(state_mean_std[0],
                                          trainable=False, dtype=tf.float32)
            self.std_state = tf.Variable(state_mean_std[1],
                                         trainable=False, dtype=tf.float32)
        else:
            self.mean_state = None
            self.std_state = None

        # ---------------------------------------------------------------------
        # 1) Build the CNN blocks, each named "NatureCNN_i".
        #    Inside each "NatureCNN_i", create a sub-block "CNN_0"
        #    with 3 Conv layers named "Conv_0", "Conv_1", "Conv_2".
        # ---------------------------------------------------------------------
        self.cnn_blocks = []
        self.downstream_denses = []
        for i in range(self.num_pixel_streams):
            nature_cnn_block = tf.keras.Sequential(name=f"CNN_{i}")
            nature_cnn_block.add(
                layers.Conv2D(32, (8, 8), strides=(4, 4), 
                              activation=self.activation, use_bias=False,
                              kernel_initializer=self.kernel_initializer,
                              name="Conv_0",
                              padding='same')
            )
            nature_cnn_block.add(
                layers.Conv2D(64, (4, 4), strides=(2, 2),
                              activation=self.activation, use_bias=False,
                              kernel_initializer=self.kernel_initializer,
                              name="Conv_1",
                              padding='same')
            )
            nature_cnn_block.add(
                layers.Conv2D(64, (3, 3), strides=(1, 1),
                              activation=self.activation, use_bias=False,
                              kernel_initializer=self.kernel_initializer,
                              name="Conv_2",
                              padding='same')
            )

            # Add the sub-block to the "NatureCNN_i" block
            self.cnn_blocks.append(nature_cnn_block)

            # 2) Each CNN output is projected to 128 dims via a Dense_i
            proj_dense = layers.Dense(
                latent_dense_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name=f"Dense_{i}"
            )

            self.downstream_denses.append(proj_dense)

        # Append two more blocks to downstream_denses.
        for j in range(1, 3): # TODO: less sketchy way to ensure correct names.
            proj_dense = layers.Dense(
                latent_dense_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name=f"Dense_{i+j}"
            )
            self.downstream_denses.append(proj_dense)


        # ---------------------------------------------------------------------
        # 3) Build the MLP block, named "MLP_0", containing hidden_0..hidden_n.
        # ---------------------------------------------------------------------
        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=self.activation, 
                kernel_initializer=self.kernel_initializer, 
                name=f"hidden_{i}"
            )
            self.mlp_block.add(dense_layer)

            # Optionally add layer normalization after each hidden
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))

        # Remove activation from the final Dense if activate_final is False
        if not self.activate_final and len(self.mlp_block.layers) > 0:
            last_layer = self.mlp_block.layers[-1]
            if hasattr(last_layer, 'activation') and (last_layer.activation is not None):
                last_layer.activation = None

    def normalise_image_channels(self, x):
        """
        Apply per-channel normalization over the spatial dimensions (H, W).
        Matches JAX linen.LayerNorm(reduction_axes=(-1, -2)) for NHWC format:
        compute mean/var over H,W for each N,C.
        """
        # x shape: [N, H, W, C]
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + 1e-5)
        return x

    def call(self, pixel_streams, states=None, depth_latent_streams=None, training=False):
        """
        Forward pass through the model.

        Args:
            pixel_streams: A list (or single Tensor) of pixel inputs,
                each shape [N, C, H, W].
            states: A Tensor of shape [N, k], representing additional state info.
        """
        # If a single tensor is provided, wrap it in a list
        if isinstance(pixel_streams, tf.Tensor):
            pixel_streams = [pixel_streams]

        if len(pixel_streams) != self.num_pixel_streams:
            raise ValueError(f"Expected {self.num_pixel_streams} pixel streams, "
                             f"but got {len(pixel_streams)}")

        # Per-channel normalization if enabled: convert to NHWC format first.
        if self.normalise_channels:
            pixel_streams = [
                self.normalise_image_channels(
                    tf.keras.layers.Permute((2, 3, 1))(x)
                )
                for x in pixel_streams
            ]
        else:
            # Just permute to [N, H, W, C] if not normalizing
            pixel_streams = [
                tf.keras.layers.Permute((2, 3, 1))(x) for x in pixel_streams
            ]

        projected_cnn_outs = []     # For the policy head
        for i, x in enumerate(pixel_streams):
            # Pass through "NatureCNN_i"
            x = self.cnn_blocks[i](x, training=training)
            # Global average pool over H,W; shape becomes [N, filters]
            x_pooled = tf.reduce_mean(x, axis=[1, 2])
            x_proj = self.downstream_denses[i](x_pooled)
            projected_cnn_outs.append(x_proj)

        ##### ORDER ####
        # RGB | DEPTH | Proprio
        ################
        if depth_latent_streams is not None:
            assert len(depth_latent_streams) == 2, "depth_latent_streams must have 2 streams."
            for j, x in enumerate(depth_latent_streams):
                x_proj = self.downstream_denses[self.num_pixel_streams + j](x)
                projected_cnn_outs.append(x_proj)
        
        # Optionally normalize states and append them.
        if states is not None:
            if (self.mean_state is not None) and (self.std_state is not None):
                # import ipdb; ipdb.set_trace()
                proprio_size = (states.shape[0], 33)
                assert states.shape == proprio_size
                assert self.mean_state.shape == (33,)
                assert self.std_state.shape == (33,)
                states = (states - self.mean_state) / (self.std_state)
            projected_cnn_outs.append(states)

        # Concatenate projected CNN outputs (and state if provided)
        hidden = tf.concat(projected_cnn_outs, axis=-1)
        hidden = self.mlp_block(hidden)

        # Compute the policy head output.
        # (Here we assume the MLP produces an output of size 2*action_size,
        # and we use the first half (after tanh) as the policy output.)
        # Adjust this postprocessing as needed.
        if hidden.shape[-1] != 2 * self.action_size:
            raise ValueError(f"Expected hidden dim to be {2*self.action_size}, got {hidden.shape[-1]}")

        return tf.math.tanh(hidden[:, :self.action_size])


def transfer_jax_params_to_tf(jax_params: tuple[dict, dict], _tf_model: tf.keras.Model):
    tf_model = _tf_model.get_layer(name='tf_vision_mlp')
    
    encoder_params, policy_params = jax_params
    # First, transfer the encoder params.
    for block_name, block_content in encoder_params.items(): # keys = {Nature_CNN_i}
        try:
            # e.g. block_name = "NatureCNN_0" or "Dense_0" or "MLP_0"
            tf_block = tf_model.get_layer(name=block_name)
        except ValueError:
            print(f"[Warning] No layer named '{block_name}' found in TF model.")
            # Available layers:
            print("Available layers:", [l.name for l in tf_model.layers])
            continue

        # ---------------------------------------------------------------------
        # CASE 1: A top-level NatureCNN_i block, which is a tf.keras.Sequential
        #         containing a sub-block named "CNN_0".
        # ---------------------------------------------------------------------
        if block_name.startswith("NatureCNN_") and isinstance(tf_block, tf.keras.Sequential):
            for sub_block_name, sub_block_layers in block_content.items():
                # sub_block_name should be "CNN_0"
                try:
                    tf_sub_block = tf_block.get_layer(name=sub_block_name)
                except ValueError:
                    print(f"  [Warning] No sub-layer '{sub_block_name}' in '{block_name}'.")
                    continue

                # Now sub_block_layers might be {"Conv_0": {kernel/bias}, "Conv_1": ..., ...}
                for layer_name, layer_params in sub_block_layers.items():
                    try:
                        tf_layer = tf_sub_block.get_layer(name=layer_name)
                    except ValueError:
                        print(f"  [Warning] No layer '{layer_name}' in sub-block '{sub_block_name}'.")
                        continue

                    # e.g. layer_name = "Conv_0"
                    if isinstance(tf_layer, tf.keras.layers.Conv2D):
                        kernel = np.array(layer_params["kernel"])
                        # Some Conv2D might have bias if use_bias=True,
                        # but your example used use_bias=False, so we skip it here.
                        tf_layer.set_weights([kernel])
                        print(f"Transferred Conv2D weights to {block_name}/{sub_block_name}/{layer_name}")
                    else:
                        print(f"  [Warning] Unhandled layer type in {block_name}/{sub_block_name}/{layer_name}")
    # Then, transfer the policy params.
    for block_name, block_content in policy_params.items(): # keys = {Dense_i, hidden_i}
        try:
            if block_name.startswith("Dense_"):
                tf_block = tf_model.get_layer(name=block_name)
            elif block_name == "MLP_0":
                tf_block = tf_model.get_layer(name="MLP_0")
            else:
                # Must be a hidden layer within MLP_0
                continue
        except ValueError:
            print(f"[Warning] No layer named '{block_name}' found in TF model.")
            print("Available layers:", [l.name for l in tf_model.layers])
            continue

        # ---------------------------------------------------------------------
        # CASE A: A top-level Dense_i block, which is a single Dense layer.
        # ---------------------------------------------------------------------
        if isinstance(tf_block, tf.keras.layers.Dense):
            # block_content should have "kernel" and "bias"
            kernel = np.array(block_content["kernel"])
            bias = np.array(block_content["bias"])
            tf_block.set_weights([kernel, bias])
            print(f"Transferred Dense weights to {block_name}")

        # ---------------------------------------------------------------------
        # CASE B: The MLP_0 block, which contains multiple hidden Dense layers.
        # ---------------------------------------------------------------------
        elif isinstance(tf_block, tf.keras.Sequential):
            for layer_name, layer_params in block_content.items():
                try:
                    tf_layer = tf_block.get_layer(name=layer_name)
                except ValueError:
                    print(f"  [Warning] No layer '{layer_name}' in MLP_0 block.")
                    continue

                if isinstance(tf_layer, tf.keras.layers.Dense):
                    kernel = np.array(layer_params["kernel"])
                    bias = np.array(layer_params["bias"])
                    tf_layer.set_weights([kernel, bias])
                    print(f"Transferred Dense weights to MLP_0/{layer_name}")
                else:
                    print(f"  [Warning] Unhandled layer type in MLP_0/{layer_name}")

        else:
            raise ValueError(f"Unhandled block '{block_name}' of type {type(tf_block)}")
    print("Weight transfer complete.")
