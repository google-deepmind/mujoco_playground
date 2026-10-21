# Sim2Sim Transfer

In this directory, we demonstrate how to deploy a Go1 joystick controller trained with Playground in native C MuJoCo and interact with it using a joystick.

## Usage

```bash
python play_go1_joystick.py
```

<a href="https://youtu.be/XwF2lkT2gqo" target="_blank">
 <img src="http://img.youtube.com/vi/XwF2lkT2gqo/hqdefault.jpg" alt="Watch the video" width="560" height="315"/>
</a>

## Requirements

We'll need 2 additional dependencies:

1. `onnxruntime` for running the ONNX model.
2. `hidapi` for reading the joystick.

```bash
uv pip install onnxruntime hidapi
```

On macOS, you'll need to install `hidapi` with brew and correctly set the `DYLD_LIBRARY_PATH` environment variable.

```bash
brew install hidapi
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hidapi/0.14.0/lib:$DYLD_LIBRARY_PATH
```

## Joystick

We use a Logitech G F710 Wireless Gamepad in this example. You can buy one on [Amazon](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW) for $40. In principle, you can use any joystick of your choice, but you'll need to modify `gamepad_reader.py` to support it.

<img src="assets/f710.jpg" alt="Logitech F710 Gamepad" width="200">

- Why not use `inputs`? It didn't seem to read any joystick on macOS.
- Why not use `pygame`? PyGame and MuJoCo's viewer don't place nice together. pygame needs to run on the main thread on macOS and we use the managed viewer in MuJoCo which runs the policy in a callback thread.

## Exporting a trained policy

We have a notebook for exporting trained policies to ONNX format. See `mujoco_playground/experimental/brax_network_to_onnx.ipynb`.
