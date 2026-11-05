# YAM

A single [I2RT YAM Pro](https://doc.i2rt.com/products/yam-pro.html) arm bolted
to the desk, in sim and on hardware.

## Hardware

| | |
| --- | --- |
| Arm | I2RT YAM Pro (`yam_pro` in I2RT's SDK), 6 DOF |
| Gripper | Linear gripper on a DM4310 (`linear_4310`), confirmed from the bus |
| Motors | Damiao, MIT mode: DM4340 on joints 1-3, DM4310 on joints 4-6 and the gripper; CAN ids 1-7 in joint order |
| Bus | USB-CAN adapter (gs_usb / candleLight) as SocketCAN `can0`, 1 Mbit/s |
| Camera | Intel RealSense D405 on I2RT's wrist bracket (`wrist_cam` on `wrist_camera`) |

## Setup

### Linux

Install the `yam` extra and bring up the `can0` SocketCAN interface (requires
root and resets on reboot or unplug; to persist, install the `udev` rule via
`sudo sh devices/install_devices.sh` from a clone of `i2rt-robotics/i2rt`):

```bash
uv sync --extra yam
sudo ip link set can0 up type can bitrate 1000000
```

### macOS

Install `libusb` (used in user space for CAN communication, so no `ip link`
setup is needed) and sync the `yam` extra:

```bash
brew install libusb
uv sync --extra yam
```

### Identify the wrist camera

The command below saves one preview image per candidate camera to
`.local/camera_previews/` and writes the selection to `.local/yam_cameras.json`.

```bash
uv run --extra yam python -m mujoco_playground.robots.yam.real.get_cameras --identify
```

> [!NOTE]
> The Intel RealSense D405 exposes multiple color UVC streams on both Linux and
> macOS. Pick the **ISP RGB stream** (bright, `640×480` `4:3` with no corner
> vignetting — `...-video-index4`), **not** the raw left stereo sensor
> (dark, `424×240` `16:9` with heavy corner vignetting — `...-video-index2`).
