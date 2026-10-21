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

Install the driver and camera extra (the I2RT SDK and OpenCV):

```bash
CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17" uv sync --extra cu124 --extra yam --group dev
```

Bring up the CAN interface. This needs root, and doesn't persist across
replugs; for that, I2RT's repo has `devices/install_devices.sh`:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

Identify the wrist camera (saves one preview image per candidate V4L2 node to
`.local/camera_previews/` and writes the selection to `.local/yam_cameras.json`):

```bash
uv run --extra yam python -m mujoco_playground.robots.yam.real.get_cameras --identify
```

> **Note:** On Linux, the Intel RealSense D405 exposes multiple V4L2 nodes that
> produce color images. Pick the **ISP RGB stream** (`...-video-index4`,
> typically `/dev/video4` — bright, `640×480` `4:3` with no corner vignetting),
> **not** the raw left stereo sensor (`...-video-index2`, typically
> `/dev/video2` — dark, `424×240` `16:9` with heavy corner vignetting).
