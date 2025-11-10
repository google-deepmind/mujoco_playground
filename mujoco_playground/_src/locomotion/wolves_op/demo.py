import mujoco
import mujoco.viewer
import glfw
import numpy as np
import pickle

# ---------------------------------------------------------
# Model laden
# ---------------------------------------------------------
model = mujoco.MjModel.from_xml_path("wolves_op.xml")
data = mujoco.MjData(model)

# ---------------------------------------------------------
# Policy laden
# ---------------------------------------------------------
policy = pickle.load(open("?", "rb")) # Pfad zur Policy-Datei angeben

# Keyboard
keys = {}

def key_callback(window, key, scancode, action, mods):
    keys[key] = (action != glfw.RELEASE)

# Viewer
viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

# Control targets
desired_forward_vel = 0.0
desired_turn_vel = 0.0

while viewer.is_running():

    # -------------------------------------------
    # Keyboard input definieren
    # -------------------------------------------
    desired_forward_vel = 0.0
    desired_turn_vel = 0.0

    if keys.get(glfw.KEY_W):
        desired_forward_vel = 1.0
    if keys.get(glfw.KEY_S):
        desired_forward_vel = -1.0   # rückwärts
    if keys.get(glfw.KEY_A):
        desired_turn_vel = 1.0
    if keys.get(glfw.KEY_D):
        desired_turn_vel = -1.0

    # Beschleunigen mit Shift
    if keys.get(glfw.KEY_LEFT_SHIFT):
        desired_forward_vel *= 2.0

    # -------------------------------------------
    # Build observation: proprioception + intent
    # -------------------------------------------
    obs = np.concatenate([
        data.qpos,
        data.qvel,
        np.array([desired_forward_vel, desired_turn_vel])
    ])

    # -------------------------------------------
    # RL Policy predicts motor torques
    # -------------------------------------------
    action = policy(obs)
    data.ctrl[:] = action

    # Sim step
    mujoco.mj_step(model, data)
