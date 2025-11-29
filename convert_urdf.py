from dm_control.utils import urdf

# URDF laden
model = urdf.load_urdf(
    "mujoco_playground/locomotion/wolves_op/humanoid_v49.urdf",
    fix_inertia=True,                     # bessere Stabilität
    decompose_concave_bodies=True         # falls komplexe Meshes
)

# speichern
model.save("mujoco_playground/locomotion/wolves_op/humanoid_converted.xml")
print("Konvertierung abgeschlossen!")
