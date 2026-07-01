import importlib


def test_mujoco_mjx_imports_without_softjax():
    module = importlib.import_module("mujoco.mjx")

    assert hasattr(module, "Model")
    assert hasattr(module, "Data")
