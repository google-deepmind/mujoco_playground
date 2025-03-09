"""Base classes for ZBot Humanoid."""

from typing import Any

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from mujoco_playground._src.locomotion.zbot import zbot_constants as consts


def get_assets() -> dict[str, bytes]:
    assets = {}

    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")

    path = mjx_env.MENAGERIE_PATH / "zbot"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")

    return assets


class ZbotEnv(mjx_env.MjxEnv):
    """Base class for ZBot environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._mj_model = mujoco.MjModel.from_xml_string(epath.Path(xml_path).read_text(), assets=get_assets())
        self._mj_model.opt.timestep = self.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Return the gravity vector in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        """Return the position of the feet in the world frame."""
        return jp.vstack(
            [mjx_env.get_sensor_data(self.mj_model, data, sensor_name) for sensor_name in consts.FEET_POS_SENSOR]
        )

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
