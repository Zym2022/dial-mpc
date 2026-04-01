from dataclasses import dataclass, field
from typing import Any, Sequence, Union, List
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import jax
import jax.numpy as jnp

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import State
from brax.io import mjcf

import mujoco

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class SpiritMoz1PathTrackEnvConfig(BaseEnvConfig):
    collision_mode: str = "full"
    chassis_model: str = "wheel_pd"
    waist_leg_action_range_scale: float = 0.20
    arm_action_range_scale: float = 0.25
    command_ramp_time: float = 1.5
    mobile_root_ref_lpf_alpha: float = 0.18
    body_velocity_scale: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.75, 0.75, 1.5])
    )
    body_command_correction: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.eye(3)
    )
    waist_leg_ref_delta: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.zeros(6)
    )
    arm_ref_delta: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.zeros(14)
    )

    # Omnidirectional wheelbase geometry and wheel PD.
    wheel_radius: float = 0.08
    wheel_base_half_x: float = 0.2423
    wheel_base_half_y: float = 0.2423
    wheel_kp: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([18.0, 18.0, 18.0, 18.0])
    )
    wheel_kd: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([1.5, 1.5, 1.5, 1.5])
    )
    wheel_speed_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([20.0, 20.0, 20.0, 20.0])
    )
    wheel_tau_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([40.0, 40.0, 40.0, 40.0])
    )
    planar_wrench_kp: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([450.0, 450.0, 180.0])
    )
    planar_wrench_kd: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([60.0, 60.0, 30.0])
    )
    planar_wrench_ki: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([80.0, 80.0, 25.0])
    )
    planar_control_period: float = 0.02
    planar_velocity_estimate_period: float = 0.02
    planar_integrator_deadband: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.015, 0.015, 0.02])
    )
    planar_integrator_leak: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.08, 0.08, 0.10])
    )
    planar_velocity_lpf_alpha: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.10, 0.10, 0.12])
    )
    planar_wrench_slew_rate: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([40000.0, 40000.0, 12000.0])
    )
    planar_force_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([700.0, 700.0])
    )
    planar_torque_limit: float = 260.0
    planar_cmd_ramp_time: float = 0.25
    tip_roll_limit: float = 0.70
    tip_pitch_limit: float = 0.70
    min_base_height: float = 0.03

    # PD gains for waist+leg (6 joints)
    waist_leg_kp: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([7000.0, 8000.0, 7000.0, 6000.0, 5000.0, 5000.0])
    )
    waist_leg_kd: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([600.0, 400.0, 400.0, 600.0, 400.0, 300.0])
    )

    # PD gains for each arm (7 joints, mirrored to both sides)
    arm_kp: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([200.0, 200.0, 150.0, 150.0, 60.0, 60.0, 60.0])
    )
    arm_kd: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([20.0, 20.0, 15.0, 15.0, 6.0, 6.0, 6.0])
    )
    waist_leg_tau_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([400.0, 400.0, 350.0, 300.0, 250.0, 250.0])
    )
    arm_tau_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([80.0, 80.0, 60.0, 60.0, 30.0, 30.0, 30.0])
    )
    head_kp: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([80.0, 80.0])
    )
    head_kd: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([8.0, 8.0])
    )
    head_tau_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([20.0, 20.0])
    )

    # Figure-8 path settings: x=cx+ax*sin(wt), y=cy+0.5*ay*sin(2wt)
    path_amp_x: float = 1.0
    path_amp_y: float = 0.8
    path_omega: float = 0.35
    path_center_xy: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    z_ref: float = 0.08
    reward_pos_weight: float = 2.0
    reward_yaw_weight: float = 1.0
    reward_vel_weight: float = 1.5
    reward_upright_weight: float = 1.0
    reward_height_weight: float = 0.5
    reward_ang_vel_weight: float = 0.0
    reward_arm_pose_weight: float = 0.05
    reward_arm_limit_weight: float = 0.0
    reward_waist_pose_weight: float = 0.05
    reward_waist_qd_weight: float = 0.0
    reward_waist_limit_weight: float = 0.0
    reward_ctrl_weight: float = 0.01
    reward_ctrl_rate_weight: float = 0.01
    limit_margin_scale: float = 0.15
    stand_success_xy_tol: float = 0.01
    stand_success_z_tol: float = 0.005
    stand_success_yaw_tol: float = 0.05
    stand_success_waist_tol: float = 0.08
    stand_success_arm_tol: float = 0.10
    stand_success_joint_vel_tol: float = 0.35
    stand_success_body_ang_vel_tol: float = 0.25
    stand_success_steps: int = 50


class SpiritMoz1PathTrackEnv(BaseEnv):
    _LITE_COLLISION_GEOMS = {
        "floor",
        "wheel01_collision",
        "wheel02_collision",
        "wheel03_collision",
        "wheel04_collision",
    }

    _PRIMITIVE_WHEEL_BODIES = {"wheel01", "wheel02", "wheel03", "wheel04"}
    _SIMPLIFIED_COLLISION_SPECS = {
        "base_link": [
            {"name": "base_link_simplified_collision", "type": "box", "size": "0.33 0.26 0.20", "pos": "0 0 0.17"},
        ],
        "wheel01": [
            {"name": "wheel01_simplified_collision", "type": "cylinder", "size": "0.08 0.05", "euler": "1.5707963 0 0"},
        ],
        "wheel02": [
            {"name": "wheel02_simplified_collision", "type": "cylinder", "size": "0.08 0.05", "euler": "1.5707963 0 0"},
        ],
        "wheel03": [
            {"name": "wheel03_simplified_collision", "type": "cylinder", "size": "0.08 0.05", "euler": "1.5707963 0 0"},
        ],
        "wheel04": [
            {"name": "wheel04_simplified_collision", "type": "cylinder", "size": "0.08 0.05", "euler": "1.5707963 0 0"},
        ],
        "leg01": [
            {"name": "leg01_simplified_collision", "type": "capsule", "fromto": "0 0 0.02 0 0 0.16", "size": "0.07"},
        ],
        "leg02": [
            {"name": "leg02_simplified_collision", "type": "capsule", "fromto": "0 0 0.02 0 0 0.27", "size": "0.075"},
        ],
        "leg03": [
            {"name": "leg03_simplified_collision", "type": "capsule", "fromto": "0 0 0.03 0 0 0.30", "size": "0.075"},
        ],
        "waist01": [
            {"name": "waist01_simplified_collision", "type": "capsule", "fromto": "0 0 0.01 0 0 0.16", "size": "0.07"},
        ],
        "waist02": [
            {"name": "waist02_simplified_collision", "type": "capsule", "fromto": "0 0 0.02 0 0 0.18", "size": "0.065"},
        ],
        "waist03": [
            {"name": "waist03_simplified_collision", "type": "capsule", "fromto": "0 0 -0.22 0 0 0.08", "size": "0.11"},
        ],
        "left01": [
            {"name": "left01_simplified_collision", "type": "capsule", "fromto": "0 0 0 0.10 0 0", "size": "0.045"},
        ],
        "left02": [
            {"name": "left02_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.12", "size": "0.04"},
        ],
        "left03": [
            {"name": "left03_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.16", "size": "0.04"},
        ],
        "left04": [
            {"name": "left04_simplified_collision", "type": "capsule", "fromto": "0 0 -0.01 0 0 -0.11", "size": "0.035"},
        ],
        "left05": [
            {"name": "left05_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.14", "size": "0.035"},
        ],
        "left06": [
            {"name": "left06_simplified_collision", "type": "capsule", "fromto": "0 0 -0.01 0 0 -0.10", "size": "0.03"},
        ],
        "left07": [
            {"name": "left07_simplified_collision", "type": "sphere", "size": "0.04"},
        ],
        "right01": [
            {"name": "right01_simplified_collision", "type": "capsule", "fromto": "0 0 0 0.10 0 0", "size": "0.045"},
        ],
        "right02": [
            {"name": "right02_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.12", "size": "0.04"},
        ],
        "right03": [
            {"name": "right03_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.16", "size": "0.04"},
        ],
        "right04": [
            {"name": "right04_simplified_collision", "type": "capsule", "fromto": "0 0 -0.01 0 0 -0.11", "size": "0.035"},
        ],
        "right05": [
            {"name": "right05_simplified_collision", "type": "capsule", "fromto": "0 0 -0.02 0 0 -0.14", "size": "0.035"},
        ],
        "right06": [
            {"name": "right06_simplified_collision", "type": "capsule", "fromto": "0 0 -0.01 0 0 -0.10", "size": "0.03"},
        ],
        "right07": [
            {"name": "right07_simplified_collision", "type": "sphere", "size": "0.04"},
        ],
        "head21": [
            {"name": "head21_simplified_collision", "type": "capsule", "fromto": "0 0 0 0 0 0.10", "size": "0.05"},
        ],
        "head22": [
            {"name": "head22_simplified_collision", "type": "capsule", "fromto": "0 0 0 0 0 0.08", "size": "0.045"},
        ],
        "head23": [
            {"name": "head23_simplified_collision", "type": "sphere", "size": "0.06"},
        ],
    }

    def __init__(self, config: SpiritMoz1PathTrackEnvConfig):
        super().__init__(config)

        self._base_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base_link"
        )

        # Actuator / joint grouping (matches moz1.xml ordering)
        self._wheel_idx = jnp.arange(0, 4)
        self._waist_leg_idx = jnp.arange(4, 10)
        self._left_arm_idx = jnp.arange(10, 17)
        self._right_arm_idx = jnp.arange(17, 24)
        self._arm_idx = jnp.arange(10, 24)
        self._head_idx = jnp.arange(24, 26)
        self._action_joint_idx = jnp.arange(4, 24)

        self._init_q = jnp.array(self.sys.mj_model.keyframe("stand").qpos)
        self._q_ref = self._init_q[7:]

        waist_leg_kp = jnp.array(config.waist_leg_kp)
        waist_leg_kd = jnp.array(config.waist_leg_kd)
        arm_kp = jnp.array(config.arm_kp)
        arm_kd = jnp.array(config.arm_kd)

        self._joint_kp = jnp.concatenate([waist_leg_kp, arm_kp, arm_kp])
        self._joint_kd = jnp.concatenate([waist_leg_kd, arm_kd, arm_kd])
        self._joint_tau_limit = jnp.concatenate(
            [
                jnp.array(config.waist_leg_tau_limit),
                jnp.array(config.arm_tau_limit),
                jnp.array(config.arm_tau_limit),
            ]
        )

        self._head_kp = jnp.array(config.head_kp)
        self._head_kd = jnp.array(config.head_kd)
        self._head_tau_limit = jnp.array(config.head_tau_limit)
        self._wheel_kp = jnp.array(config.wheel_kp)
        self._wheel_kd = jnp.array(config.wheel_kd)
        self._wheel_tau_limit = jnp.array(config.wheel_tau_limit)
        self._wheel_speed_limit = jnp.array(config.wheel_speed_limit)
        self._body_velocity_scale = jnp.array(config.body_velocity_scale)
        self._body_command_correction = jnp.array(config.body_command_correction)
        self._wheel_cmd_matrix = self._build_wheel_command_matrix()

        waist_ref = self._q_ref[self._waist_leg_idx] + jnp.array(config.waist_leg_ref_delta)
        arm_ref = self._q_ref[self._arm_idx] + jnp.array(config.arm_ref_delta)
        head_ref = self._q_ref[self._head_idx]
        self._waist_ref = jnp.clip(
            waist_ref,
            self.physical_joint_range[self._waist_leg_idx, 0],
            self.physical_joint_range[self._waist_leg_idx, 1],
        )
        self._arm_ref = jnp.clip(
            arm_ref,
            self.physical_joint_range[self._arm_idx, 0],
            self.physical_joint_range[self._arm_idx, 1],
        )
        self._head_ref = jnp.clip(
            head_ref,
            self.physical_joint_range[self._head_idx, 0],
            self.physical_joint_range[self._head_idx, 1],
        )
        self._joint_ref = jnp.concatenate([self._waist_ref, self._arm_ref])
        self._joint_range_valid = (
            self.physical_joint_range[:, 1] - self.physical_joint_range[:, 0]
        ) > 1e-6
        self._stand_xy_ref = jnp.array(self._init_q[:2])
        self._stand_yaw_ref = math.quat_to_euler(self._init_q[3:7])[2]

        joint_physical_low = self.physical_joint_range[self._action_joint_idx, 0]
        joint_physical_high = self.physical_joint_range[self._action_joint_idx, 1]
        joint_physical_half = 0.5 * (joint_physical_high - joint_physical_low)
        joint_scale = jnp.concatenate(
            [
                jnp.full((len(self._waist_leg_idx),), config.waist_leg_action_range_scale),
                jnp.full((len(self._arm_idx),), config.arm_action_range_scale),
            ]
        )
        joint_half = joint_physical_half * joint_scale
        joint_half = jnp.minimum(joint_half, self._joint_ref - joint_physical_low)
        joint_half = jnp.minimum(joint_half, joint_physical_high - self._joint_ref)
        joint_half = jnp.maximum(joint_half, 0.0)
        self._action_joint_low = self._joint_ref - joint_half
        self._action_joint_high = self._joint_ref + joint_half

        self.joint_range = self.physical_joint_range
        self.joint_range = self.joint_range.at[self._action_joint_idx, 0].set(self._action_joint_low)
        self.joint_range = self.joint_range.at[self._action_joint_idx, 1].set(self._action_joint_high)

    @property
    def action_size(self) -> int:
        return 3 + len(self._action_joint_idx)

    def _load_spirit_model(self, filename: str, collision_mode: str) -> System:
        model_path = Path(get_model_path("spirit_moz1", filename))
        xml_text = model_path.read_text()
        xml_text = xml_text.replace("../meshes/", "meshes/")
        xml_text = self._apply_collision_mode(xml_text, collision_mode)

        mesh_dir = model_path.parent / "meshes"
        assets = {
            f"meshes/{mesh_file.name}": mesh_file.read_bytes()
            for mesh_file in mesh_dir.glob("*")
            if mesh_file.is_file()
        }
        if collision_mode == "primitive":
            assets = {}

        mj_model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
        return mjcf.load_model(mj_model)

    def make_system(self, config: SpiritMoz1PathTrackEnvConfig) -> System:
        self._visualization_sys = None
        sys = self._load_spirit_model("mjx_moz1.xml", config.collision_mode)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def make_visualization_system(self) -> System:
        if self._visualization_sys is None:
            self._visualization_sys = self._load_spirit_model("moz1.xml", "visual_only")
            self._visualization_sys = self._visualization_sys.tree_replace(
                {"opt.timestep": self._config.timestep}
            )
        return self._visualization_sys

    @classmethod
    def _disable_collision_geoms(cls, root: ET.Element) -> None:
        for body in root.iter("body"):
            for geom in list(body.findall("geom")):
                name = geom.attrib.get("name", "")
                if "collision" not in name:
                    continue
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")

    @classmethod
    def _append_simplified_geoms(cls, root: ET.Element, include_visual: bool) -> None:
        for body in root.iter("body"):
            body_name = body.attrib.get("name", "")
            specs = cls._SIMPLIFIED_COLLISION_SPECS.get(body_name)
            if not specs:
                continue
            for spec in specs:
                attrib = {k: v for k, v in spec.items() if k != "name"}
                attrib["name"] = spec["name"]
                attrib["class"] = "collision"
                body.append(ET.Element("geom", attrib))
                if include_visual:
                    visual_attrib = dict(attrib)
                    visual_attrib["name"] = spec["name"].replace("_collision", "_visual")
                    visual_attrib["class"] = "visual"
                    visual_attrib["rgba"] = "0.5 0.5 0.5 1"
                    body.append(ET.Element("geom", visual_attrib))

    @classmethod
    def _apply_collision_mode(cls, xml_text: str, collision_mode: str) -> str:
        if collision_mode == "full":
            return xml_text
        if collision_mode not in {"lite", "primitive", "simplified", "visual_only"}:
            raise ValueError(
                f"Unsupported Spirit MOZ1 collision_mode={collision_mode!r}. Expected 'full', 'lite', 'simplified', 'visual_only', or 'primitive'."
            )

        root = ET.fromstring(xml_text)

        if collision_mode == "visual_only":
            cls._disable_collision_geoms(root)
            return ET.tostring(root, encoding="unicode")

        if collision_mode == "primitive":
            asset = root.find("asset")
            if asset is not None:
                for mesh in list(asset.findall("mesh")):
                    asset.remove(mesh)

        for body in root.iter("body"):
            for geom in list(body.findall("geom")):
                name = geom.attrib.get("name", "")
                geom_type = geom.attrib.get("type", "")
                if collision_mode == "primitive" and geom_type == "mesh":
                    body.remove(geom)
                    continue
                if collision_mode == "simplified" and "collision" in name:
                    body.remove(geom)
                    continue
                if name in cls._LITE_COLLISION_GEOMS:
                    continue
                if "collision" not in name:
                    continue
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")

        if collision_mode == "lite":
            base_link = next(
                (body for body in root.iter("body") if body.attrib.get("name") == "base_link"),
                None,
            )
            if base_link is None:
                raise ValueError("Spirit MOZ1 XML is missing base_link body")
            base_link.insert(
                1,
                ET.Element(
                    "geom",
                    {
                        "name": "base_link_lite_collision",
                        "type": "box",
                        "size": "0.33 0.26 0.20",
                        "pos": "0 0 0.17",
                        "class": "collision",
                    },
                ),
            )
        elif collision_mode == "simplified":
            cls._append_simplified_geoms(root, include_visual=False)
        elif collision_mode == "primitive":
            cls._append_simplified_geoms(root, include_visual=True)
        return ET.tostring(root, encoding="unicode")

    def _command_scale(self, t: jax.Array) -> jax.Array:
        ramp_time = self._config.command_ramp_time
        if ramp_time <= 0.0:
            return jnp.array(1.0)
        return jnp.clip(t / ramp_time, 0.0, 1.0)

    def _path_targets(self, t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self._config.task_name == "stand_hold":
            pos_tar = jnp.array([self._stand_xy_ref[0], self._stand_xy_ref[1], self._config.z_ref])
            vel_tar_world = jnp.zeros(3)
            yaw_tar = self._stand_yaw_ref
            return pos_tar, vel_tar_world, yaw_tar

        cfg = self._config
        wt = cfg.path_omega * t
        x_tar = cfg.path_center_xy[0] + cfg.path_amp_x * jnp.sin(wt)
        y_tar = cfg.path_center_xy[1] + 0.5 * cfg.path_amp_y * jnp.sin(2.0 * wt)

        dx_dt = cfg.path_amp_x * cfg.path_omega * jnp.cos(wt)
        dy_dt = cfg.path_amp_y * cfg.path_omega * jnp.cos(2.0 * wt)
        ramp = self._command_scale(t)
        yaw_tar = jnp.atan2(dy_dt, dx_dt) * ramp

        pos_tar = jnp.array([x_tar, y_tar, cfg.z_ref])
        vel_tar_world = jnp.array([dx_dt, dy_dt, 0.0]) * ramp
        return pos_tar, vel_tar_world, yaw_tar

    def _build_wheel_command_matrix(self) -> jax.Array:
        wheel_body_names = ["wheel01", "wheel02", "wheel03", "wheel04"]
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rows = []
        for body_name in wheel_body_names:
            body_id = mujoco.mj_name2id(
                self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name
            )
            joint_id = int(self.sys.mj_model.body_jntadr[body_id])
            axis_local = np.array(self.sys.mj_model.jnt_axis[joint_id], dtype=np.float64)
            quat = np.array(self.sys.mj_model.body_quat[body_id], dtype=np.float64)
            axis_body = np.zeros(3, dtype=np.float64)
            mujoco.mju_rotVecQuat(axis_body, axis_local, quat)
            # The wheel hinge axis points along the axle.  The rolling
            # direction is the tangent direction produced by a positive wheel
            # angle increase, which is axis x up rather than up x axis.
            roll_dir = np.cross(axis_body, up)
            roll_dir /= np.linalg.norm(roll_dir)
            pos = np.array(self.sys.mj_model.body_pos[body_id], dtype=np.float64)
            yaw_coeff = -roll_dir[0] * pos[1] + roll_dir[1] * pos[0]
            rows.append([roll_dir[0], roll_dir[1], yaw_coeff])
        return jnp.array(rows) / self._config.wheel_radius

    def _action_to_body_command(self, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        body_cmd = action[:3] * self._config.action_scale * self._body_velocity_scale
        vel_cmd = jnp.array([body_cmd[0], body_cmd[1], 0.0])
        ang_cmd = jnp.array([0.0, 0.0, body_cmd[2]])
        return vel_cmd, ang_cmd

    def _yaw_to_quat(self, yaw: jax.Array) -> jax.Array:
        half_yaw = 0.5 * yaw
        return jnp.array([jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)])

    def _apply_ideal_planar_chassis(
        self,
        prev_pipeline_state: base.State,
        pipeline_state: base.State,
        ctrl: jax.Array,
        wheel_speed_tar: jax.Array,
        wheel_q_ref: jax.Array,
        vel_cmd_body: jax.Array,
        ang_cmd_body: jax.Array,
    ) -> base.State:
        prev_base_rot = prev_pipeline_state.x.rot[self._base_idx - 1]
        prev_yaw = math.quat_to_euler(prev_base_rot)[2]
        prev_base_pos = prev_pipeline_state.q[:3]

        yaw_quat = self._yaw_to_quat(prev_yaw)
        vel_cmd_world = math.rotate(vel_cmd_body, yaw_quat)
        next_yaw = prev_yaw + ang_cmd_body[2] * self.dt
        next_quat = self._yaw_to_quat(next_yaw)
        next_pos = jnp.array(
            [
                prev_base_pos[0] + vel_cmd_world[0] * self.dt,
                prev_base_pos[1] + vel_cmd_world[1] * self.dt,
                self._config.z_ref,
            ]
        )

        q = pipeline_state.q
        q = q.at[:3].set(next_pos)
        q = q.at[3:7].set(next_quat)
        q = q.at[self._wheel_idx + 7].set(wheel_q_ref)

        qd = pipeline_state.qd
        qd = qd.at[:3].set(jnp.array([vel_cmd_world[0], vel_cmd_world[1], 0.0]))
        qd = qd.at[3:6].set(jnp.array([0.0, 0.0, ang_cmd_body[2]]))
        qd = qd.at[self._wheel_idx + 6].set(wheel_speed_tar)

        pipeline_state = self.pipeline_init(q, qd)
        return pipeline_state.replace(ctrl=ctrl)

    def _body_command_to_wheel_speed(self, vel_cmd: jax.Array, ang_cmd: jax.Array) -> jax.Array:
        corrected_body_cmd = self._body_command_correction @ jnp.array(
            [vel_cmd[0], vel_cmd[1], ang_cmd[2]]
        )
        wheel_speed = self._wheel_cmd_matrix @ corrected_body_cmd
        return jnp.clip(wheel_speed, -self._wheel_speed_limit, self._wheel_speed_limit)

    def reset(self, rng: jax.Array) -> State:
        init_q = self._init_q
        if self._config.task_name == "stand_hold":
            init_q = init_q.at[self._action_joint_idx + 7].set(self._joint_ref)
            init_q = init_q.at[self._head_idx + 7].set(self._head_ref)
        pipeline_state = self.pipeline_init(init_q, jnp.zeros(self._nv))

        pos_tar, vel_tar_world, yaw_tar = self._path_targets(jnp.array(0.0))

        state_info = {
            "rng": rng,
            "step": 0,
            "path_phase": 0.0,
            "pos_tar": pos_tar,
            "vel_tar_world": vel_tar_world,
            "vel_tar": jnp.zeros(3),
            "yaw_tar": yaw_tar,
            "q_arm_ref": self._arm_ref,
            "q_waist_ref": self._waist_ref,
            "q_head_ref": self._head_ref,
            "wheel_q_ref": pipeline_state.q[7:11],
            "last_ctrl": jnp.zeros(self.sys.nu),
            "stable_steps": jnp.array(0, dtype=jnp.int32),
            "terminal_code": jnp.array(0, dtype=jnp.int32),
        }

        state_info["vel_tar"] = global_to_body_velocity(
            vel_tar_world, pipeline_state.x.rot[self._base_idx - 1]
        )

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        q = state.pipeline_state.q[7:]
        qd = state.pipeline_state.qd[6:]

        vel_cmd_body, ang_cmd_body = self._action_to_body_command(action)
        if self._config.task_name == "stand_hold":
            vel_cmd_body = jnp.zeros_like(vel_cmd_body)
            ang_cmd_body = jnp.zeros_like(ang_cmd_body)
        wheel_speed_tar = self._body_command_to_wheel_speed(vel_cmd_body, ang_cmd_body)
        wheel_q_ref = state.info["wheel_q_ref"] + wheel_speed_tar * self.dt
        if self._config.chassis_model == "wheel_pd":
            wheel_q = q[self._wheel_idx]
            wheel_qd = qd[self._wheel_idx]
            wheel_tau = self._wheel_kp * (wheel_q_ref - wheel_q) + self._wheel_kd * (wheel_speed_tar - wheel_qd)
            wheel_tau = jnp.clip(wheel_tau, -self._wheel_tau_limit, self._wheel_tau_limit)
        elif self._config.chassis_model == "ideal_planar":
            wheel_tau = jnp.zeros_like(wheel_speed_tar)
        else:
            raise ValueError(
                f"Unsupported Spirit MOZ1 chassis_model={self._config.chassis_model!r}. Expected 'wheel_pd' or 'ideal_planar'."
            )

        # 2) Waist+arms use action-parameterized joint targets with pure PD tracking.
        if self._config.task_name == "stand_hold":
            joint_targets = self._joint_ref
        else:
            joint_action = action[3:]
            joint_norm = (joint_action * self._config.action_scale + 1.0) / 2.0
            joint_targets = self._action_joint_low + joint_norm * (self._action_joint_high - self._action_joint_low)
            joint_targets = jnp.clip(
                joint_targets,
                self.physical_joint_range[self._action_joint_idx, 0],
                self.physical_joint_range[self._action_joint_idx, 1],
            )
        joint_q = q[self._action_joint_idx]
        joint_qd = qd[self._action_joint_idx]
        joint_tau = self._joint_kp * (joint_targets - joint_q) - self._joint_kd * joint_qd
        joint_tau = jnp.clip(joint_tau, -self._joint_tau_limit, self._joint_tau_limit)

        # 3) Heads: fixed-target PD so every actuated joint is actively driven.
        head_q = q[self._head_idx]
        head_qd = qd[self._head_idx]
        head_tau = self._head_kp * (state.info["q_head_ref"] - head_q) - self._head_kd * head_qd
        head_tau = jnp.clip(head_tau, -self._head_tau_limit, self._head_tau_limit)

        ctrl = jnp.zeros(self.sys.nu)
        ctrl = ctrl.at[self._wheel_idx].set(wheel_tau)
        ctrl = ctrl.at[self._action_joint_idx].set(joint_tau)
        ctrl = ctrl.at[self._head_idx].set(head_tau)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        if self._config.chassis_model == "ideal_planar":
            pipeline_state = self._apply_ideal_planar_chassis(
                state.pipeline_state,
                pipeline_state,
                ctrl,
                wheel_speed_tar,
                wheel_q_ref,
                vel_cmd_body,
                ang_cmd_body,
            )
        x, xd = pipeline_state.x, pipeline_state.xd

        next_step = state.info["step"] + 1
        t = next_step * self.dt
        pos_tar, vel_tar_world, yaw_tar = self._path_targets(t)
        vel_tar_body = global_to_body_velocity(vel_tar_world, x.rot[self._base_idx - 1])

        pos = x.pos[self._base_idx - 1]
        yaw = math.quat_to_euler(x.rot[self._base_idx - 1])[2]
        vb = global_to_body_velocity(xd.vel[self._base_idx - 1], x.rot[self._base_idx - 1])
        wb = global_to_body_velocity(
            xd.ang[self._base_idx - 1] * jnp.pi / 180.0,
            x.rot[self._base_idx - 1],
        )

        up_world = jnp.array([0.0, 0.0, 1.0])
        up_body = math.rotate(up_world, x.rot[self._base_idx - 1])

        q_act = pipeline_state.q[7:]
        q_waist = q_act[self._waist_leg_idx]
        q_arm = q_act[self._arm_idx]

        reward_pos = -jnp.sum((pos[:2] - pos_tar[:2]) ** 2)
        d_yaw = jnp.atan2(jnp.sin(yaw - yaw_tar), jnp.cos(yaw - yaw_tar))
        reward_yaw = -jnp.square(d_yaw)
        reward_vel = -jnp.sum((vb[:2] - vel_tar_body[:2]) ** 2)
        reward_upright = -jnp.sum((up_body - up_world) ** 2)
        reward_height = -jnp.square(pos[2] - self._config.z_ref)
        reward_ang_vel = -jnp.sum(wb[:2] ** 2)
        reward_arm_pose = -jnp.sum((q_arm - state.info["q_arm_ref"]) ** 2)

        arm_low = self.physical_joint_range[self._arm_idx, 0]
        arm_high = self.physical_joint_range[self._arm_idx, 1]
        arm_margin = (arm_high - arm_low) * self._config.limit_margin_scale
        arm_margin = jnp.maximum(arm_margin, 1e-3)
        arm_low_pen = jnp.maximum(0.0, arm_low + arm_margin - q_arm) / arm_margin
        arm_high_pen = jnp.maximum(0.0, q_arm - (arm_high - arm_margin)) / arm_margin
        reward_arm_limit = -jnp.sum(arm_low_pen ** 2 + arm_high_pen ** 2)

        reward_waist_pose = -jnp.sum((q_waist - state.info["q_waist_ref"]) ** 2)
        reward_waist_qd = -jnp.sum(qd[self._waist_leg_idx] ** 2)
        waist_low = self.physical_joint_range[self._waist_leg_idx, 0]
        waist_high = self.physical_joint_range[self._waist_leg_idx, 1]
        waist_margin = (waist_high - waist_low) * self._config.limit_margin_scale
        waist_margin = jnp.maximum(waist_margin, 1e-3)
        waist_low_pen = jnp.maximum(0.0, waist_low + waist_margin - q_waist) / waist_margin
        waist_high_pen = jnp.maximum(0.0, q_waist - (waist_high - waist_margin)) / waist_margin
        reward_waist_limit = -jnp.sum(waist_low_pen ** 2 + waist_high_pen ** 2)

        ctrl_scale = jnp.concatenate([self._wheel_tau_limit, self._joint_tau_limit, self._head_tau_limit])
        reward_ctrl = -jnp.sum((ctrl / ctrl_scale) ** 2)
        reward_ctrl_rate = -jnp.sum(((ctrl - state.info["last_ctrl"]) / ctrl_scale) ** 2)

        reward = (
            reward_pos * self._config.reward_pos_weight
            + reward_yaw * self._config.reward_yaw_weight
            + reward_vel * self._config.reward_vel_weight
            + reward_upright * self._config.reward_upright_weight
            + reward_height * self._config.reward_height_weight
            + reward_ang_vel * self._config.reward_ang_vel_weight
            + reward_arm_pose * self._config.reward_arm_pose_weight
            + reward_arm_limit * self._config.reward_arm_limit_weight
            + reward_waist_pose * self._config.reward_waist_pose_weight
            + reward_waist_qd * self._config.reward_waist_qd_weight
            + reward_waist_limit * self._config.reward_waist_limit_weight
            + reward_ctrl * self._config.reward_ctrl_weight
            + reward_ctrl_rate * self._config.reward_ctrl_rate_weight
        )

        failed = jnp.dot(up_body, up_world) < 0.0
        failed |= pos[2] < max(0.03, self._config.z_ref * 0.5)
        failed |= jnp.any((q_act < self.physical_joint_range[:, 0]) & self._joint_range_valid)
        failed |= jnp.any((q_act > self.physical_joint_range[:, 1]) & self._joint_range_valid)
        failed |= ~jnp.isfinite(jnp.sum(pipeline_state.q))
        failed |= ~jnp.isfinite(jnp.sum(pipeline_state.qd))

        terminal_code = jnp.array(0, dtype=jnp.int32)
        stable_steps = jnp.array(0, dtype=jnp.int32)
        success = jnp.array(False)
        if self._config.task_name == "stand_hold":
            xy_err = jnp.linalg.norm(pos[:2] - pos_tar[:2])
            z_err = jnp.abs(pos[2] - self._config.z_ref)
            waist_err = jnp.max(jnp.abs(q_waist - state.info["q_waist_ref"]))
            arm_err = jnp.max(jnp.abs(q_arm - state.info["q_arm_ref"]))
            joint_vel_rms = jnp.sqrt(jnp.mean(qd[self._action_joint_idx] ** 2))
            body_ang_vel_mag = jnp.linalg.norm(wb)
            stable_now = xy_err <= self._config.stand_success_xy_tol
            stable_now &= z_err <= self._config.stand_success_z_tol
            stable_now &= jnp.abs(d_yaw) <= self._config.stand_success_yaw_tol
            stable_now &= waist_err <= self._config.stand_success_waist_tol
            stable_now &= arm_err <= self._config.stand_success_arm_tol
            stable_now &= joint_vel_rms <= self._config.stand_success_joint_vel_tol
            stable_now &= body_ang_vel_mag <= self._config.stand_success_body_ang_vel_tol
            stable_steps = jnp.where(stable_now, state.info["stable_steps"] + 1, 0)
            success = stable_steps >= self._config.stand_success_steps
            terminal_code = jnp.where(success, jnp.array(1, dtype=jnp.int32), terminal_code)

        terminal_code = jnp.where(failed, jnp.array(-1, dtype=jnp.int32), terminal_code)
        done = (failed | success).astype(jnp.float32)

        state.info["step"] = next_step
        state.info["path_phase"] = self._config.path_omega * t
        state.info["pos_tar"] = pos_tar
        state.info["vel_tar_world"] = vel_tar_world
        state.info["vel_tar"] = vel_tar_body
        state.info["yaw_tar"] = yaw_tar
        state.info["wheel_q_ref"] = wheel_q_ref
        state.info["last_ctrl"] = ctrl
        state.info["stable_steps"] = stable_steps
        state.info["terminal_code"] = terminal_code

        obs = self._get_obs(pipeline_state, state.info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        base_pos = x.pos[self._base_idx - 1]
        base_rot = x.rot[self._base_idx - 1]
        vb = global_to_body_velocity(xd.vel[self._base_idx - 1], base_rot)
        wb = global_to_body_velocity(xd.ang[self._base_idx - 1] * jnp.pi / 180.0, base_rot)

        pos_err = state_info["pos_tar"] - base_pos
        yaw = math.quat_to_euler(base_rot)[2]
        yaw_err = jnp.atan2(
            jnp.sin(state_info["yaw_tar"] - yaw),
            jnp.cos(state_info["yaw_tar"] - yaw),
        )

        obs = jnp.concatenate(
            [
                pos_err,
                jnp.array([yaw_err]),
                state_info["vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                wb,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "front_camera"
        return super().render(trajectory, camera=camera, width=width, height=height)


brax_envs.register_environment("spirit_moz1_path_track", SpiritMoz1PathTrackEnv)
