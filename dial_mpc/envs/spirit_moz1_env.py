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

    # Wheel torque is directly scaled from action and clipped by actuator limits.
    wheel_torque_scale: float = 1.0
    wheel_tau_limit: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([40.0, 40.0, 40.0, 40.0])
    )

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

    # Figure-8 path settings: x=cx+ax*sin(wt), y=cy+0.5*ay*sin(2wt)
    path_amp_x: float = 1.0
    path_amp_y: float = 0.8
    path_omega: float = 0.35
    path_center_xy: Union[float, jax.Array] = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    z_ref: float = 0.463


class SpiritMoz1PathTrackEnv(BaseEnv):
    _LITE_COLLISION_GEOMS = {
        "floor",
        "wheel01_collision",
        "wheel02_collision",
        "wheel03_collision",
        "wheel04_collision",
    }

    _PRIMITIVE_WHEEL_BODIES = {"wheel01", "wheel02", "wheel03", "wheel04"}

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
        self._pd_idx = jnp.arange(4, 24)

        self._init_q = jnp.array(self.sys.mj_model.keyframe("stand").qpos)
        self._q_ref = self._init_q[7:]

        # Build PD gains for controlled joints [waist+leg, left arm, right arm]
        waist_leg_kp = jnp.array(config.waist_leg_kp)
        waist_leg_kd = jnp.array(config.waist_leg_kd)
        arm_kp = jnp.array(config.arm_kp)
        arm_kd = jnp.array(config.arm_kd)

        self._pd_kp = jnp.concatenate([waist_leg_kp, arm_kp, arm_kp])
        self._pd_kd = jnp.concatenate([waist_leg_kd, arm_kd, arm_kd])
        self._pd_tau_limit = jnp.concatenate(
            [
                jnp.array(config.waist_leg_tau_limit),
                jnp.array(config.arm_tau_limit),
                jnp.array(config.arm_tau_limit),
            ]
        )
        self._wheel_tau_limit = jnp.array(config.wheel_tau_limit)

        self._waist_ref = self._q_ref[self._waist_leg_idx]
        self._arm_ref = self._q_ref[self._arm_idx]
        self._joint_range_valid = (
            self.physical_joint_range[:, 1] - self.physical_joint_range[:, 0]
        ) > 1e-6

        # Limit controllable joints to physical ranges but keep wheels/head unchanged.
        self.joint_range = self.physical_joint_range

    def make_system(self, config: SpiritMoz1PathTrackEnvConfig) -> System:
        model_path = get_model_path("spirit_moz1", "moz1.xml")
        model_path = Path(model_path)

        # The source XML references meshes with "../meshes/..", which does not
        # match this repo layout under dial_mpc/models/spirit_moz1.
        xml_text = model_path.read_text()
        xml_text = xml_text.replace("../meshes/", "meshes/")
        xml_text = self._apply_collision_mode(xml_text, config.collision_mode)

        mesh_dir = model_path.parent / "meshes"
        assets = {
            f"meshes/{mesh_file.name}": mesh_file.read_bytes()
            for mesh_file in mesh_dir.glob("*")
            if mesh_file.is_file()
        }

        mj_model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    @classmethod
    def _apply_collision_mode(cls, xml_text: str, collision_mode: str) -> str:
        if collision_mode == "full":
            return xml_text
        if collision_mode not in {"lite", "primitive"}:
            raise ValueError(
                f"Unsupported Spirit MOZ1 collision_mode={collision_mode!r}. Expected 'full', 'lite', or 'primitive'."
            )

        root = ET.fromstring(xml_text)

        if collision_mode == "primitive":
            asset = root.find("asset")
            if asset is not None:
                for mesh in list(asset.findall("mesh")):
                    asset.remove(mesh)

        for body in root.iter("body"):
            body_name = body.attrib.get("name", "")
            for geom in list(body.findall("geom")):
                name = geom.attrib.get("name", "")
                geom_type = geom.attrib.get("type", "")
                if collision_mode == "primitive" and geom_type == "mesh":
                    body.remove(geom)
                    continue
                if name in cls._LITE_COLLISION_GEOMS:
                    continue
                if "collision" not in name:
                    continue
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")

        base_link = next(
            (body for body in root.iter("body") if body.attrib.get("name") == "base_link"),
            None,
        )
        if base_link is None:
            raise ValueError("Spirit MOZ1 XML is missing base_link body")

        lite_geom = ET.Element(
            "geom",
            {
                "name": "base_link_lite_collision",
                "type": "box",
                "size": "0.33 0.26 0.20",
                "pos": "0 0 0.17",
                "class": "collision",
            },
        )
        base_link.insert(1, lite_geom)

        if collision_mode == "primitive":
            primitive_visual = ET.Element(
                "geom",
                {
                    "name": "base_link_primitive_visual",
                    "type": "box",
                    "size": "0.33 0.26 0.20",
                    "pos": "0 0 0.17",
                    "class": "visual",
                    "rgba": "0.6 0.6 0.6 1",
                },
            )
            base_link.insert(2, primitive_visual)
            for body in root.iter("body"):
                body_name = body.attrib.get("name", "")
                if body_name not in cls._PRIMITIVE_WHEEL_BODIES:
                    continue
                body.append(
                    ET.Element(
                        "geom",
                        {
                            "name": f"{body_name}_primitive_collision",
                            "type": "cylinder",
                            "size": "0.08 0.05",
                            "class": "collision",
                            "euler": "1.5707963 0 0",
                        },
                    )
                )
                body.append(
                    ET.Element(
                        "geom",
                        {
                            "name": f"{body_name}_primitive_visual",
                            "type": "cylinder",
                            "size": "0.08 0.05",
                            "class": "visual",
                            "euler": "1.5707963 0 0",
                            "rgba": "0.2 0.2 0.2 1",
                        },
                    )
                )
        return ET.tostring(root, encoding="unicode")

    def _path_targets(self, t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        cfg = self._config
        wt = cfg.path_omega * t
        x_tar = cfg.path_center_xy[0] + cfg.path_amp_x * jnp.sin(wt)
        y_tar = cfg.path_center_xy[1] + 0.5 * cfg.path_amp_y * jnp.sin(2.0 * wt)

        dx_dt = cfg.path_amp_x * cfg.path_omega * jnp.cos(wt)
        dy_dt = cfg.path_amp_y * cfg.path_omega * jnp.cos(2.0 * wt)
        yaw_tar = jnp.atan2(dy_dt, dx_dt)

        pos_tar = jnp.array([x_tar, y_tar, cfg.z_ref])
        vel_tar_world = jnp.array([dx_dt, dy_dt, 0.0])
        return pos_tar, vel_tar_world, yaw_tar

    def reset(self, rng: jax.Array) -> State:
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

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

        # 1) Wheels: direct torque control
        wheel_tau = (
            action[self._wheel_idx]
            * self._config.action_scale
            * self._config.wheel_torque_scale
            * self._wheel_tau_limit
        )
        wheel_tau = jnp.clip(wheel_tau, -self._wheel_tau_limit, self._wheel_tau_limit)

        # 2) Waist+arms: PD control to action-parameterized joint targets
        pd_action = action[self._pd_idx]
        pd_norm = (pd_action * self._config.action_scale + 1.0) / 2.0
        pd_joint_low = self.physical_joint_range[self._pd_idx, 0]
        pd_joint_high = self.physical_joint_range[self._pd_idx, 1]
        pd_targets = pd_joint_low + pd_norm * (pd_joint_high - pd_joint_low)

        pd_q = q[self._pd_idx]
        pd_qd = qd[self._pd_idx]
        pd_tau = self._pd_kp * (pd_targets - pd_q) - self._pd_kd * pd_qd
        pd_tau = jnp.clip(pd_tau, -self._pd_tau_limit, self._pd_tau_limit)

        # 3) Heads: fixed (not optimized)
        head_tau = jnp.zeros(2)

        ctrl = jnp.zeros(self.sys.nu)
        ctrl = ctrl.at[self._wheel_idx].set(wheel_tau)
        ctrl = ctrl.at[self._pd_idx].set(pd_tau)
        ctrl = ctrl.at[self._head_idx].set(head_tau)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Update path target at next control instant.
        next_step = state.info["step"] + 1
        t = next_step * self.dt
        pos_tar, vel_tar_world, yaw_tar = self._path_targets(t)
        vel_tar_body = global_to_body_velocity(vel_tar_world, x.rot[self._base_idx - 1])

        pos = x.pos[self._base_idx - 1]
        yaw = math.quat_to_euler(x.rot[self._base_idx - 1])[2]
        vb = global_to_body_velocity(xd.vel[self._base_idx - 1], x.rot[self._base_idx - 1])

        up_world = jnp.array([0.0, 0.0, 1.0])
        up_body = math.rotate(up_world, x.rot[self._base_idx - 1])

        q_act = pipeline_state.q[7:]
        q_waist = q_act[self._waist_leg_idx]
        q_arm = q_act[self._arm_idx]

        # Rewards
        reward_pos = -jnp.sum((pos[:2] - pos_tar[:2]) ** 2)
        d_yaw = jnp.atan2(jnp.sin(yaw - yaw_tar), jnp.cos(yaw - yaw_tar))
        reward_yaw = -jnp.square(d_yaw)
        reward_vel = -jnp.sum((vb[:2] - vel_tar_body[:2]) ** 2)
        reward_upright = -jnp.sum((up_body - up_world) ** 2)
        reward_height = -jnp.square(pos[2] - self._config.z_ref)
        reward_arm_pose = -jnp.sum((q_arm - state.info["q_arm_ref"]) ** 2)
        reward_waist_pose = -jnp.sum((q_waist - state.info["q_waist_ref"]) ** 2)
        ctrl_scale = jnp.concatenate([self._wheel_tau_limit, self._pd_tau_limit, jnp.ones(2)])
        reward_ctrl = -jnp.sum((ctrl / ctrl_scale) ** 2)

        reward = (
            reward_pos * 2.0
            + reward_yaw * 1.0
            + reward_vel * 1.5
            + reward_upright * 1.0
            + reward_height * 0.5
            + reward_arm_pose * 0.05
            + reward_waist_pose * 0.05
            + reward_ctrl * 0.005
        )

        # Done conditions
        done = jnp.dot(up_body, up_world) < 0.0
        done |= pos[2] < 0.20
        done |= jnp.any((q_act < self.joint_range[:, 0]) & self._joint_range_valid)
        done |= jnp.any((q_act > self.joint_range[:, 1]) & self._joint_range_valid)
        done |= ~jnp.isfinite(jnp.sum(pipeline_state.q))
        done |= ~jnp.isfinite(jnp.sum(pipeline_state.qd))
        done = done.astype(jnp.float32)

        state.info["step"] = next_step
        state.info["path_phase"] = self._config.path_omega * t
        state.info["pos_tar"] = pos_tar
        state.info["vel_tar_world"] = vel_tar_world
        state.info["vel_tar"] = vel_tar_body
        state.info["yaw_tar"] = yaw_tar

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
        yaw_err = jnp.atan2(jnp.sin(state_info["yaw_tar"] - yaw), jnp.cos(state_info["yaw_tar"] - yaw))

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
