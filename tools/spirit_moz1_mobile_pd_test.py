#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dial_mpc.envs.spirit_moz1_env import SpiritMoz1PathTrackEnv, SpiritMoz1PathTrackEnvConfig
from dial_mpc.utils.io_utils import load_dataclass_from_dict
from tools.spirit_moz1_mujoco_viewer import (
    get_joint_indices,
    load_config,
    load_spirit_mj_model,
    parse_override,
)


WAIST_NAMES = [f"LegWaist-{i}" for i in range(6)]
LEFT_ARM_NAMES = [f"LeftArm-{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"RightArm-{i}" for i in range(7)]
HEAD_NAMES = [f"Head-{i}" for i in range(2)]
WHEEL_NAMES = [f"Base-{i}" for i in range(4)]
ACTION_JOINT_NAMES = WAIST_NAMES + LEFT_ARM_NAMES + RIGHT_ARM_NAMES


def _deg_array(values: list[float]) -> np.ndarray:
    return np.deg2rad(np.array(values, dtype=np.float64))


def _first_crossing_time(progress: np.ndarray, threshold: float, time_s: np.ndarray) -> float | None:
    idx = np.flatnonzero(progress >= threshold)
    if idx.size == 0:
        return None
    return float(time_s[idx[0]])


def _settling_time(abs_error: np.ndarray, tol: float, time_s: np.ndarray) -> float | None:
    within = abs_error <= tol
    if not np.any(within):
        return None
    suffix_all = np.flip(np.cumprod(np.flip(within.astype(np.int32)))) == 1
    idx = np.flatnonzero(suffix_all)
    if idx.size == 0:
        return None
    return float(time_s[idx[0]])


def _signal_metrics(name: str, signal: np.ndarray, ref_pre: float, ref_post: float, time_after_s: np.ndarray, steady_window_steps: int, settle_tol: float) -> dict:
    step = ref_post - ref_pre
    steady_actual = float(np.mean(signal[-steady_window_steps:]))
    steady_err = steady_actual - ref_post
    pre_step_err = signal[0] - ref_pre
    abs_error = np.abs(signal - ref_post)
    if abs(step) < 1e-9:
        rise_time = 0.0
        overshoot = 0.0
        overshoot_pct = 0.0
    else:
        direction = np.sign(step)
        progress = direction * (signal - ref_pre)
        target = abs(step)
        t10 = _first_crossing_time(progress, 0.1 * target, time_after_s)
        t90 = _first_crossing_time(progress, 0.9 * target, time_after_s)
        rise_time = None if t10 is None or t90 is None else float(t90 - t10)
        overshoot = max(0.0, float(np.max(progress) - target))
        overshoot_pct = 100.0 * overshoot / target
    settling_time = _settling_time(abs_error, settle_tol, time_after_s)
    return {
        "name": name,
        "ref_pre": float(ref_pre),
        "ref_post": float(ref_post),
        "step": float(step),
        "pre_step_error": float(pre_step_err),
        "steady_state_error": float(steady_err),
        "steady_state_abs_error": float(abs(steady_err)),
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "overshoot": float(overshoot),
        "overshoot_pct": float(overshoot_pct),
    }


def _plot_joint_group(output_path: Path, title: str, time_s: np.ndarray, actual_deg: np.ndarray, ref_deg: np.ndarray, names: list[str]):
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.5 * n, 6)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name, actual, ref in zip(axes, names, actual_deg.T, ref_deg.T):
        ax.plot(time_s, actual, label=f"{name} q")
        ax.plot(time_s, ref, "--", label=f"{name} ref")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_chassis(output_path: Path, time_s: np.ndarray, vel_body: np.ndarray, vel_ref: np.ndarray):
    names = ["vx_body [m/s]", "vy_body [m/s]", "yaw_rate [rad/s]"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time_s, vel_body[:, idx], label="actual")
        ax.plot(time_s, vel_ref[:, idx], "--", label="ref")
        ax.set_ylabel(names[idx])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Chassis Velocity Tracking")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_chassis_pose(
    output_path: Path,
    time_s: np.ndarray,
    base_height: np.ndarray,
    roll_pitch_deg: np.ndarray,
    z_ref: float,
    tip_roll_limit_deg: float,
    tip_pitch_limit_deg: float,
    min_base_height: float,
):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(time_s, base_height, label="z")
    axes[0].axhline(z_ref, linestyle="--", label="z_ref")
    axes[0].axhline(min_base_height, linestyle=":", color="tab:red", label="min height")
    axes[0].set_ylabel("z [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(time_s, roll_pitch_deg[:, 0], label="roll")
    axes[1].axhline(tip_roll_limit_deg, linestyle=":", color="tab:red", label="+tip limit")
    axes[1].axhline(-tip_roll_limit_deg, linestyle=":", color="tab:red", label="-tip limit")
    axes[1].set_ylabel("roll [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(time_s, roll_pitch_deg[:, 1], label="pitch")
    axes[2].axhline(tip_pitch_limit_deg, linestyle=":", color="tab:red", label="+tip limit")
    axes[2].axhline(-tip_pitch_limit_deg, linestyle=":", color="tab:red", label="-tip limit")
    axes[2].set_ylabel("pitch [deg]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    fig.suptitle("Chassis Attitude / Height")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _quat_to_euler_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _command_scale(command_elapsed_s: float, ramp_time_s: float) -> float:
    if ramp_time_s <= 0.0:
        return 1.0
    return float(np.clip(command_elapsed_s / ramp_time_s, 0.0, 1.0))


def _angle_diff(target: float, source: float) -> float:
    return float(np.arctan2(np.sin(target - source), np.cos(target - source)))


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--example", type=str, default="spirit_moz1_mobile_joint_track")
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--command-time", type=float, default=3.0)
    parser.add_argument("--vel-settle-tol", type=float, default=0.03)
    parser.add_argument("--yaw-settle-tol", type=float, default=0.08)
    parser.add_argument("--joint-settle-tol-deg", type=float, default=0.15)
    parser.add_argument("--steady-window", type=float, default=0.4)
    parser.add_argument("--vx-cmd", type=float, default=0.25)
    parser.add_argument("--vy-cmd", type=float, default=0.0)
    parser.add_argument("--yaw-rate-cmd", type=float, default=0.0)
    parser.add_argument("--waist-delta-deg", type=float, nargs=6, default=[2.0, 3.0, -3.0, 3.0, 2.0, 5.0])
    parser.add_argument("--larm-delta-deg", type=float, nargs=7, default=[4.0, 6.0, 5.0, 6.0, 4.0, 4.0, 3.0])
    parser.add_argument("--rarm-delta-deg", type=float, nargs=7, default=[-4.0, 6.0, -5.0, -6.0, -4.0, 4.0, -3.0])
    parser.add_argument("--head-delta-deg", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_mobile_pd_test")
    args = parser.parse_args()

    cfg = load_config(args.example, args.config, parse_override(args.override))
    env_cfg = load_dataclass_from_dict(
        SpiritMoz1PathTrackEnvConfig,
        cfg,
        convert_list_to_array=True,
    )
    env = SpiritMoz1PathTrackEnv(env_cfg)
    model = load_spirit_mj_model("moz1.xml", fixed_base=False, gravity_off=False)
    data = mujoco.MjData(model)

    wheel_ids, wheel_qpos_adr, wheel_qvel_adr, wheel_act_ids = get_joint_indices(model, WHEEL_NAMES)
    _, action_qpos_adr, action_qvel_adr, action_act_ids = get_joint_indices(model, ACTION_JOINT_NAMES)
    _, head_qpos_adr, head_qvel_adr, head_act_ids = get_joint_indices(model, HEAD_NAMES)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    init_q = np.array(env._init_q, dtype=np.float64)
    default_joint_ref = np.asarray(env._joint_ref, dtype=np.float64).copy()
    default_head_ref = np.asarray(env._head_ref, dtype=np.float64).copy()
    init_q[np.asarray(env._action_joint_idx) + 7] = default_joint_ref
    init_q[np.asarray(env._head_idx) + 7] = default_head_ref
    data.qpos[:] = init_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    joint_ref_cmd = default_joint_ref.copy()
    joint_ref_cmd[:6] += _deg_array(args.waist_delta_deg)
    joint_ref_cmd[6:13] += _deg_array(args.larm_delta_deg)
    joint_ref_cmd[13:20] += _deg_array(args.rarm_delta_deg)
    head_ref_cmd = default_head_ref.copy()
    head_ref_cmd += _deg_array(args.head_delta_deg)

    wheel_kp = np.asarray(env._wheel_kp, dtype=np.float64)
    wheel_kd = np.asarray(env._wheel_kd, dtype=np.float64)
    wheel_tau_limit = np.asarray(env._wheel_tau_limit, dtype=np.float64)
    joint_kp = np.asarray(env._joint_kp, dtype=np.float64)
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64)
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64)
    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)
    chassis_model = str(cfg.get("chassis_model", "wheel_pd"))
    if chassis_model not in {"wheel_pd", "planar_wrench_servo"}:
        raise ValueError(
            "spirit_moz1_mobile_pd_test.py expects "
            f"chassis_model in {{'wheel_pd', 'planar_wrench_servo'}}, got {chassis_model!r}"
        )
    planar_wrench_kp = np.asarray(env_cfg.planar_wrench_kp, dtype=np.float64)
    planar_wrench_kd = np.asarray(env_cfg.planar_wrench_kd, dtype=np.float64)
    planar_wrench_ki = np.asarray(env_cfg.planar_wrench_ki, dtype=np.float64)
    planar_integrator_deadband = np.asarray(env_cfg.planar_integrator_deadband, dtype=np.float64)
    planar_integrator_leak = np.asarray(env_cfg.planar_integrator_leak, dtype=np.float64)
    planar_velocity_lpf_alpha = np.asarray(env_cfg.planar_velocity_lpf_alpha, dtype=np.float64)
    planar_wrench_slew_rate = np.asarray(env_cfg.planar_wrench_slew_rate, dtype=np.float64)
    planar_force_limit = np.asarray(env_cfg.planar_force_limit, dtype=np.float64)
    planar_torque_limit = float(env_cfg.planar_torque_limit)
    planar_cmd_ramp_time = float(env_cfg.planar_cmd_ramp_time)
    tip_roll_limit = float(env_cfg.tip_roll_limit)
    tip_pitch_limit = float(env_cfg.tip_pitch_limit)
    min_base_height = float(env_cfg.min_base_height)
    mobile_root_ref_lpf_alpha = float(env_cfg.mobile_root_ref_lpf_alpha)

    dt = float(model.opt.timestep)
    settle_steps = int(round(args.settle_time / dt))
    command_steps = int(round(args.command_time / dt))
    total_steps = settle_steps + command_steps
    steady_window_steps = max(1, int(round(args.steady_window / dt)))

    wheel_q_ref = data.qpos[wheel_qpos_adr].copy()
    body_cmd = np.array([args.vx_cmd, args.vy_cmd, args.yaw_rate_cmd], dtype=np.float64)
    time_s = np.arange(total_steps, dtype=np.float64) * dt
    vel_body_trace = np.zeros((total_steps, 3), dtype=np.float64)
    vel_ref_trace = np.zeros_like(vel_body_trace)
    vel_body_filt_trace = np.zeros_like(vel_body_trace)
    wrench_body_trace = np.zeros_like(vel_body_trace)
    wheel_speed_trace = np.zeros((total_steps, 4), dtype=np.float64)
    wheel_speed_ref_trace = np.zeros_like(wheel_speed_trace)
    wheel_tau_trace = np.zeros_like(wheel_speed_trace)
    q_trace = np.zeros((total_steps, len(ACTION_JOINT_NAMES)), dtype=np.float64)
    qd_trace = np.zeros_like(q_trace)
    tau_trace = np.zeros_like(q_trace)
    ref_trace = np.zeros_like(q_trace)
    head_q_trace = np.zeros((total_steps, len(HEAD_NAMES)), dtype=np.float64)
    head_tau_trace = np.zeros_like(head_q_trace)
    base_height_trace = np.zeros(total_steps, dtype=np.float64)
    base_rpy_trace = np.zeros((total_steps, 3), dtype=np.float64)

    base_vel_buf = np.zeros(6, dtype=np.float64)
    vel_int_error = np.zeros(3, dtype=np.float64)
    prev_vel_error = np.zeros(3, dtype=np.float64)
    prev_wrench_body = np.zeros(3, dtype=np.float64)
    vel_body_filt = np.zeros(3, dtype=np.float64)
    joint_ref_filt = default_joint_ref.copy()
    steps_run = total_steps
    failure_reason = None

    for step in range(total_steps):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, base_vel_buf, 1)
        vel_body_now = np.array([base_vel_buf[3], base_vel_buf[4], base_vel_buf[2]], dtype=np.float64)
        base_pos_now = np.array(data.xpos[base_body_id], dtype=np.float64)
        base_quat_now = np.array(data.xquat[base_body_id], dtype=np.float64)
        base_rpy_now = _quat_to_euler_wxyz(base_quat_now)
        if step == 0:
            vel_body_filt = vel_body_now.copy()
        else:
            vel_body_filt = (
                planar_velocity_lpf_alpha * vel_body_now
                + (1.0 - planar_velocity_lpf_alpha) * vel_body_filt
            )

        if step < settle_steps:
            joint_ref = default_joint_ref
            head_ref = default_head_ref
            wheel_speed_tar = np.zeros(4, dtype=np.float64)
            active_body_cmd = np.zeros(3, dtype=np.float64)
            wrench_body = np.zeros(3, dtype=np.float64)
            vel_int_error[:] = 0.0
            prev_vel_error[:] = 0.0
            prev_wrench_body[:] = 0.0
            joint_ref_filt[:] = default_joint_ref
        else:
            root_alpha = np.clip(mobile_root_ref_lpf_alpha, 0.0, 1.0)
            joint_ref_filt[:6] = (1.0 - root_alpha) * joint_ref_filt[:6] + root_alpha * joint_ref_cmd[:6]
            joint_ref_filt[6:] = joint_ref_cmd[6:]
            joint_ref = joint_ref_filt
            head_ref = head_ref_cmd
            cmd_elapsed_s = (step - settle_steps) * dt
            cmd_scale = _command_scale(cmd_elapsed_s, planar_cmd_ramp_time)
            active_body_cmd = body_cmd * cmd_scale
            wheel_speed_tar = np.asarray(
                env._body_command_to_wheel_speed(
                    np.array([active_body_cmd[0], active_body_cmd[1], 0.0], dtype=np.float64),
                    np.array([0.0, 0.0, active_body_cmd[2]], dtype=np.float64),
                ),
                dtype=np.float64,
            )
            if chassis_model == "planar_wrench_servo":
                vel_error = active_body_cmd - vel_body_filt
                active_integrator = np.abs(vel_error) >= planar_integrator_deadband
                vel_int_error = vel_int_error * (1.0 - planar_integrator_leak * dt)
                vel_int_error += np.where(active_integrator, vel_error, 0.0) * dt
                force_int_limit = planar_force_limit / np.maximum(planar_wrench_ki[:2], 1e-6)
                torque_int_limit = planar_torque_limit / max(planar_wrench_ki[2], 1e-6)
                vel_int_error[:2] = np.clip(vel_int_error[:2], -force_int_limit, force_int_limit)
                vel_int_error[2] = np.clip(vel_int_error[2], -torque_int_limit, torque_int_limit)
                vel_error_dot = (vel_error - prev_vel_error) / dt
                prev_vel_error = vel_error
                desired_wrench_body = (
                    planar_wrench_kp * vel_error
                    + planar_wrench_ki * vel_int_error
                    + planar_wrench_kd * vel_error_dot
                )
                desired_wrench_body[:2] = np.clip(
                    desired_wrench_body[:2], -planar_force_limit, planar_force_limit
                )
                desired_wrench_body[2] = np.clip(
                    desired_wrench_body[2], -planar_torque_limit, planar_torque_limit
                )
                saturated = np.array(
                    [
                        np.isclose(abs(desired_wrench_body[0]), planar_force_limit[0]),
                        np.isclose(abs(desired_wrench_body[1]), planar_force_limit[1]),
                        np.isclose(abs(desired_wrench_body[2]), planar_torque_limit),
                    ],
                    dtype=bool,
                )
                if np.any(saturated):
                    vel_int_error[saturated] *= 0.92
                wrench_delta_limit = planar_wrench_slew_rate * dt
                wrench_body = prev_wrench_body + np.clip(
                    desired_wrench_body - prev_wrench_body,
                    -wrench_delta_limit,
                    wrench_delta_limit,
                )
                wrench_body[:2] = np.clip(wrench_body[:2], -planar_force_limit, planar_force_limit)
                wrench_body[2] = np.clip(wrench_body[2], -planar_torque_limit, planar_torque_limit)
                prev_wrench_body = wrench_body
            else:
                wrench_body = np.zeros(3, dtype=np.float64)

        wheel_q_ref = wheel_q_ref + wheel_speed_tar * dt

        wheel_q = data.qpos[wheel_qpos_adr]
        wheel_qd = data.qvel[wheel_qvel_adr]
        if chassis_model == "wheel_pd":
            wheel_tau = wheel_kp * (wheel_q_ref - wheel_q) + wheel_kd * (wheel_speed_tar - wheel_qd)
            wheel_tau = np.clip(wheel_tau, -wheel_tau_limit, wheel_tau_limit)
        else:
            wheel_tau = np.zeros_like(wheel_qd)

        joint_q = data.qpos[action_qpos_adr]
        joint_qd = data.qvel[action_qvel_adr]
        joint_tau = joint_kp * (joint_ref - joint_q) - joint_kd * joint_qd
        joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

        head_q = data.qpos[head_qpos_adr]
        head_qd = data.qvel[head_qvel_adr]
        head_tau = head_kp * (head_ref - head_q) - head_kd * head_qd
        head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

        vel_body_trace[step] = vel_body_now
        vel_ref_trace[step] = active_body_cmd
        vel_body_filt_trace[step] = vel_body_filt
        wrench_body_trace[step] = wrench_body
        wheel_speed_trace[step] = wheel_qd
        wheel_speed_ref_trace[step] = wheel_speed_tar
        wheel_tau_trace[step] = wheel_tau
        q_trace[step] = joint_q
        qd_trace[step] = joint_qd
        tau_trace[step] = joint_tau
        ref_trace[step] = joint_ref
        head_q_trace[step] = head_q
        head_tau_trace[step] = head_tau
        base_height_trace[step] = base_pos_now[2]
        base_rpy_trace[step] = base_rpy_now

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.ctrl[wheel_act_ids] = wheel_tau
        data.ctrl[action_act_ids] = joint_tau
        data.ctrl[head_act_ids] = head_tau
        if chassis_model == "planar_wrench_servo":
            base_rot = np.array(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
            force_world = base_rot @ np.array([wrench_body[0], wrench_body[1], 0.0], dtype=np.float64)
            torque_world = base_rot @ np.array([0.0, 0.0, wrench_body[2]], dtype=np.float64)
            mujoco.mj_applyFT(
                model,
                data,
                force_world,
                torque_world,
                np.array(data.xipos[base_body_id], dtype=np.float64),
                base_body_id,
                data.qfrc_applied,
            )
        mujoco.mj_step(model, data)

        post_quat = np.array(data.xquat[base_body_id], dtype=np.float64)
        post_rpy = _quat_to_euler_wxyz(post_quat)
        post_pos = np.array(data.xpos[base_body_id], dtype=np.float64)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or abs(post_rpy[0]) > tip_roll_limit
            or abs(post_rpy[1]) > tip_pitch_limit
            or post_pos[2] < min_base_height
        ):
            steps_run = step + 1
            failure_reason = {
                "roll_deg": float(np.rad2deg(post_rpy[0])),
                "pitch_deg": float(np.rad2deg(post_rpy[1])),
                "base_height_m": float(post_pos[2]),
            }
            break

    time_s = time_s[:steps_run]
    vel_body_trace = vel_body_trace[:steps_run]
    vel_ref_trace = vel_ref_trace[:steps_run]
    vel_body_filt_trace = vel_body_filt_trace[:steps_run]
    wrench_body_trace = wrench_body_trace[:steps_run]
    wheel_speed_trace = wheel_speed_trace[:steps_run]
    wheel_speed_ref_trace = wheel_speed_ref_trace[:steps_run]
    wheel_tau_trace = wheel_tau_trace[:steps_run]
    q_trace = q_trace[:steps_run]
    qd_trace = qd_trace[:steps_run]
    tau_trace = tau_trace[:steps_run]
    ref_trace = ref_trace[:steps_run]
    head_q_trace = head_q_trace[:steps_run]
    head_tau_trace = head_tau_trace[:steps_run]
    base_height_trace = base_height_trace[:steps_run]
    base_rpy_trace = base_rpy_trace[:steps_run]

    if steps_run <= settle_steps:
        raise RuntimeError(
            f"Mobile-base run terminated before command phase. failure_reason={failure_reason}"
        )

    time_after_s = time_s[settle_steps:] - time_s[settle_steps]

    base_metrics = [
        _signal_metrics(
            "vx_body_mps",
            vel_body_trace[settle_steps:, 0],
            0.0,
            args.vx_cmd,
            time_after_s,
            steady_window_steps,
            args.vel_settle_tol,
        ),
        _signal_metrics(
            "vy_body_mps",
            vel_body_trace[settle_steps:, 1],
            0.0,
            args.vy_cmd,
            time_after_s,
            steady_window_steps,
            args.vel_settle_tol,
        ),
        _signal_metrics(
            "yaw_rate_rps",
            vel_body_trace[settle_steps:, 2],
            0.0,
            args.yaw_rate_cmd,
            time_after_s,
            steady_window_steps,
            args.yaw_settle_tol,
        ),
    ]

    joint_metrics = []
    for idx, joint_name in enumerate(ACTION_JOINT_NAMES):
        joint_metrics.append(
            {
                "joint": joint_name,
                **_signal_metrics(
                    joint_name,
                    np.rad2deg(q_trace[settle_steps:, idx]),
                    float(np.rad2deg(default_joint_ref[idx])),
                    float(np.rad2deg(joint_ref_cmd[idx])),
                    time_after_s,
                    steady_window_steps,
                    args.joint_settle_tol_deg,
                ),
                "peak_tau_nm": float(np.max(np.abs(tau_trace[settle_steps:, idx]))),
            }
        )

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "trace_data.npz",
        time_s=time_s,
        vel_body_trace=vel_body_trace,
        vel_ref_trace=vel_ref_trace,
        vel_body_filt_trace=vel_body_filt_trace,
        wrench_body_trace=wrench_body_trace,
        wheel_speed_trace=wheel_speed_trace,
        wheel_speed_ref_trace=wheel_speed_ref_trace,
        wheel_tau_trace=wheel_tau_trace,
        ref_trace=ref_trace,
        q_trace=q_trace,
        qd_trace=qd_trace,
        tau_trace=tau_trace,
        head_q_trace=head_q_trace,
        head_tau_trace=head_tau_trace,
        base_height_trace=base_height_trace,
        base_rpy_trace=base_rpy_trace,
    )

    summary = {
        "example": args.example,
        "config": cfg,
        "test": {
            "settle_time_s": args.settle_time,
            "command_time_s": args.command_time,
            "vx_cmd_mps": args.vx_cmd,
            "vy_cmd_mps": args.vy_cmd,
            "yaw_rate_cmd_rps": args.yaw_rate_cmd,
            "waist_delta_deg": args.waist_delta_deg,
            "larm_delta_deg": args.larm_delta_deg,
            "rarm_delta_deg": args.rarm_delta_deg,
            "head_delta_deg": args.head_delta_deg,
        },
        "base_metrics": base_metrics,
        "joint_metrics": joint_metrics,
        "aggregate": {
            "worst_base_steady_abs_error": float(max(metric["steady_state_abs_error"] for metric in base_metrics)),
            "worst_base_overshoot_pct": float(max(metric["overshoot_pct"] for metric in base_metrics)),
            "worst_joint_steady_abs_error_deg": float(max(metric["steady_state_abs_error"] for metric in joint_metrics)),
            "worst_joint_overshoot_pct": float(max(metric["overshoot_pct"] for metric in joint_metrics)),
            "worst_joint_peak_tau_nm": float(max(metric["peak_tau_nm"] for metric in joint_metrics)),
            "worst_wheel_tau_nm": float(np.max(np.abs(wheel_tau_trace[settle_steps:]))),
            "peak_roll_deg": float(np.max(np.abs(np.rad2deg(base_rpy_trace[:, 0])))),
            "peak_pitch_deg": float(np.max(np.abs(np.rad2deg(base_rpy_trace[:, 1])))),
            "min_base_height_m": float(np.min(base_height_trace)),
            "peak_planar_wrench_force_n": float(np.max(np.abs(wrench_body_trace[:, :2]))),
            "peak_planar_wrench_torque_nm": float(np.max(np.abs(wrench_body_trace[:, 2]))),
            "terminated_early": bool(failure_reason is not None),
        },
        "failure_reason": failure_reason,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_chassis(run_dir / "chassis_velocity_tracking.png", time_s, vel_body_trace, vel_ref_trace)
    _plot_chassis_pose(
        run_dir / "chassis_pose_tracking.png",
        time_s,
        base_height_trace,
        np.rad2deg(base_rpy_trace[:, :2]),
        float(env_cfg.z_ref),
        float(np.rad2deg(tip_roll_limit)),
        float(np.rad2deg(tip_pitch_limit)),
        min_base_height,
    )
    _plot_joint_group(
        run_dir / "waist_tracking.png",
        "Waist Tracking (Mobile Base)",
        time_s,
        np.rad2deg(q_trace[:, :6]),
        np.rad2deg(ref_trace[:, :6]),
        WAIST_NAMES,
    )
    _plot_joint_group(
        run_dir / "left_arm_tracking.png",
        "Left Arm Tracking (Mobile Base)",
        time_s,
        np.rad2deg(q_trace[:, 6:13]),
        np.rad2deg(ref_trace[:, 6:13]),
        LEFT_ARM_NAMES,
    )
    _plot_joint_group(
        run_dir / "right_arm_tracking.png",
        "Right Arm Tracking (Mobile Base)",
        time_s,
        np.rad2deg(q_trace[:, 13:20]),
        np.rad2deg(ref_trace[:, 13:20]),
        RIGHT_ARM_NAMES,
    )

    print("saved_run_dir", run_dir)
    print("worst_base_steady_abs_error", summary["aggregate"]["worst_base_steady_abs_error"])
    print("worst_base_overshoot_pct", summary["aggregate"]["worst_base_overshoot_pct"])
    print("worst_joint_steady_abs_error_deg", summary["aggregate"]["worst_joint_steady_abs_error_deg"])
    print("worst_joint_overshoot_pct", summary["aggregate"]["worst_joint_overshoot_pct"])
    print("worst_wheel_tau_nm", summary["aggregate"]["worst_wheel_tau_nm"])
    print("peak_roll_deg", summary["aggregate"]["peak_roll_deg"])
    print("peak_pitch_deg", summary["aggregate"]["peak_pitch_deg"])
    print("min_base_height_m", summary["aggregate"]["min_base_height_m"])
    print("terminated_early", summary["aggregate"]["terminated_early"])


if __name__ == "__main__":
    main()
