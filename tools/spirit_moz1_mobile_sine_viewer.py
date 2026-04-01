#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
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
from tools.spirit_moz1_mobile_pd_test import (
    _plot_chassis,
    _plot_chassis_pose,
    _plot_joint_group,
)


WAIST_NAMES = [f"LegWaist-{i}" for i in range(6)]
LEFT_ARM_NAMES = [f"LeftArm-{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"RightArm-{i}" for i in range(7)]
HEAD_NAMES = [f"Head-{i}" for i in range(2)]
WHEEL_NAMES = [f"Base-{i}" for i in range(4)]
ACTION_JOINT_NAMES = WAIST_NAMES + LEFT_ARM_NAMES + RIGHT_ARM_NAMES


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


def _body_sine_command(t: float, args: argparse.Namespace) -> np.ndarray:
    omega = 2.0 * math.pi * args.freq_hz
    return np.array(
        [
            args.vx_amp * math.sin(omega * t),
            args.vy_amp * math.cos(omega * t),
            args.yaw_rate_amp * math.sin(omega * t + args.yaw_phase_rad),
        ],
        dtype=np.float64,
    )


def _joint_sine_offsets(t: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * math.pi * args.joint_freq_hz

    waist_pattern_deg = np.array([0.30, 0.60, -1.00, 0.70, 0.35, 0.50], dtype=np.float64)
    larm_pattern_deg = np.array([0.60, 1.00, 0.80, 0.90, 0.45, 0.35, 0.30], dtype=np.float64)
    rarm_pattern_deg = np.array([-0.60, 1.00, -0.80, -0.90, -0.45, 0.35, -0.30], dtype=np.float64)
    head_pattern_deg = np.array([0.70, 0.45], dtype=np.float64)

    phase_a = omega * t
    phase_b = omega * t + 0.5 * math.pi
    phase_c = omega * t + math.pi

    waist_offset = np.deg2rad(args.waist_amp_deg) * waist_pattern_deg * math.sin(phase_a)
    larm_offset = np.deg2rad(args.arm_amp_deg) * larm_pattern_deg * math.sin(phase_b)
    rarm_offset = np.deg2rad(args.arm_amp_deg) * rarm_pattern_deg * math.sin(phase_c)
    head_offset = np.deg2rad(args.head_amp_deg) * head_pattern_deg * math.sin(phase_b)

    joint_offset = np.concatenate([waist_offset, larm_offset, rarm_offset])
    return joint_offset, head_offset


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Spirit MOZ1 mobile-base tracking under sinusoidal commands."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--example", type=str, default="spirit_moz1_mobile_joint_track")
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--duration", type=float, default=30.0, help="Viewer run time in seconds.")
    parser.add_argument("--freq-hz", type=float, default=0.12, help="Command sine frequency.")
    parser.add_argument("--vx-amp", type=float, default=0.25, help="Sine amplitude for vx [m/s].")
    parser.add_argument("--vy-amp", type=float, default=0.12, help="Cosine amplitude for vy [m/s].")
    parser.add_argument(
        "--yaw-rate-amp",
        type=float,
        default=0.25,
        help="Sine amplitude for yaw rate [rad/s].",
    )
    parser.add_argument(
        "--yaw-phase-deg",
        type=float,
        default=90.0,
        help="Yaw-rate sine phase shift in degrees.",
    )
    parser.add_argument(
        "--joint-freq-hz",
        type=float,
        default=0.18,
        help="Upper-joint sine frequency.",
    )
    parser.add_argument(
        "--waist-amp-deg",
        type=float,
        default=6.0,
        help="Scale factor for waist sine motion in degrees.",
    )
    parser.add_argument(
        "--arm-amp-deg",
        type=float,
        default=10.0,
        help="Scale factor for left/right arm sine motion in degrees.",
    )
    parser.add_argument(
        "--head-amp-deg",
        type=float,
        default=4.0,
        help="Scale factor for head sine motion in degrees.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Console print interval in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spirit_moz1_mobile_sine_viewer",
        help="Directory used to save tracking curves after the viewer exits.",
    )
    args = parser.parse_args()
    args.yaw_phase_rad = math.radians(args.yaw_phase_deg)

    cfg = load_config(args.example, args.config, parse_override(args.override))
    env_cfg = load_dataclass_from_dict(
        SpiritMoz1PathTrackEnvConfig,
        cfg,
        convert_list_to_array=True,
    )
    env = SpiritMoz1PathTrackEnv(env_cfg)

    chassis_model = str(cfg.get("chassis_model", "wheel_pd"))
    if chassis_model not in {"wheel_pd", "planar_wrench_servo"}:
        raise ValueError(
            "spirit_moz1_mobile_sine_viewer.py expects "
            f"chassis_model in {{'wheel_pd', 'planar_wrench_servo'}}, got {chassis_model!r}"
        )

    model = load_spirit_mj_model("moz1.xml", fixed_base=False, gravity_off=False)
    data = mujoco.MjData(model)

    _, wheel_qpos_adr, wheel_qvel_adr, wheel_act_ids = get_joint_indices(model, WHEEL_NAMES)
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

    wheel_q_ref = data.qpos[wheel_qpos_adr].copy()
    joint_ref_filt = default_joint_ref.copy()

    wheel_kp = np.asarray(env._wheel_kp, dtype=np.float64)
    wheel_kd = np.asarray(env._wheel_kd, dtype=np.float64)
    wheel_tau_limit = np.asarray(env._wheel_tau_limit, dtype=np.float64)
    joint_kp = np.asarray(env._joint_kp, dtype=np.float64)
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64)
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64)
    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)

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
    mobile_root_ref_lpf_alpha = float(env_cfg.mobile_root_ref_lpf_alpha)

    vel_body_filt = np.zeros(3, dtype=np.float64)
    vel_int_error = np.zeros(3, dtype=np.float64)
    prev_vel_error = np.zeros(3, dtype=np.float64)
    prev_wrench_body = np.zeros(3, dtype=np.float64)
    base_vel_buf = np.zeros(6, dtype=np.float64)
    dt = float(model.opt.timestep)
    next_print_time = 0.0
    time_trace = []
    vel_body_trace = []
    vel_ref_trace = []
    base_height_trace = []
    base_rpy_trace = []
    wrench_body_trace = []
    waist_q_trace = []
    waist_ref_trace = []
    left_arm_q_trace = []
    left_arm_ref_trace = []
    right_arm_q_trace = []
    right_arm_ref_trace = []
    head_q_trace = []
    head_ref_trace = []

    print("Viewer controls: drag to rotate camera, scroll to zoom, close window to exit.")
    print(
        "Sine command:"
        f" vx_amp={args.vx_amp:.3f}, vy_amp={args.vy_amp:.3f},"
        f" yaw_rate_amp={args.yaw_rate_amp:.3f}, freq={args.freq_hz:.3f} Hz"
    )
    print(
        "Joint sine:"
        f" waist_amp={args.waist_amp_deg:.1f}deg,"
        f" arm_amp={args.arm_amp_deg:.1f}deg,"
        f" head_amp={args.head_amp_deg:.1f}deg,"
        f" joint_freq={args.joint_freq_hz:.3f} Hz"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.perf_counter()
        while viewer.is_running():
            sim_t = float(data.time)
            if sim_t >= args.duration:
                break

            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, base_vel_buf, 1
            )
            vel_body_now = np.array(
                [base_vel_buf[3], base_vel_buf[4], base_vel_buf[2]],
                dtype=np.float64,
            )
            base_quat = np.array(data.xquat[base_body_id], dtype=np.float64)
            base_rpy = _quat_to_euler_wxyz(base_quat)

            if sim_t <= dt:
                vel_body_filt = vel_body_now.copy()
            else:
                vel_body_filt = (
                    planar_velocity_lpf_alpha * vel_body_now
                    + (1.0 - planar_velocity_lpf_alpha) * vel_body_filt
                )

            cmd_scale = _command_scale(sim_t, planar_cmd_ramp_time)
            body_cmd = _body_sine_command(sim_t, args) * cmd_scale
            wheel_speed_tar = np.asarray(
                env._body_command_to_wheel_speed(
                    np.array([body_cmd[0], body_cmd[1], 0.0], dtype=np.float64),
                    np.array([0.0, 0.0, body_cmd[2]], dtype=np.float64),
                ),
                dtype=np.float64,
            )
            wheel_q_ref = wheel_q_ref + wheel_speed_tar * dt

            if chassis_model == "planar_wrench_servo":
                vel_error = body_cmd - vel_body_filt
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
                wheel_tau = np.zeros(4, dtype=np.float64)
            else:
                wheel_q = data.qpos[wheel_qpos_adr]
                wheel_qd = data.qvel[wheel_qvel_adr]
                wheel_tau = wheel_kp * (wheel_q_ref - wheel_q) + wheel_kd * (wheel_speed_tar - wheel_qd)
                wheel_tau = np.clip(wheel_tau, -wheel_tau_limit, wheel_tau_limit)
                wrench_body = np.zeros(3, dtype=np.float64)

            joint_ref_cmd = default_joint_ref.copy()
            head_ref_cmd = default_head_ref.copy()
            joint_offset, head_offset = _joint_sine_offsets(sim_t, args)
            joint_ref_cmd += joint_offset
            head_ref_cmd += head_offset
            root_alpha = float(np.clip(mobile_root_ref_lpf_alpha, 0.0, 1.0))
            joint_ref_filt[:6] = (1.0 - root_alpha) * joint_ref_filt[:6] + root_alpha * joint_ref_cmd[:6]
            joint_ref_filt[6:] = joint_ref_cmd[6:]

            joint_q = data.qpos[action_qpos_adr]
            joint_qd = data.qvel[action_qvel_adr]
            joint_tau = joint_kp * (joint_ref_filt - joint_q) - joint_kd * joint_qd
            joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

            head_q = data.qpos[head_qpos_adr]
            head_qd = data.qvel[head_qvel_adr]
            head_tau = head_kp * (head_ref_cmd - head_q) - head_kd * head_qd
            head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

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

            time_trace.append(sim_t)
            vel_body_trace.append(vel_body_now.copy())
            vel_ref_trace.append(body_cmd.copy())
            base_height_trace.append(float(data.xpos[base_body_id][2]))
            base_rpy_trace.append(base_rpy.copy())
            wrench_body_trace.append(wrench_body.copy())
            waist_q_trace.append(joint_q[:6].copy())
            waist_ref_trace.append(joint_ref_filt[:6].copy())
            left_arm_q_trace.append(joint_q[6:13].copy())
            left_arm_ref_trace.append(joint_ref_filt[6:13].copy())
            right_arm_q_trace.append(joint_q[13:20].copy())
            right_arm_ref_trace.append(joint_ref_filt[13:20].copy())
            head_q_trace.append(head_q.copy())
            head_ref_trace.append(head_ref_cmd.copy())

            mujoco.mj_step(model, data)
            viewer.sync()

            if sim_t >= next_print_time:
                print(
                    f"t={sim_t:5.2f}s"
                    f" cmd=[{body_cmd[0]: .3f}, {body_cmd[1]: .3f}, {body_cmd[2]: .3f}]"
                    f" actual=[{vel_body_now[0]: .3f}, {vel_body_now[1]: .3f}, {vel_body_now[2]: .3f}]"
                    f" waist2_ref={np.rad2deg(joint_ref_filt[2]): .2f}deg"
                    f" roll={np.rad2deg(base_rpy[0]): .2f}deg"
                    f" pitch={np.rad2deg(base_rpy[1]): .2f}deg"
                )
                next_print_time += args.print_every

            elapsed = time.perf_counter() - start_wall
            target = data.time
            sleep_time = target - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    if not time_trace:
        print("No samples recorded; viewer exited before simulation started.")
        return

    time_s = np.asarray(time_trace, dtype=np.float64)
    vel_body_trace = np.asarray(vel_body_trace, dtype=np.float64)
    vel_ref_trace = np.asarray(vel_ref_trace, dtype=np.float64)
    base_height_trace = np.asarray(base_height_trace, dtype=np.float64)
    base_rpy_trace = np.asarray(base_rpy_trace, dtype=np.float64)
    wrench_body_trace = np.asarray(wrench_body_trace, dtype=np.float64)
    waist_q_trace = np.asarray(waist_q_trace, dtype=np.float64)
    waist_ref_trace = np.asarray(waist_ref_trace, dtype=np.float64)
    left_arm_q_trace = np.asarray(left_arm_q_trace, dtype=np.float64)
    left_arm_ref_trace = np.asarray(left_arm_ref_trace, dtype=np.float64)
    right_arm_q_trace = np.asarray(right_arm_q_trace, dtype=np.float64)
    right_arm_ref_trace = np.asarray(right_arm_ref_trace, dtype=np.float64)
    head_q_trace = np.asarray(head_q_trace, dtype=np.float64)
    head_ref_trace = np.asarray(head_ref_trace, dtype=np.float64)

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "trace_data.npz",
        time_s=time_s,
        vel_body_trace=vel_body_trace,
        vel_ref_trace=vel_ref_trace,
        base_height_trace=base_height_trace,
        base_rpy_trace=base_rpy_trace,
        wrench_body_trace=wrench_body_trace,
        waist_q_trace=waist_q_trace,
        waist_ref_trace=waist_ref_trace,
        left_arm_q_trace=left_arm_q_trace,
        left_arm_ref_trace=left_arm_ref_trace,
        right_arm_q_trace=right_arm_q_trace,
        right_arm_ref_trace=right_arm_ref_trace,
        head_q_trace=head_q_trace,
        head_ref_trace=head_ref_trace,
    )

    summary = {
        "example": args.example,
        "duration_s": float(time_s[-1]),
        "command": {
            "freq_hz": args.freq_hz,
            "vx_amp_mps": args.vx_amp,
            "vy_amp_mps": args.vy_amp,
            "yaw_rate_amp_rps": args.yaw_rate_amp,
            "yaw_phase_deg": args.yaw_phase_deg,
            "joint_freq_hz": args.joint_freq_hz,
            "waist_amp_deg": args.waist_amp_deg,
            "arm_amp_deg": args.arm_amp_deg,
            "head_amp_deg": args.head_amp_deg,
        },
        "aggregate": {
            "peak_abs_vx_error": float(np.max(np.abs(vel_ref_trace[:, 0] - vel_body_trace[:, 0]))),
            "peak_abs_vy_error": float(np.max(np.abs(vel_ref_trace[:, 1] - vel_body_trace[:, 1]))),
            "peak_abs_yaw_rate_error": float(np.max(np.abs(vel_ref_trace[:, 2] - vel_body_trace[:, 2]))),
            "peak_roll_deg": float(np.max(np.abs(np.rad2deg(base_rpy_trace[:, 0])))),
            "peak_pitch_deg": float(np.max(np.abs(np.rad2deg(base_rpy_trace[:, 1])))),
            "min_base_height_m": float(np.min(base_height_trace)),
            "peak_planar_wrench_force_n": float(np.max(np.abs(wrench_body_trace[:, :2]))),
            "peak_planar_wrench_torque_nm": float(np.max(np.abs(wrench_body_trace[:, 2]))),
        },
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
        float(np.rad2deg(env_cfg.tip_roll_limit)),
        float(np.rad2deg(env_cfg.tip_pitch_limit)),
        float(env_cfg.min_base_height),
    )
    _plot_joint_group(
        run_dir / "waist_tracking.png",
        "Waist Tracking (Sine Viewer)",
        time_s,
        np.rad2deg(waist_q_trace),
        np.rad2deg(waist_ref_trace),
        WAIST_NAMES,
    )
    _plot_joint_group(
        run_dir / "left_arm_tracking.png",
        "Left Arm Tracking (Sine Viewer)",
        time_s,
        np.rad2deg(left_arm_q_trace),
        np.rad2deg(left_arm_ref_trace),
        LEFT_ARM_NAMES,
    )
    _plot_joint_group(
        run_dir / "right_arm_tracking.png",
        "Right Arm Tracking (Sine Viewer)",
        time_s,
        np.rad2deg(right_arm_q_trace),
        np.rad2deg(right_arm_ref_trace),
        RIGHT_ARM_NAMES,
    )
    _plot_joint_group(
        run_dir / "head_tracking.png",
        "Head Tracking (Sine Viewer)",
        time_s,
        np.rad2deg(head_q_trace),
        np.rad2deg(head_ref_trace),
        HEAD_NAMES,
    )

    print(f"saved_run_dir {run_dir}")
    print(f"saved_plot {run_dir / 'chassis_velocity_tracking.png'}")
    print(f"saved_plot {run_dir / 'chassis_pose_tracking.png'}")
    print(f"saved_plot {run_dir / 'waist_tracking.png'}")
    print(f"saved_plot {run_dir / 'left_arm_tracking.png'}")
    print(f"saved_plot {run_dir / 'right_arm_tracking.png'}")
    print(f"saved_plot {run_dir / 'head_tracking.png'}")


if __name__ == "__main__":
    main()
