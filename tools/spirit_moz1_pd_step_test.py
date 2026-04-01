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


def _deg_array(values: list[float]) -> np.ndarray:
    return np.deg2rad(np.array(values, dtype=np.float64))


def _build_targets(default_joint_ref: np.ndarray, default_head_ref: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    joint_ref = default_joint_ref.copy()
    head_ref = default_head_ref.copy()

    joint_ref[:6] += _deg_array(args.waist_delta_deg)
    joint_ref[6:13] += _deg_array(args.larm_delta_deg)
    joint_ref[13:20] += _deg_array(args.rarm_delta_deg)
    head_ref += _deg_array(args.head_delta_deg)

    return joint_ref, head_ref


def _first_crossing_time(progress: np.ndarray, threshold: float, time_s: np.ndarray) -> float | None:
    idx = np.flatnonzero(progress >= threshold)
    if idx.size == 0:
        return None
    return float(time_s[idx[0]])


def _settling_time(abs_error_deg: np.ndarray, tol_deg: float, time_s: np.ndarray) -> float | None:
    within = abs_error_deg <= tol_deg
    if not np.any(within):
        return None
    suffix_all = np.flip(np.cumprod(np.flip(within.astype(np.int32)))) == 1
    idx = np.flatnonzero(suffix_all)
    if idx.size == 0:
        return None
    return float(time_s[idx[0]])


def _joint_metrics(
    joint_name: str,
    signal_rad: np.ndarray,
    ref_pre_rad: float,
    ref_post_rad: float,
    time_after_s: np.ndarray,
    steady_window_steps: int,
    settle_tol_deg: float,
    tau_nm: np.ndarray,
) -> dict:
    signal_deg = np.rad2deg(signal_rad)
    ref_pre_deg = float(np.rad2deg(ref_pre_rad))
    ref_post_deg = float(np.rad2deg(ref_post_rad))
    step_deg = ref_post_deg - ref_pre_deg
    steady_actual_deg = float(np.mean(signal_deg[-steady_window_steps:]))
    steady_err_deg = steady_actual_deg - ref_post_deg
    pre_step_err_deg = signal_deg[0] - ref_pre_deg
    abs_error_deg = np.abs(signal_deg - ref_post_deg)

    if abs(step_deg) < 1e-6:
        rise_time_s = 0.0
        overshoot_deg = 0.0
        overshoot_pct = 0.0
    else:
        direction = np.sign(step_deg)
        progress_deg = direction * (signal_deg - ref_pre_deg)
        target_deg = abs(step_deg)
        t10 = _first_crossing_time(progress_deg, 0.1 * target_deg, time_after_s)
        t90 = _first_crossing_time(progress_deg, 0.9 * target_deg, time_after_s)
        rise_time_s = None if t10 is None or t90 is None else float(t90 - t10)
        overshoot_deg = max(0.0, float(np.max(progress_deg) - target_deg))
        overshoot_pct = 100.0 * overshoot_deg / target_deg

    settling_time_s = _settling_time(abs_error_deg, settle_tol_deg, time_after_s)

    return {
        "joint": joint_name,
        "ref_pre_deg": ref_pre_deg,
        "ref_post_deg": ref_post_deg,
        "step_deg": step_deg,
        "pre_step_error_deg": float(pre_step_err_deg),
        "steady_state_error_deg": float(steady_err_deg),
        "steady_state_abs_error_deg": float(abs(steady_err_deg)),
        "rise_time_s": rise_time_s,
        "settling_time_s": settling_time_s,
        "overshoot_deg": overshoot_deg,
        "overshoot_pct": float(overshoot_pct),
        "peak_tau_nm": float(np.max(np.abs(tau_nm))),
    }


def _plot_group(output_path: Path, title: str, time_s: np.ndarray, actual_deg: np.ndarray, ref_deg: np.ndarray, names: list[str]):
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


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--model", choices=["visual", "planning"], default="visual")
    parser.add_argument("--settle-time", type=float, default=1.0, help="Seconds to hold the initial pose before the new command.")
    parser.add_argument("--command-time", type=float, default=2.5, help="Seconds to hold the new command.")
    parser.add_argument("--settle-tol-deg", type=float, default=0.1, help="Settling tolerance used in metrics.")
    parser.add_argument("--steady-window", type=float, default=0.4, help="Last-N-second window used for steady-state metrics.")
    parser.add_argument("--waist-delta-deg", type=float, nargs=6, default=[2.0, 3.0, -3.0, 3.0, 2.0, 5.0])
    parser.add_argument("--larm-delta-deg", type=float, nargs=7, default=[4.0, 6.0, 5.0, 6.0, 4.0, 4.0, 3.0])
    parser.add_argument("--rarm-delta-deg", type=float, nargs=7, default=[-4.0, 6.0, -5.0, -6.0, -4.0, 4.0, -3.0])
    parser.add_argument("--head-delta-deg", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_pd_step_test")
    args = parser.parse_args()

    cfg = load_config(args.example, args.config, parse_override(args.override))
    env_cfg = load_dataclass_from_dict(
        SpiritMoz1PathTrackEnvConfig,
        cfg,
        convert_list_to_array=True,
    )
    env = SpiritMoz1PathTrackEnv(env_cfg)

    model_name = "moz1.xml" if args.model == "visual" else "mjx_moz1.xml"
    model = load_spirit_mj_model(model_name, fixed_base=True, gravity_off=False)
    data = mujoco.MjData(model)

    waist_names = [f"LegWaist-{i}" for i in range(6)]
    left_arm_names = [f"LeftArm-{i}" for i in range(7)]
    right_arm_names = [f"RightArm-{i}" for i in range(7)]
    head_names = [f"Head-{i}" for i in range(2)]
    action_joint_names = waist_names + left_arm_names + right_arm_names

    _, action_qpos_adr, action_qvel_adr, action_act_ids = get_joint_indices(model, action_joint_names)
    _, head_qpos_adr, head_qvel_adr, head_act_ids = get_joint_indices(model, head_names)

    init_q = np.array(env._init_q[7:], dtype=np.float64)
    default_joint_ref = np.asarray(env._joint_ref, dtype=np.float64).copy()
    default_head_ref = np.asarray(env._head_ref, dtype=np.float64).copy()
    init_q[np.asarray(env._action_joint_idx)] = default_joint_ref
    init_q[np.asarray(env._head_idx)] = default_head_ref

    data.qpos[:] = init_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    joint_ref_cmd, head_ref_cmd = _build_targets(default_joint_ref, default_head_ref, args)

    joint_kp = np.asarray(env._joint_kp, dtype=np.float64)
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64)
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64)
    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)

    dt = float(model.opt.timestep)
    settle_steps = int(round(args.settle_time / dt))
    command_steps = int(round(args.command_time / dt))
    total_steps = settle_steps + command_steps
    steady_window_steps = max(1, int(round(args.steady_window / dt)))

    time_s = np.arange(total_steps, dtype=np.float64) * dt
    ref_trace = np.zeros((total_steps, len(action_joint_names)), dtype=np.float64)
    head_ref_trace = np.zeros((total_steps, len(head_names)), dtype=np.float64)
    q_trace = np.zeros_like(ref_trace)
    qd_trace = np.zeros_like(ref_trace)
    tau_trace = np.zeros_like(ref_trace)
    head_q_trace = np.zeros((total_steps, len(head_names)), dtype=np.float64)
    head_tau_trace = np.zeros_like(head_q_trace)

    for step in range(total_steps):
        if step < settle_steps:
            joint_ref = default_joint_ref
            head_ref = default_head_ref
        else:
            joint_ref = joint_ref_cmd
            head_ref = head_ref_cmd

        joint_q = data.qpos[action_qpos_adr]
        joint_qd = data.qvel[action_qvel_adr]
        joint_tau = joint_kp * (joint_ref - joint_q) - joint_kd * joint_qd
        joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

        head_q = data.qpos[head_qpos_adr]
        head_qd = data.qvel[head_qvel_adr]
        head_tau = head_kp * (head_ref - head_q) - head_kd * head_qd
        head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

        ref_trace[step] = joint_ref
        head_ref_trace[step] = head_ref
        q_trace[step] = joint_q
        qd_trace[step] = joint_qd
        tau_trace[step] = joint_tau
        head_q_trace[step] = head_q
        head_tau_trace[step] = head_tau

        data.ctrl[:] = 0.0
        data.ctrl[action_act_ids] = joint_tau
        data.ctrl[head_act_ids] = head_tau
        mujoco.mj_step(model, data)

    time_after_s = time_s[settle_steps:] - time_s[settle_steps]

    metrics = []
    post_q = q_trace[settle_steps:]
    post_tau = tau_trace[settle_steps:]
    for idx, joint_name in enumerate(action_joint_names):
        metrics.append(
            _joint_metrics(
                joint_name=joint_name,
                signal_rad=post_q[:, idx],
                ref_pre_rad=default_joint_ref[idx],
                ref_post_rad=joint_ref_cmd[idx],
                time_after_s=time_after_s,
                steady_window_steps=steady_window_steps,
                settle_tol_deg=args.settle_tol_deg,
                tau_nm=post_tau[:, idx],
            )
        )

    head_post_q = head_q_trace[settle_steps:]
    head_post_tau = head_tau_trace[settle_steps:]
    for idx, joint_name in enumerate(head_names):
        metrics.append(
            _joint_metrics(
                joint_name=joint_name,
                signal_rad=head_post_q[:, idx],
                ref_pre_rad=default_head_ref[idx],
                ref_post_rad=head_ref_cmd[idx],
                time_after_s=time_after_s,
                steady_window_steps=steady_window_steps,
                settle_tol_deg=args.settle_tol_deg,
                tau_nm=head_post_tau[:, idx],
            )
        )

    output_root = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "trace_data.npz",
        time_s=time_s,
        ref_trace=ref_trace,
        q_trace=q_trace,
        qd_trace=qd_trace,
        tau_trace=tau_trace,
        head_ref_trace=head_ref_trace,
        head_q_trace=head_q_trace,
        head_tau_trace=head_tau_trace,
    )

    summary = {
        "example": args.example,
        "config": cfg,
        "test": {
            "settle_time_s": args.settle_time,
            "command_time_s": args.command_time,
            "settle_tol_deg": args.settle_tol_deg,
            "steady_window_s": args.steady_window,
            "waist_delta_deg": args.waist_delta_deg,
            "larm_delta_deg": args.larm_delta_deg,
            "rarm_delta_deg": args.rarm_delta_deg,
            "head_delta_deg": args.head_delta_deg,
        },
        "metrics": metrics,
        "aggregate": {
            "worst_steady_abs_error_deg": float(max(m["steady_state_abs_error_deg"] for m in metrics)),
            "worst_overshoot_pct": float(max(m["overshoot_pct"] for m in metrics)),
            "worst_settling_time_s": max((m["settling_time_s"] for m in metrics if m["settling_time_s"] is not None), default=None),
            "worst_peak_tau_nm": float(max(m["peak_tau_nm"] for m in metrics)),
        },
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_group(
        run_dir / "waist_tracking.png",
        "Waist Tracking",
        time_s,
        np.rad2deg(q_trace[:, :6]),
        np.rad2deg(ref_trace[:, :6]),
        waist_names,
    )
    _plot_group(
        run_dir / "left_arm_tracking.png",
        "Left Arm Tracking",
        time_s,
        np.rad2deg(q_trace[:, 6:13]),
        np.rad2deg(ref_trace[:, 6:13]),
        left_arm_names,
    )
    _plot_group(
        run_dir / "right_arm_tracking.png",
        "Right Arm Tracking",
        time_s,
        np.rad2deg(q_trace[:, 13:20]),
        np.rad2deg(ref_trace[:, 13:20]),
        right_arm_names,
    )
    _plot_group(
        run_dir / "head_tracking.png",
        "Head Tracking",
        time_s,
        np.rad2deg(head_q_trace),
        np.rad2deg(head_ref_trace),
        head_names,
    )

    print("saved_run_dir", run_dir)
    print("worst_steady_abs_error_deg", summary["aggregate"]["worst_steady_abs_error_deg"])
    print("worst_overshoot_pct", summary["aggregate"]["worst_overshoot_pct"])
    print("worst_settling_time_s", summary["aggregate"]["worst_settling_time_s"])
    print("worst_peak_tau_nm", summary["aggregate"]["worst_peak_tau_nm"])


if __name__ == "__main__":
    main()
