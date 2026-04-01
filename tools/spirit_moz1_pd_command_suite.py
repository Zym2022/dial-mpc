#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

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
ACTION_JOINT_NAMES = WAIST_NAMES + LEFT_ARM_NAMES + RIGHT_ARM_NAMES
ALL_JOINT_NAMES = ACTION_JOINT_NAMES + HEAD_NAMES

WAIST_SMALL = np.array([2.0, 3.0, -3.0, 3.0, 2.0, 5.0], dtype=np.float64)
WAIST_LARGE = np.array([4.0, 6.0, -6.0, 6.0, 4.0, 10.0], dtype=np.float64)
LEFT_ARM_SMALL = np.array([4.0, 6.0, 5.0, 6.0, 4.0, 4.0, 3.0], dtype=np.float64)
LEFT_ARM_LARGE = np.array([8.0, 12.0, 10.0, 12.0, 8.0, 8.0, 6.0], dtype=np.float64)
RIGHT_ARM_SMALL = np.array([-4.0, 6.0, -5.0, -6.0, -4.0, 4.0, -3.0], dtype=np.float64)
RIGHT_ARM_LARGE = np.array([-8.0, 12.0, -10.0, -12.0, -8.0, 8.0, -6.0], dtype=np.float64)
HEAD_SMALL = np.array([4.0, 4.0], dtype=np.float64)
HEAD_LARGE = np.array([8.0, 8.0], dtype=np.float64)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    joint_delta_deg: np.ndarray
    head_delta_deg: np.ndarray


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


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _build_case_specs(
    include_single_joint: bool,
    include_head: bool,
    joint_amplitude_scale: float,
    head_amplitude_scale: float,
) -> list[CaseSpec]:
    zeros_joint = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float64)
    zeros_head = np.zeros(len(HEAD_NAMES), dtype=np.float64)
    cases = [
        CaseSpec(
            name="full_small",
            description="Small whole-body command.",
            joint_delta_deg=np.concatenate([WAIST_SMALL, LEFT_ARM_SMALL, RIGHT_ARM_SMALL]) * joint_amplitude_scale,
            head_delta_deg=HEAD_SMALL * head_amplitude_scale if include_head else zeros_head.copy(),
        ),
        CaseSpec(
            name="full_large",
            description="Large whole-body command.",
            joint_delta_deg=np.concatenate([WAIST_LARGE, LEFT_ARM_LARGE, RIGHT_ARM_LARGE]) * joint_amplitude_scale,
            head_delta_deg=HEAD_LARGE * head_amplitude_scale if include_head else zeros_head.copy(),
        ),
        CaseSpec(
            name="full_large_reverse",
            description="Reverse-direction large whole-body command.",
            joint_delta_deg=-np.concatenate([WAIST_LARGE, LEFT_ARM_LARGE, RIGHT_ARM_LARGE]) * joint_amplitude_scale,
            head_delta_deg=-HEAD_LARGE * head_amplitude_scale if include_head else zeros_head.copy(),
        ),
        CaseSpec(
            name="waist_large_only",
            description="Large waist-only command.",
            joint_delta_deg=np.concatenate([WAIST_LARGE, np.zeros(14, dtype=np.float64)]) * joint_amplitude_scale,
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="waist_large_reverse_only",
            description="Reverse-direction large waist-only command.",
            joint_delta_deg=np.concatenate([-WAIST_LARGE, np.zeros(14, dtype=np.float64)]) * joint_amplitude_scale,
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="arms_large_only",
            description="Large bilateral arm-only command.",
            joint_delta_deg=np.concatenate([np.zeros(6, dtype=np.float64), LEFT_ARM_LARGE, RIGHT_ARM_LARGE]) * joint_amplitude_scale,
            head_delta_deg=zeros_head.copy(),
        ),
    ]

    if include_head:
        cases.append(
            CaseSpec(
                name="head_large_only",
                description="Large head-only command.",
                joint_delta_deg=zeros_joint.copy(),
                head_delta_deg=HEAD_LARGE * head_amplitude_scale,
            )
        )

    if include_single_joint:
        single_joint_mag = np.concatenate(
            [
                np.abs(WAIST_LARGE),
                np.abs(LEFT_ARM_LARGE),
                np.abs(RIGHT_ARM_LARGE),
            ]
        ) * joint_amplitude_scale
        for idx, joint_name in enumerate(ACTION_JOINT_NAMES):
            for sign_name, sign in [("pos", 1.0), ("neg", -1.0)]:
                delta = zeros_joint.copy()
                delta[idx] = sign * single_joint_mag[idx]
                cases.append(
                    CaseSpec(
                        name=f"{joint_name}_{sign_name}",
                        description=f"Single-joint {sign_name} step for {joint_name}.",
                        joint_delta_deg=delta,
                        head_delta_deg=zeros_head.copy(),
                    )
                )
        if include_head:
            head_mag = np.abs(HEAD_LARGE) * head_amplitude_scale
            for idx, joint_name in enumerate(HEAD_NAMES):
                for sign_name, sign in [("pos", 1.0), ("neg", -1.0)]:
                    delta = zeros_head.copy()
                    delta[idx] = sign * head_mag[idx]
                    cases.append(
                        CaseSpec(
                            name=f"{joint_name}_{sign_name}",
                            description=f"Single-joint {sign_name} step for {joint_name}.",
                            joint_delta_deg=zeros_joint.copy(),
                            head_delta_deg=delta,
                        )
                    )

    return cases


def _joint_metric(
    *,
    joint_name: str,
    signal_rad: np.ndarray,
    ref_pre_rad: float,
    ref_post_rad: float,
    time_after_s: np.ndarray,
    steady_window_steps: int,
    settle_tol_deg: float,
    tau_nm: np.ndarray,
    tau_limit_nm: float,
    saturation_eps: float,
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
    sat_mask = np.abs(tau_nm) >= max(0.0, float(tau_limit_nm) - saturation_eps)
    sat_count = int(np.count_nonzero(sat_mask))
    sat_fraction = float(sat_count / max(1, len(sat_mask)))
    dt = time_after_s[1] - time_after_s[0] if len(time_after_s) > 1 else 0.0
    longest_sat_s = float(_longest_true_run(sat_mask) * dt)
    peak_tau = float(np.max(np.abs(tau_nm)))
    peak_tau_ratio = peak_tau / float(tau_limit_nm) if tau_limit_nm > 0 else 0.0

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
        "peak_tau_nm": peak_tau,
        "tau_limit_nm": float(tau_limit_nm),
        "peak_tau_ratio": float(peak_tau_ratio),
        "saturation_count": sat_count,
        "saturation_fraction": sat_fraction,
        "longest_saturation_s": longest_sat_s,
    }


def _metric_score(metric: dict, command_time: float) -> float:
    settle = metric["settling_time_s"] if metric["settling_time_s"] is not None else command_time
    return (
        10.0 * metric["steady_state_abs_error_deg"]
        + 0.08 * metric["overshoot_pct"]
        + 2.0 * settle
        + 25.0 * metric["saturation_fraction"]
        + 3.0 * metric["longest_saturation_s"]
        + 2.0 * max(0.0, metric["peak_tau_ratio"] - 0.98)
    )


def _make_recommendation(joint_summary: dict) -> str:
    if joint_summary["max_peak_tau_ratio"] >= 0.995 and (
        joint_summary["max_steady_state_abs_error_deg"] > 0.1 or joint_summary["max_saturation_fraction"] > 0.05
    ):
        return "torque_limit_or_damping"
    if joint_summary["max_overshoot_pct"] > 25.0:
        return "more_damping_or_less_kp"
    if joint_summary["worst_settling_time_s"] is not None and joint_summary["worst_settling_time_s"] > 1.0:
        return "faster_settle"
    return "baseline_ok"


def _run_case(
    *,
    model: mujoco.MjModel,
    init_q: np.ndarray,
    action_qpos_adr: np.ndarray,
    action_qvel_adr: np.ndarray,
    action_act_ids: np.ndarray,
    head_qpos_adr: np.ndarray,
    head_qvel_adr: np.ndarray,
    head_act_ids: np.ndarray,
    default_joint_ref: np.ndarray,
    default_head_ref: np.ndarray,
    joint_kp: np.ndarray,
    joint_kd: np.ndarray,
    joint_tau_limit: np.ndarray,
    head_kp: np.ndarray,
    head_kd: np.ndarray,
    head_tau_limit: np.ndarray,
    case: CaseSpec,
    settle_time: float,
    command_time: float,
    settle_tol_deg: float,
    steady_window: float,
    saturation_eps: float,
) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = init_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    cmd_joint_ref = default_joint_ref + np.deg2rad(case.joint_delta_deg)
    cmd_head_ref = default_head_ref + np.deg2rad(case.head_delta_deg)

    dt = float(model.opt.timestep)
    settle_steps = int(round(settle_time / dt))
    command_steps = int(round(command_time / dt))
    total_steps = settle_steps + command_steps
    steady_window_steps = max(1, int(round(steady_window / dt)))

    q_trace = np.zeros((total_steps, len(ACTION_JOINT_NAMES)), dtype=np.float64)
    qd_trace = np.zeros_like(q_trace)
    tau_trace = np.zeros_like(q_trace)
    head_q_trace = np.zeros((total_steps, len(HEAD_NAMES)), dtype=np.float64)
    head_qd_trace = np.zeros_like(head_q_trace)
    head_tau_trace = np.zeros_like(head_q_trace)

    for step in range(total_steps):
        if step < settle_steps:
            joint_ref = default_joint_ref
            head_ref = default_head_ref
        else:
            joint_ref = cmd_joint_ref
            head_ref = cmd_head_ref

        joint_q = data.qpos[action_qpos_adr]
        joint_qd = data.qvel[action_qvel_adr]
        joint_tau = joint_kp * (joint_ref - joint_q) - joint_kd * joint_qd
        joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

        head_q = data.qpos[head_qpos_adr]
        head_qd = data.qvel[head_qvel_adr]
        head_tau = head_kp * (head_ref - head_q) - head_kd * head_qd
        head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

        q_trace[step] = joint_q
        qd_trace[step] = joint_qd
        tau_trace[step] = joint_tau
        head_q_trace[step] = head_q
        head_qd_trace[step] = head_qd
        head_tau_trace[step] = head_tau

        data.ctrl[:] = 0.0
        data.ctrl[action_act_ids] = joint_tau
        data.ctrl[head_act_ids] = head_tau
        mujoco.mj_step(model, data)

    time_s = np.arange(total_steps, dtype=np.float64) * dt
    time_after_s = time_s[settle_steps:] - time_s[settle_steps]

    metrics = []
    for idx, joint_name in enumerate(ACTION_JOINT_NAMES):
        metrics.append(
            _joint_metric(
                joint_name=joint_name,
                signal_rad=q_trace[settle_steps:, idx],
                ref_pre_rad=default_joint_ref[idx],
                ref_post_rad=cmd_joint_ref[idx],
                time_after_s=time_after_s,
                steady_window_steps=steady_window_steps,
                settle_tol_deg=settle_tol_deg,
                tau_nm=tau_trace[settle_steps:, idx],
                tau_limit_nm=joint_tau_limit[idx],
                saturation_eps=saturation_eps,
            )
        )
    for idx, joint_name in enumerate(HEAD_NAMES):
        metrics.append(
            _joint_metric(
                joint_name=joint_name,
                signal_rad=head_q_trace[settle_steps:, idx],
                ref_pre_rad=default_head_ref[idx],
                ref_post_rad=cmd_head_ref[idx],
                time_after_s=time_after_s,
                steady_window_steps=steady_window_steps,
                settle_tol_deg=settle_tol_deg,
                tau_nm=head_tau_trace[settle_steps:, idx],
                tau_limit_nm=head_tau_limit[idx],
                saturation_eps=saturation_eps,
            )
        )

    active_metrics = [metric for metric in metrics if abs(metric["step_deg"]) > 1e-6]
    for metric in active_metrics:
        metric["score"] = _metric_score(metric, command_time)

    aggregate = {
        "active_joint_count": len(active_metrics),
        "worst_joint": max(active_metrics, key=lambda item: item["score"])["joint"] if active_metrics else None,
        "worst_score": float(max((item["score"] for item in active_metrics), default=0.0)),
        "worst_steady_abs_error_deg": float(max((item["steady_state_abs_error_deg"] for item in active_metrics), default=0.0)),
        "worst_overshoot_pct": float(max((item["overshoot_pct"] for item in active_metrics), default=0.0)),
        "worst_peak_tau_ratio": float(max((item["peak_tau_ratio"] for item in active_metrics), default=0.0)),
        "worst_saturation_fraction": float(max((item["saturation_fraction"] for item in active_metrics), default=0.0)),
    }

    return {
        "name": case.name,
        "description": case.description,
        "joint_delta_deg": case.joint_delta_deg.tolist(),
        "head_delta_deg": case.head_delta_deg.tolist(),
        "metrics": metrics,
        "aggregate": aggregate,
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--model", choices=["visual", "planning"], default="visual")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--command-time", type=float, default=2.5)
    parser.add_argument("--settle-tol-deg", type=float, default=0.1)
    parser.add_argument("--steady-window", type=float, default=0.4)
    parser.add_argument("--saturation-eps", type=float, default=1e-3)
    parser.add_argument("--joint-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--head-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--skip-single-joint", action="store_true")
    parser.add_argument("--skip-head", action="store_true")
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_pd_command_suite")
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

    _, action_qpos_adr, action_qvel_adr, action_act_ids = get_joint_indices(model, ACTION_JOINT_NAMES)
    _, head_qpos_adr, head_qvel_adr, head_act_ids = get_joint_indices(model, HEAD_NAMES)

    init_q = np.array(env._init_q[7:], dtype=np.float64)
    default_joint_ref = np.asarray(env._joint_ref, dtype=np.float64).copy()
    default_head_ref = np.asarray(env._head_ref, dtype=np.float64).copy()
    init_q[np.asarray(env._action_joint_idx)] = default_joint_ref
    init_q[np.asarray(env._head_idx)] = default_head_ref

    joint_kp = np.asarray(env._joint_kp, dtype=np.float64)
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64)
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64)
    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)

    cases = _build_case_specs(
        include_single_joint=not args.skip_single_joint,
        include_head=not args.skip_head,
        joint_amplitude_scale=args.joint_amplitude_scale,
        head_amplitude_scale=args.head_amplitude_scale,
    )

    case_results = []
    active_case_metrics: dict[str, list[dict]] = {joint_name: [] for joint_name in ALL_JOINT_NAMES}
    for case in cases:
        result = _run_case(
            model=model,
            init_q=init_q,
            action_qpos_adr=action_qpos_adr,
            action_qvel_adr=action_qvel_adr,
            action_act_ids=action_act_ids,
            head_qpos_adr=head_qpos_adr,
            head_qvel_adr=head_qvel_adr,
            head_act_ids=head_act_ids,
            default_joint_ref=default_joint_ref,
            default_head_ref=default_head_ref,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            joint_tau_limit=joint_tau_limit,
            head_kp=head_kp,
            head_kd=head_kd,
            head_tau_limit=head_tau_limit,
            case=case,
            settle_time=args.settle_time,
            command_time=args.command_time,
            settle_tol_deg=args.settle_tol_deg,
            steady_window=args.steady_window,
            saturation_eps=args.saturation_eps,
        )
        case_results.append(result)
        for metric in result["metrics"]:
            if abs(metric["step_deg"]) > 1e-6:
                active_case_metrics[metric["joint"]].append({"case": case.name, **metric})

    joint_summary = {}
    for joint_name, metrics in active_case_metrics.items():
        if not metrics:
            continue
        worst = max(metrics, key=lambda item: item["score"])
        joint_summary[joint_name] = {
            "case_count": len(metrics),
            "worst_case": worst["case"],
            "worst_score": float(worst["score"]),
            "max_steady_state_abs_error_deg": float(max(item["steady_state_abs_error_deg"] for item in metrics)),
            "max_overshoot_pct": float(max(item["overshoot_pct"] for item in metrics)),
            "max_peak_tau_ratio": float(max(item["peak_tau_ratio"] for item in metrics)),
            "max_saturation_fraction": float(max(item["saturation_fraction"] for item in metrics)),
            "worst_settling_time_s": max((item["settling_time_s"] for item in metrics if item["settling_time_s"] is not None), default=None),
            "max_longest_saturation_s": float(max(item["longest_saturation_s"] for item in metrics)),
            "recommendation": _make_recommendation(
                {
                    "max_steady_state_abs_error_deg": max(item["steady_state_abs_error_deg"] for item in metrics),
                    "max_overshoot_pct": max(item["overshoot_pct"] for item in metrics),
                    "max_peak_tau_ratio": max(item["peak_tau_ratio"] for item in metrics),
                    "max_saturation_fraction": max(item["saturation_fraction"] for item in metrics),
                    "worst_settling_time_s": max((item["settling_time_s"] for item in metrics if item["settling_time_s"] is not None), default=None),
                }
            ),
        }

    ranked_joints = [
        {"joint": joint_name, **summary}
        for joint_name, summary in sorted(joint_summary.items(), key=lambda item: item[1]["worst_score"], reverse=True)
    ]
    ranked_cases = sorted(case_results, key=lambda item: item["aggregate"]["worst_score"], reverse=True)

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "example": args.example,
        "config": cfg,
        "suite": {
            "settle_time_s": args.settle_time,
            "command_time_s": args.command_time,
            "settle_tol_deg": args.settle_tol_deg,
            "steady_window_s": args.steady_window,
            "joint_amplitude_scale": args.joint_amplitude_scale,
            "head_amplitude_scale": args.head_amplitude_scale,
            "include_single_joint": not args.skip_single_joint,
            "include_head": not args.skip_head,
            "case_count": len(cases),
        },
        "ranked_joints": ranked_joints,
        "ranked_cases": [
            {
                "name": case["name"],
                "description": case["description"],
                **case["aggregate"],
            }
            for case in ranked_cases
        ],
        "cases": case_results,
    }

    with open(run_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append(f"case_count: {len(cases)}")
    lines.append("worst_joints:")
    for item in ranked_joints[:12]:
        lines.append(
            "  "
            + f"{item['joint']}: score={item['worst_score']:.3f}, case={item['worst_case']}, "
            + f"err={item['max_steady_state_abs_error_deg']:.4f}deg, overshoot={item['max_overshoot_pct']:.2f}%, "
            + f"tau_ratio={item['max_peak_tau_ratio']:.3f}, sat={item['max_saturation_fraction']:.3f}, "
            + f"recommend={item['recommendation']}"
        )
    lines.append("worst_cases:")
    for item in ranked_cases[:12]:
        lines.append(
            "  "
            + f"{item['name']}: score={item['aggregate']['worst_score']:.3f}, joint={item['aggregate']['worst_joint']}, "
            + f"err={item['aggregate']['worst_steady_abs_error_deg']:.4f}deg, "
            + f"overshoot={item['aggregate']['worst_overshoot_pct']:.2f}%, "
            + f"tau_ratio={item['aggregate']['worst_peak_tau_ratio']:.3f}, "
            + f"sat={item['aggregate']['worst_saturation_fraction']:.3f}"
        )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved_run_dir", run_dir)
    if ranked_joints:
        worst_joint = ranked_joints[0]
        print("worst_joint", worst_joint["joint"])
        print("worst_joint_score", worst_joint["worst_score"])
        print("worst_joint_case", worst_joint["worst_case"])
        print("worst_joint_max_steady_abs_error_deg", worst_joint["max_steady_state_abs_error_deg"])
        print("worst_joint_max_overshoot_pct", worst_joint["max_overshoot_pct"])
        print("worst_joint_max_peak_tau_ratio", worst_joint["max_peak_tau_ratio"])
        print("worst_joint_max_saturation_fraction", worst_joint["max_saturation_fraction"])
    if ranked_cases:
        worst_case = ranked_cases[0]
        print("worst_case", worst_case["name"])
        print("worst_case_worst_joint", worst_case["aggregate"]["worst_joint"])
        print("worst_case_score", worst_case["aggregate"]["worst_score"])


if __name__ == "__main__":
    main()
