#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dial_mpc.envs.spirit_moz1_env import SpiritMoz1PathTrackEnv, SpiritMoz1PathTrackEnvConfig
from dial_mpc.utils.io_utils import load_dataclass_from_dict
from tools.spirit_moz1_mujoco_viewer import get_joint_indices, load_config, load_spirit_mj_model, parse_override

import mujoco


WAIST_NAMES = [f"LegWaist-{i}" for i in range(6)]
LEFT_ARM_NAMES = [f"LeftArm-{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"RightArm-{i}" for i in range(7)]
HEAD_NAMES = [f"Head-{i}" for i in range(2)]
ACTION_JOINT_NAMES = WAIST_NAMES + LEFT_ARM_NAMES + RIGHT_ARM_NAMES
WAIST_STEP_DEFAULT = np.array([2.0, 3.0, -3.0, 3.0, 2.0, 5.0], dtype=np.float64)


def _load_base(example: str, config_path: str | None, overrides: list[str]):
    cfg = load_config(example, config_path, parse_override(overrides))
    env_cfg = load_dataclass_from_dict(
        SpiritMoz1PathTrackEnvConfig,
        cfg,
        convert_list_to_array=True,
    )
    env = SpiritMoz1PathTrackEnv(env_cfg)
    model = load_spirit_mj_model("moz1.xml", fixed_base=True, gravity_off=False)
    _, action_qpos_adr, action_qvel_adr, action_act_ids = get_joint_indices(model, ACTION_JOINT_NAMES)
    _, head_qpos_adr, head_qvel_adr, head_act_ids = get_joint_indices(model, HEAD_NAMES)
    init_q = np.array(env._init_q[7:], dtype=np.float64)
    init_q[np.asarray(env._action_joint_idx)] = np.asarray(env._joint_ref, dtype=np.float64)
    init_q[np.asarray(env._head_idx)] = np.asarray(env._head_ref, dtype=np.float64)

    return {
        "cfg": cfg,
        "env": env,
        "model": model,
        "init_q": init_q,
        "action_qpos_adr": action_qpos_adr,
        "action_qvel_adr": action_qvel_adr,
        "action_act_ids": action_act_ids,
        "head_qpos_adr": head_qpos_adr,
        "head_qvel_adr": head_qvel_adr,
        "head_act_ids": head_act_ids,
    }


def _run_step_response(
    base,
    waist_kp: np.ndarray,
    waist_kd: np.ndarray,
    waist_tau_limit: np.ndarray,
    waist_delta_deg: np.ndarray,
    *,
    settle_time: float,
    command_time: float,
    settle_tol_deg: float,
):
    env = base["env"]
    model = base["model"]
    data = mujoco.MjData(model)
    data.qpos[:] = base["init_q"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    default_joint_ref = np.asarray(env._joint_ref, dtype=np.float64).copy()
    default_head_ref = np.asarray(env._head_ref, dtype=np.float64).copy()
    cmd_joint_ref = default_joint_ref.copy()
    cmd_joint_ref[:6] += np.deg2rad(waist_delta_deg)

    joint_kp = np.asarray(env._joint_kp, dtype=np.float64).copy()
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64).copy()
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64).copy()
    joint_kp[:6] = waist_kp
    joint_kd[:6] = waist_kd
    joint_tau_limit[:6] = waist_tau_limit

    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)

    dt = float(model.opt.timestep)
    settle_steps = int(round(settle_time / dt))
    command_steps = int(round(command_time / dt))
    total_steps = settle_steps + command_steps
    steady_window_steps = max(1, int(round(0.3 / dt)))

    q_hist = np.zeros((total_steps, 6), dtype=np.float64)
    qd_hist = np.zeros_like(q_hist)
    tau_hist = np.zeros_like(q_hist)
    ref_hist = np.zeros_like(q_hist)

    for step in range(total_steps):
        joint_ref = default_joint_ref if step < settle_steps else cmd_joint_ref
        head_ref = default_head_ref

        joint_q = data.qpos[base["action_qpos_adr"]]
        joint_qd = data.qvel[base["action_qvel_adr"]]
        joint_tau = joint_kp * (joint_ref - joint_q) - joint_kd * joint_qd
        joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

        head_q = data.qpos[base["head_qpos_adr"]]
        head_qd = data.qvel[base["head_qvel_adr"]]
        head_tau = head_kp * (head_ref - head_q) - head_kd * head_qd
        head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

        q_hist[step] = joint_q[:6]
        qd_hist[step] = joint_qd[:6]
        tau_hist[step] = joint_tau[:6]
        ref_hist[step] = joint_ref[:6]

        data.ctrl[:] = 0.0
        data.ctrl[base["action_act_ids"]] = joint_tau
        data.ctrl[base["head_act_ids"]] = head_tau
        mujoco.mj_step(model, data)

    t = np.arange(total_steps) * dt
    t_after = t[settle_steps:] - t[settle_steps]
    q_after_deg = np.rad2deg(q_hist[settle_steps:])
    ref_before_deg = np.rad2deg(default_joint_ref[:6])
    ref_after_deg = np.rad2deg(cmd_joint_ref[:6])
    ref_after_trace_deg = np.rad2deg(ref_hist[settle_steps:])
    tau_after = tau_hist[settle_steps:]

    metrics = []
    for j, name in enumerate(WAIST_NAMES):
        signal = q_after_deg[:, j]
        ref_pre = ref_before_deg[j]
        ref_post = ref_after_deg[j]
        step_deg = ref_post - ref_pre
        pre_err = float(np.rad2deg(q_hist[settle_steps, j]) - ref_pre)
        steady_actual = float(np.mean(signal[-steady_window_steps:]))
        steady_err = steady_actual - ref_post
        abs_err = np.abs(signal - ref_post)
        if abs(step_deg) < 1e-9:
            overshoot_pct = 0.0
            rise_time = 0.0
        else:
            direction = np.sign(step_deg)
            progress = direction * (signal - ref_pre)
            target = abs(step_deg)
            overshoot_pct = max(0.0, 100.0 * (float(np.max(progress)) - target) / target)
            t10 = t_after[np.flatnonzero(progress >= 0.1 * target)][0] if np.any(progress >= 0.1 * target) else None
            t90 = t_after[np.flatnonzero(progress >= 0.9 * target)][0] if np.any(progress >= 0.9 * target) else None
            rise_time = None if t10 is None or t90 is None else float(t90 - t10)
        settling_time = None
        within = abs_err <= settle_tol_deg
        if np.any(within):
            suffix_all = np.flip(np.cumprod(np.flip(within.astype(np.int32)))) == 1
            idx = np.flatnonzero(suffix_all)
            if idx.size:
                settling_time = float(t_after[idx[0]])
        metrics.append(
            {
                "joint": name,
                "step_deg": float(step_deg),
                "pre_step_error_deg": float(pre_err),
                "steady_state_error_deg": float(steady_err),
                "steady_state_abs_error_deg": float(abs(steady_err)),
                "overshoot_pct": float(overshoot_pct),
                "rise_time_s": rise_time,
                "settling_time_s": settling_time,
                "peak_tau_nm": float(np.max(np.abs(tau_after[:, j]))),
            }
        )

    return {
        "metrics": metrics,
        "t": t,
        "q_hist": q_hist,
        "qd_hist": qd_hist,
        "tau_hist": tau_hist,
        "ref_hist": ref_hist,
    }


def _score_joint(metric: dict) -> float:
    return (
        8.0 * abs(metric["pre_step_error_deg"])
        + 10.0 * metric["steady_state_abs_error_deg"]
        + 0.08 * metric["overshoot_pct"]
        + 0.4 * max(0.0, (metric["rise_time_s"] or 0.0) - 0.25)
        + 0.3 * (0.0 if metric["settling_time_s"] is not None else 5.0)
    )


def _search_joint(
    base,
    waist_kp: np.ndarray,
    waist_kd: np.ndarray,
    waist_tau_limit: np.ndarray,
    joint_idx: int,
    *,
    settle_time: float,
    command_time: float,
    settle_tol_deg: float,
):
    base_metric = _run_step_response(
        base,
        waist_kp,
        waist_kd,
        waist_tau_limit,
        np.eye(6, dtype=np.float64)[joint_idx] * WAIST_STEP_DEFAULT[joint_idx],
        settle_time=settle_time,
        command_time=command_time,
        settle_tol_deg=settle_tol_deg,
    )["metrics"][joint_idx]
    best = {
        "kp": float(waist_kp[joint_idx]),
        "kd": float(waist_kd[joint_idx]),
        "tau": float(waist_tau_limit[joint_idx]),
        "metric": base_metric,
        "score": _score_joint(base_metric),
    }

    kp_factors = [0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    kd_factors = [0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    tau_candidates = sorted(
        {
            float(waist_tau_limit[joint_idx]),
            float(waist_tau_limit[joint_idx] * 1.15),
            float(waist_tau_limit[joint_idx] * 1.3),
            float(waist_tau_limit[joint_idx] + 40.0),
            float(waist_tau_limit[joint_idx] + 80.0),
        }
    )

    for kp_factor in kp_factors:
        for kd_factor in kd_factors:
            for tau in tau_candidates:
                cand_kp = waist_kp.copy()
                cand_kd = waist_kd.copy()
                cand_tau = waist_tau_limit.copy()
                cand_kp[joint_idx] = float(np.clip(waist_kp[joint_idx] * kp_factor, 1000.0, 80000.0))
                cand_kd[joint_idx] = float(np.clip(waist_kd[joint_idx] * kd_factor, 50.0, 12000.0))
                cand_tau[joint_idx] = float(np.clip(tau, 120.0, 1200.0))
                metric = _run_step_response(
                    base,
                    cand_kp,
                    cand_kd,
                    cand_tau,
                    np.eye(6, dtype=np.float64)[joint_idx] * WAIST_STEP_DEFAULT[joint_idx],
                    settle_time=settle_time,
                    command_time=command_time,
                    settle_tol_deg=settle_tol_deg,
                )["metrics"][joint_idx]
                score = _score_joint(metric)
                if score < best["score"]:
                    best = {
                        "kp": float(cand_kp[joint_idx]),
                        "kd": float(cand_kd[joint_idx]),
                        "tau": float(cand_tau[joint_idx]),
                        "metric": metric,
                        "score": score,
                    }

    waist_kp[joint_idx] = best["kp"]
    waist_kd[joint_idx] = best["kd"]
    waist_tau_limit[joint_idx] = best["tau"]
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    parser.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_waist_tune")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--search-settle-time", type=float, default=0.6)
    parser.add_argument("--search-command-time", type=float, default=1.0)
    parser.add_argument("--final-settle-time", type=float, default=1.0)
    parser.add_argument("--final-command-time", type=float, default=2.5)
    parser.add_argument("--settle-tol-deg", type=float, default=0.1)
    args = parser.parse_args()

    base = _load_base(args.example, args.config, args.override)
    cfg = deepcopy(base["cfg"])

    waist_kp = np.array(cfg["waist_leg_kp"], dtype=np.float64)
    waist_kd = np.array(cfg["waist_leg_kd"], dtype=np.float64)
    waist_tau_limit = np.array(cfg["waist_leg_tau_limit"], dtype=np.float64)

    search_order = [3, 1, 2, 0, 4]
    search_log = []

    for round_idx in range(args.rounds):
        for joint_idx in search_order:
            best = _search_joint(
                base,
                waist_kp,
                waist_kd,
                waist_tau_limit,
                joint_idx,
                settle_time=args.search_settle_time,
                command_time=args.search_command_time,
                settle_tol_deg=args.settle_tol_deg,
            )
            search_log.append(
                {
                    "round": round_idx,
                    "joint": WAIST_NAMES[joint_idx],
                    "kp": best["kp"],
                    "kd": best["kd"],
                    "tau": best["tau"],
                    "metric": best["metric"],
                    "score": best["score"],
                }
            )

    final_waist_only = _run_step_response(
        base,
        waist_kp,
        waist_kd,
        waist_tau_limit,
        WAIST_STEP_DEFAULT,
        settle_time=args.final_settle_time,
        command_time=args.final_command_time,
        settle_tol_deg=args.settle_tol_deg,
    )

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "waist_leg_kp": waist_kp.tolist(),
        "waist_leg_kd": waist_kd.tolist(),
        "waist_leg_tau_limit": waist_tau_limit.tolist(),
        "search_log": search_log,
        "final_waist_only_metrics": final_waist_only["metrics"],
    }
    with open(run_dir / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    np.savez(
        run_dir / "final_waist_only_trace.npz",
        t=final_waist_only["t"],
        q_hist=final_waist_only["q_hist"],
        qd_hist=final_waist_only["qd_hist"],
        tau_hist=final_waist_only["tau_hist"],
        ref_hist=final_waist_only["ref_hist"],
    )

    worst = max(final_waist_only["metrics"], key=lambda m: m["steady_state_abs_error_deg"])
    print("saved_run_dir", run_dir)
    print("waist_leg_kp", waist_kp.tolist())
    print("waist_leg_kd", waist_kd.tolist())
    print("waist_leg_tau_limit", waist_tau_limit.tolist())
    print("worst_joint", worst["joint"])
    print("worst_steady_abs_error_deg", worst["steady_state_abs_error_deg"])


if __name__ == "__main__":
    main()
