#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dial_mpc.envs.spirit_moz1_env import SpiritMoz1PathTrackEnv, SpiritMoz1PathTrackEnvConfig
from dial_mpc.utils.io_utils import load_dataclass_from_dict
from tools.spirit_moz1_mujoco_viewer import get_joint_indices, load_config, load_spirit_mj_model, parse_override


WAIST_NAMES = [f"LegWaist-{i}" for i in range(6)]
LEFT_ARM_NAMES = [f"LeftArm-{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"RightArm-{i}" for i in range(7)]
HEAD_NAMES = [f"Head-{i}" for i in range(2)]
ACTION_JOINT_NAMES = WAIST_NAMES + LEFT_ARM_NAMES + RIGHT_ARM_NAMES
LEFT_STEP_DEFAULT = np.array([4.0, 6.0, 5.0, 6.0, 4.0, 4.0, 3.0], dtype=np.float64)
RIGHT_STEP_DEFAULT = np.array([-4.0, 6.0, -5.0, -6.0, -4.0, 4.0, -3.0], dtype=np.float64)


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
    arm_kp: np.ndarray,
    arm_kd: np.ndarray,
    arm_tau_limit: np.ndarray,
    left_delta_deg: np.ndarray,
    right_delta_deg: np.ndarray,
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
    cmd_joint_ref[6:13] += np.deg2rad(left_delta_deg)
    cmd_joint_ref[13:20] += np.deg2rad(right_delta_deg)

    joint_kp = np.asarray(env._joint_kp, dtype=np.float64).copy()
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64).copy()
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64).copy()
    joint_kp[6:13] = arm_kp
    joint_kp[13:20] = arm_kp
    joint_kd[6:13] = arm_kd
    joint_kd[13:20] = arm_kd
    joint_tau_limit[6:13] = arm_tau_limit
    joint_tau_limit[13:20] = arm_tau_limit

    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)

    dt = float(model.opt.timestep)
    settle_steps = int(round(settle_time / dt))
    command_steps = int(round(command_time / dt))
    total_steps = settle_steps + command_steps
    steady_window_steps = max(1, int(round(0.3 / dt)))

    q_hist = np.zeros((total_steps, 14), dtype=np.float64)
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

        q_hist[step] = np.concatenate([joint_q[6:13], joint_q[13:20]])
        qd_hist[step] = np.concatenate([joint_qd[6:13], joint_qd[13:20]])
        tau_hist[step] = np.concatenate([joint_tau[6:13], joint_tau[13:20]])
        ref_hist[step] = np.concatenate([joint_ref[6:13], joint_ref[13:20]])

        data.ctrl[:] = 0.0
        data.ctrl[base["action_act_ids"]] = joint_tau
        data.ctrl[base["head_act_ids"]] = head_tau
        mujoco.mj_step(model, data)

    t = np.arange(total_steps) * dt
    t_after = t[settle_steps:] - t[settle_steps]
    q_after_deg = np.rad2deg(q_hist[settle_steps:])
    ref_before_deg = np.rad2deg(np.concatenate([default_joint_ref[6:13], default_joint_ref[13:20]]))
    ref_after_deg = np.rad2deg(np.concatenate([cmd_joint_ref[6:13], cmd_joint_ref[13:20]]))
    tau_after = tau_hist[settle_steps:]

    metrics = []
    names = LEFT_ARM_NAMES + RIGHT_ARM_NAMES
    for j, name in enumerate(names):
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

    return {"metrics": metrics}


def _score_pair(metrics: list[dict], arm_idx: int) -> float:
    left = metrics[arm_idx]
    right = metrics[arm_idx + 7]
    vals = [
        abs(left["pre_step_error_deg"]),
        abs(right["pre_step_error_deg"]),
        left["steady_state_abs_error_deg"],
        right["steady_state_abs_error_deg"],
    ]
    overs = max(left["overshoot_pct"], right["overshoot_pct"])
    settle_penalty = 5.0 if (left["settling_time_s"] is None or right["settling_time_s"] is None) else max(left["settling_time_s"], right["settling_time_s"])
    return 8.0 * max(vals[:2]) + 10.0 * max(vals[2:]) + 0.08 * overs + 0.2 * settle_penalty


def _search_arm_idx(base, arm_kp, arm_kd, arm_tau_limit, arm_idx: int, *, settle_time, command_time, settle_tol_deg):
    left_delta = np.zeros(7, dtype=np.float64)
    right_delta = np.zeros(7, dtype=np.float64)
    left_delta[arm_idx] = LEFT_STEP_DEFAULT[arm_idx]
    right_delta[arm_idx] = RIGHT_STEP_DEFAULT[arm_idx]
    base_metrics = _run_step_response(
        base,
        arm_kp,
        arm_kd,
        arm_tau_limit,
        left_delta,
        right_delta,
        settle_time=settle_time,
        command_time=command_time,
        settle_tol_deg=settle_tol_deg,
    )["metrics"]
    best = {
        "kp": float(arm_kp[arm_idx]),
        "kd": float(arm_kd[arm_idx]),
        "tau": float(arm_tau_limit[arm_idx]),
        "score": _score_pair(base_metrics, arm_idx),
        "metrics": base_metrics,
    }

    kp_factors = [0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    kd_factors = [0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    tau_candidates = sorted(
        {
            float(arm_tau_limit[arm_idx]),
            float(arm_tau_limit[arm_idx] * 1.15),
            float(arm_tau_limit[arm_idx] * 1.3),
            float(arm_tau_limit[arm_idx] + 20.0),
            float(arm_tau_limit[arm_idx] + 40.0),
        }
    )

    for kp_factor in kp_factors:
        for kd_factor in kd_factors:
            for tau in tau_candidates:
                cand_kp = arm_kp.copy()
                cand_kd = arm_kd.copy()
                cand_tau = arm_tau_limit.copy()
                cand_kp[arm_idx] = float(np.clip(arm_kp[arm_idx] * kp_factor, 50.0, 12000.0))
                cand_kd[arm_idx] = float(np.clip(arm_kd[arm_idx] * kd_factor, 1.0, 3000.0))
                cand_tau[arm_idx] = float(np.clip(tau, 10.0, 400.0))
                metrics = _run_step_response(
                    base,
                    cand_kp,
                    cand_kd,
                    cand_tau,
                    left_delta,
                    right_delta,
                    settle_time=settle_time,
                    command_time=command_time,
                    settle_tol_deg=settle_tol_deg,
                )["metrics"]
                score = _score_pair(metrics, arm_idx)
                if score < best["score"]:
                    best = {
                        "kp": float(cand_kp[arm_idx]),
                        "kd": float(cand_kd[arm_idx]),
                        "tau": float(cand_tau[arm_idx]),
                        "score": score,
                        "metrics": metrics,
                    }

    arm_kp[arm_idx] = best["kp"]
    arm_kd[arm_idx] = best["kd"]
    arm_tau_limit[arm_idx] = best["tau"]
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    parser.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_arm_tune")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--search-settle-time", type=float, default=0.6)
    parser.add_argument("--search-command-time", type=float, default=1.0)
    parser.add_argument("--settle-tol-deg", type=float, default=0.1)
    args = parser.parse_args()

    base = _load_base(args.example, args.config, args.override)
    cfg = deepcopy(base["cfg"])
    arm_kp = np.array(cfg["arm_kp"], dtype=np.float64)
    arm_kd = np.array(cfg["arm_kd"], dtype=np.float64)
    arm_tau_limit = np.array(cfg["arm_tau_limit"], dtype=np.float64)

    search_order = [1, 0, 3, 2, 5]
    search_log = []
    for round_idx in range(args.rounds):
        for arm_idx in search_order:
            best = _search_arm_idx(
                base,
                arm_kp,
                arm_kd,
                arm_tau_limit,
                arm_idx,
                settle_time=args.search_settle_time,
                command_time=args.search_command_time,
                settle_tol_deg=args.settle_tol_deg,
            )
            search_log.append(
                {
                    "round": round_idx,
                    "arm_idx": arm_idx,
                    "kp": best["kp"],
                    "kd": best["kd"],
                    "tau": best["tau"],
                    "score": best["score"],
                }
            )

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "arm_kp": arm_kp.tolist(),
                "arm_kd": arm_kd.tolist(),
                "arm_tau_limit": arm_tau_limit.tolist(),
                "search_log": search_log,
            },
            f,
            indent=2,
        )

    print("saved_run_dir", run_dir)
    print("arm_kp", arm_kp.tolist())
    print("arm_kd", arm_kd.tolist())
    print("arm_tau_limit", arm_tau_limit.tolist())


if __name__ == "__main__":
    main()
