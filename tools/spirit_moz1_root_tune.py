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
from tools.spirit_moz1_pd_command_suite import (
    ACTION_JOINT_NAMES,
    CaseSpec,
    HEAD_NAMES,
    WAIST_LARGE,
    WAIST_NAMES,
    _metric_score,
    _run_case,
)


ROOT_JOINT_IDXS = [0, 1, 2, 3]
ROOT_JOINT_WEIGHTS = {
    "LegWaist-0": 1.0,
    "LegWaist-1": 1.4,
    "LegWaist-2": 2.0,
    "LegWaist-3": 1.5,
    "LegWaist-4": 0.5,
    "LegWaist-5": 0.2,
}
CASE_WEIGHTS = {
    "full_small": 1.0,
    "full_large": 1.3,
    "full_large_reverse": 1.3,
    "waist_large_only": 1.7,
    "waist_large_reverse_only": 1.7,
    "LegWaist-0_pos": 1.0,
    "LegWaist-0_neg": 1.0,
    "LegWaist-1_pos": 1.2,
    "LegWaist-1_neg": 1.2,
    "LegWaist-2_pos": 1.6,
    "LegWaist-2_neg": 1.6,
    "LegWaist-3_pos": 1.3,
    "LegWaist-3_neg": 1.3,
}


def _make_root_cases(amplitude_scale: float) -> list[CaseSpec]:
    zeros_joint = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float64)
    zeros_head = np.zeros(len(HEAD_NAMES), dtype=np.float64)
    waist_delta = WAIST_LARGE * amplitude_scale
    cases = [
        CaseSpec(
            name="full_small",
            description="Small root-focused full-body surrogate.",
            joint_delta_deg=np.concatenate([0.5 * waist_delta, np.zeros(14, dtype=np.float64)]),
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="full_large",
            description="Large root-focused full-body surrogate.",
            joint_delta_deg=np.concatenate([waist_delta, np.zeros(14, dtype=np.float64)]),
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="full_large_reverse",
            description="Reverse large root-focused full-body surrogate.",
            joint_delta_deg=np.concatenate([-waist_delta, np.zeros(14, dtype=np.float64)]),
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="waist_large_only",
            description="Large waist-only command.",
            joint_delta_deg=np.concatenate([waist_delta, np.zeros(14, dtype=np.float64)]),
            head_delta_deg=zeros_head.copy(),
        ),
        CaseSpec(
            name="waist_large_reverse_only",
            description="Reverse large waist-only command.",
            joint_delta_deg=np.concatenate([-waist_delta, np.zeros(14, dtype=np.float64)]),
            head_delta_deg=zeros_head.copy(),
        ),
    ]

    for joint_idx in ROOT_JOINT_IDXS:
        for sign_name, sign in [("pos", 1.0), ("neg", -1.0)]:
            delta = zeros_joint.copy()
            delta[joint_idx] = sign * abs(waist_delta[joint_idx])
            cases.append(
                CaseSpec(
                    name=f"{WAIST_NAMES[joint_idx]}_{sign_name}",
                    description=f"Single-joint root test for {WAIST_NAMES[joint_idx]} {sign_name}.",
                    joint_delta_deg=delta,
                    head_delta_deg=zeros_head.copy(),
                )
            )
    return cases


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
    default_joint_ref = np.asarray(env._joint_ref, dtype=np.float64).copy()
    default_head_ref = np.asarray(env._head_ref, dtype=np.float64).copy()
    init_q[np.asarray(env._action_joint_idx)] = default_joint_ref
    init_q[np.asarray(env._head_idx)] = default_head_ref
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
        "default_joint_ref": default_joint_ref,
        "default_head_ref": default_head_ref,
    }


def _evaluate_candidate(
    base,
    cases: list[CaseSpec],
    joint_kp: np.ndarray,
    joint_kd: np.ndarray,
    joint_tau_limit: np.ndarray,
    *,
    settle_time: float,
    command_time: float,
    settle_tol_deg: float,
    steady_window: float,
) -> dict:
    head_kp = np.asarray(base["env"]._head_kp, dtype=np.float64)
    head_kd = np.asarray(base["env"]._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(base["env"]._head_tau_limit, dtype=np.float64)

    case_results = []
    weighted_score = 0.0
    total_weight = 0.0
    root_metrics_by_joint = {name: [] for name in WAIST_NAMES[:6]}

    for case in cases:
        result = _run_case(
            model=base["model"],
            init_q=base["init_q"],
            action_qpos_adr=base["action_qpos_adr"],
            action_qvel_adr=base["action_qvel_adr"],
            action_act_ids=base["action_act_ids"],
            head_qpos_adr=base["head_qpos_adr"],
            head_qvel_adr=base["head_qvel_adr"],
            head_act_ids=base["head_act_ids"],
            default_joint_ref=base["default_joint_ref"],
            default_head_ref=base["default_head_ref"],
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            joint_tau_limit=joint_tau_limit,
            head_kp=head_kp,
            head_kd=head_kd,
            head_tau_limit=head_tau_limit,
            case=case,
            settle_time=settle_time,
            command_time=command_time,
            settle_tol_deg=settle_tol_deg,
            steady_window=steady_window,
            saturation_eps=1e-3,
        )
        case_results.append(result)
        case_weight = CASE_WEIGHTS.get(case.name, 1.0)
        for metric in result["metrics"]:
            if not metric["joint"].startswith("LegWaist"):
                continue
            if abs(metric["step_deg"]) <= 1e-6:
                continue
            metric["score"] = _metric_score(metric, command_time)
            root_metrics_by_joint[metric["joint"]].append({"case": case.name, **metric})
            if metric["joint"] in ROOT_JOINT_WEIGHTS:
                weight = case_weight * ROOT_JOINT_WEIGHTS[metric["joint"]]
                weighted_score += weight * metric["score"]
                total_weight += weight

    ranked_joints = []
    for joint_name, metrics in root_metrics_by_joint.items():
        if not metrics:
            continue
        worst = max(metrics, key=lambda item: item["score"])
        ranked_joints.append(
            {
                "joint": joint_name,
                "worst_case": worst["case"],
                "worst_score": float(worst["score"]),
                "max_steady_state_abs_error_deg": float(max(item["steady_state_abs_error_deg"] for item in metrics)),
                "max_overshoot_pct": float(max(item["overshoot_pct"] for item in metrics)),
                "max_peak_tau_ratio": float(max(item["peak_tau_ratio"] for item in metrics)),
                "max_saturation_fraction": float(max(item["saturation_fraction"] for item in metrics)),
            }
        )
    ranked_joints.sort(key=lambda item: item["worst_score"], reverse=True)

    return {
        "score": float(weighted_score / max(total_weight, 1e-9)),
        "ranked_joints": ranked_joints,
        "cases": case_results,
    }


def _candidate_grid(base_value: float, kp_factors: list[float], kd_factors: list[float], tau_values: list[float]) -> list[tuple[float, float, float]]:
    out = []
    for kp_factor in kp_factors:
        for kd_factor in kd_factors:
            for tau in tau_values:
                out.append((base_value * kp_factor, kd_factor, tau))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    parser.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default="spirit_moz1_root_tune")
    parser.add_argument("--tau-scale", type=float, default=1.5)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--command-time", type=float, default=2.5)
    parser.add_argument("--settle-tol-deg", type=float, default=0.1)
    parser.add_argument("--steady-window", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    args = parser.parse_args()

    base = _load_base(args.example, args.config, args.override)
    cfg = deepcopy(base["cfg"])

    waist_kp = np.array(cfg["waist_leg_kp"], dtype=np.float64)
    waist_kd = np.array(cfg["waist_leg_kd"], dtype=np.float64)
    waist_tau_limit = np.array(cfg["waist_leg_tau_limit"], dtype=np.float64) * args.tau_scale
    joint_kp = np.asarray(base["env"]._joint_kp, dtype=np.float64).copy()
    joint_kd = np.asarray(base["env"]._joint_kd, dtype=np.float64).copy()
    joint_tau_limit = np.asarray(base["env"]._joint_tau_limit, dtype=np.float64).copy()
    joint_kp[:6] = waist_kp
    joint_kd[:6] = waist_kd
    joint_tau_limit[:6] = waist_tau_limit

    cases = _make_root_cases(args.amplitude_scale)

    baseline_eval = _evaluate_candidate(
        base,
        cases,
        joint_kp,
        joint_kd,
        joint_tau_limit,
        settle_time=args.settle_time,
        command_time=args.command_time,
        settle_tol_deg=args.settle_tol_deg,
        steady_window=args.steady_window,
    )

    search_log = []
    search_order = [2, 3, 1, 0]
    kp_factors = {
        2: [0.9, 0.95, 1.0, 1.05],
        3: [0.85, 0.9, 0.95, 1.0],
        1: [0.9, 0.95, 1.0, 1.05],
        0: [0.9, 0.95, 1.0, 1.05],
    }
    kd_factors = {
        2: [1.0, 1.1, 1.2, 1.3, 1.4],
        3: [1.0, 1.1, 1.2, 1.3],
        1: [1.0, 1.1, 1.2, 1.3],
        0: [1.0, 1.1, 1.2],
    }
    tau_factors = {
        2: [1.0, 1.15, 1.3],
        3: [1.0, 1.1, 1.2],
        1: [1.0, 1.1, 1.2],
        0: [1.0, 1.1],
    }

    best_score = baseline_eval["score"]
    best_eval = baseline_eval

    for round_idx in range(args.rounds):
        for joint_idx in search_order:
            base_kp_value = float(waist_kp[joint_idx])
            base_kd_value = float(waist_kd[joint_idx])
            base_tau_value = float(waist_tau_limit[joint_idx])
            local_best = {
                "score": best_score,
                "kp": base_kp_value,
                "kd": base_kd_value,
                "tau": base_tau_value,
                "eval": best_eval,
            }
            for kp_factor in kp_factors[joint_idx]:
                for kd_factor in kd_factors[joint_idx]:
                    for tau_factor in tau_factors[joint_idx]:
                        cand_waist_kp = waist_kp.copy()
                        cand_waist_kd = waist_kd.copy()
                        cand_waist_tau = waist_tau_limit.copy()
                        cand_waist_kp[joint_idx] = base_kp_value * kp_factor
                        cand_waist_kd[joint_idx] = base_kd_value * kd_factor
                        cand_waist_tau[joint_idx] = base_tau_value * tau_factor
                        cand_joint_kp = joint_kp.copy()
                        cand_joint_kd = joint_kd.copy()
                        cand_joint_tau = joint_tau_limit.copy()
                        cand_joint_kp[:6] = cand_waist_kp
                        cand_joint_kd[:6] = cand_waist_kd
                        cand_joint_tau[:6] = cand_waist_tau
                        cand_eval = _evaluate_candidate(
                            base,
                            cases,
                            cand_joint_kp,
                            cand_joint_kd,
                            cand_joint_tau,
                            settle_time=args.settle_time,
                            command_time=args.command_time,
                            settle_tol_deg=args.settle_tol_deg,
                            steady_window=args.steady_window,
                        )
                        if cand_eval["score"] < local_best["score"]:
                            local_best = {
                                "score": cand_eval["score"],
                                "kp": float(cand_waist_kp[joint_idx]),
                                "kd": float(cand_waist_kd[joint_idx]),
                                "tau": float(cand_waist_tau[joint_idx]),
                                "eval": cand_eval,
                            }
            waist_kp[joint_idx] = local_best["kp"]
            waist_kd[joint_idx] = local_best["kd"]
            waist_tau_limit[joint_idx] = local_best["tau"]
            joint_kp[:6] = waist_kp
            joint_kd[:6] = waist_kd
            joint_tau_limit[:6] = waist_tau_limit
            best_score = local_best["score"]
            best_eval = local_best["eval"]
            search_log.append(
                {
                    "round": round_idx,
                    "joint": WAIST_NAMES[joint_idx],
                    "score": local_best["score"],
                    "kp": local_best["kp"],
                    "kd": local_best["kd"],
                    "tau": local_best["tau"],
                }
            )

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tau_scale": args.tau_scale,
        "baseline_score": baseline_eval["score"],
        "best_score": best_eval["score"],
        "waist_leg_kp": waist_kp.tolist(),
        "waist_leg_kd": waist_kd.tolist(),
        "waist_leg_tau_limit": waist_tau_limit.tolist(),
        "baseline_ranked_joints": baseline_eval["ranked_joints"],
        "best_ranked_joints": best_eval["ranked_joints"],
        "search_log": search_log,
    }
    with open(run_dir / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("saved_run_dir", run_dir)
    print("baseline_score", baseline_eval["score"])
    print("best_score", best_eval["score"])
    print("waist_leg_kp", waist_kp.tolist())
    print("waist_leg_kd", waist_kd.tolist())
    print("waist_leg_tau_limit", waist_tau_limit.tolist())
    if best_eval["ranked_joints"]:
        worst = best_eval["ranked_joints"][0]
        print("worst_joint", worst["joint"])
        print("worst_joint_case", worst["worst_case"])
        print("worst_joint_score", worst["worst_score"])


if __name__ == "__main__":
    main()
