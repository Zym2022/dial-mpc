#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from spirit_moz1_task_check import make_config, run_check


def load_profiles(path: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("profiles file must be a YAML list")

    profiles = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"profile #{idx} must be a mapping")
        name = item.get("name")
        overrides = item.get("overrides", {})
        if not isinstance(name, str) or not name:
            raise ValueError(f"profile #{idx} is missing a valid name")
        if not isinstance(overrides, dict):
            raise ValueError(f"profile {name!r} overrides must be a mapping")
        profiles.append({"name": name, "overrides": overrides})
    return profiles


def score(metrics: dict) -> float:
    done_penalty = 1000.0 * metrics["done_count"]
    err_penalty = 200.0 * metrics["final_xy_err"] + 120.0 * metrics["mean_xy_err"]
    tilt_penalty = (
        80.0 * metrics["roll_rms"]
        + 60.0 * metrics["pitch_rms"]
        + 30.0 * metrics["roll_max_abs"]
        + 25.0 * metrics["pitch_max_abs"]
    )
    height_penalty = 150.0 * max(0.0, metrics["base_z_minmax"][1] - 0.12)
    chatter_penalty = 0.06 * sum(metrics["waist_ctrl_rate_rms"])
    waist_load_penalty = 0.02 * sum(metrics["waist_ctrl_rms"])
    reward_bonus = -25.0 * metrics["mean_reward"]
    return (
        done_penalty
        + err_penalty
        + tilt_penalty
        + height_penalty
        + chatter_penalty
        + waist_load_penalty
        + reward_bonus
    )


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str)
    source.add_argument("--config", type=str)
    parser.add_argument("--profiles", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    base_cfg = make_config(args.example, args.config, {})
    profiles = load_profiles(args.profiles)

    results = []
    for profile in profiles:
        cfg = dict(base_cfg)
        cfg.update(profile["overrides"])
        metrics = run_check(cfg)
        summary = {
            "name": profile["name"],
            "score": score(metrics),
            "done_count": metrics["done_count"],
            "first_done_step": metrics["first_done_step"],
            "mean_reward": metrics["mean_reward"],
            "mean_xy_err": metrics["mean_xy_err"],
            "final_xy_err": metrics["final_xy_err"],
            "roll_rms": metrics["roll_rms"],
            "pitch_rms": metrics["pitch_rms"],
            "roll_max_abs": metrics["roll_max_abs"],
            "pitch_max_abs": metrics["pitch_max_abs"],
            "base_z_minmax": metrics["base_z_minmax"],
            "waist_ctrl_rms": metrics["waist_ctrl_rms"],
            "waist_ctrl_rate_rms": metrics["waist_ctrl_rate_rms"],
            "max_rss_mb": metrics["max_rss_mb"],
            "config": cfg,
        }
        results.append(summary)
        print(
            json.dumps(
                {
                    "name": summary["name"],
                    "score": round(summary["score"], 3),
                    "done_count": summary["done_count"],
                    "first_done_step": summary["first_done_step"],
                    "final_xy_err": round(summary["final_xy_err"], 4),
                    "roll_rms": round(summary["roll_rms"], 4),
                    "pitch_rms": round(summary["pitch_rms"], 4),
                    "base_z_minmax": [round(v, 4) for v in summary["base_z_minmax"]],
                },
                sort_keys=True,
            )
        )

    results.sort(key=lambda item: item["score"])
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
