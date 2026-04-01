#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

import jax
import jax.numpy as jnp
import brax.envs as brax_envs
from brax import math

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def rss_mb() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def parse_override(override_list):
    out = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"invalid override: {item}")
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def make_config(example: str | None, config_path: str | None, overrides: dict):
    if example is not None:
        cfg = yaml.safe_load(open(get_example_path(example + ".yaml"), "r", encoding="utf-8"))
    else:
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    cfg.update(overrides)
    return cfg


def to_float_list(x):
    return [float(v) for v in np.asarray(x).reshape(-1)]


def run_check(cfg: dict) -> dict:
    dial_cfg = load_dataclass_from_dict(DialConfig, cfg)
    env_cfg = load_dataclass_from_dict(
        dial_envs.get_config(dial_cfg.env_name), cfg, convert_list_to_array=True
    )

    t0 = time.time()
    env = brax_envs.get_environment(dial_cfg.env_name, config=env_cfg)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_cfg, env)

    rng = jax.random.PRNGKey(dial_cfg.seed)
    rng, rng_reset = jax.random.split(rng)
    state = reset_env(rng_reset)
    Y0 = jnp.zeros((dial_cfg.Hnode + 1, env.action_size))

    def reverse_scan(rng_Y0_state, factor):
        rng_in, Y0_in, state_in = rng_Y0_state
        rng_out, Y0_out, info = mbdpi.reverse_once(state_in, rng_in, Y0_in, factor)
        return (rng_out, Y0_out, state_in), info

    rewards = []
    done_hist = []
    base_pos_hist = []
    pos_tar_hist = []
    yaw_tar_hist = []
    yaw_hist = []
    roll_hist = []
    pitch_hist = []
    terminal_code_hist = []
    stable_steps_hist = []
    ctrl_hist = []
    qvel_hist = []
    qpos_hist = []
    up_dot_hist = []
    rss_hist = [rss_mb()]
    rews_plan = []
    joint_names = [env.sys.mj_model.joint(i).name for i in range(env.sys.mj_model.njnt)]
    act_joint_names = joint_names[1:]

    for _ in range(dial_cfg.n_steps):
        state = step_env(state, Y0[0])
        state.reward.block_until_ready()

        rewards.append(float(state.reward))
        done_hist.append(float(state.done))
        base_pos_hist.append(np.asarray(state.pipeline_state.x.pos[env._base_idx - 1]))
        pos_tar_hist.append(np.asarray(state.info["pos_tar"]))
        yaw_tar_hist.append(float(state.info["yaw_tar"]))
        rpy = math.quat_to_euler(state.pipeline_state.x.rot[env._base_idx - 1])
        roll_hist.append(float(rpy[0]))
        pitch_hist.append(float(rpy[1]))
        yaw_hist.append(float(rpy[2]))
        terminal_code_hist.append(int(state.info.get("terminal_code", jnp.array(0))))
        stable_steps_hist.append(int(state.info.get("stable_steps", jnp.array(0))))
        up_dot_hist.append(
            float(
                jnp.dot(
                    math.rotate(jnp.array([0.0, 0.0, 1.0]), state.pipeline_state.x.rot[env._base_idx - 1]),
                    jnp.array([0.0, 0.0, 1.0]),
                )
            )
        )
        ctrl_hist.append(np.asarray(state.pipeline_state.ctrl))
        qvel_hist.append(np.asarray(state.pipeline_state.qvel))
        qpos_hist.append(np.asarray(state.pipeline_state.q[7:]))
        rss_hist.append(rss_mb())

        Y0 = mbdpi.shift(Y0)
        traj_diffuse_factors = (
            mbdpi.sigma_control
            * dial_cfg.traj_diffuse_factor ** (jnp.arange(dial_cfg.Ndiffuse))[:, None]
        )
        (rng, Y0, _), info = jax.lax.scan(
            reverse_scan, (rng, Y0, state), traj_diffuse_factors
        )
        Y0.block_until_ready()
        rews_plan.append(float(jnp.mean(info["rews"][-1])))
        rss_hist.append(rss_mb())

    base_pos = np.asarray(base_pos_hist)
    pos_tar = np.asarray(pos_tar_hist)
    ctrl = np.asarray(ctrl_hist)
    qvel = np.asarray(qvel_hist)
    qpos = np.asarray(qpos_hist)
    up_dot = np.asarray(up_dot_hist)
    ctrl_rate = np.diff(ctrl, axis=0) if len(ctrl) > 1 else np.zeros_like(ctrl)
    roll = np.asarray(roll_hist)
    pitch = np.asarray(pitch_hist)

    xy_err = np.linalg.norm(base_pos[:, :2] - pos_tar[:, :2], axis=1)
    z_err = np.abs(base_pos[:, 2] - pos_tar[:, 2])
    first_done = next((i for i, d in enumerate(done_hist) if d > 0.5), None)
    first_done_reasons = []
    first_done_joint_names = []
    first_done_joint_details = []
    if first_done is not None:
        terminal_code = terminal_code_hist[first_done]
        if terminal_code == 1:
            first_done_reasons.append("success_stable")
        q_act = qpos[first_done]
        low_hits = (q_act < np.asarray(env.physical_joint_range[:, 0])) & np.asarray(env._joint_range_valid)
        high_hits = (q_act > np.asarray(env.physical_joint_range[:, 1])) & np.asarray(env._joint_range_valid)
        if terminal_code != 1 and up_dot[first_done] < 0.0:
            first_done_reasons.append("upside_down")
        if terminal_code != 1 and base_pos[first_done, 2] < max(0.03, float(env._config.z_ref) * 0.5):
            first_done_reasons.append("low_height")
        if terminal_code != 1 and np.any(low_hits):
            first_done_reasons.append("joint_low")
            first_done_joint_names.extend(
                act_joint_names[i] for i in np.where(low_hits)[0]
            )
            first_done_joint_details.extend(
                {
                    "name": act_joint_names[i],
                    "kind": "low",
                    "q": float(q_act[i]),
                    "limit": float(env.physical_joint_range[i, 0]),
                }
                for i in np.where(low_hits)[0]
            )
        if terminal_code != 1 and np.any(high_hits):
            first_done_reasons.append("joint_high")
            first_done_joint_names.extend(
                act_joint_names[i] for i in np.where(high_hits)[0]
            )
            first_done_joint_details.extend(
                {
                    "name": act_joint_names[i],
                    "kind": "high",
                    "q": float(q_act[i]),
                    "limit": float(env.physical_joint_range[i, 1]),
                }
                for i in np.where(high_hits)[0]
            )
        if terminal_code != 1 and not np.isfinite(q_act).all():
            first_done_reasons.append("q_nonfinite")
        if terminal_code != 1 and not np.isfinite(qvel[first_done]).all():
            first_done_reasons.append("qd_nonfinite")

    metrics = {
        "config": cfg,
        "elapsed_sec": time.time() - t0,
        "max_rss_mb": float(max(rss_hist)),
        "final_rss_mb": float(rss_hist[-1]),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_plan_reward": float(np.mean(rews_plan)) if rews_plan else None,
        "done_count": int(sum(d > 0.5 for d in done_hist)),
        "first_done_step": first_done,
        "first_done_reasons": first_done_reasons,
        "first_done_terminal_code": terminal_code_hist[first_done] if first_done is not None else 0,
        "first_done_joint_names": first_done_joint_names,
        "first_done_joint_details": first_done_joint_details,
        "max_stable_steps": int(max(stable_steps_hist)) if stable_steps_hist else 0,
        "base_xy_span": to_float_list(np.ptp(base_pos[:, :2], axis=0)),
        "base_z_minmax": [float(base_pos[:, 2].min()), float(base_pos[:, 2].max())],
        "final_base_pos": to_float_list(base_pos[-1]),
        "mean_xy_err": float(np.mean(xy_err)),
        "max_xy_err": float(np.max(xy_err)),
        "final_xy_err": float(xy_err[-1]),
        "mean_z_err": float(np.mean(z_err)),
        "max_z_err": float(np.max(z_err)),
        "roll_rms": float(np.sqrt(np.mean(roll ** 2))),
        "pitch_rms": float(np.sqrt(np.mean(pitch ** 2))),
        "roll_max_abs": float(np.max(np.abs(roll))),
        "pitch_max_abs": float(np.max(np.abs(pitch))),
        "wheel_ctrl_rms": to_float_list(np.sqrt(np.mean(ctrl[:, :4] ** 2, axis=0))),
        "waist_ctrl_rms": to_float_list(np.sqrt(np.mean(ctrl[:, 4:10] ** 2, axis=0))),
        "arm_ctrl_rms": to_float_list(np.sqrt(np.mean(ctrl[:, 10:24] ** 2, axis=0))),
        "waist_ctrl_rate_rms": to_float_list(np.sqrt(np.mean(ctrl_rate[:, 4:10] ** 2, axis=0))) if len(ctrl_rate) else [0.0] * 6,
        "wheel_sign_changes": [int(np.sum(np.diff(np.signbit(ctrl[:, i])) != 0)) for i in range(4)],
        "waist_sign_changes": [int(np.sum(np.diff(np.signbit(ctrl[:, 4 + i])) != 0)) for i in range(6)],
        "joint_vel_rms_first10": to_float_list(np.sqrt(np.mean(qvel[:, 6:16] ** 2, axis=0))),
        "base_ang_vel_rms": to_float_list(np.sqrt(np.mean(qvel[:, 3:6] ** 2, axis=0))),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str)
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--log-path", type=str, default=None)
    args = parser.parse_args()

    cfg = make_config(args.example, args.config, parse_override(args.override))
    metrics = run_check(cfg)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
