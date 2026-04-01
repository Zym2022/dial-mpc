#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import jax
import jax.numpy as jnp
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from brax import math

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


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


def run_probe(cfg: dict, command: np.ndarray, n_steps: int, label: str) -> dict:
    dial_cfg = load_dataclass_from_dict(DialConfig, cfg)
    env_cfg = load_dataclass_from_dict(
        dial_envs.get_config(dial_cfg.env_name), cfg, convert_list_to_array=True
    )
    env = brax_envs.get_environment(dial_cfg.env_name, config=env_cfg)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    action = np.zeros(env.action_size, dtype=np.float32)
    action[:3] = command.astype(np.float32)
    action = jnp.array(action)

    rng = jax.random.PRNGKey(dial_cfg.seed)
    state = reset_env(rng)

    base_pos_hist = []
    yaw_hist = []
    ctrl_hist = []
    done_hist = []
    vel_body_hist = []
    ang_body_hist = []

    for _ in range(n_steps):
        state = step_env(state, action)
        state.reward.block_until_ready()

        base_pos_hist.append(np.asarray(state.pipeline_state.x.pos[env._base_idx - 1]))
        yaw_hist.append(float(math.quat_to_euler(state.pipeline_state.x.rot[env._base_idx - 1])[2]))
        ctrl_hist.append(np.asarray(state.pipeline_state.ctrl))
        done_hist.append(float(state.done))
        vel_body_hist.append(
            np.asarray(
                global_to_body_velocity(
                    state.pipeline_state.xd.vel[env._base_idx - 1],
                    state.pipeline_state.x.rot[env._base_idx - 1],
                )
            )
        )
        ang_body_hist.append(
            np.asarray(
                global_to_body_velocity(
                    state.pipeline_state.xd.ang[env._base_idx - 1] * jnp.pi / 180.0,
                    state.pipeline_state.x.rot[env._base_idx - 1],
                )
            )
        )

    base_pos = np.asarray(base_pos_hist)
    ctrl = np.asarray(ctrl_hist)
    vel_body = np.asarray(vel_body_hist)
    ang_body = np.asarray(ang_body_hist)

    return {
        "label": label,
        "command": to_float_list(command),
        "n_steps": n_steps,
        "done_count": int(sum(d > 0.5 for d in done_hist)),
        "final_base_pos": to_float_list(base_pos[-1]),
        "base_delta_world": to_float_list(base_pos[-1] - base_pos[0]),
        "yaw_delta": float(yaw_hist[-1] - yaw_hist[0]),
        "mean_body_vel": to_float_list(np.mean(vel_body, axis=0)),
        "mean_body_ang_vel": to_float_list(np.mean(ang_body, axis=0)),
        "wheel_ctrl_rms": to_float_list(np.sqrt(np.mean(ctrl[:, :4] ** 2, axis=0))),
        "waist_ctrl_rms": to_float_list(np.sqrt(np.mean(ctrl[:, 4:10] ** 2, axis=0))),
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str)
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--log-path", type=str, default=None)
    args = parser.parse_args()

    cfg = make_config(args.example, args.config, parse_override(args.override))
    probes = [
        ("vx+", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("vy+", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("yaw+", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("vx-", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        ("vy-", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        ("yaw-", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    ]
    results = [run_probe(cfg, command, args.n_steps, label) for label, command in probes]
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
