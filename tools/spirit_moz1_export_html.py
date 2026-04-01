#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import yaml
import jax
import jax.numpy as jnp
import brax.envs as brax_envs
from brax.io import html

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict


def parse_override(override_list):
    out = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"invalid override: {item}")
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def load_config(example: str | None, config_path: str | None, overrides: dict):
    if example is not None:
        cfg = yaml.safe_load(open(get_example_path(example + ".yaml"), "r", encoding="utf-8"))
    else:
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    cfg.update(overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", type=str)
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    args = parser.parse_args()

    cfg = load_config(args.example, args.config, parse_override(args.override))
    dial_cfg = load_dataclass_from_dict(DialConfig, cfg)
    env_cfg = load_dataclass_from_dict(
        dial_envs.get_config(dial_cfg.env_name), cfg, convert_list_to_array=True
    )

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

    rollout = []
    rewards = []
    for _ in range(dial_cfg.n_steps):
        state = step_env(state, Y0[0])
        state.reward.block_until_ready()
        rollout.append(state.pipeline_state)
        rewards.append(float(state.reward))

        Y0 = mbdpi.shift(Y0)
        traj_diffuse_factors = (
            mbdpi.sigma_control
            * dial_cfg.traj_diffuse_factor ** (jnp.arange(dial_cfg.Ndiffuse))[:, None]
        )
        (rng, Y0, _), _ = jax.lax.scan(
            reverse_scan, (rng, Y0, state), traj_diffuse_factors
        )
        Y0.block_until_ready()

    output_dir = Path(dial_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    render_sys = (
        env.make_visualization_system() if hasattr(env, "make_visualization_system") else env.sys
    )
    webpage = html.render(
        render_sys.tree_replace({"opt.timestep": env.dt}),
        rollout,
        1080,
        True,
    )

    html_path = output_dir / f"{timestamp}_brax_visualization.html"
    html_path.write_text(webpage, encoding="utf-8")

    states = []
    for i, pipeline_state in enumerate(rollout):
        states.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
    jnp.save(output_dir / f"{timestamp}_states.npy", jnp.array(states))

    print(f"html_path={html_path}")
    print(f"mean_reward={sum(rewards) / len(rewards):.6f}")


if __name__ == "__main__":
    main()
