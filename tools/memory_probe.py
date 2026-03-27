#!/usr/bin/env python3
import argparse
import time
import threading
import yaml

import jax
import jax.numpy as jnp
import brax.envs as brax_envs
import dial_mpc.envs as dial_envs

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import MBDPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict, get_example_path


def rss_mb() -> float:
    with open('/proc/self/status', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return 0.0


def parse_override(override_list):
    out = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f'invalid override: {item}')
        k, v = item.split('=', 1)
        out[k] = yaml.safe_load(v)
    return out




def print_system_stats(env):
    sys = env.sys
    mj = sys.mj_model
    print(
        "system "
        f"nq={sys.nq} nv={sys.nv} nu={sys.nu} "
        f"nbody={sys.nbody} njnt={sys.njnt} ngeom={sys.ngeom}",
        flush=True,
    )
    print(
        "mj_model "
        f"nbody={mj.nbody} njnt={mj.njnt} ngeom={mj.ngeom} "
        f"nmesh={mj.nmesh} npair={mj.npair} nexclude={mj.nexclude}",
        flush=True,
    )


def run_probe(example: str, overrides: dict, phase: str):
    cfg = yaml.safe_load(open(get_example_path(example + '.yaml'), 'r', encoding='utf-8'))
    cfg.update(overrides)

    dial_cfg = load_dataclass_from_dict(DialConfig, cfg)
    env_cfg = load_dataclass_from_dict(
        dial_envs.get_config(dial_cfg.env_name), cfg, convert_list_to_array=True
    )

    log = []
    stop = False

    def monitor():
        while not stop:
            log.append((time.time(), rss_mb()))
            time.sleep(0.1)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    t0 = time.time()
    print(f'start rss={rss_mb():.1f}MB', flush=True)

    s = time.time()
    env = brax_envs.get_environment(dial_cfg.env_name, config=env_cfg)
    print(f'env_built dt={time.time()-s:.2f}s rss={rss_mb():.1f}MB', flush=True)
    print_system_stats(env)

    if phase == 'env':
        stop = True
        th.join(timeout=1)
        peak = max(x for _, x in log) if log else rss_mb()
        print(f'peak rss={peak:.1f}MB total dt={time.time()-t0:.2f}s', flush=True)
        return

    s = time.time()
    mbdpi = MBDPI(dial_cfg, env)
    print(f'mbdpi_init dt={time.time()-s:.2f}s rss={rss_mb():.1f}MB', flush=True)

    if phase == 'planner':
        stop = True
        th.join(timeout=1)
        peak = max(x for _, x in log) if log else rss_mb()
        print(f'peak rss={peak:.1f}MB total dt={time.time()-t0:.2f}s', flush=True)
        return

    rng = jax.random.PRNGKey(dial_cfg.seed)
    rng, rr, rp = jax.random.split(rng, 3)

    s = time.time()
    state = jax.jit(env.reset)(rr)
    _ = float(jnp.sum(state.obs))
    print(f'reset_jit dt={time.time()-s:.2f}s rss={rss_mb():.1f}MB', flush=True)

    if phase == 'reset':
        stop = True
        th.join(timeout=1)
        peak = max(x for _, x in log) if log else rss_mb()
        print(f'peak rss={peak:.1f}MB total dt={time.time()-t0:.2f}s', flush=True)
        return

    Y0 = jnp.zeros((dial_cfg.Hnode + 1, env.action_size))
    s = time.time()
    _, Y1, info = mbdpi.reverse_once(state, rp, Y0, mbdpi.sigma_control[:dial_cfg.Hnode + 1])
    rew_mean = float(jnp.mean(info['rews']))
    _ = float(jnp.sum(Y1))
    print(f'reverse_once dt={time.time()-s:.2f}s rss={rss_mb():.1f}MB rew_mean={rew_mean:.4f}', flush=True)

    stop = True
    th.join(timeout=1)
    peak = max(x for _, x in log) if log else rss_mb()
    print(f'peak rss={peak:.1f}MB total dt={time.time()-t0:.2f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--example', required=True)
    p.add_argument(
        '--phase',
        choices=['env', 'planner', 'reset', 'reverse'],
        default='reverse',
        help='stop the probe after the selected stage',
    )
    p.add_argument('--override', action='append', default=[], help='key=value YAML literal')
    args = p.parse_args()
    run_probe(args.example, parse_override(args.override), args.phase)
