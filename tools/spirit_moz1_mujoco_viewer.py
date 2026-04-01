#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import threading
import queue
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import mujoco
import mujoco.viewer

from dial_mpc.envs.spirit_moz1_env import SpiritMoz1PathTrackEnv, SpiritMoz1PathTrackEnvConfig
from dial_mpc.utils.io_utils import get_example_path, get_model_path, load_dataclass_from_dict


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


def get_joint_indices(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    qpos_adr = model.jnt_qposadr[joint_ids]
    qvel_adr = model.jnt_dofadr[joint_ids]
    actuator_ids = []
    for joint_id in joint_ids:
        actuator_id = -1
        for i in range(model.nu):
            if int(model.actuator_trnid[i, 0]) == int(joint_id):
                actuator_id = i
                break
        if actuator_id < 0:
            raise ValueError(f"Could not find actuator for joint id {joint_id}")
        actuator_ids.append(actuator_id)
    return joint_ids, qpos_adr, qvel_adr, np.array(actuator_ids, dtype=np.int32)


def _rewrite_xml_for_debug(xml_text: str, fixed_base: bool, gravity_off: bool) -> str:
    if not fixed_base and not gravity_off:
        return xml_text

    root = ET.fromstring(xml_text)

    if gravity_off:
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("gravity", "0 0 0")

    if fixed_base:
        base_body = root.find("./worldbody/body[@name='base_link']")
        if base_body is None:
            raise ValueError("Could not find base_link body in XML")

        freejoint = base_body.find("freejoint")
        if freejoint is not None:
            base_body.remove(freejoint)

        keyframe = root.find("./keyframe/key[@name='stand']")
        if keyframe is not None and "qpos" in keyframe.attrib:
            qpos = keyframe.attrib["qpos"].split()
            if len(qpos) >= 7:
                base_body.set("pos", " ".join(qpos[:3]))
                base_body.set("quat", " ".join(qpos[3:7]))

        for key in root.findall("./keyframe/key"):
            qpos = key.attrib.get("qpos", "").split()
            if len(qpos) >= 7:
                key.attrib["qpos"] = " ".join(qpos[7:])

    return ET.tostring(root, encoding="unicode")


def load_spirit_mj_model(
    model_name: str,
    *,
    fixed_base: bool = False,
    gravity_off: bool = False,
) -> mujoco.MjModel:
    model_path = Path(get_model_path("spirit_moz1", model_name))
    xml_text = model_path.read_text(encoding="utf-8")
    xml_text = _rewrite_xml_for_debug(xml_text, fixed_base=fixed_base, gravity_off=gravity_off)
    xml_text = xml_text.replace("../meshes/", "meshes/")
    mesh_dir = model_path.parent / "meshes"
    assets = {
        f"meshes/{mesh_file.name}": mesh_file.read_bytes()
        for mesh_file in mesh_dir.glob("*")
        if mesh_file.is_file()
    }
    return mujoco.MjModel.from_xml_string(xml_text, assets=assets)


def _parse_deg_list(values: list[str], expected: int, label: str) -> np.ndarray:
    if len(values) != expected:
        raise ValueError(f"{label} expects {expected} values, got {len(values)}")
    return np.deg2rad(np.array([float(v) for v in values], dtype=np.float64))


def _start_command_reader(cmd_queue: "queue.SimpleQueue[list[str] | None]") -> threading.Thread:
    def reader():
        while True:
            try:
                line = input()
            except EOFError:
                cmd_queue.put(None)
                return
            cmd_queue.put(line.strip().split())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--example", type=str, default="spirit_moz1_stand_hold")
    source.add_argument("--config", type=str)
    parser.add_argument("--override", action="append", default=[], help="key=value YAML literal")
    parser.add_argument(
        "--model",
        choices=["visual", "planning"],
        default="visual",
        help="Use the full visual model or the planning model.",
    )
    parser.add_argument(
        "--base-mode",
        choices=["fixed", "locked", "free"],
        default="fixed",
        help="Use a true fixed-base model, the old pseudo-locked floating base, or a fully free floating base.",
    )
    parser.add_argument(
        "--free-base",
        action="store_true",
        help="Backward-compatible alias for --base-mode free.",
    )
    parser.add_argument(
        "--with-gravity",
        action="store_true",
        help="Enable gravity. By default gravity is disabled for pure joint-PD debugging.",
    )
    parser.add_argument("--print-every", type=float, default=0.5, help="Seconds between console status prints.")
    args = parser.parse_args()
    if args.free_base:
        args.base_mode = "free"
    gravity_off = not args.with_gravity

    cfg = load_config(args.example, args.config, parse_override(args.override))
    env_cfg = load_dataclass_from_dict(
        SpiritMoz1PathTrackEnvConfig,
        cfg,
        convert_list_to_array=True,
    )
    env = SpiritMoz1PathTrackEnv(env_cfg)

    model_name = "moz1.xml" if args.model == "visual" else "mjx_moz1.xml"
    model = load_spirit_mj_model(
        model_name,
        fixed_base=args.base_mode == "fixed",
        gravity_off=gravity_off,
    )
    data = mujoco.MjData(model)

    wheel_names = [f"Base-{i}" for i in range(4)]
    waist_names = [f"LegWaist-{i}" for i in range(6)]
    left_arm_names = [f"LeftArm-{i}" for i in range(7)]
    right_arm_names = [f"RightArm-{i}" for i in range(7)]
    head_names = [f"Head-{i}" for i in range(2)]
    action_joint_names = waist_names + left_arm_names + right_arm_names

    _, wheel_qpos_adr, wheel_qvel_adr, wheel_act_ids = get_joint_indices(model, wheel_names)
    _, action_qpos_adr, action_qvel_adr, action_act_ids = get_joint_indices(model, action_joint_names)
    _, head_qpos_adr, head_qvel_adr, head_act_ids = get_joint_indices(model, head_names)

    qpos_base_offset = 0 if args.base_mode == "fixed" else 7
    init_q = np.array(env._init_q, dtype=np.float64)
    if args.base_mode == "fixed":
        init_q = init_q[7:]
    if env_cfg.task_name == "stand_hold":
        init_q[qpos_base_offset + np.asarray(env._action_joint_idx)] = np.asarray(env._joint_ref, dtype=np.float64)
        init_q[qpos_base_offset + np.asarray(env._head_idx)] = np.asarray(env._head_ref, dtype=np.float64)

    data.qpos[:] = init_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    base_qpos_ref = data.qpos[:7].copy() if args.base_mode != "fixed" else None
    base_qvel_ref = data.qvel[:6].copy() if args.base_mode != "fixed" else None
    wheel_q_ref = data.qpos[wheel_qpos_adr].copy()

    joint_ref = np.asarray(env._joint_ref, dtype=np.float64)
    head_ref = np.asarray(env._head_ref, dtype=np.float64)
    joint_kp = np.asarray(env._joint_kp, dtype=np.float64)
    joint_kd = np.asarray(env._joint_kd, dtype=np.float64)
    joint_tau_limit = np.asarray(env._joint_tau_limit, dtype=np.float64)
    head_kp = np.asarray(env._head_kp, dtype=np.float64)
    head_kd = np.asarray(env._head_kd, dtype=np.float64)
    head_tau_limit = np.asarray(env._head_tau_limit, dtype=np.float64)
    default_joint_ref = joint_ref.copy()
    default_head_ref = head_ref.copy()

    waist_slice = slice(0, 6)
    left_arm_slice = slice(6, 13)
    right_arm_slice = slice(13, 20)
    joint_name_to_idx = {name: i for i, name in enumerate(action_joint_names)}
    head_name_to_idx = {name: i for i, name in enumerate(head_names)}

    cmd_queue: queue.SimpleQueue[list[str] | None] = queue.SimpleQueue()
    _start_command_reader(cmd_queue)
    print("Interactive commands:")
    print("  show")
    print("  reset")
    print("  waist <6 deg values>")
    print("  larm <7 deg values>")
    print("  rarm <7 deg values>")
    print("  head <2 deg values>")
    print("  joint <JointName> <deg>")
    print("Examples:")
    print("  waist 0 60 -90 30 0 0")
    print("  larm -9 -50 -20 -90 -35 8 -7")
    print("  joint RightArm-3 100")

    def lock_base():
        if args.base_mode == "fixed":
            return
        data.qpos[:7] = base_qpos_ref
        data.qvel[:6] = base_qvel_ref
        mujoco.mj_forward(model, data)

    def apply_command(parts: list[str] | None):
        nonlocal joint_ref, head_ref
        if parts is None or len(parts) == 0:
            return
        cmd = parts[0].lower()
        try:
            if cmd == "show":
                print("waist_ref_deg", np.round(np.rad2deg(joint_ref[waist_slice]), 3).tolist())
                print("larm_ref_deg", np.round(np.rad2deg(joint_ref[left_arm_slice]), 3).tolist())
                print("rarm_ref_deg", np.round(np.rad2deg(joint_ref[right_arm_slice]), 3).tolist())
                print("head_ref_deg", np.round(np.rad2deg(head_ref), 3).tolist())
            elif cmd == "reset":
                joint_ref[:] = default_joint_ref
                head_ref[:] = default_head_ref
                print("targets reset to default stand pose")
            elif cmd == "waist":
                joint_ref[waist_slice] = _parse_deg_list(parts[1:], 6, "waist")
                print("updated waist_ref_deg", np.round(np.rad2deg(joint_ref[waist_slice]), 3).tolist())
            elif cmd == "larm":
                joint_ref[left_arm_slice] = _parse_deg_list(parts[1:], 7, "larm")
                print("updated larm_ref_deg", np.round(np.rad2deg(joint_ref[left_arm_slice]), 3).tolist())
            elif cmd == "rarm":
                joint_ref[right_arm_slice] = _parse_deg_list(parts[1:], 7, "rarm")
                print("updated rarm_ref_deg", np.round(np.rad2deg(joint_ref[right_arm_slice]), 3).tolist())
            elif cmd == "head":
                head_ref[:] = _parse_deg_list(parts[1:], 2, "head")
                print("updated head_ref_deg", np.round(np.rad2deg(head_ref), 3).tolist())
            elif cmd == "joint":
                if len(parts) != 3:
                    raise ValueError("joint expects: joint <JointName> <deg>")
                name = parts[1]
                value = np.deg2rad(float(parts[2]))
                if name in joint_name_to_idx:
                    joint_ref[joint_name_to_idx[name]] = value
                    print("updated", name, "to", round(float(np.rad2deg(value)), 3), "deg")
                elif name in head_name_to_idx:
                    head_ref[head_name_to_idx[name]] = value
                    print("updated", name, "to", round(float(np.rad2deg(value)), 3), "deg")
                else:
                    raise ValueError(f"unknown joint name: {name}")
            else:
                raise ValueError(f"unknown command: {parts[0]}")
        except Exception as exc:
            print("command error:", exc)

    last_print = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while viewer.is_running():
            step_start = time.time()

            while not cmd_queue.empty():
                apply_command(cmd_queue.get())

            if args.base_mode == "locked":
                lock_base()

            joint_q = data.qpos[action_qpos_adr]
            joint_qd = data.qvel[action_qvel_adr]
            joint_tau = joint_kp * (joint_ref - joint_q) - joint_kd * joint_qd
            joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)

            head_q = data.qpos[head_qpos_adr]
            head_qd = data.qvel[head_qvel_adr]
            head_tau = head_kp * (head_ref - head_q) - head_kd * head_qd
            head_tau = np.clip(head_tau, -head_tau_limit, head_tau_limit)

            data.ctrl[:] = 0.0
            data.ctrl[action_act_ids] = joint_tau
            data.ctrl[head_act_ids] = head_tau

            mujoco.mj_step(model, data)

            if args.base_mode == "locked":
                wheel_q_ref[:] = data.qpos[wheel_qpos_adr]
                lock_base()

            now = time.time()
            if now - last_print >= args.print_every:
                waist_err = np.rad2deg(joint_ref[:6] - joint_q[:6])
                arm_err = np.rad2deg(joint_ref[6:] - joint_q[6:])
                joint_vel_rms = float(np.sqrt(np.mean(joint_qd ** 2)))
                print(
                    "waist_err_deg_max="
                    f"{np.max(np.abs(waist_err)):.3f} "
                    "arm_err_deg_max="
                    f"{np.max(np.abs(arm_err)):.3f} "
                    "joint_vel_rms="
                    f"{joint_vel_rms:.4f}"
                )
                last_print = now

            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
