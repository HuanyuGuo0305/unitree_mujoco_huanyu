#!/usr/bin/env python3
"""Acceptance criterion 17: the adapters must reproduce the original controllers.

The original deployment scripts are executed as subprocesses in headless mode
with their native constant ``base_command`` and their native EE sampler, and
each adapter is driven by a *replay* scenario that reproduces that native
sampler exactly. Both start from the identical initial state, so the whole
trajectory must agree.

Comparison uses the scripts' own periodic console diagnostics (printed every
200 physics steps): base height, roll, pitch, blend, per-group action min/max,
the three PLB keypoint errors, max |d.ctrl|, and the EE command. That is a
strict fingerprint of the full state, the policy output and the actuation.

Run:
    python3 benchmark/tests/test_controller_parity.py            # C1,C2,C3,C5
    python3 benchmark/tests/test_controller_parity.py --baselines C1 --duration 6
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
import yaml

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark.controllers.registry import BASELINES, build_adapter, load_baseline_config  # noqa: E402
from benchmark.core import frames  # noqa: E402


class NativeReplayScenario:
    """Reproduces the native command sources bit-for-bit.

    * base: the constant ``base_command`` from the baseline's own YAML
    * EE:   the native ``PresampledKeypointsDirectCommandPLBSim`` /
            ``PresampledDirectPLBCommand`` sample-and-hold, same table, same
            ``ee_command_seed``, same ``ee_cycle_duration_s``, same off-by-one
            counter, same ``sample_first=True`` reset.
    """

    kind = "native_replay"
    description = {"experiment": "parity", "condition": "native_replay"}

    def __init__(self, cfg: dict, n_ticks: int, control_dt: float):
        self.base_command = np.asarray(cfg["base_command"], dtype=np.float32).reshape(3)
        table_rel = cfg.get("ee_command_path") or cfg["ee_command_table"]
        table = np.load(os.path.join(REPO_ROOT, table_rel), allow_pickle=False).astype(np.float32)
        rng = np.random.default_rng(int(cfg.get("ee_command_seed", 0)))
        cycle_steps = max(1, int(round(float(cfg.get("ee_cycle_duration_s", 8.0)) / control_dt)))

        # sample_first=True draws row 0 at reset; update() then resamples when
        # the per-cycle counter reaches cycle_steps.
        self.reset_command = table[int(rng.integers(0, len(table)))].copy()
        commands = np.empty((n_ticks, 9), dtype=np.float32)
        current = self.reset_command.copy()
        step_in_cycle = 0
        for k in range(n_ticks):
            step_in_cycle += 1
            if step_in_cycle >= cycle_steps:
                current = table[int(rng.integers(0, len(table)))].copy()
                step_in_cycle = 0
            commands[k] = current
        self.commands = commands

    def prepare(self, adapter, initial_ee_kp_plb):
        pass

    def settle_commands(self, adapter, initial_ee_kp_plb):
        # The native scripts seed the observation histories with the FIRST
        # sampled command, not with the measured EE pose.
        return self.base_command.copy(), self.reset_command.copy()

    def command(self, scenario_tick, adapter):
        return self.base_command.copy(), self.commands[scenario_tick], None

    def arrays(self):
        return {"base_commands": np.repeat(self.base_command[None, :], len(self.commands), 0),
                "ee_commands_plb": self.commands}


# --------------------------------------------------------------------------
# run the adapter and collect the same diagnostics the scripts print
# --------------------------------------------------------------------------
def run_adapter(baseline: str, duration_s: float, report_every: int = 200) -> dict:
    import torch

    # Thread count changes the reduction order inside the TorchScript UAN and
    # inside ONNX Runtime, which perturbs the last bits of every torque. Both
    # sides of this comparison must therefore be pinned the same way: the
    # subprocess gets OMP_NUM_THREADS=1 and this process gets
    # torch.set_num_threads(1). The benchmark runner pins the same way, so
    # benchmark episodes are bit-reproducible.
    torch.set_num_threads(1)
    cfg, _ = load_baseline_config(baseline)
    adapter = build_adapter(baseline)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()

    scenario = NativeReplayScenario(
        cfg, int(round(duration_s / adapter.control_dt)) + 2, adapter.control_dt
    )
    base0, ee0 = scenario.settle_commands(adapter, None)
    adapter.reset(base0, ee0)

    n_steps = int(round(duration_s / adapter.simulation_dt))
    sim_time, tick = 0.0, 0
    samples = {}
    ee_cmd_current = ee0.copy()

    for counter in range(n_steps):
        adapter.pre_physics(sim_time)
        mujoco.mj_step(model, data)
        sim_time += adapter.simulation_dt
        if counter % adapter.control_decimation == 0:
            base_cmd, ee_cmd_current, _ = scenario.command(tick, adapter)
            adapter.policy_tick(base_cmd, ee_cmd_current, sim_time)
            tick += 1
        if counter % report_every == 0:
            ee_act = frames.ee_keypoints_plb_from_state(
                data.xpos[adapter.ee_bid], data.xmat[adapter.ee_bid].reshape(3, 3),
                data.qpos[0:3], data.qpos[3:7], adapter.ee_ground_z,
                adapter.ee_kp_dx, adapter.ee_kp_dz,
            )
            kp_err = np.linalg.norm((ee_cmd_current - ee_act).reshape(3, 3), axis=1)
            roll, pitch, _ = frames.roll_pitch_yaw(data.qpos[3:7])
            samples[counter] = {
                "z": float(data.qpos[2]),
                "roll_deg": float(np.degrees(roll)),
                "pitch_deg": float(np.degrees(pitch)),
                "kp_err": kp_err.astype(float).tolist(),
                "max_ctrl": float(np.max(np.abs(data.ctrl))),
            }
    return samples


# --------------------------------------------------------------------------
# run the original script headless and parse its diagnostics
# --------------------------------------------------------------------------
_C1_LINE = re.compile(
    r"\[\s*(\d+)\]\s+t=\s*([\d.]+)s \| z=\s*([-\d.]+) \| roll=\s*([-+\d.]+)deg \| "
    r"pitch=\s*([-+\d.]+)deg"
)
_C1_ERR = re.compile(
    r"max_leg_err=[\d.]+ \| max_arm_err=[\d.]+ \| "
    r"ee_err_plb=\[([\d.e+-]+),([\d.e+-]+),([\d.e+-]+)\] \| max_ctrl=([\d.]+)"
)
# C3/C4 print the keypoint errors with np.array2string, which drops trailing
# zeros and pads ("[0.017 0.018 0.02 ]", "[0. 0. 0.]"), so the bracket content
# is captured whole and split rather than matched field by field. A regex that
# assumed three fixed fields silently skipped those lines and misaligned the
# whole comparison.
_C3_LINE = re.compile(
    r"\[C[34] t=\s*([\d.]+)s\] z=([\d.-]+) ee_kp_err=\[([^\]]*)\]"
)
_C5_LINE = re.compile(
    r"\[C[56] S2 t=\s*([\d.]+)s\] z=([\d.-]+) kp_errors=\[([^\]]*)\].*?"
    r"pitch=([-+\d.]+)deg roll=([-+\d.]+)deg"
)


def _split_array(text: str) -> list:
    return [float(v) for v in text.replace(",", " ").split()]


def run_original(baseline: str, duration_s: float) -> dict:
    script = os.path.join(SIM_ROOT, BASELINES[baseline]["deploy_script"])
    cfg_path = os.path.join(SIM_ROOT, BASELINES[baseline]["config"])
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["simulation_duration"] = float(duration_s)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cfg, handle)
        tmp_cfg = handle.name

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        BENCHMARK_ORT_THREADS="1",
    )
    cmd = [sys.executable, script, tmp_cfg, "--headless"]
    proc = subprocess.run(cmd, cwd=SIM_ROOT, capture_output=True, text=True, env=env, timeout=900)
    os.unlink(tmp_cfg)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Original {baseline} failed (rc={proc.returncode}):\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )

    samples = {}
    text = proc.stdout
    if baseline in ("C1", "C2"):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = _C1_LINE.search(line)
            if not m:
                continue
            counter = int(m.group(1))
            entry = {
                "z": float(m.group(3)),
                "roll_deg": float(m.group(4)),
                "pitch_deg": float(m.group(5)),
            }
            for follow in lines[i + 1 : i + 3]:
                e = _C1_ERR.search(follow)
                if e:
                    entry["kp_err"] = [float(e.group(j)) for j in (1, 2, 3)]
                    entry["max_ctrl"] = float(e.group(4))
                    break
            samples[counter] = entry
    elif baseline in ("C3", "C4"):
        for m in _C3_LINE.finditer(text):
            # Index by the PRINTED time, so a missed line cannot silently
            # shift every later sample.
            counter = int(round(float(m.group(1)) / 0.004))
            samples[counter] = {
                "z": float(m.group(2)),
                "kp_err": _split_array(m.group(3)),
            }
    else:
        for m in _C5_LINE.finditer(text):
            counter = int(round(float(m.group(1)) / 0.004))
            samples[counter] = {
                "z": float(m.group(2)),
                "kp_err": _split_array(m.group(3)),
                "pitch_deg": float(m.group(4)),
                "roll_deg": float(m.group(5)),
            }
    if not samples:
        raise RuntimeError(f"Could not parse {baseline} diagnostics (tail):\n{text[-800:]}")
    return samples


# --------------------------------------------------------------------------
# The comparison floor is the scripts' own print precision, not the simulator:
# a value printed with "%.2f" cannot be compared more tightly than 0.5e-2.
# Half an ULP of the format each script actually uses, per baseline.
#   C1/C2: z "%.3f", roll/pitch "%+6.2f", max_ctrl "%.2f", kp_err "%.4f"
#   C3/C4: z "%.3f", kp_err via np.array2string(precision=3)
#   C5/C6: z "%.3f", pitch/roll "%+.1f", kp_err via np.round(.,3)
_DEFAULT_ULP = {"z": 0.5e-3, "roll_deg": 0.5e-2, "pitch_deg": 0.5e-2, "max_ctrl": 0.5e-2}
PRINT_HALF_ULP = {
    "C1": dict(_DEFAULT_ULP),
    "C2": dict(_DEFAULT_ULP),
    "C3": {"z": 0.5e-3},
    "C4": {"z": 0.5e-3},
    "C5": {"z": 0.5e-3, "roll_deg": 0.5e-1, "pitch_deg": 0.5e-1},
    "C6": {"z": 0.5e-3, "roll_deg": 0.5e-1, "pitch_deg": 0.5e-1},
}
KP_ERR_HALF_ULP = {"C1": 0.5e-4, "C2": 0.5e-4, "C3": 0.5e-3, "C4": 0.5e-3,
                   "C5": 0.5e-3, "C6": 0.5e-3}


def compare(baseline: str, duration_s: float, slack: float) -> tuple[bool, str]:
    """Compare every parsed field against its own print-precision floor.

    ``slack`` is a multiplier on that floor: 1.0 means "indistinguishable at
    the precision the script prints", which is the strongest claim these logs
    can support. Bit-exactness is verified separately and much more strictly by
    ``test_bitexact_parity.py``, which dumps full state arrays.
    """
    native = run_original(baseline, duration_s)
    adapter = run_adapter(baseline, duration_s)

    shared = sorted(set(native) & set(adapter))
    if len(shared) < 3:
        return False, f"only {len(shared)} comparable sample points"
    missed = sorted(set(native) - set(adapter)) + sorted(set(adapter) - set(native))
    expected = int(duration_s / 0.004) // 200
    if len(shared) < expected - 1:
        return False, (
            f"parsed only {len(shared)} of ~{expected} diagnostic lines "
            f"(unmatched indices {missed[:5]}) -- the comparison would be misaligned"
        )

    worst_ratio = 0.0
    worst_desc = ""
    worst_abs = 0.0
    n_compared = 0
    for counter in shared:
        a, b = native[counter], adapter[counter]
        floors = PRINT_HALF_ULP[baseline]
        checks = [(key, floors[key]) for key in floors if key in a and key in b]
        for key, floor in checks:
            delta = abs(a[key] - b[key])
            n_compared += 1
            ratio = delta / floor
            if ratio > worst_ratio:
                worst_ratio, worst_abs = ratio, delta
                worst_desc = f"step {counter} {key}: {a[key]} vs {b[key]}"
        if "kp_err" in a and "kp_err" in b:
            floor = KP_ERR_HALF_ULP[baseline]
            for i, (x, y) in enumerate(zip(a["kp_err"], b["kp_err"])):
                delta = abs(x - y)
                n_compared += 1
                ratio = delta / floor
                if ratio > worst_ratio:
                    worst_ratio, worst_abs = ratio, delta
                    worst_desc = f"step {counter} kp_err[{i}]: {x} vs {y}"

    ok = worst_ratio <= slack
    return ok, (
        f"{len(shared)} sample points / {n_compared} field comparisons; "
        f"worst difference {worst_abs:.3e} = {worst_ratio:.2f} x the print "
        f"precision floor (limit {slack:g}x)"
        + (f" [{worst_desc}]" if worst_desc else "")
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default="C1,C2,C3,C5")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--slack", type=float, default=1.0,
        help="Multiplier on each field's print-precision floor. 1.0 = the "
             "adapter is indistinguishable from the original at the precision "
             "the original prints.",
    )
    args = parser.parse_args(argv)

    failures = []
    for baseline in [b.strip() for b in args.baselines.split(",")]:
        print(f"--- parity: {baseline} ({args.duration:g} s) ---", flush=True)
        try:
            ok, detail = compare(baseline, args.duration, args.slack)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {baseline}: {detail}", flush=True)
        if not ok:
            failures.append(baseline)

    print("=" * 60)
    if failures:
        print(f"PARITY FAILED for {failures}")
        return 1
    print("PARITY PASS: every adapter reproduces its original controller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
