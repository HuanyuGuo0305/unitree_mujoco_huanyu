#!/usr/bin/env python3
"""Bit-exact parity: the adapters must reproduce the originals exactly.

Acceptance criterion 17, strict version. The console-log comparison in
``test_controller_parity.py`` is limited by the scripts' print precision; this
test removes that limit by dumping full float64 state arrays.

Method
------
A *throwaway copy* of each original deployment script is created in a temporary
directory inside ``deploy_mujoco`` (so its ``__file__``-relative asset
resolution still works), with a single probe line inserted that appends
``[sim_time, qpos, qvel, ctrl, last_action, arm_target, leg_target]`` to a list
and saves it at the end. The original scripts on disk are never modified. The
copy is deleted afterwards.

The adapter is then driven by the same ``NativeReplayScenario`` from the same
initial state and the two arrays are compared element-wise. Any nonzero
difference in qpos, qvel, ctrl, the raw policy action or the joint targets
fails the test.

Both sides are pinned to single-threaded math (thread count changes the
floating-point reduction order inside TorchScript and ONNX Runtime).

Run:
    python3 benchmark/tests/test_bitexact_parity.py
    python3 benchmark/tests/test_bitexact_parity.py --baselines C1 --duration 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.controllers.registry import BASELINES, build_adapter, load_baseline_config  # noqa: E402
from test_controller_parity import NativeReplayScenario  # noqa: E402

PROBE_CALL = (
    "            if {counter} < 40 or {counter} % {stride} == 0:\n"
    "                _bench_probe.append(np.concatenate([[{simtime}], d.qpos.copy(), "
    "d.qvel.copy(), d.ctrl.copy(), np.asarray({action}, dtype=np.float64).ravel(), "
    "np.asarray({arm_target}, dtype=np.float64).ravel(), "
    "np.asarray({leg_target}, dtype=np.float64).ravel()]))\n"
)

# Per-script loop-variable names and the unique line the probe is inserted
# before. Verified against the current files; the instrumenter fails loudly if
# an anchor is missing or ambiguous, so a refactor of a deployment script
# cannot silently turn this test into a no-op.
PROBE_SPECS = {
    "C1": dict(anchor="            counter += 1\n", counter="counter", simtime="sim_time",
               action="last_action", arm_target="arm_target", leg_target="leg_target"),
    "C2": dict(anchor="            counter += 1\n", counter="counter", simtime="sim_time",
               action="last_action", arm_target="arm_target", leg_target="leg_target"),
    "C3": dict(anchor="            counter += 1\n", counter="counter", simtime="sim_time",
               action="last_action", arm_target="arm_target", leg_target="leg_target"),
    "C4": dict(anchor="            counter += 1\n", counter="counter", simtime="sim_time",
               action="last_action", arm_target="arm_target", leg_target="leg_target"),
    "C5": dict(anchor="            count += 1\n", counter="count", simtime="time_s",
               action="last_loco", arm_target="arm_target", leg_target="leg_target"),
    "C6": dict(anchor="            count += 1\n", counter="count", simtime="time_s",
               action="last_loco", arm_target="arm_target", leg_target="leg_target"),
}


def _instrument(baseline: str, stride: int) -> str:
    """Write an instrumented throwaway copy next to the original.

    The original file on disk is never touched. ``_bench_probe`` is declared at
    module level so it is reachable from both the ``main()``-based scripts
    (C3-C6) and the ``__main__``-block scripts (C1/C2).
    """
    spec = PROBE_SPECS[baseline]
    src = os.path.join(SIM_ROOT, BASELINES[baseline]["deploy_script"])
    with open(src, "r", encoding="utf-8") as handle:
        text = handle.read()

    anchor = spec["anchor"]
    if text.count(anchor) != 1:
        raise RuntimeError(
            f"{baseline}: probe anchor {anchor!r} appears {text.count(anchor)} times, "
            "expected exactly 1 -- the deployment script changed shape"
        )
    probe = PROBE_CALL.format(
        stride=stride, counter=spec["counter"], simtime=spec["simtime"],
        action=spec["action"], arm_target=spec["arm_target"],
        leg_target=spec["leg_target"],
    )
    text = text.replace(anchor, probe + anchor, 1)

    main_guard = "\nif __name__"
    if main_guard not in text:
        raise RuntimeError(f"{baseline}: no __main__ guard to anchor the probe list")
    idx = text.index(main_guard)
    text = text[: idx + 1] + "_bench_probe = []\n\n" + text[idx + 1 :]

    text = text.rstrip() + (
        "\n\nimport os as _bench_os\n"
        "import numpy as _bench_np\n"
        "_bench_np.save(_bench_os.environ['BENCH_PROBE_OUT'], "
        "_bench_np.stack(_bench_probe))\n"
    )

    fd, path = tempfile.mkstemp(
        prefix=f"_bench_probe_{baseline}_", suffix=".py",
        dir=os.path.join(SIM_ROOT, "deploy_mujoco"),
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def run_original_probe(baseline: str, duration_s: float, stride: int) -> np.ndarray:
    import yaml

    probe_script = _instrument(baseline, stride)
    cfg_path = os.path.join(SIM_ROOT, BASELINES[baseline]["config"])
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["simulation_duration"] = float(duration_s)
    tmp_cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp_cfg)
    tmp_cfg.close()
    out = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    out.close()

    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", BENCH_PROBE_OUT=out.name,
    )
    try:
        proc = subprocess.run(
            [sys.executable, probe_script, tmp_cfg.name, "--headless"],
            cwd=SIM_ROOT, capture_output=True, text=True, env=env, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"instrumented {baseline} failed (rc={proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return np.load(out.name)
    finally:
        for path in (probe_script, tmp_cfg.name, out.name):
            try:
                os.unlink(path)
            except OSError:
                pass


def run_adapter_probe(baseline: str, duration_s: float, stride: int) -> np.ndarray:
    import torch

    torch.set_num_threads(1)
    cfg, _ = load_baseline_config(baseline)
    adapter = build_adapter(baseline)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()

    n_ticks = int(round(duration_s / adapter.control_dt)) + 4
    scenario = NativeReplayScenario(cfg, n_ticks, adapter.control_dt)
    base0, ee0 = scenario.settle_commands(adapter, None)
    adapter.reset(base0, ee0)

    probe = []
    sim_time, tick = 0.0, 0
    for counter in range(int(round(duration_s / adapter.simulation_dt))):
        adapter.pre_physics(sim_time)
        mujoco.mj_step(model, data)
        sim_time += adapter.simulation_dt
        if counter % adapter.control_decimation == 0:
            base_cmd, ee_cmd, _ = scenario.command(tick, adapter)
            adapter.policy_tick(base_cmd, ee_cmd, sim_time)
            tick += 1
        if counter < 40 or counter % stride == 0:
            probe.append(
                np.concatenate(
                    [
                        [sim_time], data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                        np.asarray(adapter.last_policy_action(), dtype=np.float64).ravel(),
                        np.asarray(adapter.arm_target, dtype=np.float64).ravel(),
                        np.asarray(adapter.leg_target, dtype=np.float64).ravel(),
                    ]
                )
            )
    return np.stack(probe)


def compare(baseline: str, duration_s: float, stride: int) -> tuple[bool, str]:
    native = run_original_probe(baseline, duration_s, stride)
    adapter = run_adapter_probe(baseline, duration_s, stride)
    n = min(len(native), len(adapter))
    if n < 10:
        return False, f"only {n} probe rows"
    native, adapter = native[:n], adapter[:n]
    if native.shape[1] != adapter.shape[1]:
        return False, f"column mismatch {native.shape} vs {adapter.shape}"

    delta = np.abs(native - adapter)
    worst = float(delta.max())
    row = int(np.unravel_index(int(delta.argmax()), delta.shape)[0]) if worst > 0 else -1
    col = int(np.unravel_index(int(delta.argmax()), delta.shape)[1]) if worst > 0 else -1
    first_bad = -1
    rowmax = delta.max(axis=1)
    if worst > 0:
        first_bad = int(np.argmax(rowmax > 0))
    return (
        worst == 0.0,
        f"{n} probe rows x {native.shape[1]} float64 columns; "
        f"max |difference| = {worst:.3e}"
        + ("" if worst == 0 else f" (first differing probe row {first_bad}, "
                                f"worst at row {row} column {col})"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default="C1,C2,C3,C4,C5,C6")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=50,
                        help="Probe every N physics steps (plus the first 40)")
    args = parser.parse_args(argv)

    failures = []
    for baseline in [b.strip() for b in args.baselines.split(",")]:
        print(f"--- bit-exact parity: {baseline} ({args.duration:g} s) ---", flush=True)
        try:
            ok, detail = compare(baseline, args.duration, args.stride)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}  {baseline}: {detail}", flush=True)
        if not ok:
            failures.append(baseline)

    print("=" * 70)
    if failures:
        print(f"BIT-EXACT PARITY FAILED for {failures}")
        return 1
    print("BIT-EXACT PARITY PASS: every adapter reproduces its original exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
