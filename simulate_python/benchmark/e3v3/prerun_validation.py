#!/usr/bin/env python3
"""E3 v3 pre-run validation (before the official 15,000 episodes).

  V1 command timing     switch ticks exactly every T s from t = 3 s; Fixed constant
  V2 no interpolation   every EE command equals one of the 5 frozen waypoints
  V3 sequences          line ping-pong / circle cyclic, start/direction counterbalanced
  V4 paired schedules   EE identical across base levels; base identical across EE
                        levels and patterns; both a function of the seed only
  V5 base bounds        |vx|,|vy|,|wz| <= B; B = 0 exact zeros; U(2,4) s segments
  V6 pilot independence pilot seeds disjoint from official and showcase pools;
                        selection inputs contain no between-controller quantity
  V7 frozen config      configs/E3.yaml SHA-256 equals the value recorded at freeze
  V8 metric definitions RMSEs recomputed from per-tick logs of trial episodes
                        equal metrics.json (run separately: --episodes DIR)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from benchmark.round3 import common
from benchmark.core import scenario as sc
from benchmark.scenarios import experiments as E

V3 = os.path.join(common.BENCH, "results", "E3_v3")


def main():
    cfg, g = E.load_config("E3")
    seeds = list(range(cfg["official_seeds"]["start"],
                       cfg["official_seeds"]["start"] + cfg["official_seeds"]["count"]))
    conds = E.e3_conditions(cfg)
    wp = {p: np.asarray(cfg["ee"]["waypoints"][p]["keypoints_plb"], np.float32)
          for p in cfg["ee"]["patterns"]}
    n = E.n_eval_ticks(cfg)
    res, fails = {}, []
    starts = {p: np.zeros((5, 2), int) for p in wp}
    for seed in seeds:
        sch = {c: E.build_e3(cfg, c, seed) for c, *_ in conds}
        for cid, B, T, pat in conds:
            s = sch[cid]
            ee, base = s.ee.commands, s.base.commands
            # V2: every command is a frozen waypoint (bit-exact)
            match = (np.abs(ee[:, None, :] - wp[pat][None]).max(-1) == 0)
            if not match.any(1).all():
                fails.append(f"V2 {cid} seed {seed}: command not a waypoint")
            idx = match.argmax(1)
            # V1: timing
            ch = np.nonzero(np.any(np.diff(ee, axis=0) != 0, axis=1))[0] + 1
            if T is None:
                if len(ch):
                    fails.append(f"V1 {cid} seed {seed}: Fixed level changes")
            else:
                steps = int(round(T / 0.02))
                expect = np.arange(steps, n, steps)
                if not np.array_equal(ch, expect) or not np.array_equal(s.ee.switch_ticks, expect):
                    fails.append(f"V1 {cid} seed {seed}: switch ticks {ch[:4]} != {expect[:4]}")
            # V3: sequence
            st, d = sc.e3v3_start_direction(seed)
            seg = idx[np.r_[0, ch]] if len(ch) else idx[:1]
            want = sc.e3v3_sequence(pat, st, d, len(seg))
            if not np.array_equal(seg, want):
                fails.append(f"V3 {cid} seed {seed}: sequence {seg[:6]} != {want[:6]}")
            if idx[0] != st:
                fails.append(f"V3 {cid} seed {seed}: start {idx[0]} != {st}")
            # V5: bounds
            if B == 0.0:
                if np.any(base != 0):
                    fails.append(f"V5 {cid} seed {seed}: B=0 base not zero")
            elif np.abs(base).max() > B + 1e-6:
                fails.append(f"V5 {cid} seed {seed}: base exceeds bound")
            dur = s.base.segment_durations_s[:-1]
            if len(dur) and (dur.min() < 2.0 - 1e-9 or dur.max() > 4.0 + 1e-9):
                fails.append(f"V5 {cid} seed {seed}: segment duration out of [2,4]")
        for pat in wp:
            st, d = sc.e3v3_start_direction(seed)
            starts[pat][st, 0 if d > 0 else 1] += 1
        # V4: pairing
        for bi, B in enumerate(cfg["base"]["levels"]):
            ref = sch[f"B{bi}T0_line"].base
            for ti in range(5):
                for pat in wp:
                    b = sch[f"B{bi}T{ti}_{pat}"].base
                    if not (np.array_equal(b.commands, ref.commands)
                            and np.array_equal(b.switch_ticks, ref.switch_ticks)):
                        fails.append(f"V4 seed {seed}: base differs across EE/pattern at B{bi}")
        for ti in range(5):
            for pat in wp:
                ref = sch[f"B0T{ti}_{pat}"].ee.commands
                for bi in range(1, 5):
                    if not np.array_equal(sch[f"B{bi}T{ti}_{pat}"].ee.commands, ref):
                        fails.append(f"V4 seed {seed}: EE differs across base levels")
        # base waveform shared across levels (same seed): commands = waveform * B
        w = sch["B4T0_line"].base.waveform
        for bi, B in enumerate(cfg["base"]["levels"][1:], 1):
            if not np.allclose(sch[f"B{bi}T0_line"].base.commands, (w * np.float32(B)), atol=1e-6):
                fails.append(f"V4 seed {seed}: B{bi} not the scaled common waveform")
    # determinism
    a = E.build_e3(cfg, "B3T4_circle", 17).arrays()
    b = E.build_e3(cfg, "B3T4_circle", 17).arrays()
    if not all(np.array_equal(a[k], b[k]) for k in a):
        fails.append("V4: schedule not deterministic")
    res["counterbalance_start_x_direction"] = {p: starts[p].tolist() for p in starts}
    # V6
    pilot_seeds = set(cfg["pilot"]["seeds"])
    off = set(seeds)
    sc_pool = set(range(g["showcase_seeds"]["start"], g["showcase_seeds"]["start"] + g["showcase_seeds"]["count"]))
    if pilot_seeds & (off | sc_pool):
        fails.append("V6: pilot seeds overlap official/showcase")
    import csv
    with open(os.path.join(V3, "pilot", "pilot_screening.csv")) as h:
        hdr = next(csv.reader(h))
    banned = [c for c in hdr if "diff" in c or "minus" in c or "advantage" in c or "rel" in c]
    if banned:
        fails.append(f"V6: selection table has between-controller columns {banned}")
    res["pilot_seeds"] = sorted(pilot_seeds)
    res["selection_inputs"] = ["span_m", "ori_range_deg", "F", "W (max over controllers)"]
    # V7
    sha = hashlib.sha256(open(os.path.join(common.BENCH, "configs", "E3.yaml"), "rb").read()).hexdigest()
    frozen = open(os.path.join(V3, "protocol", "E3_config_sha256_frozen.txt")).read().split()[0]
    if sha != frozen:
        fails.append("V7: configs/E3.yaml changed after freeze")
    res["config_sha256"] = sha
    res["n_conditions"] = len(conds)
    res["n_seeds"] = len(seeds)
    res["failures"] = fails
    res["passed"] = not fails
    os.makedirs(os.path.join(V3, "protocol"), exist_ok=True)
    with open(os.path.join(V3, "protocol", "prerun_validation.json"), "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
    print(f"{len(fails)} failures")
    for f in fails[:20]:
        print("  ", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
