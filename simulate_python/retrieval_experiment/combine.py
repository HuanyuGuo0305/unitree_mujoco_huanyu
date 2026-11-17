"""Combine the immutable 4-6 cm base run with the OOD extension run -> official E5 (3-7 cm) analysis.

    cd unitree_mujoco/simulate_python
    python3 -m retrieval_experiment.combine retrieval_experiment/results/<ood_extension_run>

Design (config.yaml: official). Four final 1-cm groups, each the union of two half-cm strata:

    stratum   [3,4] OOD-small   [4,5] ID      [5,6] ID      [6,7] OOD-large
      A       [3.0, 3.5)        [4.0, 4.5)    [5.0, 5.5)    [6.0, 6.5)
      B       [3.5, 4.0)        [4.5, 5.0)    [5.5, 6.0)    [6.5, 7.0]

Every cell replays the SAME 250 base scenarios (same size quantile u_size inside the stratum, same
density quantile, materials, reset state, cube placement, target and noise seed). The 4-6 cm cells
come from the base run (never re-simulated); the OOD cells from the extension run. Pairing key:
(scenario_id, stratum). Statistical inference clusters on scenario_id (250 clusters).

Fails with an explicit error if any source datum is missing, altered (SHA-256), incomplete,
duplicated, generated with a different configuration, or if the reused base data do not reproduce
the regression counts or their stored One-Shot classification.

Writes <ood_run>/combined/:
    episodes_combined.csv      all 2000 episodes with provenance (source run, trajectory file, ...)
    summary_table.csv/.tex     official four-row paper table
    group_metrics.csv          per final group: rates, Wilson CIs, completion time, grasp / drop / re-grasp
    stratum_metrics.csv        per half-cm stratum (diagnostic)
    failure_breakdown.csv, stage_decomposition.csv
    ood_vs_id.json             paired cluster bootstrap effects (OOD - ID), permutation p-values, Cochran's Q per stratum
    physics_diagnostics.csv/.json   failure correlates (size, density, mass, aperture, penetration, slip)
    detector_validation.json   retention-detector gap populations and One-Shot sensitivity on OOD data
    provenance.json            source runs, hashes, verification results
    COMBINED.md                human-readable report
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from retrieval_experiment import retention as RT  # noqa: E402
from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.analyze import cochran_q, mcnemar_exact  # noqa: E402
from retrieval_experiment.metrics import FAILURE_REASONS, mean_std, wilson_ci  # noqa: E402
from retrieval_experiment.simulator import REPO_ROOT, resolve_repo_path  # noqa: E402

HL_DT = 0.1
N_BOOT = 10000


class SourceDataError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _key_iv(iv):
    return (round(float(iv[0]), 6), round(float(iv[1]), 6))


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------
def _run_config(run_dir):
    p = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(p):
        raise SourceDataError(f"missing {p}")
    return yaml.safe_load(open(p))


def _check_complete(run_dir, n_scen, n_groups, what):
    ep = os.path.join(run_dir, "episodes.csv")
    if not os.path.isfile(ep):
        raise SourceDataError(f"{what}: missing {ep}")
    rows = list(csv.DictReader(open(ep)))
    keys = [(int(r["scenario_id"]), int(r["group"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise SourceDataError(f"{what}: duplicated episodes in {ep}")
    expected = {(i, g) for i in range(n_scen) for g in range(1, n_groups + 1)}
    if set(keys) != expected:
        raise SourceDataError(f"{what}: episodes.csv is incomplete or has unexpected entries "
                              f"({len(set(keys) & expected)} of {len(expected)} expected episodes)")
    for i, g in keys:
        t = os.path.join(run_dir, "trajectories", f"s{i:04d}_g{g}.npz")
        if not os.path.isfile(t):
            raise SourceDataError(f"{what}: missing trajectory {t}")
    return rows


def verify_base_run(base_dir: str, exp: dict) -> dict:
    """Integrity and completeness of the immutable base run; returns its description."""
    off = exp["official"]
    if not os.path.isdir(base_dir):
        raise SourceDataError(f"base run not found: {base_dir} -- it must not be regenerated")
    sha = {}
    for f, want in off["base_run_sha256"].items():
        p = os.path.join(base_dir, f)
        if not os.path.isfile(p):
            raise SourceDataError(f"base run file missing: {p}")
        sha[f] = sha256(p)
        if sha[f] != want:
            raise SourceDataError(f"base run file altered: {p} (sha256 {sha[f][:16]} != {want[:16]})")
    cfg = _run_config(base_dir)["experiment_config"]
    ivs = [list(map(float, iv)) for iv in cfg["size_intervals_cm"]]
    _, scs = scen.load(os.path.join(base_dir, "scenarios.json"))
    rows = _check_complete(base_dir, len(scs), len(ivs), "base run")
    if len(rows) != int(off["base_episodes"]):
        raise SourceDataError(f"base run has {len(rows)} episodes, expected {off['base_episodes']}")
    z = np.load(os.path.join(base_dir, "trajectories", "s0000_g1.npz"))
    if "ph_q" not in z.files:
        raise SourceDataError("base run has no 250-Hz retention log (ph_*)")
    return {"sha256": sha, "intervals_cm": ivs, "n_scenarios": len(scs), "config": cfg}


def assert_disjoint(base_ivs, new_ivs):
    for a in base_ivs:
        for b in new_ivs:
            if max(a[0], b[0]) < min(a[1], b[1]):
                raise SourceDataError(f"interval {b} overlaps base interval {a}: it would be simulated twice")


def _comparable(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    # interval / bank size / worker count and analysis-only blocks (official, retention) may differ
    for k in ("size_intervals_cm", "trials_per_group", "workers", "official", "retention"):
        c.pop(k, None)
    return c


# ---------------------------------------------------------------------------
# Episode records
# ---------------------------------------------------------------------------
def load_records(run_dir, source, ret_cfg):
    cfg = _run_config(run_dir)["experiment_config"]
    ivs = [list(map(float, iv)) for iv in cfg["size_intervals_cm"]]
    recs, gaps = [], []
    for r in csv.DictReader(open(os.path.join(run_dir, "episodes.csv"))):
        sid, g = int(r["scenario_id"]), int(r["group"])
        tpath = os.path.join(run_dir, "trajectories", f"s{sid:04d}_g{g}.npz")
        z = np.load(tpath)
        side, rho = float(r["cube_side_m"]), float(r["cube_density"])
        succ = r["success"] == "True"
        ret = RT.from_npz(z, succ, ret_cfg)
        strict = RT.from_npz(z, succ, ret_cfg, settled=False)
        gaps += [(source, sid, g) + tuple(x) for x in ret["gaps"]]
        recs.append({
            "source": source, "source_run": os.path.relpath(run_dir, REPO_ROOT),
            "trajectory_file": os.path.relpath(tpath, REPO_ROOT), "source_group": g,
            "interval": _key_iv(ivs[g - 1]), "scenario_id": sid,
            "size_u": _f(r["size_u"]), "density_u": _f(r["density_u"]),
            "side": side, "density": rho, "mass": _f(r["cube_mass_kg"]), "mujoco_mass": _f(r["mujoco_mass_kg"]),
            "mujoco_inertia": _f(r["mujoco_inertia"]), "expected_inertia": _f(r["expected_inertia"]),
            "task_success": succ, "one_shot": bool(ret["one_shot_success"]), "one_shot_strict": bool(strict["one_shot_success"]),
            "completion_time_s": _f(r["completion_time_s"]), "failure_reason": r["failure_reason"],
            "hl_steps": int(r["hl_steps"]), "grasp_confirmed": r["grasp_confirmed"] == "True",
            "reached_stage1": r["reached_stage1"] == "True", "time_to_grasp_s": _f(r["time_to_grasp_s"]),
            "acquired": bool(ret["acquired"]), "first_acquisition_s": ret["first_acquisition_s"],
            "acquisition_jaw_angle_deg": ret["acquisition_jaw_angle_deg"],
            "attempts_before_acquisition": ret["attempts_before_acquisition"],
            "drops": ret["releases"], "drop_times_s": ";".join(f"{t:.3f}" for t in ret["release_times_s"]),
            "regrasps": ret["reacquisitions"], "regrasp_times_s": ";".join(f"{t:.3f}" for t in ret["reacquisition_times_s"]),
            "flicker_gaps": ret["flicker_gaps"], "close_to_open_cmd_while_held": ret["close_to_open_cmd_while_held"],
            "close_to_open_cmd_after_acquisition": ret["close_to_open_cmd_after_acquisition"],
            "s2_max_penetration_mm": _f(r["s2_max_penetration_mm"]), "s2_cube_slip_mm": _f(r["s2_cube_slip_mm"]),
            "s2_mean_grip_force_N": 0.5 * (_f(r["s2_mean_f_stator"]) + _f(r["s2_mean_f_mover"])),
            "cube_mu_used": _f(r["cube_mu_used"]), "object_r": _f(r["object_r"]), "target_radius": _f(r["target_radius"]),
            "_ph": {k: z[k] for k in ("ph_f_stator", "ph_f_mover", "ph_q", "ph_rel_pos", "ph_cmd", "ph_dt")},
        })
    return recs, gaps


# ---------------------------------------------------------------------------
# Statistics helpers (cluster = base scenario_id)
# ---------------------------------------------------------------------------
def _per_scenario(recs, key):
    d = defaultdict(list)
    for r in recs:
        d[r["scenario_id"]].append(float(r[key]))
    return d


def cluster_rate_diff(a_recs, b_recs, key, rng, n_boot=N_BOOT):
    """Paired cluster bootstrap of rate(a) - rate(b) [percentage points]; clusters = scenario ids."""
    A, B = _per_scenario(a_recs, key), _per_scenario(b_recs, key)
    ids = sorted(set(A) & set(B))
    pa = np.array([np.mean(A[i]) for i in ids])
    pb = np.array([np.mean(B[i]) for i in ids])
    d = 100.0 * (pa - pb)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    bs = d[idx].mean(1)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, len(ids)))
    perm = np.abs((signs * d).mean(1))
    p = float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n_boot + 1))
    fa, fb = 1 - pa, 1 - pb
    rr_bs = (fa[idx].mean(1) + 1e-12) / (fb[idx].mean(1) + 1e-12)
    return {"n_clusters": len(ids), "rate_a_pct": 100 * pa.mean(), "rate_b_pct": 100 * pb.mean(),
            "diff_pp": float(d.mean()), "ci95_pp": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "failure_ratio": float((fa.mean() + 1e-12) / (fb.mean() + 1e-12)),
            "failure_ratio_ci95": [float(np.percentile(rr_bs, 2.5)), float(np.percentile(rr_bs, 97.5))],
            "sign_flip_permutation_p": p}


def cluster_time_diff(a_recs, b_recs, rng, n_boot=N_BOOT):
    """Cluster bootstrap of mean completion time (successful episodes) a - b [s]."""
    def by(recs):
        d = defaultdict(list)
        for r in recs:
            if r["task_success"]:
                d[r["scenario_id"]].append(r["completion_time_s"])
        return d
    A, B = by(a_recs), by(b_recs)
    ids = sorted({r["scenario_id"] for r in a_recs} | {r["scenario_id"] for r in b_recs})
    sa = np.array([sum(A.get(i, [])) for i in ids]); na = np.array([len(A.get(i, [])) for i in ids])
    sb = np.array([sum(B.get(i, [])) for i in ids]); nb = np.array([len(B.get(i, [])) for i in ids])
    est = sa.sum() / na.sum() - sb.sum() / nb.sum()
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    bs = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
    return {"mean_a_s": float(sa.sum() / na.sum()), "mean_b_s": float(sb.sum() / nb.sum()), "diff_s": float(est),
            "ci95_s": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def _mw(a, b):
    from scipy.stats import mannwhitneyu
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "p": float("nan")}
    return {"n_a": len(a), "n_b": len(b), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "p": float(mannwhitneyu(a, b).pvalue)}


# ---------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    allow_partial = "--allow-partial" in argv
    argv = [a for a in argv if a != "--allow-partial"]
    if not argv:
        raise SystemExit("usage: python3 -m retrieval_experiment.combine <ood_extension_run> [--allow-partial]")
    ood_dir = os.path.abspath(argv[0])
    exp = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    off = exp["official"]
    prov_p = os.path.join(ood_dir, "extension_provenance.json")
    if not os.path.isfile(prov_p):
        raise SourceDataError(f"{ood_dir} is not an OOD extension run (missing extension_provenance.json)")
    prov = json.load(open(prov_p))
    base_dir = resolve_repo_path(prov["base_run"])
    if os.path.abspath(base_dir) != os.path.abspath(resolve_repo_path(off["base_run"])):
        raise SourceDataError("extension run refers to a different base run than config.yaml: official.base_run")
    base = verify_base_run(base_dir, exp)
    ood_cfg_all = _run_config(ood_dir)
    ood_cfg = ood_cfg_all["experiment_config"]
    ood_ivs = [list(map(float, iv)) for iv in ood_cfg["size_intervals_cm"]]
    if [_key_iv(iv) for iv in ood_ivs] != [_key_iv(iv) for iv in off["ood_intervals_cm"]]:
        raise SourceDataError(f"extension run intervals {ood_ivs} != official OOD intervals {off['ood_intervals_cm']}")
    assert_disjoint(base["intervals_cm"], ood_ivs)
    _, base_scs = scen.load(os.path.join(base_dir, "scenarios.json"))
    _, ood_scs = scen.load(os.path.join(ood_dir, "scenarios.json"))
    if ood_scs != base_scs[:len(ood_scs)]:
        raise SourceDataError("extension run scenario bank differs from the base run's bank")
    if len(ood_scs) != len(base_scs) and not allow_partial:
        raise SourceDataError(f"extension run covers {len(ood_scs)} of {len(base_scs)} base scenarios (pilot?)")
    if _comparable(ood_cfg) != _comparable(base["config"]):
        diff = sorted(k for k in set(_comparable(ood_cfg)) | set(_comparable(base["config"]))
                      if _comparable(ood_cfg).get(k) != _comparable(base["config"]).get(k))
        raise SourceDataError(f"experiment configuration differs between base and extension run: {diff}")
    if ood_cfg_all["deploy_config"] != _run_config(base_dir)["deploy_config"]:
        raise SourceDataError("deploy configuration differs between base and extension run")
    pm_b = json.load(open(os.path.join(base_dir, "policy_manifest.json")))
    pm_o = json.load(open(os.path.join(ood_dir, "policy_manifest.json")))
    for k in ("high_level", "low_level", "uan", "scene_xml_sha256", "robot_xml_sha256", "deploy_config_sha256"):
        vb = pm_b[k]["sha256_verified"] if isinstance(pm_b[k], dict) else pm_b[k]
        vo = pm_o[k]["sha256_verified"] if isinstance(pm_o[k], dict) else pm_o[k]
        if vb != vo:
            raise SourceDataError(f"{k} differs between base and extension run")
    _check_complete(ood_dir, len(ood_scs), len(ood_ivs), "extension run")

    ret_cfg = dict(RT.DEFAULTS, **exp.get("retention", {}))
    base_recs, base_gaps = load_records(base_dir, "base_run", ret_cfg)
    ood_recs, ood_gaps = load_records(ood_dir, "ood_extension", ret_cfg)
    # the reused base data must reproduce their stored official One-Shot classification exactly
    stored = {(int(r["scenario_id"]), int(r["group"])): r
              for r in csv.DictReader(open(os.path.join(base_dir, "analysis", "episodes_retention.csv")))}
    for r in base_recs:
        s = stored[(r["scenario_id"], r["source_group"])]
        if (str(r["one_shot"]) != s["one_shot_success"] or int(s["drops"]) != r["drops"]
                or int(s["regrasps"]) != r["regrasps"] or str(r["task_success"]) != s["task_success"]):
            raise SourceDataError(f"base episode {r['scenario_id']}/G{r['source_group']}: retention re-evaluation "
                                  "differs from the stored official classification")

    # ---- map to final groups / strata
    cell = {}
    for gi, fg in enumerate(off["final_groups"]):
        for si, iv in enumerate(fg["strata"]):
            cell[_key_iv(iv)] = (gi, fg["label"], fg["type"], "AB"[si], f"[{iv[0]:.1f}, {iv[1]:.1f}]")
    recs = base_recs + ood_recs
    for r in recs:
        if r["interval"] not in cell:
            raise SourceDataError(f"episode interval {r['interval']} is not an official stratum")
        gi, lab, typ, st, sub = cell[r["interval"]]
        r.update(group_index=gi, size_group=lab, eval_type=typ, stratum=st, stratum_interval=sub)
        lo, hi = r["interval"]
        cm = 100 * r["side"]
        last = (lo, hi) == _key_iv(off["final_groups"][-1]["strata"][-1])
        if not (lo <= cm < hi or (last and lo <= cm <= hi)) or abs(cm - (lo + r["size_u"] * (hi - lo))) > 1e-9:
            raise SourceDataError(f"episode {r['scenario_id']} side {cm} cm inconsistent with stratum {lo}-{hi}")
        rho = exp["cube"]["density_range"]
        if abs(r["density"] - (rho[0] + r["density_u"] * (rho[1] - rho[0]))) > 1e-9:
            raise SourceDataError(f"episode {r['scenario_id']}: density not reproducible from density_u")
        if abs(r["mujoco_mass"] - r["density"] * r["side"] ** 3) > 1e-12 or \
                abs(r["mujoco_inertia"] - r["mass"] * r["side"] ** 2 / 6.0) > 1e-9 * r["mujoco_inertia"]:
            raise SourceDataError(f"episode {r['scenario_id']}: MuJoCo mass/inertia read-back mismatch")
    keys = [(r["scenario_id"], r["group_index"], r["stratum"]) for r in recs]
    if len(set(keys)) != len(keys):
        raise SourceDataError("duplicate (scenario_id, group, stratum) episodes")
    n_scen = len(base_scs)
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group_index"]].append(r)
    for gi, fg in enumerate(off["final_groups"]):
        n_exp = 2 * (n_scen if fg["type"] == "in_distribution" else len(ood_scs))
        if len(by_g[gi]) != n_exp:
            raise SourceDataError(f"group {fg['label']} has {len(by_g[gi])} episodes, expected {n_exp}")
    for lab, want in off["regression"].items():
        rs = [r for r in recs if r["size_group"] == lab]
        got = {"task_success": sum(r["task_success"] for r in rs), "one_shot": sum(r["one_shot"] for r in rs)}
        if got != want:
            raise SourceDataError(f"regression check failed for {lab}: {got} != {want}")
    for r in recs:
        if r["one_shot"] and not r["task_success"]:
            raise SourceDataError("one-shot success without task success")

    out = os.path.join(ood_dir, "combined")
    os.makedirs(out, exist_ok=True)

    def wcsv(name, rows):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- episode-level provenance table
    cols = ["source_run", "trajectory_file", "source_group", "scenario_id", "stratum", "size_group", "eval_type",
            "stratum_interval", "size_u", "density_u", "side", "density", "mass", "mujoco_mass", "mujoco_inertia",
            "task_success", "one_shot", "one_shot_strict", "completion_time_s", "failure_reason", "hl_steps",
            "grasp_confirmed", "time_to_grasp_s", "acquired", "first_acquisition_s", "acquisition_jaw_angle_deg",
            "attempts_before_acquisition", "drops", "drop_times_s", "regrasps", "regrasp_times_s", "flicker_gaps",
            "close_to_open_cmd_while_held", "s2_max_penetration_mm", "s2_cube_slip_mm", "s2_mean_grip_force_N",
            "cube_mu_used"]
    rename = {"side": "actual_cube_side_m", "density": "actual_cube_density", "mass": "actual_cube_mass_kg",
              "one_shot": "one_shot_success", "completion_time_s": "completion_time_s"}
    ordered = sorted(recs, key=lambda r: (r["group_index"], r["stratum"], r["scenario_id"]))
    wcsv("episodes_combined.csv", [{rename.get(c, c): r[c] for c in cols} for r in ordered])

    # ---- per group
    summary, gm = [], []
    for gi, fg in enumerate(off["final_groups"]):
        rs = by_g[gi]
        n = len(rs)
        k = sum(r["task_success"] for r in rs)
        k1 = sum(r["one_shot"] for r in rs)
        ct = [r["completion_time_s"] for r in rs if r["task_success"]]
        m, sd, nct = mean_std(ct)
        lo, hi = wilson_ci(k, n)
        lo1, hi1 = wilson_ci(k1, n)
        acq = [r for r in rs if r["acquired"]]
        dropped = [r for r in acq if r["drops"] > 0]
        conf = sum(r["grasp_confirmed"] for r in rs)
        summary.append({"size_interval_cm": fg["label"], "evaluation": fg["type"], "n_trials": n,
                        "task_successes": k, "one_shot_successes": k1,
                        "task_success_rate_pct": 100 * k / n, "one_shot_success_rate_pct": 100 * k1 / n,
                        "completion_time_mean_s": m, "completion_time_sd_s": sd, "completion_time_n": nct})
        gm.append({
            "size_interval_cm": fg["label"], "evaluation": fg["type"], "n_trials": n,
            "sources": "+".join(sorted({r["source"] for r in rs})),
            "mass_kg_min": min(r["mass"] for r in rs), "mass_kg_max": max(r["mass"] for r in rs),
            "task_successes": k, "task_pct": 100 * k / n, "task_ci95_lo": 100 * lo, "task_ci95_hi": 100 * hi,
            "one_shot_successes": k1, "one_shot_pct": 100 * k1 / n, "one_shot_ci95_lo": 100 * lo1, "one_shot_ci95_hi": 100 * hi1,
            "one_shot_strict_successes": sum(r["one_shot_strict"] for r in rs),
            "completion_time_mean_s": m, "completion_time_sd_s": sd, "completion_time_n": nct,
            "completion_time_median_s": float(np.median(ct)) if ct else float("nan"),
            "grasp_confirmation_pct": 100 * conf / n, "physical_acquisition_pct": 100 * len(acq) / n,
            "time_to_first_acquisition_mean_s": mean_std([r["first_acquisition_s"] for r in acq])[0],
            "acquisition_jaw_angle_mean_deg": mean_std([r["acquisition_jaw_angle_deg"] for r in acq])[0],
            "episodes_with_drop": len(dropped), "drop_pct_of_acquired": 100 * len(dropped) / len(acq) if acq else float("nan"),
            "drop_events": sum(r["drops"] for r in rs), "regrasp_events": sum(r["regrasps"] for r in rs),
            "regrasp_episodes": sum(r["regrasps"] > 0 for r in rs),
            "successful_after_drop_and_regrasp": sum(r["task_success"] and r["drops"] > 0 for r in rs),
            "training_grasp_loss_terminations": sum(r["failure_reason"] == "grasp_lost_after_confirmation" for r in rs),
            "episodes_with_failed_pinch_attempts": sum(r["attempts_before_acquisition"] > 0 for r in acq),
            "training_confirmed_never_settled": sum(r["grasp_confirmed"] and not r["acquired"] for r in rs),
            "s2_max_penetration_mm_p95": float(np.nanpercentile([r["s2_max_penetration_mm"] for r in rs if r["grasp_confirmed"]], 95)),
            "s2_cube_slip_mm_median": float(np.nanmedian([r["s2_cube_slip_mm"] for r in rs if r["grasp_confirmed"]])),
            "s2_mean_grip_force_N": mean_std([r["s2_mean_grip_force_N"] for r in rs if r["grasp_confirmed"]])[0],
        })
    wcsv("summary_table.csv", summary)
    wcsv("group_metrics.csv", gm)
    sm = []
    for gi, fg in enumerate(off["final_groups"]):
        for st in "AB":
            rs = [r for r in by_g[gi] if r["stratum"] == st]
            k = sum(r["task_success"] for r in rs)
            m, sd, nct = mean_std([r["completion_time_s"] for r in rs if r["task_success"]])
            sm.append({"size_group": fg["label"], "stratum": st, "stratum_interval": rs[0]["stratum_interval"],
                       "source": rs[0]["source"], "n": len(rs), "task_successes": k,
                       "one_shot_successes": sum(r["one_shot"] for r in rs), "task_pct": 100 * k / len(rs),
                       "one_shot_pct": 100 * sum(r["one_shot"] for r in rs) / len(rs),
                       "completion_time_mean_s": m, "completion_time_sd_s": sd,
                       "drop_episodes": sum(r["drops"] > 0 for r in rs)})
    wcsv("stratum_metrics.csv", sm)
    fb, sd_rows = [], []
    for gi, fg in enumerate(off["final_groups"]):
        rs = by_g[gi]
        c = Counter(r["failure_reason"] for r in rs)
        fb.append({"size_interval_cm": fg["label"], "n_trials": len(rs), **{k: c.get(k, 0) for k in FAILURE_REASONS}})
        fail = [r for r in rs if not r["task_success"]]
        sd_rows.append({"size_interval_cm": fg["label"], "n_failures": len(fail),
                        "approach_never_stage1": sum(not r["reached_stage1"] for r in fail),
                        "grasp_never_acquired": sum(r["reached_stage1"] and not r["acquired"] for r in fail),
                        "acquired_not_confirmed": sum(r["acquired"] and not r["grasp_confirmed"] for r in fail),
                        "post_grasp_drop": sum(r["acquired"] and r["drops"] > 0 and r["failure_reason"] != "success" for r in fail),
                        "held_but_not_delivered": sum(r["acquired"] and r["drops"] == 0 for r in fail),
                        "other_termination": sum(r["failure_reason"] in ("base_contact_failure", "grasp_error_out_of_bounds",
                                                                          "numerical_failure") for r in fail)})
    wcsv("failure_breakdown.csv", fb)
    wcsv("stage_decomposition.csv", sd_rows)

    # ---- OOD vs ID (paired cluster bootstrap over scenario ids)
    rng = np.random.default_rng(20261117)
    lab = {fg["type"] + ("" if fg["type"] != "in_distribution" else str(gi)): gi for gi, fg in enumerate(off["final_groups"])}
    small = by_g[[gi for gi, fg in enumerate(off["final_groups"]) if fg["type"] == "ood_small"][0]]
    large = by_g[[gi for gi, fg in enumerate(off["final_groups"]) if fg["type"] == "ood_large"][0]]
    id_g = [gi for gi, fg in enumerate(off["final_groups"]) if fg["type"] == "in_distribution"]
    idr = by_g[id_g[0]] + by_g[id_g[1]]
    del lab
    comps = {}
    for name, a, b in (("small_OOD_minus_ID[4,6]", small, idr), ("large_OOD_minus_ID[4,6]", large, idr),
                       ("small_OOD_minus_[4,5]", small, by_g[id_g[0]]), ("large_OOD_minus_[5,6]", large, by_g[id_g[1]]),
                       ("[5,6]_minus_[4,5]", by_g[id_g[1]], by_g[id_g[0]]), ("large_OOD_minus_small_OOD", large, small)):
        comps[name] = {"task_success": cluster_rate_diff(a, b, "task_success", rng),
                       "one_shot": cluster_rate_diff(a, b, "one_shot", rng),
                       "completion_time": cluster_time_diff(a, b, rng)}
    per_stratum = {}
    for st in "AB":
        mat = {}
        ids = sorted({r["scenario_id"] for r in by_g[0]})
        for key in ("task_success", "one_shot"):
            look = {(r["scenario_id"], r["group_index"]): r[key] for r in recs if r["stratum"] == st}
            M = np.array([[look[(i, gi)] for gi in range(len(off["final_groups"]))] for i in ids], dtype=int)
            q, p = cochran_q(M)
            mc = {}
            for (x, y) in ((0, 1), (2, 3), (0, 3)):
                n01, n10, pv = mcnemar_exact(M[:, x].astype(bool), M[:, y].astype(bool))
                mc[f"{off['final_groups'][x]['label']}_vs_{off['final_groups'][y]['label']}"] = {
                    "only_second_success": n01, "only_first_success": n10, "p_exact": pv}
            mat[key] = {"cochran_q": q, "df": len(off["final_groups"]) - 1, "p": p, "mcnemar": mc}
        per_stratum[st] = mat
    ood_vs_id = {"method": ("paired cluster bootstrap over the base scenario ids (each id contributes its episodes in "
                            "both strata of a group); 95 % percentile CIs, 10000 resamples; sign-flip permutation test "
                            "of per-scenario differences; failure_ratio = (1 - rate_a) / (1 - rate_b)"),
                 "comparisons": comps, "per_stratum_tests": per_stratum}
    with open(os.path.join(out, "ood_vs_id.json"), "w") as f:
        json.dump(ood_vs_id, f, indent=1)

    # ---- physics diagnostics
    phys_rows, phys = [], {}
    for gi, fg in enumerate(off["final_groups"]):
        rs = by_g[gi]
        acq = [r for r in rs if r["acquired"]]
        drop = [r for r in acq if r["drops"] > 0]
        keep = [r for r in acq if r["drops"] == 0]
        d = {}
        for var in ("side", "density", "mass", "acquisition_jaw_angle_deg", "s2_max_penetration_mm", "s2_cube_slip_mm",
                    "cube_mu_used", "object_r"):
            d[f"{var}: dropped vs kept (acquired)"] = _mw([r[var] for r in drop], [r[var] for r in keep])
            d[f"{var}: task failure vs success"] = _mw([r[var] for r in rs if not r["task_success"]],
                                                        [r[var] for r in rs if r["task_success"]])
        phys[fg["label"]] = d
        for var, nb in (("side", 4), ("density", 4), ("mass", 4)):
            vals = np.array([r[var] for r in rs])
            edges = np.quantile(vals, np.linspace(0, 1, nb + 1))
            for i in range(nb):
                sel = [r for r in rs if (edges[i] <= r[var] < edges[i + 1]) or (i == nb - 1 and r[var] == edges[-1])]
                if not sel:
                    continue
                k = sum(r["task_success"] for r in sel)
                a_ = [r for r in sel if r["acquired"]]
                phys_rows.append({"size_group": fg["label"], "variable": var, "bin": f"Q{i + 1}",
                                  "lo": float(edges[i]), "hi": float(edges[i + 1]), "n": len(sel),
                                  "task_pct": 100 * k / len(sel), "one_shot_pct": 100 * sum(r["one_shot"] for r in sel) / len(sel),
                                  "acquired_pct": 100 * len(a_) / len(sel),
                                  "drop_pct_of_acquired": 100 * sum(r["drops"] > 0 for r in a_) / len(a_) if a_ else float("nan")})
    wcsv("physics_diagnostics.csv", phys_rows)
    with open(os.path.join(out, "physics_diagnostics.json"), "w") as f:
        json.dump(phys, f, indent=1, default=float)

    # ---- retention detector re-validation on the new data
    def gapstats(gs):
        fl = [g for g in gs if not g[6]]
        rel = [g for g in gs if g[6]]
        return {"n_flicker": len(fl), "n_release": len(rel),
                "flicker_max_ms": max([g[3] for g in fl], default=0.0), "flicker_max_disp_mm": max([g[4] for g in fl], default=0.0),
                "flicker_max_jaw_deg": max([g[5] for g in fl], default=0.0),
                "release_min_ms": min([g[3] for g in rel], default=float("nan")),
                "gaps_between_32_and_56_ms": sum(1 for g in gs if 32.0 < g[3] < 56.0),
                "releases_decided_by_geometry_only": sum(1 for g in rel if g[3] < 1000 * ret_cfg["release_min_gap_s"])}
    det = {"base_run": gapstats(base_gaps), "ood_extension": gapstats(ood_gaps)}
    variants = [("official", {}), ("hold_force 0.3 N", {"hold_force_threshold": 0.3}),
                ("hold_force 30 N", {"hold_force_threshold": 30.0}), ("hold_force 100 N", {"hold_force_threshold": 100.0}),
                ("release gap 24 ms", {"release_min_gap_s": 0.024}), ("release gap 96 ms", {"release_min_gap_s": 0.096}),
                ("jaw closing 3 deg", {"release_jaw_closing_deg": 3.0}), ("jaw closing 8 deg", {"release_jaw_closing_deg": 8.0}),
                ("displacement side/4", {"release_rel_disp_fraction_of_side": 0.25}),
                ("displacement 1 side", {"release_rel_disp_fraction_of_side": 1.0}),
                ("acquisition jaw range 2 deg", {"acquisition_max_jaw_range_deg": 2.0}),
                ("acquisition rel. disp 5 mm", {"acquisition_max_rel_disp_m": 0.005})]
    sens = []
    for name, ov in variants:
        c = dict(ret_cfg, **ov)
        row = {"variant": name}
        for gi, fg in enumerate(off["final_groups"]):
            row[fg["label"]] = sum(RT.analyze_retention(r["_ph"], r["side"], r["task_success"], c)["one_shot_success"]
                                   for r in by_g[gi])
        sens.append(row)
    det["one_shot_sensitivity"] = sens
    det["hold_force_note"] = ("hold_force_threshold = 10 N was justified for <= 0.259 kg; the heaviest OOD cube (0.412 kg) "
                              "needs 5.0 N per jaw against gravity at jaw friction 0.4 (15.1 N with 2 g extra). Settled "
                              "grasps squeeze with ~400 N, so the classification is insensitive to this threshold "
                              "(see one_shot_sensitivity).")
    with open(os.path.join(out, "detector_validation.json"), "w") as f:
        json.dump(det, f, indent=1, default=float)

    # ---- provenance
    provenance = {"base_run": os.path.relpath(base_dir, REPO_ROOT), "base_run_sha256": base["sha256"],
                  "extension_run": os.path.relpath(ood_dir, REPO_ROOT),
                  "extension_sha256": {f: sha256(os.path.join(ood_dir, f)) for f in ("scenarios.json", "episodes.csv", "config.yaml")},
                  "episodes_from_base": len(base_recs), "episodes_from_extension": len(ood_recs), "total": len(recs),
                  "checks": ["base run SHA-256 (scenarios, episodes, config) == config.yaml official.base_run_sha256",
                             "base and extension complete (every scenario x interval once, trajectories present)",
                             "identical scenario bank, experiment config (except intervals), deploy config, policy/UAN/XML hashes",
                             "OOD strata disjoint from base strata (no episode simulated twice)",
                             "base One-Shot re-evaluation == stored official classification (every episode)",
                             "side/density reproducible from the shared quantiles; MuJoCo mass/inertia read-back",
                             "500 episodes per final group (2000 total); regression counts of the reused groups",
                             "one-shot <= task success per episode"],
                  "allow_partial": allow_partial}
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=1)

    # ---- LaTeX (manuscript format)
    def ct(s):
        if s["completion_time_n"] == 0:
            return "--"
        return f"{s['completion_time_mean_s']:.2f} $\\pm$ {s['completion_time_sd_s']:.2f}"
    tex = [
        "% E5 dynamic object retrieval (3-7 cm) -- generated by retrieval_experiment/combine.py",
        f"% sources: {provenance['base_run']} (4-6 cm, reused) + {provenance['extension_run']} (OOD strata)",
        "% each row: 2 half-cm strata x the same paired base scenarios; density U(400,1200) kg/m^3; training sizes 4-6 cm",
        "% SR: task success; One-Shot: task success with continuous physical retention after first acquisition;",
        "% Time: completion time mean +- SD over successful trials.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dynamic object retrieval performance across different object-size intervals.}",
        "\\label{tab:retrieval_size}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular*}{\\columnwidth}",
        "    {@{\\extracolsep{\\fill}}lccc@{}}",
        "\\toprule",
        "Size (cm) & SR (\\%) $\\uparrow$ & One-Shot (\\%) $\\uparrow$ & Time (s) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for s in summary:
        tex.append(f"{{{s['size_interval_cm']}}} & {s['task_success_rate_pct']:.1f} & {s['one_shot_success_rate_pct']:.1f} & {ct(s)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    with open(os.path.join(out, "summary_table.tex"), "w") as f:
        f.write("\n".join(tex))

    # ---- report
    def ci(lo, hi):
        return f"{lo:.1f}–{hi:.1f}"
    md = [f"# E5 dynamic object retrieval, 3–7 cm (combined)", "",
          f"- Sources: `{provenance['base_run']}` (4–6 cm, {len(base_recs)} episodes, reused unchanged) + "
          f"`{provenance['extension_run']}` (OOD strata, {len(ood_recs)} episodes). All source checks passed (provenance.json).",
          f"- Each 1-cm group = 2 half-cm strata (A, B) replaying the same {n_scen} base scenarios; inference clusters on scenario id.",
          "- Training sizes 4–6 cm: [3, 4] and [6, 7] are out-of-distribution. Density U(400, 1200) kg/m³ in every group "
          "(training: 800); mass = density × side³ changes with size, so size effects include mass effects.", "",
          "## Official table", "",
          "| Size (cm) | Type | SR (%) ↑ [95% CI] | One-Shot (%) ↑ [95% CI] | Time (s) ↓ | succ. / one-shot / n |",
          "|---|---|---:|---:|---:|---:|"]
    for s, x in zip(summary, gm):
        md.append(f"| {s['size_interval_cm']} | {s['evaluation']} | {s['task_success_rate_pct']:.1f} [{ci(x['task_ci95_lo'], x['task_ci95_hi'])}] "
                  f"| {s['one_shot_success_rate_pct']:.1f} [{ci(x['one_shot_ci95_lo'], x['one_shot_ci95_hi'])}] "
                  f"| {s['completion_time_mean_s']:.2f} ± {s['completion_time_sd_s']:.2f} (n={s['completion_time_n']}) "
                  f"| {s['task_successes']} / {s['one_shot_successes']} / {s['n_trials']} |")
    md += ["", "## Grasp and retention", "",
           "| Size (cm) | mass (kg) | grasp confirmed (%) | physically acquired (%) | jaw angle at acquisition (deg) | drop (% of acquired) | drop events | re-grasp events | success after drop+re-grasp | training grasp-loss term. | strict one-shot |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in gm:
        md.append(f"| {x['size_interval_cm']} | {x['mass_kg_min']:.4f}–{x['mass_kg_max']:.4f} | {x['grasp_confirmation_pct']:.1f} | "
                  f"{x['physical_acquisition_pct']:.1f} | {x['acquisition_jaw_angle_mean_deg']:.1f} | {x['drop_pct_of_acquired']:.1f} | "
                  f"{x['drop_events']} | {x['regrasp_events']} | {x['successful_after_drop_and_regrasp']} | "
                  f"{x['training_grasp_loss_terminations']} | {x['one_shot_strict_successes']} |")
    md += ["", "Half-cm strata (diagnostic):", "", "| group | stratum | interval | source | task % | one-shot % | time (s) | drop episodes |",
           "|---|---|---|---|---:|---:|---:|---:|"]
    for x in sm:
        md.append(f"| {x['size_group']} | {x['stratum']} | {x['stratum_interval']} | {x['source']} | {x['task_pct']:.1f} | "
                  f"{x['one_shot_pct']:.1f} | {x['completion_time_mean_s']:.2f} ± {x['completion_time_sd_s']:.2f} | {x['drop_episodes']} |")
    md += ["", "## Failure breakdown", "", "| Size (cm) | " + " | ".join(FAILURE_REASONS) + " |", "|---|" + "---:|" * len(FAILURE_REASONS)]
    for x in fb:
        md.append(f"| {x['size_interval_cm']} | " + " | ".join(str(x[k]) for k in FAILURE_REASONS) + " |")
    md += ["", "| Size (cm) | failures | never Stage 1 | never acquired | acquired, not confirmed | dropped after acquisition | held, not delivered | other |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in sd_rows:
        md.append(f"| {x['size_interval_cm']} | {x['n_failures']} | {x['approach_never_stage1']} | {x['grasp_never_acquired']} | "
                  f"{x['acquired_not_confirmed']} | {x['post_grasp_drop']} | {x['held_but_not_delivered']} | {x['other_termination']} |")
    md += ["", "## OOD vs in-distribution (paired cluster bootstrap over scenario ids)", "",
           "| comparison | Δ task SR (pp) [95% CI] | perm. p | failure ratio [95% CI] | Δ one-shot (pp) [95% CI] | perm. p | Δ time (s) [95% CI] |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in comps.items():
        t, o, tm = c["task_success"], c["one_shot"], c["completion_time"]
        md.append(f"| {name} | {t['diff_pp']:+.1f} [{t['ci95_pp'][0]:+.1f}, {t['ci95_pp'][1]:+.1f}] | {t['sign_flip_permutation_p']:.3f} | "
                  f"{t['failure_ratio']:.2f} [{t['failure_ratio_ci95'][0]:.2f}, {t['failure_ratio_ci95'][1]:.2f}] | "
                  f"{o['diff_pp']:+.1f} [{o['ci95_pp'][0]:+.1f}, {o['ci95_pp'][1]:+.1f}] | {o['sign_flip_permutation_p']:.3f} | "
                  f"{tm['diff_s']:+.2f} [{tm['ci95_s'][0]:+.2f}, {tm['ci95_s'][1]:+.2f}] |")
    md += ["", "Per-stratum Cochran's Q over the four groups (250 paired scenarios each):", ""]
    for st, mat in per_stratum.items():
        md.append(f"- stratum {st}: task Q = {mat['task_success']['cochran_q']:.2f} (p = {mat['task_success']['p']:.3g}); "
                  f"one-shot Q = {mat['one_shot']['cochran_q']:.2f} (p = {mat['one_shot']['p']:.3g})")
    md += ["", "## Physics diagnostics (quartile bins within each group)", "",
           "| group | variable | bin | range | n | task % | acquired % | drop % of acquired |", "|---|---|---|---|---:|---:|---:|---:|"]
    for x in phys_rows:
        fmt = "{:.4f}" if x["variable"] in ("side", "mass") else "{:.0f}"
        md.append(f"| {x['size_group']} | {x['variable']} | {x['bin']} | {fmt.format(x['lo'])}–{fmt.format(x['hi'])} | {x['n']} | "
                  f"{x['task_pct']:.1f} | {x['acquired_pct']:.1f} | {x['drop_pct_of_acquired']:.1f} |")
    md += ["", "Mann–Whitney (dropped vs kept among acquired; failure vs success) per group: physics_diagnostics.json.", "",
           "## Retention-detector re-validation", "", "```json", json.dumps({k: v for k, v in det.items() if k != "one_shot_sensitivity"}, indent=1, default=float), "```", "",
           "| One-Shot variant | " + " | ".join(fg["label"] for fg in off["final_groups"]) + " |", "|---|" + "---:|" * len(off["final_groups"])]
    for x in sens:
        md.append(f"| {x['variant']} | " + " | ".join(str(x[fg['label']]) for fg in off["final_groups"]) + " |")
    with open(os.path.join(out, "COMBINED.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:14]))
    print(f"[E5 combine] {len(recs)} episodes ({len(base_recs)} reused + {len(ood_recs)} new) -> {out}")
    return out


if __name__ == "__main__":
    main()
