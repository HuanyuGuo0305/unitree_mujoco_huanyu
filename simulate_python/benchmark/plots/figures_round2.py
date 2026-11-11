"""Round-2 figures: E1/E2 illustrative cases (E3 figures removed with E3 v1).

Design decisions, so they are not re-litigated per figure:

* **Family pooling uses a cluster bootstrap over seeds.** Where a panel shows
  one curve per baseline rather than one per trajectory family, the three
  families of a given seed are first averaged into ONE number for that seed,
  and the bootstrap then resamples seeds. The three families are three
  different tasks run on the same seeded initial state, so resampling episodes
  would treat them as independent replicates when they are not. Per-family
  figures are emitted alongside for the supplement.
* **Paired panels difference per (seed, family) first**, then average over
  families, then bootstrap seeds -- pairing is never broken by aggregating
  first.
* **Signed quantities get the diverging scale and an explicit zero line.** The
  zero point is drawn, labelled and never clipped; negative values are shown as
  negative.
* Axis limits are the data's own range padded symmetrically about zero for
  signed quantities. No axis is truncated to magnify a small difference.
"""

from __future__ import annotations

import numpy as np

from benchmark.plots import style
from benchmark.plots.style import save

# ==========================================================================
# Selected illustrative cases
#
# Every one of these figures is built from a seed chosen by
# benchmark/select_cases.py out of a candidate pool DISJOINT from the official
# statistical pool. The footer of each figure says so, because a reader who
# sees only the figure must not mistake it for average performance.
# ==========================================================================
import csv as _csv  # noqa: E402
import os as _os  # noqa: E402

ILLUSTRATIVE_FOOTER = (
    "Selected illustrative trials from a separately evaluated candidate seed "
    "pool (seeds 100-149), chosen by a pre-declared ranking; see the selection "
    "manifest. NOT benchmark average performance -- the official statistics use "
    "every seed 0-59."
)
COMMAND_INK = style.TEXT_PRIMARY


def _row_header(axes_row, title, metric_summary):
    """One centred two-line header per row, over the middle panel.

    Per-axes left- and right-aligned titles collide once the method summary is
    long enough to be useful, so the header spans the row instead.
    """
    # pad lifts the header clear of the neighbouring panels' y-axis labels,
    # which otherwise sit in the same horizontal band.
    axes_row[len(axes_row) // 2].set_title(
        title + ("\n" + metric_summary if metric_summary else ""),
        loc="center", color=style.TEXT_SECONDARY, fontsize=6.6, linespacing=1.5,
        pad=14,
    )


def _figure_legend(fig, axes_row, footer, n_rows):
    """A single figure-level legend under the panels, plus the footer note.

    Inside a panel the legend lands on top of the traces in at least one row;
    below the figure it never does, and one legend for the whole figure is less
    ink than one per row.
    """
    handles, labels = [], []
    for ax in axes_row:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.045 - 0.004 * n_rows),
                   ncol=min(len(handles), 5), fontsize=7, frameon=False)
    fig.text(0.0, -0.025 - 0.012 * n_rows, footer, fontsize=6.5,
             color=style.TEXT_MUTED, wrap=True)


def _load_table(path, filename):
    full = path if _os.path.isabs(path) else _os.path.join(SIM_ROOT_FOR_PATHS, path)
    target = _os.path.join(full, filename)
    if not _os.path.isfile(target):
        return {}
    columns: dict = {}
    with open(target, newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        names = reader.fieldnames or []
        for name in names:
            columns[name] = []
        for row in reader:
            for name in names:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in columns.items()}


SIM_ROOT_FOR_PATHS = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..")
)


def _evaluated(table):
    if not table:
        return np.zeros(0, dtype=bool)
    return table["evaluated"] > 0.5


def _leg_arm_power(path, tolerance=1e-6):
    """Exact 250 Hz leg/arm |tau * qdot| reconstructed from ``physics.npz``.

    ``physics.npz`` stores one row per EVALUATED physics step (``record_physics``
    returns early while ``sim_time < settle_s``), laid out as

        [t] qpos_leg qpos_arm | qvel_leg qvel_arm | qacc_leg qacc_arm | tau_leg tau_arm

    so each of the three vector blocks is ``n_leg + n_arm`` wide. The time mean
    of the reconstruction must therefore equal ``metrics.json``'s ``leg_power``
    and ``arm_power`` exactly; this function CHECKS that and raises if it does
    not, because a silent off-by-one-block slice reads accelerations as
    velocities and inflates the curve by orders of magnitude without looking
    obviously wrong.
    """
    full = path if _os.path.isabs(path) else _os.path.join(SIM_ROOT_FOR_PATHS, path)
    target = _os.path.join(full, "physics.npz")
    if not _os.path.isfile(target):
        return None
    data = np.load(target)["data"].astype(np.float64)
    n_arm = 6
    n_total = (data.shape[1] - 1) // 4
    n_leg = n_total - n_arm
    if n_leg <= 0 or 1 + 4 * n_total != data.shape[1]:
        raise ValueError(f"unexpected physics.npz layout: {data.shape} columns")
    qvel = data[:, 1 + n_total: 1 + 2 * n_total]
    tau = data[:, 1 + 3 * n_total: 1 + 4 * n_total]
    leg_power = np.sum(np.abs(tau[:, :n_leg] * qvel[:, :n_leg]), axis=1)
    arm_power = np.sum(np.abs(tau[:, n_leg:] * qvel[:, n_leg:]), axis=1)

    metrics_path = _os.path.join(full, "metrics.json")
    if _os.path.isfile(metrics_path):
        import json as _json

        with open(metrics_path, encoding="utf-8") as handle:
            metrics = _json.load(handle)
        for name, series in (("leg_power", leg_power), ("arm_power", arm_power)):
            expected = float(metrics[name])
            got = float(np.mean(series))
            # float32 storage, so compare at a relative tolerance rather than
            # exactly; a wrong block is off by orders of magnitude, not ppm.
            if abs(got - expected) > max(tolerance, 1e-3 * abs(expected)):
                raise ValueError(
                    f"reconstructed {name} = {got:.6g} W does not match "
                    f"metrics.json {expected:.6g} W for {path}; the "
                    f"physics.npz column layout has changed"
                )
    return {
        "t": data[:, 0],
        "leg_power": leg_power,
        "arm_power": arm_power,
        "n_leg": n_leg, "n_arm": n_arm,
    }


def _smooth(x, window):
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def e1_selected_cases(cases, out_dir, name="E1_selected_cases"):
    """One row per selected trial: base vx, yaw rate, EE position error."""
    import matplotlib.pyplot as plt

    style.apply_style()
    cases = [c for c in cases if c.get("selected")]
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 3, figsize=(style.PAGE_WIDTH, 1.75 * len(cases)),
                             squeeze=False)
    for r, case in enumerate(cases):
        sel = case["selected"]
        baselines = list(sel["metrics"].keys())
        tables = {b: (_load_table(sel["paths"][b], "commands.csv"),
                      _load_table(sel["paths"][b], "states.csv")) for b in baselines}
        ref_cmd, _ = tables[baselines[0]]
        mask_c = _evaluated(ref_cmd)
        t0 = ref_cmd["t"][mask_c][0]

        for col, (cmd_key, act_key, ylabel) in enumerate([
            ("vx_cmd", "vx_act", r"Forward velocity $v_x$ (m/s)"),
            ("wz_cmd", "wz_act", r"Yaw rate $\omega_z$ (rad/s)"),
            (None, "ee_pos_err", "EE position error (m)"),
        ]):
            ax = axes[r][col]
            if cmd_key is not None:
                ax.step(ref_cmd["t"][mask_c] - t0, ref_cmd[cmd_key][mask_c],
                        where="post", color=COMMAND_INK, linewidth=1.0,
                        linestyle=(0, (4, 2)), label="command", zorder=5)
            for b in baselines:
                _, states = tables[b]
                mask = _evaluated(states)
                kwargs = style.series_kwargs(b)
                kwargs["marker"] = None
                kwargs["label"] = style.SHORT[b]
                kwargs["linewidth"] = 1.0
                ax.plot(states["t"][mask] - t0, states[act_key][mask], **kwargs)
            if col == 2:
                # Mark where the EE target switched: the transients matter and
                # are inside the evaluated window on purpose.
                ee_cols = [f"ee_cmd_kp{i}" for i in range(9)]
                ee = np.column_stack([ref_cmd[c][mask_c] for c in ee_cols])
                switches = np.nonzero(np.any(np.abs(np.diff(ee, axis=0)) > 1e-9, axis=1))[0]
                for s in switches:
                    ax.axvline(ref_cmd["t"][mask_c][s + 1] - t0, color=style.GRID,
                               linewidth=0.7, zorder=0)
            ax.set_ylabel(ylabel, fontsize=7)
            if r == len(cases) - 1:
                ax.set_xlabel("Time in evaluated window (s)")
        _row_header(
            axes[r],
            f"({chr(97 + r)}) {case['title']} -- seed {sel['seed']}, {sel['condition']}",
            "  ".join(
                f"{b}: $E_v$ {sel['metrics'][b]['base_vxy_rmse']:.3f},  "
                f"$E_p$ {sel['metrics'][b]['ee_pos_rmse']:.3f}"
                for b in baselines),
        )
    _figure_legend(fig, axes[0], ILLUSTRATIVE_FOOTER
                   + " $E_v$ in m/s, $E_p$ in m; every metric for every method "
                     "is in the companion table.", len(cases))
    fig.subplots_adjust(hspace=0.85, wspace=0.34)
    return save(fig, out_dir, name)


def e2_selected_cases(cases, out_dir, name="E2_selected_cases"):
    """One row per selected trial: base vx tracking, leg power, leg acceleration."""
    import matplotlib.pyplot as plt

    style.apply_style()
    cases = [c for c in cases if c.get("selected")]
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 3, figsize=(style.PAGE_WIDTH, 1.75 * len(cases)),
                             squeeze=False)
    missing_physics = False
    for r, case in enumerate(cases):
        sel = case["selected"]
        baselines = list(sel["metrics"].keys())
        ref_cmd = _load_table(sel["paths"][baselines[0]], "commands.csv")
        mask_c = _evaluated(ref_cmd)
        t0 = ref_cmd["t"][mask_c][0]

        ax = axes[r][0]
        ax.step(ref_cmd["t"][mask_c] - t0, ref_cmd["vx_cmd"][mask_c], where="post",
                color=COMMAND_INK, linewidth=1.0, linestyle=(0, (4, 2)),
                label="command", zorder=5)
        for b in baselines:
            states = _load_table(sel["paths"][b], "states.csv")
            mask = _evaluated(states)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(states["t"][mask] - t0, states["vx_act"][mask], **kwargs)
        ax.set_ylabel(r"Forward velocity $v_x$ (m/s)", fontsize=7)

        ax = axes[r][1]
        any_power = False
        for b in baselines:
            power = _leg_arm_power(sel["paths"][b])
            if power is None:
                continue
            any_power = True
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            # 0.2 s moving average: the raw 250 Hz proxy is spiky at footfall
            # and the comparison of interest is the sustained level.
            ax.plot(power["t"] - t0, _smooth(power["leg_power"], 50), **kwargs)
        if any_power:
            ax.set_ylabel("Leg power proxy (W)", fontsize=7)
            ax.set_xlim(0, None)
        else:
            missing_physics = True
            ax.text(0.5, 0.5, "physics.npz not logged\nfor this episode",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.5, color=style.TEXT_MUTED)
            ax.set_axis_off()

        ax = axes[r][2]
        for b in baselines:
            act = _load_table(sel["paths"][b], "actuation.csv")
            if not act:
                continue
            mask = _evaluated(act)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(act["t"][mask] - t0, _smooth(act["leg_acc_abs_mean"][mask], 10), **kwargs)
        ax.set_ylabel(r"Leg joint accel. (rad/s$^2$)", fontsize=7)

        for col in range(3):
            if r == len(cases) - 1:
                axes[r][col].set_xlabel("Time in evaluated window (s)")
        _row_header(
            axes[r],
            f"({chr(97 + r)}) {case['title']} -- seed {sel['seed']}, {sel['condition']}",
            "  ".join(
                f"{b}: $P_{{tot}}$ {sel['metrics'][b]['total_power']:.1f} W,  "
                f"$a_{{leg}}$ {sel['metrics'][b]['leg_acc_rms']:.1f},  "
                f"$E_v$ {sel['metrics'][b]['base_vxy_rmse']:.3f}"
                for b in baselines),
        )
    footer = ILLUSTRATIVE_FOOTER
    if missing_physics:
        footer += (" Leg power panels need the 250 Hz physics log; "
                   "re-run the selected episode with --log-physics.")
    footer += (" Centre panel is the leg mechanical power proxy "
               "sum|tau*qdot| at 250 Hz, 0.2 s moving average -- not battery power.")
    _figure_legend(fig, axes[0], footer, len(cases))
    fig.subplots_adjust(hspace=0.85, wspace=0.38)
    return save(fig, out_dir, name)


