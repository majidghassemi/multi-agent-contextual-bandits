#!/usr/bin/env python3
# =====================================================================
#  make_figures_v3.py -- AISTATS-quality figures for the CEC / D-CESA
#  reconciliation experiments (reproduce_v34.py).
#
#  Reads   out_v34/results/*.json  (+ out_v34/checkpoints/random/ for
#          the per-instance histogram) and writes vector PDF + 400-dpi
#          PNG for each experiment under out_v34/figures/.
#
#  Design goals
#    * Column-width figures (~3.3in) whose text stays legible even when
#      the figure is shrunk further -- large fonts relative to axes,
#      thick 2.2pt lines, 8pt markers with white halos.
#    * Times-family text (Nimbus Roman) + STIX math == AISTATS look.
#    * Colour-blind-safe Wong palette, colour bound to *entity* and held
#      fixed across every panel (oracle is always blue, naive-mean always
#      vermillion, D-CESA always green, ...).
#
#  Usage
#    python3 make_figures_v3.py                 # reads ./out_v34
#    python3 make_figures_v3.py --outdir runs/A
# =====================================================================
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import LogLocator, NullFormatter

# --------------------------------------------------------------- palette
# A custom, colour-blind-safe qualitative palette (validated with the
# dataviz six-check validator: adjacent-pair CVD deltaE >= 12, chroma and
# lightness bands pass).  Deliberately avoids the default matplotlib/
# seaborn blue+orange.  One fixed colour per *entity*, reused in every
# figure that shows it, so identity never depends on position.
TEAL = "#109485"     # D-CESA == the majority-vote gate
CORAL = "#D1495B"    # naive mean / tracking modulus / label corruption
GOLD = "#C88A1E"     # trimmed mean / OGD-logistic gate
PLUM = "#8E5AA8"     # coordinate median / perceptron gate / 90th pct
GRAPHITE = "#4A5560"  # oracle -- the neutral best-case reference
# ordered pair for the (ordered) bias level c_beta: light -> dark == more bias
TEAL_LO, TEAL_HI = "#5FB3A8", "#0C6E62"
TEAL_PALE = "#A9D2CB"  # histogram fill
NEUTRAL_LO = "#B4BCC4"  # "clean" level in the corruption comparison

INK = "#1a1a1a"          # primary text
MUTE = "#7a7a7a"         # secondary text / reference lines
GRID = "#dcdcdc"

# One fixed colour per named entity, reused in every figure that shows it.
CMETHOD = dict(oracle=GRAPHITE, mean=CORAL, trim=GOLD, median=PLUM,
               dcesa_mv=TEAL)
CGATE = dict(mv=TEAL, perceptron=PLUM, ogd=GOLD)
CROLE = dict(contraction=TEAL, tracking=CORAL)

LABELS = dict(oracle="Oracle", mean="Naive mean", trim="Trimmed mean",
              median="Coord. median", dcesa_mv="D-CESA")
GATE_LABELS = dict(mv="Majority vote", perceptron="Perceptron",
                   ogd="OGD-logistic")


# ------------------------------------------------------------- rc / style
def set_style():
    serif = [f.name for f in fm.fontManager.ttflist]
    pick = next((n for n in ("Nimbus Roman", "Times New Roman",
                             "TeX Gyre Termes", "DejaVu Serif")
                 if n in serif), "DejaVu Serif")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [pick, "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        # large-relative-to-axes type: legible when the figure is shrunk
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.0,
        # thick, confident marks
        "lines.linewidth": 2.2,
        "lines.markersize": 7.5,
        "lines.markeredgewidth": 1.1,
        "axes.linewidth": 1.0,
        "patch.linewidth": 0.8,
        # recessive frame + grid
        "axes.edgecolor": "#3a3a3a",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#3a3a3a",
        "ytick.color": "#3a3a3a",
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#c9c9c9",
        "legend.borderpad": 0.5,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.1,
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,           # embed TrueType so text stays selectable
        "ps.fonttype": 42,
    })


def finish(ax):
    """De-clutter: drop top/right spines, push grid behind data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=4)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _marker_kw(color):
    return dict(marker="o", color=color, markerfacecolor=color,
                markeredgecolor="white")


# ==================================================================== load
def load(resdir, name):
    p = os.path.join(resdir, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ================================================================ figures
def fig_certify(S, outdir):
    rows = S["rows"]
    cb = np.array([r["c_beta"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    kap = np.array([r["kappa_tilde"] for r in rows])
    # last c_beta where BOTH certificates hold -> certified region boundary
    ok = [r["contraction_ok"] and r["tracking_ok"] for r in rows]
    xstar = cb[np.array(ok)].max() if any(ok) else cb.min()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axvspan(cb.min() - 0.02, xstar, color=TEAL, alpha=0.10, zorder=0)
    ax.text(xstar, kap.max(), "certified\nregion", va="top", ha="left",
            fontsize=8.5, color="#1f6f63")
    ax.axhline(1.0, ls=(0, (4, 3)), lw=1.4, color=MUTE, zorder=1)
    ax.text(cb.max(), 1.0, " threshold", va="bottom", ha="right",
            fontsize=8.5, color=MUTE)

    ax.plot(cb, rho, **_marker_kw(CROLE["contraction"]),
            label=r"contraction $\varrho$", zorder=4)
    ax.plot(cb, kap, marker="s", color=CROLE["tracking"],
            markerfacecolor=CROLE["tracking"], markeredgecolor="white",
            label=r"tracking $\tilde{\kappa}$", zorder=4)

    ax.set_yscale("log")
    ax.set_xlabel(r"bias magnitude $c_\beta$")
    ax.set_ylabel("certificate modulus")
    ax.legend(loc="lower right")
    finish(ax)
    save(fig, outdir, "fig_certify")


def fig_epoch(S, outdir):
    per = {int(n): v for n, v in S["per_N"].items()}
    Ns = np.array(sorted(per))
    med = np.array([per[n]["median"] for n in Ns])
    slope = S.get("slope")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    # individual seed runs as faint scatter -> shows the spread behind median
    for n in Ns:
        ts = [t for t in per[n]["T_star"] if t]
        ax.scatter([n] * len(ts), ts, s=16, color=TEAL, alpha=0.30,
                   edgecolor="none", zorder=2)
    # ideal 1/N reference anchored at the first median
    ref = med[0] * Ns[0] / Ns
    ax.plot(Ns, ref, ls=(0, (4, 3)), lw=1.5, color=MUTE, zorder=3,
            label=r"ideal $T^\star\!\propto\!N^{-1}$")
    ax.plot(Ns, med, **_marker_kw(TEAL), zorder=4, label="median $T^\\star$")
    if slope is not None:
        ax.text(0.04, 0.06, f"slope ${slope:+.2f}$", transform=ax.transAxes,
                fontsize=9.5, color=INK, va="bottom", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"number of agents $N$")
    ax.set_ylabel(r"collapse time $T^\star$")
    ax.legend(loc="upper right")
    finish(ax)
    save(fig, outdir, "fig_epoch")


def fig_certified(S, outdir):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    keys = sorted(S, key=float)
    # c_beta is ordered -> sequential teal, darker == more bias
    cols = ([TEAL_LO, TEAL_HI] if len(keys) == 2
            else [TEAL_HI] * len(keys))
    for i, k in enumerate(keys):
        v = S[k]
        c = cols[i % len(cols)]
        t = np.array(v["traj_t"])
        d = np.array(v["traj_d"])
        ax.plot(t, d, color=c, zorder=4, label=rf"$c_\beta={k}$")
        ax.axhline(v["picard"], ls=(0, (4, 3)), lw=1.4, color=c, alpha=0.85,
                   zorder=2)
    # label the dashed asymptotes as the predicted biased fixed points
    ylo = min(S[k]["picard"] for k in keys)
    ax.annotate(r"biased fixed point $\theta_\infty$",
                xy=(t[-1], ylo), xytext=(t[-1] * 0.30, ylo * 0.55),
                fontsize=8.8, color=MUTE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel(r"round $t$")
    ax.set_ylabel(r"$\|\hat{\theta}_t-\theta^\star\|$")
    ax.set_ylim(0, max(S[k]["picard"] for k in keys) * 1.18)
    ax.legend(loc="lower right")
    finish(ax)
    save(fig, outdir, "fig_certified")


def fig_baselines(S, outdir):
    adv = S["adversary"]
    floor = S["floor"]
    order = ["oracle", "mean", "trim", "median", "dcesa_mv"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    # (a) adversary -- log-y so the median blow-up doesn't crush the rest
    ks = [k for k in order if k in adv]
    vals = [adv[k]["reg"] for k in ks]
    err = [adv[k]["sem"] for k in ks]
    cols = [CMETHOD[k] for k in ks]
    x = np.arange(len(ks))
    axes[0].bar(x, vals, yerr=err, color=cols, capsize=3, width=0.68,
                edgecolor="white", error_kw=dict(lw=1.2, ecolor="#444"))
    for xi, v in zip(x, vals):
        axes[0].text(xi, v * 1.06, f"{v:.0f}", ha="center", va="bottom",
                     fontsize=8.5, color=INK)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[k] for k in ks], rotation=28, ha="right")
    axes[0].set_ylabel("latent regret")
    axes[0].set_title("(a)", fontsize=12, loc="left")
    finish(axes[0])

    # (b) tolerance floor -- clean but heterogeneous, near-equal bars
    ks2 = [k for k in order if k in floor]
    vals2 = [floor[k] for k in ks2]
    cols2 = [CMETHOD[k] for k in ks2]
    x2 = np.arange(len(ks2))
    axes[1].bar(x2, vals2, color=cols2, width=0.68, edgecolor="white")
    for xi, v in zip(x2, vals2):
        axes[1].text(xi, v + max(vals2) * 0.012, f"{v:.0f}", ha="center",
                     va="bottom", fontsize=8.5, color=INK)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([LABELS[k] for k in ks2], rotation=28, ha="right")
    axes[1].set_ylabel("latent regret")
    axes[1].set_title("(b)", fontsize=12, loc="left")
    finish(axes[1])

    fig.tight_layout(w_pad=1.6)
    save(fig, outdir, "fig_baselines")


def fig_gate_exponent(S, outdir):
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    # q=0 curves coincide across gates -> plot one shared "no corruption" line
    q0 = S.get("mv_q0.0")
    if q0:
        p, e = zip(*q0["points"])
        ax.plot(p, e, ls=(0, (4, 3)), color=MUTE, marker="o",
                markerfacecolor="white", markeredgecolor=MUTE, zorder=3,
                label="no corruption ($q{=}0$)")
    for g in ("mv", "perceptron", "ogd"):
        v = S.get(f"{g}_q0.2")
        if not v:
            continue
        p, e = zip(*v["points"])
        ax.plot(p, e, **_marker_kw(CGATE[g]), zorder=4,
                label=f"{GATE_LABELS[g]} ($q{{=}}0.2$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"audit budget $\bar{p}$")
    ax.set_ylabel("clean-agent excess regret")
    ax.legend(loc="upper right", fontsize=8.8)
    finish(ax)
    save(fig, outdir, "fig_gate_exponent")


def fig_random(S, outdir, ckptdir):
    # reconstruct the per-instance distribution from checkpoints
    mu = []
    for f in glob.glob(os.path.join(ckptdir, "random", "draw*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("ok"):
            mu.append(d["mu_dis"])
    mu = np.array(mu)
    pk = S.get("pickled_value", 0.43)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if len(mu):
        ax.hist(mu, bins=28, color=TEAL_PALE, edgecolor="white", lw=0.6,
                alpha=0.98, zorder=2)
        med = float(np.median(mu))
        q90 = float(np.quantile(mu, 0.9))
        ax.axvline(med, color=TEAL, lw=2.0, zorder=4,
                   label=f"median {med:.3f}")
        ax.axvline(q90, color=PLUM, lw=2.0, ls=(0, (1, 1.2)),
                   zorder=4, label=f"90th pct {q90:.3f}")
    ax.axvline(pk, color=CORAL, lw=2.0, ls=(0, (4, 3)),
               zorder=4, label=f"design target {pk:.2f}")

    ax.set_xlim(0, pk * 1.08)
    ax.set_xlabel(r"decision-flip mass $\mu(\tilde{a}\neq a^\star)$")
    ax.set_ylabel("random instances")
    ax.legend(loc="upper right", fontsize=9)
    finish(ax)
    save(fig, outdir, "fig_random")


def fig_waiting(S, outdir):
    per = S["per_pbar"]
    ps = np.array(sorted(float(k) for k in per))
    med = np.array([per[f"{p:g}"]["median"] if f"{p:g}" in per
                    else per[str(p)]["median"] for p in ps])
    pred = np.array([per[f"{p:g}"]["pred"] if f"{p:g}" in per
                     else per[str(p)]["pred"] for p in ps])
    slope = S.get("slope")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(ps, pred, ls=(0, (4, 3)), lw=1.6, color=MUTE, zorder=3,
            label=r"theory $\log|E|/\bar{p}$")
    ax.plot(ps, med, **_marker_kw(TEAL), zorder=4, label="observed median")
    if slope is not None:
        ax.text(0.04, 0.06, f"slope ${slope:+.2f}$", transform=ax.transAxes,
                fontsize=9.5, color=INK, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"audit budget $\bar{p}$")
    ax.set_ylabel(r"waiting time $t^\star$")
    ax.legend(loc="upper right")
    finish(ax)
    save(fig, outdir, "fig_waiting")


def fig_mistakes(S, outdir):
    gates = ["mv", "perceptron", "ogd"]
    qs = [("0.0", NEUTRAL_LO, "$q=0$"),
          ("0.2", CORAL, "$q=0.2$")]
    x = np.arange(len(gates))
    w = 0.36

    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    for j, (q, col, lab) in enumerate(qs):
        vals, errs, xs = [], [], []
        for i, g in enumerate(gates):
            rec = S.get(f"{g}_q{q}")
            if not rec:
                continue
            vals.append(rec["mistaken_rounds"])
            errs.append(rec.get("sem", 0.0))
            xs.append(x[i] + (j - 0.5) * w)
        ax.bar(xs, vals, w, yerr=errs, color=col, edgecolor="white",
               capsize=3, error_kw=dict(lw=1.1, ecolor="#444"), label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, v * 1.06, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8.0, color=INK)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([GATE_LABELS[g] for g in gates], rotation=18, ha="right")
    ax.set_ylabel("mistaken trust rounds")
    ax.legend(loc="upper left", ncol=2)
    finish(ax)
    save(fig, outdir, "fig_mistakes")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_v34")
    a = ap.parse_args()
    resdir = os.path.join(a.outdir, "results")
    figdir = os.path.join(a.outdir, "figures")
    ckptdir = os.path.join(a.outdir, "checkpoints")
    set_style()
    print(f"reading {resdir} -> writing {figdir}")

    builders = [
        ("certify", fig_certify),
        ("epoch", fig_epoch),
        ("certified", fig_certified),
        ("baselines", fig_baselines),
        ("gate_exponent", fig_gate_exponent),
        ("waiting", fig_waiting),
        ("mistakes", fig_mistakes),
    ]
    for name, fn in builders:
        S = load(resdir, name)
        if S is None:
            print(f"  (skip {name}: no results/{name}.json)")
            continue
        fn(S, figdir)
    rnd = load(resdir, "random")
    if rnd is not None:
        fig_random(rnd, figdir, ckptdir)
    print("done.")


if __name__ == "__main__":
    main()
