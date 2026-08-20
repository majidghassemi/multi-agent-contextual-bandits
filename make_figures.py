#!/usr/bin/env python3
# =====================================================================
#  make_figures.py -- AISTATS-quality figures for the CEC / D-CESA
#  experiments (reproduce_v35.py).  Single source of truth for every
#  figure the paper includes.
#
#  Reads   <outdir>/results/*.json and writes vector PDF + 400-dpi PNG
#          for each experiment under <outdir>/figures/.
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
#    python3 make_figures.py                    # reads ./out_v4
#    python3 make_figures.py --outdir runs/A
# =====================================================================
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import NullFormatter

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




# The band the paper certifies.  NOT the same as the smallest c_beta on the
# grid: c_beta=0.05 is the level the appendix calls unresolved, because the
# kappa~ estimate there straddles 1 (F1).
CERTIFIED_BAND = (0.15, 0.18)


def _rho_loc(r):
    return r.get("rho_loc", r.get("rho"))


def _reg_sem(v):
    """floor/adversary records: dict(reg,sem) now, bare float in old runs."""
    if isinstance(v, dict):
        return float(v["reg"]), float(v.get("sem") or 0.0)
    return float(v), 0.0


# ================================================================ figures
def fig_certify(S, outdir):
    """F1 rebuild.  Three faults in the old version, all fixed here:
       * it shaded the certified region at c_beta~0.05, which is the level
         the appendix calls UNRESOLVED -- shade the paper's band [0.15,0.18];
       * it labelled varrho_loc as "varrho", but the epoch theorem consumes
         the global modulus varrho_{B_R} -- plot both, distinctly labelled;
       * it drew kappa~ as a point estimate, when the paper's claim is that
         kappa~ STRADDLES 1 -- draw the per-batch range.
    """
    rows = sorted(S["rows"], key=lambda r: r["c_beta"])
    cb = np.array([r["c_beta"] for r in rows])
    rl = np.array([_rho_loc(r) for r in rows], dtype=float)
    rg = np.array([r.get("rho_global", np.nan) for r in rows], dtype=float)
    kap = np.array([r["kappa_tilde"] for r in rows], dtype=float)
    kmin = np.array([r.get("kappa_tilde_min", np.nan) for r in rows])
    kmax = np.array([r.get("kappa_tilde_max", np.nan) for r in rows])

    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    lo, hi = CERTIFIED_BAND
    ax.axvspan(lo, hi, color=TEAL, alpha=0.13, zorder=0, lw=0)
    # x in data coords, y in axes coords, so the label pins to the top
    from matplotlib.transforms import blended_transform_factory
    _tf = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(hi, 0.985, " certified band", transform=_tf, va="top", ha="left",
            fontsize=8.5, color="#1f6f63")
    ax.axhline(1.0, ls=(0, (4, 3)), lw=1.4, color=MUTE, zorder=1)

    # kappa~ spread across independent batches -- the point of the figure
    if np.isfinite(kmin).all() and np.isfinite(kmax).all():
        ax.fill_between(cb, kmin, kmax, color=CORAL, alpha=0.22, lw=0,
                        zorder=2, label=r"$\tilde{\kappa}$ across batches")
    ax.plot(cb, kap, marker="s", color=CORAL, markerfacecolor=CORAL,
            markeredgecolor="white", zorder=5,
            label=r"tracking $\tilde{\kappa}$ (max)")

    ax.plot(cb, rl, **_marker_kw(TEAL_LO), zorder=4,
            label=r"local $\varrho_{\mathrm{loc}}$")
    if np.isfinite(rg).any():
        ax.plot(cb, rg, marker="D", color=TEAL_HI, markerfacecolor=TEAL_HI,
                markeredgecolor="white", zorder=4,
                label=r"global $\varrho_{B_R}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"bias magnitude $c_\beta$")
    ax.set_ylabel("certificate modulus")
    ax.legend(loc="lower right", fontsize=8.2)
    finish(ax)
    save(fig, outdir, "fig_certify")


def fig_epoch(S, outdir):
    """F4: the 'ideal' reference sat exactly under the median line and was
    invisible in the legend.  Offset it and say so."""
    per = {int(n): v for n, v in S["per_N"].items()}
    Ns = np.array(sorted(per))
    med = np.array([per[n]["median"] for n in Ns])
    slope = S.get("slope")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for n in Ns:
        ts = [t for t in per[n]["T_star"] if t]
        ax.scatter([n] * len(ts), ts, s=16, color=TEAL, alpha=0.30,
                   edgecolor="none", zorder=2)
    # offset x1.35 so it is visible; it otherwise coincides with the median
    ref = 1.35 * med[0] * Ns[0] / Ns
    ax.plot(Ns, ref, ls=(0, (4, 3)), lw=1.5, color=MUTE, zorder=3,
            label=r"ideal $N^{-1}$ (offset $\times1.35$)")
    ax.plot(Ns, med, **_marker_kw(TEAL), zorder=4, label=r"median $T^\star$")
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
    ax.legend(loc="upper right", fontsize=8.6)
    finish(ax)
    save(fig, outdir, "fig_epoch")


def fig_certified(S, outdir):
    """F2: the annotation leader was drawn at data-series weight and read as
    a third run.  Use a thin straight leader."""
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    keys = sorted(S, key=float)
    cols = ([TEAL_LO, TEAL_HI] if len(keys) == 2 else [TEAL_HI] * len(keys))
    for i, k in enumerate(keys):
        v = S[k]
        c = cols[i % len(cols)]
        t = np.array(v["traj_t"])
        d = np.array(v["traj_d"])
        ax.plot(t, d, color=c, zorder=4, label=rf"$c_\beta={k}$")
        ax.axhline(v["picard"], ls=(0, (4, 3)), lw=1.4, color=c, alpha=0.85,
                   zorder=2)
    ylo = min(S[k]["picard"] for k in keys)
    ax.annotate(r"biased fixed point $\theta_\infty$",
                xy=(t[-1] * 0.62, ylo), xytext=(t[-1] * 0.62, ylo * 0.42),
                fontsize=8.8, color=MUTE, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.7,
                                shrinkA=2, shrinkB=2))

    ax.set_xlabel(r"round $t$")
    ax.set_ylabel(r"$\|\hat{\theta}_t-\theta^\star\|$")
    ax.set_ylim(0, max(S[k]["picard"] for k in keys) * 1.18)
    ax.legend(loc="lower right")
    finish(ax)
    save(fig, outdir, "fig_certified")


def fig_baselines(S, outdir):
    """F3: (b) had no error bars, so D-CESA 36 'beat' oracle 38 by eye when
    the gap is inside one SEM.  And (a) used a log axis to hide the median
    blow-up; a broken linear axis keeps the other four comparable."""
    adv, floor = S["adversary"], S["floor"]
    order = ["oracle", "mean", "trim", "median", "dcesa_mv"]

    ks = [k for k in order if k in adv]
    vals = np.array([_reg_sem(adv[k])[0] for k in ks])
    errs = np.array([_reg_sem(adv[k])[1] for k in ks])
    cols = [CMETHOD[k] for k in ks]
    x = np.arange(len(ks))

    big = vals.max()
    rest = np.sort(vals)[-2] if len(vals) > 1 else big
    broken = big > 3.0 * rest

    fig = plt.figure(figsize=(6.9, 3.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5], hspace=0.10,
                          wspace=0.34)
    if broken:
        axT, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        panes = [axT, axB]
    else:
        axB = fig.add_subplot(gs[:, 0]); axT = None; panes = [axB]

    for axp in panes:
        axp.bar(x, vals, yerr=errs, color=cols, capsize=3, width=0.68,
                edgecolor="white", error_kw=dict(lw=1.2, ecolor="#444"))
    if broken:
        axT.set_ylim(big * 0.86, big * 1.14)
        axB.set_ylim(0, rest * 1.42)
        axT.spines["bottom"].set_visible(False)
        axB.spines["top"].set_visible(False)
        axT.tick_params(bottom=False, labelbottom=False)
        # break marks
        dk = 0.014
        for axp, ys in ((axT, 0.0), (axB, 1.0)):
            axp.plot([-dk, dk], [ys - dk * 2.2, ys + dk * 2.2], transform=axp.transAxes,
                     color="#3a3a3a", lw=1.0, clip_on=False, zorder=10)
            axp.plot([1 - dk, 1 + dk], [ys - dk * 2.2, ys + dk * 2.2],
                     transform=axp.transAxes, color="#3a3a3a", lw=1.0,
                     clip_on=False, zorder=10)
        _i = int(np.argmax(vals))
        axT.text(x[_i], big + errs[_i] + big * 0.012, f"{big:.0f}",
                 ha="center", va="bottom", fontsize=8.5, color=INK)
    for xi, v, e in zip(x, vals, errs):
        if broken and v == big:
            continue
        axB.text(xi, v + e + rest * 0.04, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=8.5, color=INK)
    axB.set_xticks(x)
    axB.set_xticklabels([LABELS[k] for k in ks], rotation=28, ha="right")
    axB.set_ylabel("latent regret")
    (axT or axB).set_title("(a) adversary", fontsize=11.5, loc="left")
    for axp in panes:
        finish(axp)

    # (b) tolerance floor -- now WITH SEMs (C3)
    axb = fig.add_subplot(gs[:, 1])
    ks2 = [k for k in order if k in floor]
    v2 = np.array([_reg_sem(floor[k])[0] for k in ks2])
    e2 = np.array([_reg_sem(floor[k])[1] for k in ks2])
    x2 = np.arange(len(ks2))
    axb.bar(x2, v2, yerr=e2, color=[CMETHOD[k] for k in ks2], width=0.68,
            edgecolor="white", capsize=3,
            error_kw=dict(lw=1.2, ecolor="#444"))
    for xi, v, e in zip(x2, v2, e2):
        axb.text(xi, v + e + v2.max() * 0.03, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=8.5, color=INK)
    axb.set_xticks(x2)
    axb.set_xticklabels([LABELS[k] for k in ks2], rotation=28, ha="right")
    axb.set_ylabel("latent regret")
    axb.set_ylim(0, (v2 + e2).max() * 1.20)
    axb.set_title("(b) tolerance floor", fontsize=11.5, loc="left")
    finish(axb)
    save(fig, outdir, "fig_baselines")


def fig_mistakes(S, outdir):
    gates = ["mv", "perceptron", "ogd"]
    qs = [("0.0", NEUTRAL_LO, "$q=0$"), ("0.2", CORAL, "$q=0.2$")]
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


# ---------------------------------------------------- F6: Exps. 1-5 figures
def fig_fixedpoint(S, outdir):
    """Exp. 1 -- the plateau against N.  Cooperation buys no escape from the
    biased fixed point: ||theta-theta*|| flattens at the Picard value."""
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    cols = [TEAL_LO, TEAL_HI, CORAL]
    for i, cb in enumerate(sorted(S, key=float)):
        v = S[cb]
        per = v["per_N"]
        Ns = sorted(int(n) for n in per)
        get = lambda n: per[str(n)] if str(n) in per else per[n]
        y = [get(n)["d_star"] for n in Ns]
        e = [get(n)["d_star_sem"] for n in Ns]
        c = cols[i % len(cols)]
        ax.errorbar(Ns, y, yerr=e, marker="o", color=c, markerfacecolor=c,
                    markeredgecolor="white", capsize=3, zorder=4,
                    label=rf"$c_\beta={cb}$")
        ax.axhline(v["picard"], ls=(0, (4, 3)), lw=1.3, color=c, alpha=0.85,
                   zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"number of agents $N$")
    ax.set_ylabel(r"$\|\hat{\theta}-\theta^\star\|$")
    ax.text(0.03, 0.06, "dashed: Picard prediction", transform=ax.transAxes,
            fontsize=8.4, color=MUTE, va="bottom")
    ax.legend(loc="upper right", fontsize=8.6)
    finish(ax)
    save(fig, outdir, "fig_fixedpoint")


def fig_gap(S, outdir):
    """Exp. 2 -- the four-arm gamma(W) sweep.  Carries the flatness result and
    the cooperation-versus-isolation comparison in one panel."""
    pts = S["points"]
    g = [p["gamma"] for p in pts]
    series = [("isolated", PLUM, "isolated (common bias)"),
              ("naive", CORAL, "naive coop. (common bias)"),
              ("naive_adv", GOLD, "naive coop. (adversary)"),
              ("dcesa", TEAL, "D-CESA mv (adversary)")]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for f, c, lab in series:
        if f not in pts[0]:
            continue
        ax.errorbar(g, [p[f] for p in pts],
                    yerr=[p.get(f + "_sem", 0.0) for p in pts],
                    marker="o", color=c, markerfacecolor=c,
                    markeredgecolor="white", capsize=3, label=lab, zorder=4)
    flat = S.get("naive_flatness")
    if flat is not None:
        ax.text(0.04, 0.50,
                rf"naive flatness: CV $={flat:.3f}$",
                transform=ax.transAxes, fontsize=8.6, color=INK, va="center",
                ha="left")
    ax.set_xlabel(r"spectral gap $\gamma(W)$")
    ax.set_ylabel("final latent regret")
    ax.legend(loc="best", fontsize=8.2)
    finish(ax)
    save(fig, outdir, "fig_gap")


def fig_prop(S, outdir):
    """Exp. 3 -- certification propagation, with the lemma's lower bound
    overlaid so that its VACUITY over the useful range is visible."""
    rows = S["rows"]
    Ks = [r["K"] for r in rows]
    eff = [r["p_eff_antipodal"] for r in rows]
    bound = {b["K"]: b["lower"] for b in S.get("lemma_bound", [])}
    bl = [bound.get(k, 0.0) for k in Ks]

    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    ax.semilogx(Ks, eff, **_marker_kw(TEAL), zorder=4,
                label="antipodal agent (measured)")
    ax.axhline(S["uniform_limit"], ls=(0, (4, 3)), lw=1.4, color=MUTE,
               zorder=2, label=r"uniform limit $\bar p|S|/N$")
    ax.semilogx(Ks, bl, marker="v", color=CORAL, markerfacecolor=CORAL,
                markeredgecolor="white", ls=(0, (1, 2)), lw=1.6, zorder=3,
                label="lemma lower bound")
    nz = [k for k, b in zip(Ks, bl) if b > 0]
    k0 = min(nz) if nz else None
    ax.text(0.03, 0.93,
            ("bound is vacuous ($=0$) for all $K$ shown" if k0 is None
             else rf"bound vacuous ($=0$) for $K<{k0}$"),
            transform=ax.transAxes, fontsize=8.4, color=CORAL, va="top")
    ax.set_xlabel(r"gossip rounds $K$")
    ax.set_ylabel(r"effective rate $p^{\mathrm{eff}}$")
    ax.legend(loc="center right", fontsize=8.0)
    finish(ax)
    save(fig, outdir, "fig_prop")


def fig_gate_onset(S, outdir):
    """Exp. 4 -- measured gate onset against pbar, with the fitted exponent.
    Censored points are drawn hollow and excluded from the fit."""
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    i = 0
    for k, v in S.items():
        if not isinstance(v, dict) or "points" not in v:
            continue
        c = [CGATE.get(k.split("_")[0], TEAL), CORAL, GOLD, PLUM][i % 4]
        cens = v.get("censored", [0.0] * len(v["points"]))
        good = [(p, o) for (p, o), cc in zip(v["points"], cens)
                if cc == 0.0 and o > 0]
        bad = [(p, o) for (p, o), cc in zip(v["points"], cens)
               if cc != 0.0 and o > 0]
        lab = f"{k}: slope {v['slope']:+.2f}" if v.get("slope") is not None \
            else f"{k}: n/a"
        if good:
            ax.loglog(*zip(*good), marker="o", color=c, markerfacecolor=c,
                      markeredgecolor="white", zorder=4, label=lab)
        if bad:
            ax.loglog(*zip(*bad), marker="o", color=c, markerfacecolor="white",
                      markeredgecolor=c, ls="none", zorder=4)
        i += 1
    ax.text(0.03, 0.06, "hollow: censored, excluded from fit",
            transform=ax.transAxes, fontsize=8.2, color=MUTE, va="bottom")
    ax.set_xlabel(r"audit rate $\bar p$")
    ax.set_ylabel("measured gate onset")
    ax.legend(loc="upper right", fontsize=7.8)
    finish(ax)
    save(fig, outdir, "fig_gate_onset")


def fig_collapse(S, outdir):
    """Exp. 5 -- the censored behavioural fraction, with the anytime bonus
    against gamma/3.  The bonus staying above gamma/3 is why behavioural
    collapse is not reached, and why no 1/N slope is fitted."""
    per = S["per_N"]
    Ns = sorted(int(n) for n in per)
    get = lambda n: per[str(n)] if str(n) in per else per[n]
    frac = [get(n).get("behav_frac_final") for n in Ns]
    bonus = [get(n).get("bonus_T") for n in Ns]
    g3 = next((get(n).get("gamma_over_3") for n in Ns
               if get(n).get("gamma_over_3") is not None), None)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.95))
    axes[0].plot(Ns, frac, **_marker_kw(TEAL), zorder=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(Ns); axes[0].set_xticklabels([str(n) for n in Ns])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel(r"number of agents $N$")
    axes[0].set_ylabel("behavioural fraction at $T$")
    axes[0].set_title("(a)", fontsize=11.5, loc="left")
    ncens = S.get("behav_censored_count", 0)
    if ncens:
        axes[0].text(0.03, 0.06, f"collapse censored at {ncens}/{len(Ns)} $N$",
                     transform=axes[0].transAxes, fontsize=8.4, color=CORAL,
                     va="bottom")
    finish(axes[0])

    axes[1].plot(Ns, bonus, **_marker_kw(CORAL), zorder=4,
                 label=r"anytime bonus at $T$")
    if g3 is not None:
        axes[1].axhline(g3, ls=(0, (4, 3)), lw=1.5, color=MUTE, zorder=2,
                        label=r"$\gamma/3$")
        axes[1].text(Ns[len(Ns) // 2], g3, r" bonus stays above $\gamma/3$",
                     fontsize=8.4, color=MUTE, va="bottom", ha="center")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(Ns); axes[1].set_xticklabels([str(n) for n in Ns])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel(r"number of agents $N$")
    axes[1].set_ylabel("exploration bonus")
    axes[1].set_title("(b)", fontsize=11.5, loc="left")
    axes[1].legend(loc="best", fontsize=8.4)
    finish(axes[1])

    fig.tight_layout(w_pad=1.6)
    save(fig, outdir, "fig_collapse")


# ==================================================================== main
# name -> builder, in the order the paper presents them.
BUILDERS = [
    ("fixedpoint", fig_fixedpoint),    # Exp. 1
    ("gap", fig_gap),                  # Exp. 2
    ("prop", fig_prop),                # Exp. 3
    ("gate_onset", fig_gate_onset),    # Exp. 4
    ("collapse", fig_collapse),        # Exp. 5
    ("epoch", fig_epoch),              # R1
    ("certified", fig_certified),      # R2
    ("baselines", fig_baselines),      # R3
    ("mistakes", fig_mistakes),        # R4
    ("certify", fig_certify),          # certification appendix
]


def build_all(outdir):
    """Build every publication figure from <outdir>/results/*.json."""
    resdir = os.path.join(outdir, "results")
    figdir = os.path.join(outdir, "figures")
    set_style()
    print(f"reading {resdir} -> writing {figdir}")
    made = 0
    for name, fn in BUILDERS:
        S = load(resdir, name)
        if S is None:
            print(f"  (skip {name}: no results/{name}.json)")
            continue
        fn(S, figdir)
        made += 1
    print(f"done: {made} figures.")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_v4")
    a = ap.parse_args()
    build_all(a.outdir)


if __name__ == "__main__":
    main()
