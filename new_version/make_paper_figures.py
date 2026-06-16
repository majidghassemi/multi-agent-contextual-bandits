#!/usr/bin/env python3
"""
make_paper_figures.py  --  regenerate all CEC figures in publication style.

Reads exp{1..7}_data.pkl from --indir (default: current dir), writes
fig_expN.{pdf,png} to --outdir using paper_style.py.  No experiments are
re-run; this only re-plots saved results, so it is fast and safe to iterate on.

Robust to BOTH pkl key formats:
  * Exp 2 with the original 5 N-values or the extended {...,64,128,256}.
  * Exp 7 with named-graph keys ('Path',...) or lazy-walk keys ('beta=...').

    python make_paper_figures.py --indir . --outdir figs
"""
import argparse, pickle, os
import numpy as np
import paper_style as ps


def load(indir, n):
    with open(f'{indir}/exp{n}_data.pkl', 'rb') as f:
        return pickle.load(f)

def num_keys(d):
    return sorted([k for k in d if isinstance(k, (int, float)) and not isinstance(k, bool)])

def graph_keys(d):
    return [k for k in d if not str(k).startswith('_')]


# ---------------------------------------------------------------- Exp 1
def fig_exp1(d, outdir):
    fig, axes = ps.figure(width='double', height_ratio=0.42, ncols=2)
    Ns = num_keys(d)
    for i, N in enumerate(Ns):
        c = ps.CYCLE[i % len(ps.CYCLE)]
        te, td = d[N]['te'], d[N]['td']
        t = np.arange(te.shape[1])
        axes[0].plot(t, te.mean(0), color=c, label=f'$N={N}$')
        axes[1].plot(t, td.mean(0), color=c, label=f'$N={N}$')
    axes[0].axhline(0, color=ps.PALETTE['grey'], lw=0.6, ls=':')
    ps.finalize(axes[0], xlabel='Round $t$',
                ylabel=r'$\|\hat\theta_t-\theta^*\|$',
                title='(a) Distance to truth')
    ps.finalize(axes[1], xlabel='Round $t$',
                ylabel=r'$\|\hat\theta_t-\theta_\infty\|$',
                title=r'(b) Distance to biased fixed point')
    fig.tight_layout()
    ps.save(fig, 'fig_exp1_convergence', outdir)


# ---------------------------------------------------------------- Exp 2
def fig_exp2(d, outdir):
    Ns = num_keys(d)
    med = np.array([d[N]['median'] for N in Ns])
    err = np.array([d[N]['fixed_err'] for N in Ns])
    Ns = np.array(Ns, dtype=float)

    fig, axes = ps.figure(width='double', height_ratio=0.5, ncols=2)
    # (a) collapse time vs N with 1/N reference
    axes[0].loglog(Ns, med, 'o-', color=ps.PALETTE['blue'], label='median $t^*$')
    ref = med[0] * (Ns / Ns[0]) ** (-1.0)
    axes[0].loglog(Ns, ref, '--', color=ps.PALETTE['grey'], lw=1.0,
                   label=r'$\propto 1/N$ (theory)')
    s = d.get('_slope', float('nan'))
    ps.finalize(axes[0], xlabel='Agents $N$', ylabel='Collapse time $t^*$',
                title=f'(a) Collapse time (slope ${s:.2f}$)')
    # (b) fixed-t error vs N with 1/sqrt(N) reference -- the clean mechanism
    axes[1].loglog(Ns, err, 's-', color=ps.PALETTE['green'],
                   label=r'$\|\hat\theta_{t}-\theta_\infty\|$')
    ref2 = err[0] * (Ns / Ns[0]) ** (-0.5)
    axes[1].loglog(Ns, ref2, '--', color=ps.PALETTE['grey'], lw=1.0,
                   label=r'$\propto 1/\sqrt{N}$ (theory)')
    se = d.get('_slope_err', float('nan'))
    ps.finalize(axes[1], xlabel='Agents $N$',
                ylabel=r'Estimator error at fixed $t$',
                title=f'(b) Error decay (slope ${se:.2f}$)')
    fig.tight_layout()
    ps.save(fig, 'fig_exp2_collapse_time', outdir)


# ---------------------------------------------------------------- Exp 3
def fig_exp3(d, outdir):
    order = sorted(graph_keys(d), key=lambda k: d[k]['g'])
    gs = [d[k]['g'] for k in order]
    fig, ax = ps.figure(width='single')
    for key, lab, c, mk in [('naive', 'Naive coop.', ps.PALETTE['red'], 'o'),
                            ('iso', 'Isolated', ps.PALETTE['orange'], 's'),
                            ('dcesa', 'TWINE', ps.PALETTE['green'], '^')]:
        ax.errorbar(gs, [d[k][key] for k in order],
                    yerr=[d[k][key + '_s'] for k in order],
                    marker=mk, color=c, capsize=2, label=lab)
    ax.set_xscale('log')
    ps.finalize(ax, xlabel=r'Spectral gap $\gamma(W)$',
                ylabel=r'Network regret $R_T^{\mathrm{net}}$',
                title='Naive fails uniformly; TWINE recovers')
    fig.tight_layout()
    ps.save(fig, 'fig_exp3_phase_transition', outdir)


# ---------------------------------------------------------------- Exp 4
def fig_exp4(d, outdir):
    cfgs = graph_keys(d)
    fig, ax = ps.figure(width='single')
    style = {'naive_super': ('Naive (complete)', ps.PALETTE['red'], '-'),
             'naive_sub':   ('Naive (path)',     ps.PALETTE['orange'], '-'),
             'dcesa_super': ('TWINE (complete)', ps.PALETTE['green'], '-')}
    Tref = None
    for cfg in cfgs:
        cr = d[cfg]['cr'].mean(0)
        Tref = len(cr)
        lab, c, ls = style.get(cfg, (cfg, ps.PALETTE['blue'], '-'))
        ax.loglog(np.arange(1, len(cr) + 1), cr / 16, ls, color=c,
                  label=f"{lab} ($\\approx t^{{{d[cfg]['slope_late']:.2f}}}$)")
    tt = np.arange(1, Tref + 1)
    ax.loglog(tt, cr[10] / 16 / tt[10] * tt, ':', color=ps.PALETTE['grey'],
              lw=1.0, label=r'$\mathcal{O}(T)$ ref')
    ps.finalize(ax, xlabel='Round $t$',
                ylabel=r'Regret per agent $R_t/N$',
                title='Regret scaling', legend_loc='upper left')
    fig.tight_layout()
    ps.save(fig, 'fig_exp4_regret_scaling', outdir)


# ---------------------------------------------------------------- Exp 5
def fig_exp5(d, outdir):
    Ks = sorted([k for k in d if isinstance(k, int)])
    target = d['target']
    heat = np.array([d[K]['eff'] for K in Ks])
    fig, axes = ps.figure(width='double', height_ratio=0.5, ncols=2)
    im = axes[0].imshow(heat, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_yticks(range(len(Ks)))
    axes[0].set_yticklabels([f'{K}' for K in Ks])
    axes[0].set_xlabel('Agent index')
    axes[0].set_ylabel('Gossip depth $K$')
    axes[0].set_title('(a) Effective axiom rate')
    axes[0].grid(False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    # (b) spread vs K, log-y, showing collapse to target
    stds = [d[K]['eff'].std() for K in Ks]
    axes[1].semilogy(Ks, stds, 'o-', color=ps.PALETTE['blue'])
    axes[1].set_xlabel('Gossip depth $K$')
    axes[1].set_ylabel(r'Std. of effective rate across agents')
    axes[1].set_title('(b) Equalization')
    fig.tight_layout()
    ps.save(fig, 'fig_exp5_axiom_propagation', outdir)


# ---------------------------------------------------------------- Exp 6
def fig_exp6(d, outdir):
    ps_ = num_keys(d)
    m = [d[p]['m'] for p in ps_]
    s = [d[p]['s'] for p in ps_]
    fig, ax = ps.figure(width='single')
    ax.errorbar(ps_, m, yerr=s, marker='o', color=ps.PALETTE['blue'], capsize=2)
    ax.set_xscale('log')
    sl = d.get('_slope', float('nan'))
    ps.finalize(ax, xlabel=r'Axiom rate $\bar p$',
                ylabel=r'Network regret $R_T^{\mathrm{net}}$',
                title=f'More axioms lower regret (slope ${sl:.2f}$)')
    fig.tight_layout()
    ps.save(fig, 'fig_exp6_dcesa_axiom_rate', outdir)


# ---------------------------------------------------------------- Exp 7
def fig_exp7(d, outdir):
    order = sorted(graph_keys(d), key=lambda k: d[k]['g'])
    gs = [d[k]['g'] for k in order]
    rs = [d[k]['ratio'] for k in order]
    es = [d[k].get('ratio_s', 0.0) for k in order]
    fig, ax = ps.figure(width='single')
    ax.errorbar(gs, rs, yerr=es, marker='o', color=ps.PALETTE['green'], capsize=2)
    ax.axhline(1.0, color=ps.PALETTE['grey'], lw=0.7, ls=':')
    ax.set_xscale('log')
    sl = d.get('_slope', None)
    title = 'Spectral-gap inversion'
    if sl is not None:
        title += f' (slope ${sl:.2f}$)'
    ps.finalize(ax, xlabel=r'Spectral gap $\gamma(W)$',
                ylabel='Naive / TWINE regret ratio', title=title)
    fig.tight_layout()
    ps.save(fig, 'fig_exp7_spectral_inversion', outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='.')
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ps.set_style()
    fns = {1: fig_exp1, 2: fig_exp2, 3: fig_exp3, 4: fig_exp4,
           5: fig_exp5, 6: fig_exp6, 7: fig_exp7}
    for n, fn in fns.items():
        try:
            fn(load(args.indir, n), args.outdir)
            print(f'  fig_exp{n}: OK')
        except Exception as e:
            print(f'  fig_exp{n}: FAILED ({e})')


if __name__ == '__main__':
    main()