#!/usr/bin/env python3
"""
Coupled Exploration Collapse (CEC) - Complete Experimental Validation
Run this locally with: python run_all_local.py

Requirements: numpy, scipy, networkx, matplotlib
Install: pip install numpy scipy networkx matplotlib
"""

import numpy as np # For numerical operations
import numpy.linalg as la # For linear algebra operations
import networkx as nx # For graph generation and spectral analysis
import matplotlib.pyplot as plt # For plotting; can be removed if you only want raw data
import pickle, os, sys, time, warnings # For reproducibility and file handling
from collections import defaultdict # For structured data storage
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output; remove if you want to see them

# ============================================================================
# CONFIGURATION - Adjust these for your hardware / patience
# ============================================================================
OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 30          # Paper suggests 30; increase for publication quality
T_EXP12 = 20000       # Horizon for Experiments 1-2 (our paper: 20000)
T_EXP34 = 50000       # Horizon for Experiments 3-4 (our paper: 50000)
T_EXP67 = 20000       # Horizon for Experiments 6-7 (our paper: 20000)
VERBOSE = True        # Whether to print progress updates (can be turned off for faster runs)

# ============================================================================
# CORE FRAMEWORK AND ALGORITHMS 
# ============================================================================
d_ctx, K_acts = 4, 3 # Context dimension and number of actions per agent
d_phi = d_ctx * K_acts # Dimension of the feature space (one block of d_ctx per action)
THETA_STAR = np.zeros(d_phi) # True parameter vector (sparse: only first action has nonzero features)
THETA_STAR[0] = 1.0 # Bias scale for agent-specific biases
SIGMA = 0.3 # Noise scale for rewards

def phi_feat(x, a): # Context x (d_ctx,), action a (int) -> feature vector p (d_phi,)
    """Feature map: phi(x,a) = x ⊗ e_a"""
    p = np.zeros(d_phi)
    p[a * d_ctx:(a + 1) * d_ctx] = x
    return p

def make_bias_vecs(N, c_beta=0.5, local_scale=0.3, seed=0): # N agents, bias scale c_beta, local noise scale, random seed
    """Create agent-specific bias vectors."""
    rng = np.random.RandomState(seed)
    shared = rng.randn(d_phi)
    shared /= la.norm(shared) + 1e-10
    return [c_beta * (shared + rng.randn(d_phi) * local_scale) for _ in range(N)]

# --- Graph utilities ---
def gmat(G): # Convert a NetworkX graph to a normalized weight matrix W
    N = G.number_of_nodes()
    if N <= 1: return np.eye(max(1, N))
    L = nx.laplacian_matrix(G).astype(float).toarray()
    lm = np.linalg.eigvalsh(L)[-1]
    return np.eye(N) - L / (1 + lm) if lm > 1e-10 else np.eye(N)

def spectral_gap(W): # Spectral gap γ(W) = 1 - λ_2(W)
    e = np.sort(la.eigvalsh(W))[::-1]
    return 1.0 if len(e) < 2 else 1 - e[1]

def complete(N): return gmat(nx.complete_graph(N)) if N > 1 else np.eye(1)
def cycle(N): return gmat(nx.cycle_graph(N))
def path(N): return gmat(nx.path_graph(N))
def grid(N):
    m = max(1, int(np.sqrt(N)))
    while N % m and m > 1: m -= 1
    return gmat(nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, N // m)))
def expander(N, d=3):
    if N * d % 2: d += 1
    return gmat(nx.random_regular_graph(d, N))

# --- Optimized Naive Cooperative LinUCB ---
class NaiveLinUCB:
    def __init__(self, N, W, lam=1.0, sig=0.3):
        self.N = N; self.W = W; self.lam = lam; self.sig = sig
        self.A = np.array([lam * np.eye(d_phi) for _ in range(N)])
        self.b = np.zeros((N, d_phi)); self.th = np.zeros((N, d_phi))
        self.Ai = np.array([np.eye(d_phi) / lam for _ in range(N)])
        self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        acts = []
        for i in range(self.N):
            ba, bv = 0, -1e18
            for a in range(K_acts):
                p = phi_feat(ctx[i], a)
                v = np.dot(p, self.th[i]) + al * np.sqrt(max(0, np.dot(p, self.Ai[i] @ p)))
                if v > bv: bv = v; ba = a
            acts.append(ba)
        return acts

    def update(self, ctx, ac, rw):
        Ai2 = self.Ai.copy()
        for i in range(self.N):
            p = phi_feat(ctx[i], ac[i])
            Ap = Ai2[i] @ p
            Ai2[i] -= np.outer(Ap, Ap) / (1.0 + np.dot(p, Ap))
        for i in range(self.N):
            ps = [phi_feat(ctx[j], ac[j]) for j in range(self.N)]
            self.A[i] = sum(self.W[i,j] * (self.A[j] + np.outer(ps[j], ps[j])) for j in range(self.N))
            self.b[i] = sum(self.W[i,j] * (self.b[j] + rw[j] * ps[j]) for j in range(self.N))
            self.Ai[i] = sum(self.W[i,j] * Ai2[j] for j in range(self.N))
            self.th[i] = self.Ai[i] @ self.b[i]
        self.t += 1

# --- Simplified D-CESA ---
class DCESA:
    def __init__(self, N, W, p_i, lam=1.0, sig=0.3, K_buf=10):
        self.N = N; self.W = W; self.pi = np.array(p_i); self.lam = lam; self.sig = sig
        self.A = np.array([lam * np.eye(d_phi) for _ in range(N)])
        self.b = np.zeros((N, d_phi)); self.th = np.zeros((N, d_phi))
        self.Ai = np.array([np.eye(d_phi) / lam for _ in range(N)])
        self.trust = np.ones((N, N)) * 0.5 + 0.5 * np.eye(N)
        self.Kb = K_buf; self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        return [int(np.argmax([np.dot(phi_feat(ctx[i], a), self.th[i]) +
            al * np.sqrt(max(0, np.dot(phi_feat(ctx[i], a), self.Ai[i] @ phi_feat(ctx[i], a))))
            for a in range(K_acts)])) for i in range(self.N)]

    def update(self, ctx, ac, rw):
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.W[i,j] > 0:
                    self.trust[i,j] = min(1.0, self.trust[i,j] + 0.001)
        Wt = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i,j] > 0 or i == j: Wt[i,j] = self.trust[i,j] * self.W[i,j]
            rs = Wt[i].sum(); Wt[i] /= rs if rs > 0 else 1.0
        Ai2 = self.Ai.copy()
        for i in range(self.N):
            p = phi_feat(ctx[i], ac[i]); Ap = Ai2[i] @ p
            Ai2[i] -= np.outer(Ap, Ap) / (1.0 + np.dot(p, Ap))
        for i in range(self.N):
            ps = [phi_feat(ctx[j], ac[j]) for j in range(self.N)]
            self.A[i] = sum(Wt[i,j] * (self.A[j] + np.outer(ps[j], ps[j])) for j in range(self.N))
            self.b[i] = sum(Wt[i,j] * (self.b[j] + rw[j] * ps[j]) for j in range(self.N))
            self.Ai[i] = sum(Wt[i,j] * Ai2[j] for j in range(self.N))
            self.th[i] = self.Ai[i] @ self.b[i]
        self.t += 1

# --- Helper: compute per-round regret ---
def compute_round_regret(ctx, ac, N):
    return sum(max(np.dot(phi_feat(ctx[i], a), THETA_STAR) for a in range(K_acts))
               - np.dot(phi_feat(ctx[i], ac[i]), THETA_STAR) for i in range(N))

# --- Generic naive runner ---
def run_naive(N, W, T, seed, c_beta=0.5):
    rng = np.random.RandomState(seed)
    bv = make_bias_vecs(N, c_beta=c_beta, seed=seed)
    alg = NaiveLinUCB(N, W)
    te, cr = np.zeros(T), np.zeros(T)
    c = 0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([np.dot(phi_feat(ctx[i], ac[i]), THETA_STAR) +
                       np.dot(bv[i], phi_feat(ctx[i], ac[i])) + rng.randn() * SIGMA for i in range(N)])
        alg.update(ctx, ac, rw)
        c += compute_round_regret(ctx, ac, N)
        cr[t] = c
        te[t] = np.mean([la.norm(alg.th[i] - THETA_STAR) for i in range(N)])
    return te, cr

# --- Generic D-CESA runner ---
def run_dcesa(N, W, T, seed, p_i, Kb=10):
    rng = np.random.RandomState(seed)
    bv = make_bias_vecs(N, seed=seed)
    alg = DCESA(N, W, p_i, K_buf=Kb)
    c = 0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([np.dot(phi_feat(ctx[i], ac[i]), THETA_STAR) +
                       np.dot(bv[i], phi_feat(ctx[i], ac[i])) + rng.randn() * SIGMA for i in range(N)])
        alg.update(ctx, ac, rw)
        c += compute_round_regret(ctx, ac, N)
    return c


# ============================================================================
# EXPERIMENT 1: Biased Fixed-Point Convergence
# ============================================================================
print("=" * 60)
print("EXPERIMENT 1: Biased Fixed-Point Convergence")
print("=" * 60)

def run_exp1():
    exp1 = {}
    for N in [1, 4, 8, 16]:
        W = complete(N) if N > 1 else np.eye(1)
        te_all, cr_all = [], []
        for s in range(N_SEEDS):
            te, cr = run_naive(N, W, T_EXP12, s * 100 + N)
            te_all.append(te); cr_all.append(cr)
        exp1[N] = {'te': np.array(te_all), 'cr': np.array(cr_all)}
        print(f"  N={N:2d}: final ||θ-θ*|| = {np.mean(te_all, 0)[-1]:.4f}, final R_T = {np.mean(cr_all, 0)[-1]:.1f}")
    return exp1

exp1 = run_exp1()
with open(f'{OUTDIR}/exp1_data.pkl', 'wb') as f: pickle.dump(exp1, f)


# ============================================================================
# EXPERIMENT 2: Collapse Time ~ 1/N
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: Collapse Time ~ 1/N")
print("=" * 60)

def run_exp2():
    exp2 = {}
    THRESHOLD = 0.02
    WIN = 100
    for N in [2, 4, 8, 16, 32]:
        W = complete(N)
        cts = []
        for s in range(N_SEEDS):
            rng = np.random.RandomState(s * 200 + N)
            bv = make_bias_vecs(N, seed=s * 200 + N)
            alg = NaiveLinUCB(N, W)
            prrs = []
            for t in range(T_EXP12):
                ctx = rng.randn(N, d_ctx); ac = alg.act(ctx)
                rw = np.array([np.dot(phi_feat(ctx[i], ac[i]), THETA_STAR) +
                               np.dot(bv[i], phi_feat(ctx[i], ac[i])) + rng.randn() * SIGMA for i in range(N)])
                alg.update(ctx, ac, rw)
                prrs.append(compute_round_regret(ctx, ac, N) / N)
            prr = np.array(prrs)
            ma = np.convolve(prr, np.ones(WIN) / WIN, mode='valid')
            ct = T_EXP12
            for t in range(len(ma)):
                if ma[t] > THRESHOLD and t > 100:
                    ct = t + WIN // 2; break
            cts.append(ct)
        exp2[N] = {'ct': np.array(cts), 'm': np.mean(cts), 's': np.std(cts)}
        print(f"  N={N:2d}: t* = {exp2[N]['m']:.1f} ± {exp2[N]['s']:.1f}")
    return exp2

exp2 = run_exp2()
with open(f'{OUTDIR}/exp2_data.pkl', 'wb') as f: pickle.dump(exp2, f)


# ============================================================================
# EXPERIMENT 3: Phase Transition in γ(W)
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: Phase Transition in γ(W)")
print("=" * 60)

def run_exp3():
    N = 20
    graphs = {'Complete': complete(N), 'Expander': expander(N),
              'Grid': grid(N), 'Cycle': cycle(N), 'Path': path(N)}
    exp3 = {}
    for gn, W in graphs.items():
        g = spectral_gap(W)
        regs = []
        for s in range(N_SEEDS):
            _, cr = run_naive(N, W, T_EXP34, s * 300 + N)  # Fixed: unpack tuple, no ret_full
            regs.append(cr[-1])
        exp3[gn] = {'g': g, 'r': np.array(regs), 'm': np.mean(regs), 's': np.std(regs)}
        print(f"  {gn:12s}: γ = {g:.4f}, R_T = {exp3[gn]['m']:.1f} ± {exp3[gn]['s']:.1f}")
    return exp3

exp3 = run_exp3()
with open(f'{OUTDIR}/exp3_data.pkl', 'wb') as f: pickle.dump(exp3, f)


# ============================================================================
# EXPERIMENT 4: Regret Scaling
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: Regret Scaling (Supercritical vs Subcritical)")
print("=" * 60)

def run_exp4():
    N = 16
    configs = {'super': ('Complete', complete(N)), 'sub': ('Path', path(N))}
    exp4 = {}
    for cfg, (gn, W) in configs.items():
        crs = []
        for s in range(N_SEEDS):
            _, cr = run_naive(N, W, T_EXP34, s * 400 + N)
            crs.append(cr)
        exp4[cfg] = {'cr': np.array(crs), 'g': spectral_gap(W), 'gn': gn}
        log_t = np.log(np.arange(100, T_EXP34) + 1)
        log_r = np.log(np.mean(crs, 0)[100:])
        slope = np.polyfit(log_t, log_r, 1)[0]
        print(f"  {cfg:10s} ({gn:8s}): slope ≈ {slope:.3f}")
    return exp4

exp4 = run_exp4()
with open(f'{OUTDIR}/exp4_data.pkl', 'wb') as f: pickle.dump(exp4, f)


# ============================================================================
# EXPERIMENT 5: Root of Trust Propagation
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 5: Root of Trust Propagation")
print("=" * 60)

def run_exp5():
    N = 20; W = cycle(N)
    Kvals = [1, 5, 10, 50, 100, 200]
    p_bar = 0.1
    pb = np.zeros(N); pb[0] = p_bar
    exp5 = {}
    for Kb in Kvals:
        Wk = la.matrix_power(W, Kb) if Kb > 0 else np.eye(N)
        eff_p = Wk @ pb
        exp5[Kb] = {'eff': eff_p, 'pbar': p_bar}
        print(f"  K = {Kb:3d}: eff_p range = [{eff_p.min():.4f}, {eff_p.max():.4f}], target = {p_bar:.4f}")
    return exp5

exp5 = run_exp5()
with open(f'{OUTDIR}/exp5_data.pkl', 'wb') as f: pickle.dump(exp5, f)


# ============================================================================
# EXPERIMENT 6: D-CESA Regret vs p_bar
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 6: D-CESA Regret vs p_bar")
print("=" * 60)

def run_exp6():
    N, W = 16, complete(16)
    pvals = [0.01, 0.05, 0.1, 0.3, 1.0]
    exp6 = {}
    for p in pvals:
        pi = [p] * N
        regs = [run_dcesa(N, W, T_EXP67, s * 600 + N, pi, Kb=20) for s in range(N_SEEDS)]
        exp6[p] = {'r': np.array(regs), 'm': np.mean(regs), 's': np.std(regs)}
        print(f"  p = {p:.2f}: R_T = {exp6[p]['m']:.1f} ± {exp6[p]['s']:.1f}")
    return exp6

exp6 = run_exp6()
with open(f'{OUTDIR}/exp6_data.pkl', 'wb') as f: pickle.dump(exp6, f)


# ============================================================================
# EXPERIMENT 7: Spectral-Gap Inversion
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 7: Spectral-Gap Inversion")
print("=" * 60)

def run_exp7():
    N = 16
    graphs = {'Complete': complete(N), 'Expander': expander(N),
              'Grid': grid(N), 'Cycle': cycle(N), 'Path': path(N)}
    exp7 = {}
    for gn, W in graphs.items():
        g = spectral_gap(W)
        nregs, dregs = [], []
        for s in range(N_SEEDS):
            _, cr_naive = run_naive(N, W, T_EXP67, s * 700 + N)
            nregs.append(cr_naive[-1])
            dregs.append(run_dcesa(N, W, T_EXP67, s * 800 + N, [0.1] * N, Kb=20))
        ratios = np.array(nregs) / (np.array(dregs) + 1)
        exp7[gn] = {'g': g, 'nr': np.mean(nregs), 'dr': np.mean(dregs), 'ratio': np.mean(ratios)}
        print(f"  {gn:12s}: γ = {g:.4f}, ratio = {exp7[gn]['ratio']:.2f}x")
    return exp7

exp7 = run_exp7()
with open(f'{OUTDIR}/exp7_data.pkl', 'wb') as f: pickle.dump(exp7, f)


# ============================================================================
# GENERATE ALL FIGURES
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

def save(fig, name):
    fig.savefig(f'{OUTDIR}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

# --- Figure 1: Exp 1 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for N in [1, 4, 8, 16]:
    te = exp1[N]['te']; cr = exp1[N]['cr']
    axes[0].plot(te.mean(0), label=f'N={N}', alpha=0.8)
    axes[0].fill_between(range(len(te[0])), te.mean(0) - te.std(0), te.mean(0) + te.std(0), alpha=0.15)
    axes[1].plot(cr.mean(0), label=f'N={N}', alpha=0.8)
axes[0].set_xlabel('Round t'); axes[0].set_ylabel('||θ̂_t - θ*||'); axes[0].set_title('Distance to θ*'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Round t'); axes[1].set_ylabel('Cumulative Regret'); axes[1].set_title('Cumulative Regret'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
save(fig, 'fig_exp1_convergence.png')

# --- Figure 2: Exp 2 ---
fig, ax = plt.subplots(figsize=(7, 5))
Ns = sorted(exp2.keys())
ax.bar(range(len(Ns)), [exp2[N]['m'] for N in Ns], tick_label=[str(N) for N in Ns], color='steelblue', alpha=0.7)
ax.set_xlabel('N (agents)'); ax.set_ylabel('Collapse Time t*'); ax.set_title('Collapse Time ~ 1/N'); ax.grid(True, alpha=0.3, axis='y')
save(fig, 'fig_exp2_collapse_time.png')

# --- Figure 3: Exp 3 ---
fig, ax = plt.subplots(figsize=(8, 5))
gn_order = sorted(exp3.keys(), key=lambda k: exp3[k]['g'], reverse=True)
ax.errorbar([exp3[k]['g'] for k in gn_order], [exp3[k]['m'] for k in gn_order],
            yerr=[exp3[k]['s'] for k in gn_order], marker='o', capsize=5, linewidth=2, markersize=8)
ax.set_xscale('log'); ax.set_xlabel('Spectral Gap γ(W)'); ax.set_ylabel('Network Regret R_T'); ax.set_title('Phase Transition'); ax.grid(True, alpha=0.3)
for gn in gn_order: ax.annotate(gn, (exp3[gn]['g'], exp3[gn]['m']), textcoords="offset points", xytext=(5, 5), fontsize=9)
save(fig, 'fig_exp3_phase_transition.png')

# --- Figure 4: Exp 4 ---
fig, ax = plt.subplots(figsize=(8, 5))
for cfg in ['super', 'sub']:
    cr = exp4[cfg]['cr']; mean_cr = cr.mean(0)
    ax.loglog(range(1, len(mean_cr) + 1), mean_cr / 16, label=f"{cfg} ({exp4[cfg]['gn']})", linewidth=2)
ax.loglog(range(1, T_EXP34 + 1), 0.1 * np.arange(1, T_EXP34 + 1) ** 1.0, 'k--', alpha=0.3, label='O(T)')
ax.loglog(range(1, T_EXP34 + 1), 0.5 * np.arange(1, T_EXP34 + 1) ** 0.5, 'k:', alpha=0.3, label='O(√T)')
ax.set_xlabel('Round t'); ax.set_ylabel('Normalized Regret'); ax.set_title('Regret Scaling'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
save(fig, 'fig_exp4_regret_scaling.png')

# --- Figure 5: Exp 5 ---
fig, ax = plt.subplots(figsize=(10, 6))
Kb_sorted = sorted(exp5.keys())
heatmap = np.array([exp5[Kb]['eff'] for Kb in Kb_sorted])
im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(len(Kb_sorted))); ax.set_yticklabels([f'K={Kb}' for Kb in Kb_sorted])
ax.set_xlabel('Agent Index'); ax.set_title(f'Root of Trust Propagation (target p̄ = {exp5[Kb_sorted[-1]]["pbar"]})')
plt.colorbar(im, ax=ax, label='Effective Root of Trust Rate')
save(fig, 'fig_exp5_root_of_trust_propagation.png')

# --- Figure 6: Exp 6 ---
fig, ax = plt.subplots(figsize=(7, 5))
pvals = sorted(exp6.keys())
ax.errorbar(pvals, [exp6[p]['m'] for p in pvals], yerr=[exp6[p]['s'] for p in pvals], marker='o', capsize=5, linewidth=2, markersize=8)
ax.set_xscale('log'); ax.set_xlabel('Root of Trust Rate p̄'); ax.set_ylabel('Regret R_T'); ax.set_title('D-CESA Regret vs Root of Trust Rate'); ax.grid(True, alpha=0.3)
save(fig, 'fig_exp6_dcesa_root_of_trust.png')

# --- Figure 7: Exp 7 ---
fig, ax = plt.subplots(figsize=(7, 5))
gns = list(exp7.keys())
ax.plot([exp7[gn]['g'] for gn in gns], [exp7[gn]['ratio'] for gn in gns], marker='o', linewidth=2, markersize=10, color='green')
ax.set_xscale('log'); ax.set_xlabel('Spectral Gap γ(W)'); ax.set_ylabel('Naive / D-CESA Ratio'); ax.set_title('Spectral-Gap Inversion'); ax.grid(True, alpha=0.3)
for gn in gns: ax.annotate(gn, (exp7[gn]['g'], exp7[gn]['ratio']), textcoords="offset points", xytext=(5, 5), fontsize=10)
save(fig, 'fig_exp7_spectral_inversion.png')

# --- Summary Figure ---
fig = plt.figure(figsize=(18, 12))
# (a) Exp 1 - theta
ax1 = plt.subplot(2, 4, 1)
for N in [1, 4, 8, 16]: ax1.plot(exp1[N]['te'].mean(0), label=f'N={N}', alpha=0.8)
ax1.set_xlabel('Round t'); ax1.set_ylabel('||θ̂_t - θ*||'); ax1.set_title('(a) Biased Fixed Point'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
# (b) Exp 1 - regret
ax2 = plt.subplot(2, 4, 2)
for N in [1, 4, 8, 16]: ax2.plot(exp1[N]['cr'].mean(0), label=f'N={N}', alpha=0.8)
ax2.set_xlabel('Round t'); ax2.set_ylabel('Cumulative Regret'); ax2.set_title('(b) Regret vs N'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
# (c) Exp 2
gs = sorted(exp2.keys()); ax3 = plt.subplot(2, 4, 3)
ax3.bar(range(len(gs)), [exp2[N]['m'] for N in gs], tick_label=[str(N) for N in gs], color='steelblue', alpha=0.7)
ax3.set_xlabel('N'); ax3.set_ylabel('t*'); ax3.set_title('(c) Collapse Time'); ax3.grid(True, alpha=0.3, axis='y')
# (d) Exp 3
gno = sorted(exp3.keys(), key=lambda k: exp3[k]['g'], reverse=True); ax4 = plt.subplot(2, 4, 4)
ax4.errorbar([exp3[k]['g'] for k in gno], [exp3[k]['m'] for k in gno], yerr=[exp3[k]['s'] for k in gno], marker='o', capsize=4, linewidth=2, markersize=6)
ax4.set_xscale('log'); ax4.set_xlabel('γ(W)'); ax4.set_ylabel('R_T'); ax4.set_title('(d) Phase Transition'); ax4.grid(True, alpha=0.3)
# (e) Exp 4
ax5 = plt.subplot(2, 4, 5)
for cfg in ['super', 'sub']: ax5.loglog(range(1, T_EXP34 + 1), exp4[cfg]['cr'].mean(0) / 16, label=cfg, linewidth=2)
ax5.loglog(range(1, T_EXP34 + 1), 0.1 * np.arange(1, T_EXP34 + 1), 'k--', alpha=0.3, label='O(T)')
ax5.set_xlabel('t'); ax5.set_ylabel('R_t/N'); ax5.set_title('(e) Regret Scaling'); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, which='both')
# (f) Exp 5
ax6 = plt.subplot(2, 4, 6)
ax6.bar(range(N), exp5[200]['eff'], color='coral', alpha=0.7)
ax6.axhline(y=exp5[200]['pbar'] / 20, color='red', linestyle='--', label='target')
ax6.set_xlabel('Agent'); ax6.set_ylabel('Effective Root of Trust'); ax6.set_title('(f) Root of Trust Equalization'); ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
# (g) Exp 6
ax7 = plt.subplot(2, 4, 7)
pv = sorted(exp6.keys()); ax7.errorbar(pv, [exp6[p]['m'] for p in pv], yerr=[exp6[p]['s'] for p in pv], marker='o', capsize=4, linewidth=2, markersize=6)
ax7.set_xscale('log'); ax7.set_xlabel('Root of Trust Rate'); ax7.set_ylabel('R_T'); ax7.set_title('(g) D-CESA vs Root of Trust'); ax7.grid(True, alpha=0.3)
# (h) Exp 7
gns = list(exp7.keys()); ax8 = plt.subplot(2, 4, 8)
ax8.plot([exp7[gn]['g'] for gn in gns], [exp7[gn]['ratio'] for gn in gns], marker='o', linewidth=2, markersize=8, color='green')
ax8.set_xscale('log'); ax8.set_xlabel('γ(W)'); ax8.set_ylabel('Ratio'); ax8.set_title('(h) Gap Inversion'); ax8.grid(True, alpha=0.3)
plt.suptitle('CEC: Experimental Validation', fontsize=14, y=1.02)
plt.tight_layout()
save(fig, 'fig_summary_all.png')

print("\n" + "=" * 60)
print("ALL DONE! Check output directory:")
print(f"  {OUTDIR}")
print("=" * 60)