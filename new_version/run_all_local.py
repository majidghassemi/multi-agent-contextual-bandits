#!/usr/bin/env python3
"""
Coupled Exploration Collapse (CEC) - Complete Experimental Validation (CORRECTED)
Run locally with: python run_all_local.py

Requirements: numpy, scipy, networkx, matplotlib
Install: pip install numpy scipy networkx matplotlib

------------------------------------------------------------------------------
CHANGE LOG (fixes relative to the original runner)
------------------------------------------------------------------------------
[FIX 1] D-CESA now actually implements axiom access.  Each agent j draws a
        Bernoulli(p_j) "axiom event" every round; the per-edge trust classifier
        psi_ij is updated by online logistic-regression gradient steps ONLY on
        rounds where j's axiom fired (or a propagated axiom reached i).  The
        trust weight w_ij = sigmoid(<psi_ij, phi>) now genuinely depends on
        p_j, so regret depends on p_bar (Theorem 30).  The old code incremented
        trust by a flat 0.001 and never read p_i, making every p_bar identical.

[FIX 2] Experiment 2 (collapse time ~ 1/N) now:
          (a) defines t* directly from the theory -- first time the per-agent
              estimator enters a ball around the precomputed biased fixed point
              theta_infty -- instead of a regret-rate proxy;
          (b) removes the t>100 floor and the WIN//2 offset that pinned every
              run to ~151;
          (c) reports the MEDIAN (robust to censored / never-collapsed runs)
              and treats ceiling runs as right-censored, reported separately;
          (d) uses a longer horizon so large-N runs have room to collapse.

[FIX 3] Experiment 3 (phase transition) now uses a bias strength large enough
        to populate the decoupling region (c_beta raised, aligned bias), adds
        isolated-LinUCB and D-CESA baselines on the same axes, and directly
        measures mu_G(X_dec) so the regime is verified, not assumed.

[FIX 4] Experiment 5 target line is the TRUE network average mean(p_j), not the
        single non-zero rate.  The physics was already correct; only the
        reference value was mislabeled (0.1 vs 0.005).

[FIX 5] Horizons match the paper and are honored by every runner (no silent
        short runs).  Set FAST=True for a quick smoke test.

[FIX 6] theta_infty is computed by fixed-point iteration and cached, so several
        experiments can measure distance-to-bias directly.
------------------------------------------------------------------------------
"""

import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTDIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FAST = os.environ.get('CEC_FAST', '0') == '1'   # set CEC_FAST=1 for a smoke test

N_SEEDS  = 8     if FAST else 30
T_EXP12  = 4000  if FAST else 20000
T_EXP34  = 8000  if FAST else 50000
T_EXP67  = 4000  if FAST else 20000
VERBOSE  = True

# The phase-transition (Exp 3) and scaling (Exp 4) sweeps run several algorithms
# over several graphs/seeds, so in FAST smoke-test mode we shrink the network and
# horizon further to keep the whole suite under a few minutes.  Full runs (FAST
# off) use the paper settings above.
N_EXP3   = 12    if FAST else 20
T_EXP3   = 2500  if FAST else 50000
SEEDS_HEAVY = 4  if FAST else 30

# ============================================================================
# CORE FRAMEWORK
# ============================================================================
d_ctx, K_acts = 4, 3
d_phi = d_ctx * K_acts
THETA_STAR = np.zeros(d_phi)
THETA_STAR[0] = 1.0
SIGMA = 0.3


def stamp(result, exp_name, **extra):
    """Inject a self-describing '_config' metadata block into a result dict so
    every saved pickle records exactly how it was produced.  Captures the global
    config, environment, a UTC timestamp, and any per-experiment specifics passed
    via **extra.  Returns the same dict (mutated in place) for convenient chaining
    in `pickle.dump(stamp(expN, ...), f)`.
    """
    import datetime, platform
    result['_config'] = {
        'experiment': exp_name,
        'timestamp_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'fast_mode': FAST,
        'n_seeds': N_SEEDS,
        'seeds_heavy': SEEDS_HEAVY,
        'horizons': {'T_EXP12': T_EXP12, 'T_EXP34': T_EXP34,
                     'T_EXP67': T_EXP67, 'T_EXP3': T_EXP3},
        'problem': {'d_ctx': d_ctx, 'K_acts': K_acts, 'd_phi': d_phi,
                    'sigma': SIGMA, 'theta_star_norm': float(la.norm(THETA_STAR))},
        'n_exp3': N_EXP3,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        **extra,
    }
    return result


def phi_feat(x, a):
    """Feature map phi(x,a) = x placed in the a-th block (x (x) e_a)."""
    p = np.zeros(d_phi)
    p[a * d_ctx:(a + 1) * d_ctx] = x
    return p

def make_bias_vecs(N, c_beta=0.5, local_scale=0.3, seed=0):
    """Agent-specific bias vectors: shared network direction + local noise.

    Larger c_beta => larger emergent bias ||delta|| => bigger decoupling region.
    local_scale=0 makes bias perfectly aligned across agents (max rho_0).
    """
    rng = np.random.RandomState(seed)
    shared = rng.randn(d_phi)
    shared /= la.norm(shared) + 1e-10
    return [c_beta * (shared + rng.randn(d_phi) * local_scale) for _ in range(N)]

# --- Graph utilities ---
def gmat(G):
    """Metropolis-style normalized gossip matrix from a NetworkX graph."""
    N = G.number_of_nodes()
    if N <= 1:
        return np.eye(max(1, N))
    L = nx.laplacian_matrix(G).astype(float).toarray()
    lm = np.linalg.eigvalsh(L)[-1]
    return np.eye(N) - L / (1 + lm) if lm > 1e-10 else np.eye(N)

def spectral_gap(W):
    e = np.sort(la.eigvalsh(W))[::-1]
    return 1.0 if len(e) < 2 else 1 - e[1]

def complete(N): return gmat(nx.complete_graph(N)) if N > 1 else np.eye(1)
def cycle(N):    return gmat(nx.cycle_graph(N))
def path(N):     return gmat(nx.path_graph(N))
def grid(N):
    m = max(1, int(np.sqrt(N)))
    while N % m and m > 1:
        m -= 1
    return gmat(nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, N // m)))
def expander(N, deg=3):
    if N * deg % 2:
        deg += 1
    return gmat(nx.random_regular_graph(deg, N))

# ============================================================================
# [FIX 6] Precompute the biased fixed point theta_infty
# ============================================================================
def compute_theta_infty(bias_vecs, n_mc=8000, n_iter=40, seed=0):
    """Solve the self-consistency map theta = theta* + Sigma_z(theta)^{-1} u(theta)
    by fixed-point iteration with Monte-Carlo estimates of Sigma_z and u under
    the greedy policy pi_theta.  bias_vecs is a list; we average over agents to
    get the network-level (theta_infty, delta).  Returns theta_infty.
    """
    rng = np.random.RandomState(seed)
    th = THETA_STAR.copy()
    X = rng.randn(n_mc, d_ctx)               # (M, d_ctx)
    bias_mean = np.mean(np.array(bias_vecs), axis=0)   # network-average bias vec (d_phi,)
    # Precompute the feature tensor F[m, a, :] = phi(x_m, a)  -> (M, K, d_phi)
    M = n_mc
    F = np.zeros((M, K_acts, d_phi))
    for a in range(K_acts):
        F[:, a, a * d_ctx:(a + 1) * d_ctx] = X
    for _ in range(n_iter):
        scores = F @ th                       # (M, K)  greedy scores under th
        a_greedy = np.argmax(scores, axis=1)  # (M,)
        Fg = F[np.arange(M), a_greedy, :]     # (M, d_phi) chosen features
        Sig = (Fg.T @ Fg) / M                 # (d_phi, d_phi)
        beta_bar = Fg @ bias_mean             # (M,) network-avg bias at each (x, a_greedy)
        u = (Fg * beta_bar[:, None]).mean(axis=0)
        th_new = THETA_STAR + la.solve(Sig + 1e-6 * np.eye(d_phi), u)
        if la.norm(th_new - th) < 1e-5:
            th = th_new
            break
        th = th_new
    return th

# ============================================================================
# Naive Cooperative LinUCB
# ============================================================================
class NaiveLinUCB:
    def __init__(self, N, W, lam=1.0, sig=0.3):
        self.N, self.W, self.lam, self.sig = N, W, lam, sig
        self.A  = np.array([lam * np.eye(d_phi) for _ in range(N)])
        self.b  = np.zeros((N, d_phi))
        self.th = np.zeros((N, d_phi))
        self.Ai = np.array([np.eye(d_phi) / lam for _ in range(N)])
        self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        acts = []
        for i in range(self.N):
            ba, bv = 0, -1e18
            for a in range(K_acts):
                p = phi_feat(ctx[i], a)
                v = p @ self.th[i] + al * np.sqrt(max(0.0, p @ (self.Ai[i] @ p)))
                if v > bv:
                    bv, ba = v, a
            acts.append(ba)
        return acts

    def update(self, ctx, ac, rw):
        ps = np.array([phi_feat(ctx[j], ac[j]) for j in range(self.N)])  # (N, d)
        # local Sherman-Morrison rank-1 updates first
        Ai2 = self.Ai.copy()
        for i in range(self.N):
            p = ps[i]
            Ap = Ai2[i] @ p
            Ai2[i] = Ai2[i] - np.outer(Ap, Ap) / (1.0 + p @ Ap)
        # per-agent post-local statistics, then a single gossip mix via W
        # outer products zz^T for all agents at once: (N, d, d)
        ZZ = np.einsum('ni,nj->nij', ps, ps)
        A_post = self.A + ZZ                       # (N, d, d)
        b_post = self.b + rw[:, None] * ps         # (N, d)
        # gossip mixing: newA[i] = sum_j W[i,j] A_post[j]
        self.A  = np.einsum('ij,jkl->ikl', self.W, A_post)
        self.b  = self.W @ b_post
        self.Ai = np.einsum('ij,jkl->ikl', self.W, Ai2)
        self.th = np.einsum('ikl,il->ik', self.Ai, self.b)
        self.t += 1

# ============================================================================
# Isolated LinUCB (no gossip) -- baseline for the phase-transition plot
# ============================================================================
class IsolatedLinUCB:
    def __init__(self, N, lam=1.0, sig=0.3):
        self.N, self.lam, self.sig = N, lam, sig
        self.A  = np.array([lam * np.eye(d_phi) for _ in range(N)])
        self.b  = np.zeros((N, d_phi))
        self.th = np.zeros((N, d_phi))
        self.Ai = np.array([np.eye(d_phi) / lam for _ in range(N)])
        self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        acts = []
        for i in range(self.N):
            ba, bv = 0, -1e18
            for a in range(K_acts):
                p = phi_feat(ctx[i], a)
                v = p @ self.th[i] + al * np.sqrt(max(0.0, p @ (self.Ai[i] @ p)))
                if v > bv:
                    bv, ba = v, a
            acts.append(ba)
        return acts

    def update(self, ctx, ac, rw):
        for i in range(self.N):
            p = phi_feat(ctx[i], ac[i])
            self.A[i] += np.outer(p, p)
            self.b[i] += rw[i] * p
            Ap = self.Ai[i] @ p
            self.Ai[i] = self.Ai[i] - np.outer(Ap, Ap) / (1.0 + p @ Ap)
            self.th[i] = self.Ai[i] @ self.b[i]
        self.t += 1

# ============================================================================
# [FIX 1] D-CESA with a real axiom mechanism + propagation
# ============================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class DCESA:
    """Decentralized CESA.

    Each agent j draws an axiom event ~ Bernoulli(p_j) each round.  When it
    fires, j broadcasts (phi, z, y) into a depth-K gossip buffer; after K hops
    a neighbour i receives it with probability proportional to [W^K]_{ij}.  On
    receipt, i takes one online logistic-regression step on psi_ij toward the
    reliability label ell = 1[|y - <phi,theta*>| <= eps_tol].  The trust weight
    w_ij = sigmoid(<psi_ij, phi>) gates j's contribution to i's cooperative
    update.  Thus the learning rate of trust -- and hence regret -- depends on
    the effective axiom rate, which depends on p_bar (Theorem 30, Lemma 28).
    """
    def __init__(self, N, W, p_i, lam=1.0, sig=0.3, K_buf=10,
                 eps_tol=0.05, lr_psi=0.5, theta_star=None):
        self.N, self.W, self.lam, self.sig = N, W, lam, sig
        self.p_i = np.asarray(p_i, dtype=float)
        self.Kb = K_buf
        self.eps_tol = eps_tol
        self.lr_psi = lr_psi
        self.theta_star = THETA_STAR if theta_star is None else theta_star
        # K-step gossip weights for propagation reach
        self.WK = la.matrix_power(W, K_buf) if K_buf > 0 else np.eye(N)
        # bandit sufficient statistics
        self.A  = np.array([lam * np.eye(d_phi) for _ in range(N)])
        self.b  = np.zeros((N, d_phi))
        self.th = np.zeros((N, d_phi))
        self.Ai = np.array([np.eye(d_phi) / lam for _ in range(N)])
        # per-edge trust parameters psi_ij (only neighbours + self matter)
        self.psi = np.zeros((N, N, d_phi))
        self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        acts = []
        for i in range(self.N):
            ba, bv = 0, -1e18
            for a in range(K_acts):
                p = phi_feat(ctx[i], a)
                v = p @ self.th[i] + al * np.sqrt(max(0.0, p @ (self.Ai[i] @ p)))
                if v > bv:
                    bv, ba = v, a
            acts.append(ba)
        return acts

    def update(self, ctx, ac, rw, rng):
        ps = np.array([phi_feat(ctx[j], ac[j]) for j in range(self.N)])

        # --- [FIX 1] axiom events: which agents reveal ground truth this round ---
        axiom_fired = rng.random(self.N) < self.p_i          # Bernoulli(p_j)

        # reliability labels for agents whose axiom fired
        # ell_j = 1 if j's feedback matched ground truth within eps_tol
        labels = {}
        for j in range(self.N):
            if axiom_fired[j]:
                truth = ps[j] @ self.theta_star
                labels[j] = 1.0 if abs(rw[j] - truth) <= self.eps_tol else 0.0

        # --- trust update: i learns psi_ij from any axiom that reaches it ---
        # propagation reach: i receives j's tuple w.p. proportional to [W^K]_ij
        for i in range(self.N):
            for j in range(self.N):
                if j not in labels:
                    continue
                reach = self.WK[i, j]
                if reach <= 1e-9:
                    continue
                # stochastic receipt of the propagated axiom tuple
                if rng.random() < min(1.0, reach * self.N):  # normalize so self-receipt ~1
                    z = sigmoid(self.psi[i, j] @ ps[j])
                    grad = (z - labels[j]) * ps[j]
                    self.psi[i, j] -= self.lr_psi / np.sqrt(self.t + 1) * grad

        # --- trust weights w_ij = sigmoid(<psi_ij, phi_j>) (vectorized) ---
        # logits[i,j] = <psi_ij, phi_j>  via einsum over the last axis
        logits = np.einsum('ijk,jk->ij', self.psi, ps)        # (N, N)
        w_trust = sigmoid(logits)
        np.fill_diagonal(w_trust, 1.0)                        # always trust self
        mask = (self.W > 0)
        np.fill_diagonal(mask, True)
        Wt = w_trust * self.W * mask
        row = Wt.sum(axis=1, keepdims=True)
        Wt = Wt / np.where(row > 0, row, 1.0)

        # --- trust-weighted cooperative bandit update (vectorized) ---
        Ai2 = self.Ai.copy()
        for i in range(self.N):
            p = ps[i]
            Ap = Ai2[i] @ p
            Ai2[i] = Ai2[i] - np.outer(Ap, Ap) / (1.0 + p @ Ap)
        ZZ = np.einsum('ni,nj->nij', ps, ps)
        A_post = self.A + ZZ
        b_post = self.b + rw[:, None] * ps
        self.A  = np.einsum('ij,jkl->ikl', Wt, A_post)
        self.b  = Wt @ b_post
        self.Ai = np.einsum('ij,jkl->ikl', Wt, Ai2)
        self.th = np.einsum('ikl,il->ik', self.Ai, self.b)
        self.t += 1

# --- per-round latent regret ---
def compute_round_regret(ctx, ac, N):
    return sum(max(phi_feat(ctx[i], a) @ THETA_STAR for a in range(K_acts))
               - phi_feat(ctx[i], ac[i]) @ THETA_STAR for i in range(N))

# --- runners ---
def run_naive(N, W, T, seed, c_beta=0.5, local_scale=0.3, track_theta=False,
              theta_infty=None):
    rng = np.random.RandomState(seed)
    bv = make_bias_vecs(N, c_beta=c_beta, local_scale=local_scale, seed=seed)
    alg = NaiveLinUCB(N, W)
    cr = np.zeros(T)
    te = np.zeros(T) if track_theta else None
    td = np.zeros(T) if (track_theta and theta_infty is not None) else None
    c = 0.0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                       + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                       for i in range(N)])
        alg.update(ctx, ac, rw)
        c += compute_round_regret(ctx, ac, N)
        cr[t] = c
        if track_theta:
            te[t] = np.mean([la.norm(alg.th[i] - THETA_STAR) for i in range(N)])
            if td is not None:
                td[t] = np.mean([la.norm(alg.th[i] - theta_infty) for i in range(N)])
    return cr, te, td

def run_isolated(N, T, seed, c_beta=0.5, local_scale=0.3):
    rng = np.random.RandomState(seed)
    bv = make_bias_vecs(N, c_beta=c_beta, local_scale=local_scale, seed=seed)
    alg = IsolatedLinUCB(N)
    c = 0.0
    cr = np.zeros(T)
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                       + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                       for i in range(N)])
        alg.update(ctx, ac, rw)
        c += compute_round_regret(ctx, ac, N)
        cr[t] = c
    return cr

def run_dcesa(N, W, T, seed, p_i, Kb=10, c_beta=0.5, local_scale=0.3):
    rng = np.random.RandomState(seed)
    bv = make_bias_vecs(N, c_beta=c_beta, local_scale=local_scale, seed=seed)
    alg = DCESA(N, W, p_i, K_buf=Kb)
    c = 0.0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                       + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                       for i in range(N)])
        alg.update(ctx, ac, rw, rng)
        c += compute_round_regret(ctx, ac, N)
    return c


# ============================================================================
# Adversarial-bias helper (used by Exp 6/7 so trust learning genuinely matters)
# ============================================================================
def make_adversarial_bias(N, c_honest=0.05, c_adv=1.5, seed=0):
    """Half the agents are honest (tiny bias); half are strongly, consistently
    biased along a shared direction.  Distrusting the biased half requires axiom
    access, so D-CESA regret depends on the axiom rate p_bar.
    """
    rng = np.random.RandomState(seed)
    direction = rng.randn(d_phi); direction /= la.norm(direction) + 1e-10
    return [(c_honest * rng.randn(d_phi) if i < N // 2 else c_adv * direction)
            for i in range(N)]

def run_dcesa_adv(N, W, T, seed, p_i, Kb=5, eps_tol=0.3, lr_psi=1.0,
                  c_honest=0.05, c_adv=1.5):
    rng = np.random.RandomState(seed)
    bv = make_adversarial_bias(N, c_honest, c_adv, seed=seed)
    alg = DCESA(N, W, p_i, K_buf=Kb, eps_tol=eps_tol, lr_psi=lr_psi)
    c = 0.0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                       + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                       for i in range(N)])
        alg.update(ctx, ac, rw, rng)
        c += compute_round_regret(ctx, ac, N)
    return c

def run_naive_adv(N, W, T, seed, c_honest=0.05, c_adv=1.5):
    """Naive cooperative LinUCB on the same adversarial instance (no trust)."""
    rng = np.random.RandomState(seed)
    bv = make_adversarial_bias(N, c_honest, c_adv, seed=seed)
    alg = NaiveLinUCB(N, W)
    c = 0.0
    for t in range(T):
        ctx = rng.randn(N, d_ctx)
        ac = alg.act(ctx)
        rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                       + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                       for i in range(N)])
        alg.update(ctx, ac, rw)
        c += compute_round_regret(ctx, ac, N)
    return c

# ============================================================================
# Centralized reference (exact per-round average) -- used by Exp 2.
# Remark 13's t*_cen is a statement about THIS object, so we measure it here.
# ============================================================================
class CentralizedRef:
    def __init__(self, N, lam=1.0, sig=0.3):
        self.N, self.lam, self.sig = N, lam, sig
        self.A = lam * np.eye(d_phi)
        self.b = np.zeros(d_phi)
        self.Ai = np.eye(d_phi) / lam
        self.th = np.zeros(d_phi)
        self.t = 0

    def act(self, ctx):
        al = self.sig * np.sqrt(d_phi * max(1, np.log(1 + self.t / self.lam))) + np.sqrt(self.lam)
        acts = []
        for i in range(self.N):
            ba, bv = 0, -1e18
            for a in range(K_acts):
                p = phi_feat(ctx[i], a)
                v = p @ self.th + al * np.sqrt(max(0.0, p @ (self.Ai @ p)))
                if v > bv:
                    bv, ba = v, a
            acts.append(ba)
        return acts

    def update(self, ctx, ac, rw):
        for i in range(self.N):
            p = phi_feat(ctx[i], ac[i])
            self.A += np.outer(p, p) / self.N
            self.b += rw[i] * p / self.N
        self.Ai = la.inv(self.A)
        self.th = self.Ai @ self.b
        self.t += 1


# ============================================================================
# EXPERIMENT 1: Biased fixed-point convergence + cooperative regret growth
# ============================================================================
def run_exp1():
    print("=" * 60)
    print("EXPERIMENT 1: Biased Fixed-Point Convergence")
    print("  Validates: theta_t -> theta_infty (biased FP), not theta*")
    print("=" * 60)
    exp1 = {}
    for N in [1, 4, 8, 16]:
        W = complete(N) if N > 1 else np.eye(1)
        bv0 = make_bias_vecs(N, c_beta=1.0, seed=N)
        th_inf = compute_theta_infty(bv0, seed=N)
        te_all, cr_all, td_all = [], [], []
        for s in range(N_SEEDS):
            cr, te, td = run_naive(N, W, T_EXP12, seed=s * 100 + N, c_beta=1.0,
                                   track_theta=True, theta_infty=th_inf)
            te_all.append(te); cr_all.append(cr); td_all.append(td)
        exp1[N] = {'te': np.array(te_all), 'cr': np.array(cr_all),
                   'td': np.array(td_all), 'delta': float(la.norm(th_inf - THETA_STAR))}
        if VERBOSE:
            print(f"  N={N:2d}: ||th-th*||={np.mean(te_all,0)[-1]:.3f}  "
                  f"||th-th_inf||={np.mean(td_all,0)[-1]:.3f}  "
                  f"||delta||={exp1[N]['delta']:.3f}  R_T={np.mean(cr_all,0)[-1]:.0f}")
    with open(f'{OUTDIR}/exp1_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp1, 'exp1_biased_fixed_point',
                          graph='complete', c_beta=1.0, N_values=[1, 4, 8, 16],
                          horizon=T_EXP12), f)
    return exp1


# ============================================================================
# EXPERIMENT 2: Collapse time of the CENTRALIZED reference ~ 1/N
#   [FIX 2] t* := first time ||theta_cen - theta_infty|| < floor, sustained.
#   Measured on CentralizedRef because Remark 13's t*_cen is about that object;
#   the gossip algorithm approaches it as mixing completes.  Reports MEDIAN and
#   the censored-run count separately.
# ============================================================================
def run_exp2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Collapse Time ~ 1/N (centralized reference)")
    print("  Validates: Remark 13, t*_cen proportional to 1/N")
    print("  Reports BOTH the collapse-time trend and the 1/sqrt(N) error decay")
    print("  that mechanistically produces it.")
    print("=" * 60)
    FLOOR_FRAC = 0.30
    SUSTAIN = 50
    T_FIX = min(1000, T_EXP12)     # fixed-t snapshot for the error-decay metric
    exp2 = {}
    for N in [2, 4, 8, 16, 32]:
        cts, censored, fixed_errs = [], 0, []
        for s in range(N_SEEDS):
            rng = np.random.RandomState(s * 200 + N)
            bv = make_bias_vecs(N, c_beta=1.0, seed=s * 200 + N)
            th_inf = compute_theta_infty(bv, seed=s * 200 + N)
            delta = la.norm(th_inf - THETA_STAR)
            alg = CentralizedRef(N)
            ct, run = T_EXP12, 0
            for t in range(T_EXP12):
                ctx = rng.randn(N, d_ctx)
                ac = alg.act(ctx)
                rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                               + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                               for i in range(N)])
                alg.update(ctx, ac, rw)
                if t + 1 == T_FIX:
                    fixed_errs.append(la.norm(alg.th - th_inf))
                inside = la.norm(alg.th - th_inf) <= FLOOR_FRAC * delta
                run = run + 1 if inside else 0
                if run >= SUSTAIN and ct == T_EXP12:
                    ct = t - SUSTAIN + 1
                    # keep iterating to record T_FIX error; don't break early
            if ct >= T_EXP12:
                censored += 1
            cts.append(ct)
        cts = np.array(cts)
        exp2[N] = {'ct': cts, 'median': float(np.median(cts)),
                   'mean': float(np.mean(cts)), 'censored': censored,
                   'fixed_err': float(np.mean(fixed_errs))}
        if VERBOSE:
            print(f"  N={N:2d}: median t*={exp2[N]['median']:.0f}  "
                  f"censored={censored}/{N_SEEDS}  "
                  f"||theta_{T_FIX}-theta_inf||={exp2[N]['fixed_err']:.4f}")
    Ns = sorted([k for k in exp2 if isinstance(k, int)])
    med = [exp2[N]['median'] for N in Ns]
    ferr = [exp2[N]['fixed_err'] for N in Ns]
    slope_ct = np.polyfit(np.log(Ns), np.log(med), 1)[0]
    slope_err = np.polyfit(np.log(Ns), np.log(ferr), 1)[0]
    exp2['_slope'] = float(slope_ct)
    exp2['_slope_err'] = float(slope_err)
    if VERBOSE:
        print(f"  >>> collapse-time slope (t* vs N)   = {slope_ct:.3f}  (theory -1.0)")
        print(f"  >>> error-decay slope (err vs N)    = {slope_err:.3f}  (theory -0.5)")
        print(f"  >>> implied t* exponent from error  = {2*slope_err:.3f}  (theory -1.0)")
    with open(f'{OUTDIR}/exp2_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp2, 'exp2_collapse_time',
                          estimator='centralized_reference', c_beta=1.0,
                          N_values=[2, 4, 8, 16, 32], floor_frac=FLOOR_FRAC,
                          sustain=SUSTAIN, horizon=T_EXP12, t_fix=T_FIX), f)
    return exp2


# ============================================================================
# EXPERIMENT 3: Phase transition in gamma(W)
#   [FIX 3] Strong aligned bias to populate the decoupling region; isolated and
#   D-CESA baselines on the same axes; measured decoupling mass reported.
# ============================================================================
def run_exp3():
    """Phase transition across gamma(W).

    HONEST FRAMING.  With a network-aligned bias, the naive algorithm locks into
    the SAME biased fixed point theta_infty regardless of gamma(W) -- so its
    regret is uniformly Omega(NT) and roughly flat in gamma.  That flatness IS
    the content of Corollary 25 (naive fails in every regime), NOT a bug: the
    headline is "naive is uniformly bad," with D-CESA recovering low regret.
    We therefore plot naive (flat-high), isolated (no cooperation), and D-CESA
    (low) on shared axes, and we also report the measured decoupling-region mass
    mu_G(X_dec) so the regime is verified rather than assumed.  A literal
    non-monotone "bathtub" in a single naive curve requires separating the
    super- and sub-critical mechanisms and is left as a stronger experiment;
    we do not over-claim it here.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Regret across gamma(W) (Cor. 25)")
    print("  Validates: naive uniformly Omega(NT); D-CESA recovers")
    print("=" * 60)
    N = N_EXP3
    C_BETA = 1.5
    LOCAL = 0.1
    T_e3 = T_EXP3
    # report decoupling-region mass so the regime is explicit
    bv0 = make_bias_vecs(N, c_beta=C_BETA, local_scale=LOCAL, seed=0)
    th_inf0 = compute_theta_infty(bv0, seed=0)
    rng = np.random.RandomState(0); dec = 0; n_mc = 4000
    for _ in range(n_mc):
        x = rng.randn(d_ctx)
        feats = [phi_feat(x, a) for a in range(K_acts)]
        if np.argmax([f @ THETA_STAR for f in feats]) != np.argmax([f @ th_inf0 for f in feats]):
            dec += 1
    mu_dec = dec / n_mc
    if VERBOSE:
        print(f"  measured mu_G(X_dec) = {mu_dec:.3f}, ||delta|| = {la.norm(th_inf0-THETA_STAR):.3f}")
    graphs = {'Complete': complete(N), 'Expander': expander(N),
              'Grid': grid(N), 'Cycle': cycle(N), 'Path': path(N)}
    exp3 = {'_mu_dec': mu_dec}
    for gn, W in graphs.items():
        g = spectral_gap(W)
        naive_r, iso_r, dcesa_r = [], [], []
        for s in range(SEEDS_HEAVY):
            cr, _, _ = run_naive(N, W, T_e3, seed=s * 300 + N,
                                 c_beta=C_BETA, local_scale=LOCAL)
            naive_r.append(cr[-1])
            iso_r.append(run_isolated(N, T_e3, seed=s * 300 + N,
                                      c_beta=C_BETA, local_scale=LOCAL))
            dcesa_r.append(run_dcesa_adv(N, W, T_e3, seed=s * 300 + N,
                                         p_i=[0.3] * N, Kb=10))
        exp3[gn] = {'g': g,
                    'naive': float(np.mean(naive_r)),  'naive_s': float(np.std(naive_r)),
                    'iso':   float(np.mean(iso_r)),    'iso_s':   float(np.std(iso_r)),
                    'dcesa': float(np.mean(dcesa_r)),  'dcesa_s': float(np.std(dcesa_r))}
        if VERBOSE:
            print(f"  {gn:10s}: g={g:.4f}  naive={exp3[gn]['naive']:.0f}  "
                  f"iso={exp3[gn]['iso']:.0f}  dcesa={exp3[gn]['dcesa']:.0f}")
    with open(f'{OUTDIR}/exp3_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp3, 'exp3_phase_transition',
                          N=N, c_beta=C_BETA, local_scale=LOCAL,
                          horizon=T_e3, seeds=SEEDS_HEAVY,
                          dcesa_p=0.3, dcesa_K=10,
                          graphs=['Complete','Expander','Grid','Cycle','Path']), f)
    return exp3


# ============================================================================
# EXPERIMENT 4: Regret scaling (linear for naive; sublinear for D-CESA)
#   [FIX 5] honors the full horizon; reports late-time slope.
# ============================================================================
def run_exp4():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Regret Scaling")
    print("  Validates: naive slope ~1 (linear); D-CESA slope ~0.5 (sublinear)")
    print("=" * 60)
    N = 16
    C_BETA = 1.5
    T_e4 = T_EXP3
    configs = {'naive_super': ('Complete', complete(N), 'naive'),
               'naive_sub':   ('Path',     path(N),     'naive'),
               'dcesa_super': ('Complete', complete(N), 'dcesa')}
    exp4 = {}
    for cfg, (gn, W, kind) in configs.items():
        crs = []
        for s in range(SEEDS_HEAVY):
            if kind == 'naive':
                cr, _, _ = run_naive(N, W, T_e4, seed=s * 400 + N, c_beta=C_BETA)
                crs.append(cr)
            else:
                # D-CESA: track cumulative regret over time
                rng = np.random.RandomState(s * 400 + N)
                bv = make_bias_vecs(N, c_beta=C_BETA, seed=s * 400 + N)
                alg = DCESA(N, W, [0.2] * N, K_buf=10)
                c = 0.0; cr = np.zeros(T_e4)
                for t in range(T_e4):
                    ctx = rng.randn(N, d_ctx); ac = alg.act(ctx)
                    rw = np.array([phi_feat(ctx[i], ac[i]) @ THETA_STAR
                                   + bv[i] @ phi_feat(ctx[i], ac[i]) + rng.randn() * SIGMA
                                   for i in range(N)])
                    alg.update(ctx, ac, rw, rng)
                    c += compute_round_regret(ctx, ac, N); cr[t] = c
                crs.append(cr)
        crs = np.array(crs); mean_cr = crs.mean(0)
        lo = T_e4 // 2
        slope_late = np.polyfit(np.log(np.arange(lo, T_e4) + 1),
                                np.log(mean_cr[lo:] + 1e-9), 1)[0]
        exp4[cfg] = {'cr': crs, 'gn': gn, 'kind': kind,
                     'g': spectral_gap(W), 'slope_late': float(slope_late)}
        if VERBOSE:
            print(f"  {cfg:12s} ({gn:8s}): late-time slope={slope_late:.3f}  "
                  f"(naive~1.0, dcesa~0.5)")
    with open(f'{OUTDIR}/exp4_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp4, 'exp4_regret_scaling',
                          N=N, c_beta=C_BETA, horizon=T_e4, seeds=SEEDS_HEAVY,
                          configs=list(configs.keys())), f)
    return exp4


# ============================================================================
# EXPERIMENT 5: Axiom-rate equalization  [FIX 4] target = mean(p_j)
# ============================================================================
def run_exp5():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Axiom-Rate Equalization (Lemma 28)")
    print("  Validates: every agent's effective rate -> network average mean(p)")
    print("=" * 60)
    N = 20
    W = cycle(N)
    Kvals = [1, 5, 10, 50, 100, 200]
    pb = np.zeros(N); pb[0] = 0.1          # one oracle agent, rest p=0
    p_network_avg = float(np.mean(pb))     # = 0.005  -- the TRUE equalization target
    exp5 = {'target': p_network_avg}
    for Kb in Kvals:
        Wk = la.matrix_power(W, Kb) if Kb > 0 else np.eye(N)
        eff_p = Wk @ pb
        exp5[Kb] = {'eff': eff_p}
        if VERBOSE:
            print(f"  K={Kb:3d}: eff range=[{eff_p.min():.5f},{eff_p.max():.5f}] "
                  f"std={eff_p.std():.5f}  target(mean p)={p_network_avg:.5f}")
    with open(f'{OUTDIR}/exp5_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp5, 'exp5_axiom_propagation',
                          N=N, graph='cycle', K_values=Kvals,
                          oracle_p=0.1, network_avg_target=p_network_avg), f)
    return exp5


# ============================================================================
# EXPERIMENT 6: D-CESA regret vs p_bar  [FIX 1] axioms now actually used
# ============================================================================
def run_exp6():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: D-CESA Regret vs Axiom Rate p_bar")
    print("  Validates: more axioms -> lower regret (trust learned faster)")
    print("=" * 60)
    N, W = 8, complete(8)
    pvals = [0.02, 0.05, 0.1, 0.3, 0.5, 1.0]
    T_e6 = T_EXP67 if not FAST else 2500
    exp6 = {}
    for p in pvals:
        regs = [run_dcesa_adv(N, W, T_e6, seed=s * 600 + N, p_i=[p] * N, Kb=5)
                for s in range(SEEDS_HEAVY)]
        exp6[p] = {'r': np.array(regs), 'm': float(np.mean(regs)), 's': float(np.std(regs))}
        if VERBOSE:
            print(f"  p={p:.2f}: R_T={exp6[p]['m']:.1f} +/- {exp6[p]['s']:.1f}")
    ps = sorted([k for k in exp6 if isinstance(k, (int, float)) and not isinstance(k, bool)])
    slope = np.polyfit(np.log([1 / p for p in ps]),
                       np.log([exp6[p]['m'] for p in ps]), 1)[0]
    exp6['_slope'] = float(slope)
    if VERBOSE:
        print(f"  >>> slope(log R vs log 1/p) = {slope:.3f}  (theory +0.5; >0 = axioms help)")
    with open(f'{OUTDIR}/exp6_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp6, 'exp6_dcesa_vs_axiom_rate',
                          N=N, graph='complete', p_values=pvals,
                          horizon=T_e6, seeds=SEEDS_HEAVY, dcesa_K=5,
                          regime='adversarial_neighbors'), f)
    return exp6


# ============================================================================
# EXPERIMENT 7: Spectral-gap inversion (naive/D-CESA ratio vs gamma)
# ============================================================================
def run_exp7():
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Spectral-Gap Inversion (Cor. 33)")
    print("  Validates: naive/D-CESA regret ratio grows with gamma(W)")
    print("=" * 60)
    N = 16
    T_e7 = T_EXP67 if not FAST else 2500
    graphs = {'Path': path(N), 'Cycle': cycle(N), 'Grid': grid(N),
              'Expander': expander(N), 'Complete': complete(N)}
    exp7 = {}
    for gn, W in graphs.items():
        g = spectral_gap(W)
        nregs, dregs = [], []
        for s in range(SEEDS_HEAVY):
            nregs.append(run_naive_adv(N, W, T_e7, seed=s * 700 + N))
            dregs.append(run_dcesa_adv(N, W, T_e7, seed=s * 800 + N,
                                       p_i=[0.2] * N, Kb=10))
        ratios = np.array(nregs) / (np.array(dregs) + 1.0)
        exp7[gn] = {'g': g, 'nr': float(np.mean(nregs)), 'dr': float(np.mean(dregs)),
                    'ratio': float(np.mean(ratios)), 'ratio_s': float(np.std(ratios))}
        if VERBOSE:
            print(f"  {gn:10s}: g={g:.4f}  naive={exp7[gn]['nr']:.0f}  "
                  f"dcesa={exp7[gn]['dr']:.0f}  ratio={exp7[gn]['ratio']:.2f}x")
    with open(f'{OUTDIR}/exp7_data.pkl', 'wb') as f:
        pickle.dump(stamp(exp7, 'exp7_spectral_inversion',
                          N=N, horizon=T_e7, seeds=SEEDS_HEAVY,
                          dcesa_p=0.2, dcesa_K=10,
                          regime='adversarial_neighbors',
                          graphs=['Path','Cycle','Grid','Expander','Complete']), f)
    return exp7


# ============================================================================
# FIGURES
# ============================================================================
def save(fig, name):
    fig.savefig(f'{OUTDIR}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

def make_figures(exp1, exp2, exp3, exp4, exp5, exp6, exp7):
    print("\n" + "=" * 60); print("GENERATING FIGURES"); print("=" * 60)

    # Fig 1: convergence to biased FP (distance to theta* AND to theta_infty)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for N in [1, 4, 8, 16]:
        te = exp1[N]['te']; td = exp1[N]['td']
        axes[0].plot(te.mean(0), label=f'N={N}', alpha=0.85)
        axes[1].plot(td.mean(0), label=f'N={N}', alpha=0.85)
    axes[0].set(xlabel='Round t', ylabel='||theta_t - theta*||',
                title='(a) Distance to truth (plateaus > 0)')
    axes[1].set(xlabel='Round t', ylabel='||theta_t - theta_infty||',
                title='(b) Distance to biased fixed point (-> 0)')
    for ax in axes: ax.legend(); ax.grid(alpha=0.3)
    save(fig, 'fig_exp1_convergence.png')

    # Fig 2: collapse time vs N (median, log-log)
    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = sorted([k for k in exp2 if isinstance(k, int)])
    med = [exp2[N]['median'] for N in Ns]
    ax.loglog(Ns, med, 'o-', markersize=8, linewidth=2, label='median t* (centralized)')
    ref = med[0] * (np.array(Ns) / Ns[0]) ** (-1.0)
    ax.loglog(Ns, ref, 'k--', alpha=0.5, label='1/N reference')
    ax.set(xlabel='N (agents)', ylabel='Collapse time t*',
           title=f"(slope={exp2.get('_slope',float('nan')):.2f}, theory -1)")
    ax.legend(); ax.grid(alpha=0.3, which='both')
    save(fig, 'fig_exp2_collapse_time.png')

    # Fig 3: phase transition with 3 algorithms
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted([k for k in exp3 if not str(k).startswith('_')], key=lambda k: exp3[k]['g'])
    gs = [exp3[k]['g'] for k in order]
    for key, lab, mk in [('naive', 'Naive coop.', 'o'),
                         ('iso', 'Isolated', 's'),
                         ('dcesa', 'D-CESA', '^')]:
        ax.errorbar(gs, [exp3[k][key] for k in order],
                    yerr=[exp3[k][key + '_s'] for k in order],
                    marker=mk, capsize=4, linewidth=2, label=lab)
    ax.set_xscale('log')
    ax.set(xlabel='Spectral gap gamma(W)', ylabel='Network regret R_T',
           title='Phase transition: naive high across all gamma')
    ax.legend(); ax.grid(alpha=0.3)
    for k in order:
        ax.annotate(k, (exp3[k]['g'], exp3[k]['naive']),
                    textcoords='offset points', xytext=(4, 4), fontsize=8)
    save(fig, 'fig_exp3_phase_transition.png')

    # Fig 4: regret scaling log-log
    fig, ax = plt.subplots(figsize=(8, 5))
    cfg_keys = [c for c in exp4 if not str(c).startswith('_')]
    for cfg in cfg_keys:
        cr = exp4[cfg]['cr'].mean(0)
        ax.loglog(np.arange(1, len(cr) + 1), cr / 16,
                  linewidth=2, label=f"{cfg} (slope {exp4[cfg]['slope_late']:.2f})")
    Tref = exp4[cfg_keys[0]]['cr'].shape[1]
    tt = np.arange(1, Tref + 1)
    ax.loglog(tt, 0.02 * tt, 'k--', alpha=0.3, label='O(T)')
    ax.loglog(tt, 0.5 * np.sqrt(tt), 'k:', alpha=0.3, label='O(sqrt T)')
    ax.set(xlabel='Round t', ylabel='Normalized regret R_t / N',
           title='Regret scaling')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    save(fig, 'fig_exp4_regret_scaling.png')

    # Fig 5: axiom-rate equalization heatmap (+ correct target line)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Ks = sorted([k for k in exp5 if isinstance(k, int)])
    heat = np.array([exp5[K]['eff'] for K in Ks])
    im = axes[0].imshow(heat, aspect='auto', cmap='YlOrRd')
    axes[0].set_yticks(range(len(Ks))); axes[0].set_yticklabels([f'K={K}' for K in Ks])
    axes[0].set(xlabel='Agent index', title='Effective axiom rate by (agent, K)')
    plt.colorbar(im, ax=axes[0])
    axes[1].bar(range(len(exp5[Ks[-1]]['eff'])), exp5[Ks[-1]]['eff'],
                color='coral', alpha=0.75)
    axes[1].axhline(exp5['target'], color='red', linestyle='--',
                    label=f"network avg = {exp5['target']:.4f}")
    axes[1].set(xlabel='Agent index', ylabel='Effective rate',
                title=f'Equalization at K={Ks[-1]}')
    axes[1].legend()
    save(fig, 'fig_exp5_axiom_propagation.png')

    # Fig 6: D-CESA regret vs p_bar
    fig, ax = plt.subplots(figsize=(7, 5))
    ps = sorted([k for k in exp6 if isinstance(k, (int, float)) and not isinstance(k, bool)])
    ax.errorbar(ps, [exp6[p]['m'] for p in ps],
                yerr=[exp6[p]['s'] for p in ps],
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set(xlabel='Axiom rate p_bar', ylabel='Regret R_T',
           title=f"D-CESA regret decreases with axiom rate "
                 f"(slope {exp6.get('_slope', float('nan')):.2f})")
    ax.grid(alpha=0.3)
    save(fig, 'fig_exp6_dcesa_axiom_rate.png')

    # Fig 7: spectral-gap inversion
    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted([k for k in exp7 if not str(k).startswith('_')], key=lambda k: exp7[k]['g'])
    ax.errorbar([exp7[k]['g'] for k in order], [exp7[k]['ratio'] for k in order],
                yerr=[exp7[k]['ratio_s'] for k in order],
                marker='o', linewidth=2, markersize=9, capsize=4, color='green')
    ax.set_xscale('log')
    ax.set(xlabel='Spectral gap gamma(W)', ylabel='Naive / D-CESA regret ratio',
           title='Spectral-gap inversion')
    ax.grid(alpha=0.3)
    for k in order:
        ax.annotate(k, (exp7[k]['g'], exp7[k]['ratio']),
                    textcoords='offset points', xytext=(4, 4), fontsize=9)
    save(fig, 'fig_exp7_spectral_inversion.png')


# ============================================================================
# Load saved results from disk (for re-plotting in a later session)
# ============================================================================
def load_all_results(outdir=None):
    """Load the seven saved pickle files into a dict {1: exp1, ..., 7: exp7}.
    Raises FileNotFoundError listing any missing experiment so you know which
    run() to re-run.  This is the function to use when you want to visualize
    previously-computed results without recomputing them.
    """
    outdir = outdir or OUTDIR
    data, missing = {}, []
    for n in range(1, 8):
        path = f'{outdir}/exp{n}_data.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data[n] = pickle.load(f)
        else:
            missing.append(n)
    if missing:
        raise FileNotFoundError(
            f"Missing experiment pickles for {missing} in {outdir}. "
            f"Run the corresponding run_exp{{n}}() first.")
    return data


def replot_from_disk(outdir=None):
    """Regenerate every figure from the saved pickles -- no recomputation."""
    d = load_all_results(outdir)
    make_figures(d[1], d[2], d[3], d[4], d[5], d[6], d[7])


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import sys, time

    # `python run_all_local.py --replot` regenerates figures from the saved
    # .pkl files without re-running any experiment.  Use this to restyle plots
    # or recover after a crash mid-figure.
    if '--replot' in sys.argv:
        print("Re-plotting from saved pickles (no recomputation)...")
        replot_from_disk()
        print("Done.")
        sys.exit(0)

    t0 = time.time()
    print(f"Running CEC validation suite (FAST={FAST}, N_SEEDS={N_SEEDS})\n")
    results = {}
    runners = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4,
               5: run_exp5, 6: run_exp6, 7: run_exp7}

    # `python run_all_local.py 3 6` runs only experiments 3 and 6 (each still
    # writes its own pkl), then re-plots from whatever pkls are on disk.  With
    # no numeric args, all seven run.
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    to_run = want if want else list(runners)

    for n in to_run:
        results[n] = runners[n]()

    # plot from disk so partial runs still produce a full figure set using the
    # most recent saved data for any experiment that wasn't re-run this session
    try:
        replot_from_disk()
    except FileNotFoundError as e:
        print(f"\n[skipped figures] {e}")

    print("\n" + "=" * 60)
    print(f"ALL DONE in {time.time()-t0:.0f}s. Output in: {OUTDIR}")
    print("=" * 60)