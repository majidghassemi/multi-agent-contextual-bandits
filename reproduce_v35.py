#!/usr/bin/env python3
# =====================================================================
#  reproduce_v34.py  --  single-file reproduction of ALL experiments in
#  "When Feedback Fails and Audits Suffice" (CEC / D-CESA), manuscript
#  version v3.4 (Section `Reconciliation Experiments', Exps. 8-12, plus
#  the certification appendix and theory-binding checks).
#
#  Self-contained: numpy only (matplotlib optional for figures).
#  Supersedes run_all.py.  Checkpoints PER CONFIGURATION and resumes.
#
#  Usage
#  -----
#    python3 reproduce_v34.py --quick                 # CI scale, ~3 min
#    python3 reproduce_v34.py --paper                 # Sec. recon scale
#    python3 reproduce_v34.py --paper --only epoch,baselines
#    python3 reproduce_v34.py --paper --outdir runs/A
#    python3 reproduce_v34.py --paper --force         # ignore checkpoints
#    python3 reproduce_v34.py --paper --no-figures
#
#  Outputs (under --outdir, default ./out_v34)
#    results/<exp>.json        final per-experiment results
#    results/summary.json      everything, one file
#    checkpoints/<exp>/*.json  per-config checkpoints (atomic writes)
#    figures/<exp>.pdf         color-blind-safe figures (if matplotlib)
#    run.log                   append-only progress log
#
#  Resume semantics
#    * A finished experiment (results/<exp>.json present) is SKIPPED and
#      loaded into the summary, unless --force.
#    * Within the long sweeps (epoch over N, gate over configs, random
#      instances over draws), each unit is cached under checkpoints/;
#      a kill resumes at the first uncached unit.
#    * All checkpoint writes are atomic (tmp + os.replace), so a kill
#      mid-write cannot corrupt a checkpoint.
# =====================================================================
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

# --------------------------------------------------------------------- I/O
OUTDIR = "out_v34"
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def _p(*a):
    return os.path.join(OUTDIR, *a)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(_p("run.log"), "a") as f:
        f.write(line + "\n")


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)                      # atomic on POSIX


def result_done(name):
    return os.path.exists(_p("results", f"{name}.json"))


def result_load(name):
    with open(_p("results", f"{name}.json")) as f:
        return json.load(f)


def result_save(name, obj):
    _atomic_json(_p("results", f"{name}.json"), obj)


def ckpt_load(name, key):
    path = _p("checkpoints", name, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def ckpt_save(name, key, val):
    _atomic_json(_p("checkpoints", name, f"{key}.json"), val)


def _sem(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


# ================================================================== TOPOLOGY
#  Gossip matrices.  Convention matches v2/main.py (gmat): W = I - L/lam_max
#  with L the combinatorial Laplacian, so W is symmetric, doubly stochastic,
#  and PSD (Assumption `lazy'), and lambda_* = lambda_2.
def adjacency(n, topology, k=None):
    A = np.zeros((n, n))
    idx = np.arange(n)
    if topology == "complete":
        A = np.ones((n, n)) - np.eye(n)
    elif topology == "cycle":
        A[idx, (idx + 1) % n] = 1.0
        A[idx, (idx - 1) % n] = 1.0
    elif topology == "circulant":
        if k is None:
            raise ValueError("circulant needs k")
        for h in range(1, k + 1):
            A[idx, (idx + h) % n] = 1.0
            A[idx, (idx - h) % n] = 1.0
    elif topology == "path":
        A[idx[:-1], idx[1:]] = 1.0
        A[idx[1:], idx[:-1]] = 1.0
    elif topology == "star":
        A[0, 1:] = 1.0
        A[1:, 0] = 1.0
    elif topology == "isolated":
        pass
    else:
        raise ValueError(f"unknown topology {topology}")
    np.fill_diagonal(A, 0.0)
    return A


def gossip_matrix(n, topology="complete", k=None):
    if n == 1:
        return np.ones((1, 1))
    if topology == "isolated":
        return np.eye(n)
    A = adjacency(n, topology, k)
    L = np.diag(A.sum(1)) - A
    lam_max = np.linalg.eigvalsh(L)[-1]
    if lam_max <= 0:
        return np.eye(n)
    return np.eye(n) - L / lam_max


def spectral_data(W):
    ev = np.sort(np.linalg.eigvalsh(W))[::-1]
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    lam_star = float(max(abs(lam2), abs(ev[-1]))) if len(ev) > 1 else 0.0
    return dict(gamma=float(1.0 - lam2), lam2=lam2, lam_star=lam_star,
                gamma_dagger=float(1.0 - ev[-1]))


def circulant_family(n):
    """k -> (W, gamma) sweeping cycle (k=1) to complete (k=n//2)."""
    out = []
    for k in range(1, n // 2 + 1):
        W = gossip_matrix(n, "circulant", k)
        out.append((k, W, spectral_data(W)["gamma"]))
    return out


# ===================================================================== MODEL
#  Symmetric block instance: d_c=4, K=3, theta* rows unit vectors at
#  120deg (first two coords), scaled 1/sqrt(3) so ||theta*||=1, s0=1.
#  Contexts uniform on the unit ball.  Bias: agent adds bias[i] to the
#  observed reward of the flattered action A1.
D_C, K, A1 = 4, 3, 0
SIGMA, LAMB, R_BASIN, EPS_TOL = 0.3, 1.0, 0.35, 0.05


def theta_star_symmetric():
    ang = 2 * np.pi * np.arange(K) / K
    th = np.zeros((K, D_C))
    th[:, 0], th[:, 1] = np.cos(ang), np.sin(ang)
    return th / np.sqrt(K)


TH_STAR = theta_star_symmetric()


def contexts(rng, n):
    g = rng.standard_normal((n, D_C))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / D_C)


def theta_star_random(seed):
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((K, D_C))
    return th / np.linalg.norm(th)


# -------------------------------------------------- fixed point + certify
def _pop_stats(X, TH, c_bar, th_star=TH_STAR):
    n = len(X)
    pi = (X @ TH.T).argmax(1)
    S = np.zeros((K, D_C, D_C))
    u = np.zeros((K, D_C))
    for a in range(K):
        Xa = X[pi == a]
        S[a] = (Xa.T @ Xa) / n
        if a == A1:
            u[a] = c_bar * Xa.sum(0) / n
    return S, u


def pop_F(X, TH, c_bar, th_star=TH_STAR):
    S, u = _pop_stats(X, TH, c_bar, th_star)
    out = th_star.copy()
    for a in range(K):
        out[a] = th_star[a] + np.linalg.solve(S[a], u[a])
    return out


def picard(c_bar, th_star=TH_STAR, iters=80, n_mc=400_000, seed=0):
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    TH = th_star.copy()
    res = np.inf
    for _ in range(iters):
        TH_new = pop_F(X, TH, c_bar, th_star)
        res = float(np.linalg.norm(TH_new - TH))
        TH = TH_new
    return TH, res, res < 1e-3


def sigma_min_at(TH, n_mc=400_000, seed=1):
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    S, _ = _pop_stats(X, TH, 0.0)
    return float(min(np.linalg.eigvalsh(S[a]).min() for a in range(K)))


def contraction_modulus(c_bar, TH_inf, n_dirs=40, eps_list=(0.01, 0.03, 0.1),
                        n_mc=400_000, seed=2):
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    F0 = pop_F(X, TH_inf, c_bar)
    best = 0.0
    for eps in eps_list:
        for _ in range(n_dirs):
            U = rng.standard_normal(TH_inf.shape)
            U /= np.linalg.norm(U)
            best = max(best, np.linalg.norm(
                pop_F(X, TH_inf + eps * U, c_bar) - F0) / eps)
    return best


def _rand_in_ball(rng, shape, R):
    """A single draw uniform on the ball of radius R about the origin."""
    U = rng.standard_normal(shape)
    U /= np.linalg.norm(U)
    d = int(np.prod(shape))
    return U * R * rng.uniform(0.0, 1.0) ** (1.0 / d)


def global_contraction_modulus(c_bar, n_pairs=400, n_base=80,
                               eps_list=(0.01, 0.03, 0.1), R=None,
                               n_mc=400_000, seed=7):
    """The GLOBAL contraction modulus varrho_{B_R} of Thm. cec_epoch:

        sup { ||F(th) - F(th')|| / ||th - th'||  :  th, th' in B_R(theta*) }

    contraction_modulus() above perturbs only around theta_inf, so it is the
    LOCAL modulus varrho_loc.  varrho_loc <= varrho_{B_R} always, and it is
    varrho_{B_R} -- not varrho_loc -- that the epoch theorem consumes and that
    the certification appendix tabulates.  Estimating the two separately is
    what makes finding F2 checkable rather than asserted (C4).

    The sup is approached both by far-apart pairs and, because F is smooth, in
    the limit of short chords, so both are sampled.  One-sided: this is a
    sampled lower bound on the true sup.
    """
    R = R_BASIN if R is None else R
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    best, witness = 0.0, None

    # (a) independent uniform pairs -- probes the whole ball
    for _ in range(n_pairs):
        A = TH_STAR + _rand_in_ball(rng, TH_STAR.shape, R)
        B = TH_STAR + _rand_in_ball(rng, TH_STAR.shape, R)
        den = float(np.linalg.norm(A - B))
        if den < 1e-9:
            continue
        r = float(np.linalg.norm(pop_F(X, A, c_bar) - pop_F(X, B, c_bar))) / den
        if r > best:
            best, witness = r, dict(kind="pair", sep=den)

    # (b) short chords at random base points in the ball -- far pairs alone
    #     average over the ball and systematically understate the sup
    for _ in range(n_base):
        A = TH_STAR + _rand_in_ball(rng, TH_STAR.shape, R)
        FA = pop_F(X, A, c_bar)
        for eps in eps_list:
            U = rng.standard_normal(TH_STAR.shape)
            U /= np.linalg.norm(U)
            r = float(np.linalg.norm(
                pop_F(X, A + eps * U, c_bar) - FA)) / eps
            if r > best:
                best, witness = r, dict(kind="chord", sep=float(eps))
    return float(best), witness


def tracking_modulus(c_bar, TH_inf, sigma_min, n_dirs=40,
                     eps_list=(0.01, 0.03), radial=15, n_mc=400_000, seed=3):
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    delta = TH_inf - TH_STAR

    def gbar(TH):
        S, u = _pop_stats(X, TH, c_bar)
        return np.array([u[a] - S[a] @ delta[a] for a in range(K)])

    best = 0.0
    for eps in eps_list:
        for _ in range(n_dirs):
            U = rng.standard_normal(TH_inf.shape)
            U /= np.linalg.norm(U)
            best = max(best, np.linalg.norm(gbar(TH_inf + eps * U)) / eps)
    for _ in range(radial):
        U = rng.standard_normal(TH_inf.shape)
        U /= np.linalg.norm(U)
        t = rng.uniform(0.05, R_BASIN)
        best = max(best, np.linalg.norm(gbar(TH_inf + t * U)) / t)
    return best, 2.0 * best / sigma_min


def region_masses(TH_inf, th_star=TH_STAR, gamma=0.02, Delta=0.05,
                  n_mc=200_000, seed=4):
    rng = np.random.default_rng(seed)
    X = contexts(rng, n_mc)
    Sb, Sl = X @ TH_inf.T, X @ th_star.T
    atil, astar = Sb.argmax(1), Sl.argmax(1)
    mu_dis = float((atil != astar).mean())
    reg = Sl.max(1, keepdims=True) - Sl
    gap = np.take_along_axis(Sb, atil[:, None], 1) - Sb
    gap_min = np.where(reg < Delta, gap, np.inf).min(1)
    return mu_dis, float((gap_min >= gamma).mean())


# ===================================================================== GATES
class MVGate:
    def __init__(self, sd, n):
        self.sd, self.n = sd, n
        self.s = np.zeros((sd, n, n))
        self.c = np.zeros((sd, n, n))
        self.mist = np.zeros(sd)

    def update(self, fire, labels, feats=None):
        self.s += fire * labels
        self.c += fire

    def trusted(self):
        g = np.where(self.c > 0, self.s / np.maximum(self.c, 1) >= 0.5, True)
        idx = np.arange(self.n)
        g[:, idx, idx] = True
        return g

    def tally(self, clean):
        g = self.trusted()
        t = np.broadcast_to(clean[None, None, :] > 0.5, g.shape)
        w = (g != t).copy()
        idx = np.arange(self.n)
        w[:, idx, idx] = False
        per_seed = w.sum((1, 2))
        self.mist += per_seed
        return per_seed          # used for the measured gate-onset time


class PerceptronGate(MVGate):
    def __init__(self, sd, n):
        self.sd, self.n = sd, n
        self.psi = np.zeros((sd, n, n, 2))
        self.mist = np.zeros(sd)
        self._feats = np.zeros((sd, n, n, 2))

    def _pred(self, feats):
        return (self.psi * feats).sum(-1) >= 0.0

    def update(self, fire, labels, feats):
        pred = self._pred(feats)
        wrong = fire.astype(bool) & (pred != (labels > 0.5))
        sgn = (2.0 * labels - 1.0)[..., None]
        self.psi += wrong[..., None] * sgn * feats
        self._feats = feats

    def trusted(self):
        g = self._pred(self._feats)
        idx = np.arange(self.n)
        g[:, idx, idx] = True
        return g


class OGDLogisticGate(MVGate):
    def __init__(self, sd, n, eta=0.5):
        self.sd, self.n, self.eta = sd, n, eta
        self.psi = np.zeros((sd, n, n, 2))
        self.mist = np.zeros(sd)
        self._feats = np.zeros((sd, n, n, 2))

    def update(self, fire, labels, feats):
        m = (self.psi * feats).sum(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(m, -30, 30)))
        self.psi -= self.eta * fire[..., None] * (p - labels)[..., None] * feats
        self._feats = feats

    def trusted(self):
        g = (self.psi * self._feats).sum(-1) >= 0.0
        idx = np.arange(self.n)
        g[:, idx, idx] = True
        return g


class EdgeResidualFeatures:
    """Per-edge [1, 1-2*tanh(|z|/5)] with z the cumulative standardized
    residual of the source agent's reports under the trusting model.
    Package stand-in for infrastructure-specific message features."""
    def __init__(self, sd, n, sigma):
        self.sum = np.zeros((sd, n, n))
        self.cnt = np.zeros((sd, n, n))
        self.sigma = sigma

    def update(self, residual_ij):
        self.sum += residual_ij
        self.cnt += 1.0

    def feats(self):
        z = self.sum / (self.sigma * np.sqrt(np.maximum(self.cnt, 1.0)))
        return np.stack([np.ones_like(z),
                         1.0 - 2.0 * np.tanh(np.abs(z) / 5.0)], axis=-1)


# ================================================================ ALGORITHMS
def _alpha(t, S_bound=1.5, delta=0.05):
    d = D_C * K
    return (SIGMA * np.sqrt(d * np.log(1 + t / LAMB) + 2 * np.log(1 / delta))
            + np.sqrt(LAMB) * S_bound)


def run_anytime(bias_mag, n_agents, T, seeds, seed0, agg="mean",
                gate="mv", pbar=0.1, q=0.0, record_every=250, inv_every=5,
                topology="complete", k_circ=None, W=None,
                gate_mix="selfcomp", theta_inf=None,
                gamma_dec=0.02, Delta_dec=0.05):
    """Anytime cooperative LinUCB on an arbitrary gossip graph.

    agg in {isolated, mean, gossip, oracle, trim, median, gate}.

    Mixing follows the paper's recursion
        A_t^{(i)} = sum_j W_ij (A_{t-1}^{(j)} + dA_t^{(j)}),
    so information diffuses over t hops, not one.  `mean' pins W = J/N
    (identical to the 1/N centralized convention and to v3.4's
    behaviour); `gossip' uses the requested topology; `isolated' uses
    W = I.  `trim'/`median' are increment-level robust aggregations and
    are complete-graph baselines by construction.

    gate_mix: "selfcomp" implements the paper's row-stochastic gated
    matrix (removed mass returned to the self-weight); "renorm" is the
    v3.4 legacy rule (uniform renormalisation over trusted neighbours).

    When theta_inf is supplied the run also instruments the BEHAVIOURAL
    collapse time (first sustained window in which selected actions in
    the (gamma,Delta)-decoupling region carry latent regret >= Delta)
    and, when gating, the measured gate-onset time (first round after
    which no gate is ever wrong again).
    """
    rng = np.random.default_rng(seed0)
    Sd, N = seeds, n_agents
    bias_v = np.asarray(bias_mag, float)
    clean = (np.abs(bias_v) <= EPS_TOL).astype(float)
    cmask = clean > 0.5

    # ---------------------------------------------------------- mixing rule
    if agg == "isolated":
        mix_kind, Wb = "identity", np.eye(N)
    elif agg == "mean":
        mix_kind, Wb = "mean", np.full((N, N), 1.0 / N)
    elif agg == "oracle":
        mix_kind = "matrix"
        Wb = np.tile(clean / max(clean.sum(), 1.0), (N, 1))
    elif agg in ("gossip", "gate"):
        Wb = gossip_matrix(N, topology, k_circ) if W is None else np.asarray(W)
        mix_kind = "matrix" if agg == "gossip" else "gated"
        # fast path ONLY when the caller did not supply an explicit W;
        # `topology' still holds its default "complete" in that case,
        # which previously silently overrode a passed-in graph.
        if agg == "gossip" and W is None and topology == "complete":
            mix_kind = "mean"
    elif agg in ("trim", "median"):
        mix_kind, Wb = "none", np.eye(N)
    else:
        raise ValueError(agg)

    def mix(arr, Wt=None):
        if mix_kind == "identity" or N == 1:
            return arr
        if mix_kind == "mean":
            return np.repeat(arr.mean(1, keepdims=True), N, axis=1)
        if mix_kind == "gated":
            return np.einsum('sij,sj...->si...', Wt, arr)
        return np.einsum('ij,sj...->si...', Wb, arr)

    A = np.tile(LAMB * np.eye(D_C), (Sd, N, K, 1, 1))
    b = np.zeros((Sd, N, K, D_C))
    Ainv = np.linalg.inv(A)
    TH = np.einsum('snkij,snkj->snki', Ainv, b)

    gate_obj = feat = None
    if agg == "gate":
        gate_obj = dict(mv=MVGate, perceptron=PerceptronGate,
                        ogd=OGDLogisticGate)[gate](Sd, N)
        if gate != "mv":
            feat = EdgeResidualFeatures(Sd, N, SIGMA)

    reg = np.zeros(Sd)
    reg_clean = np.zeros(Sd)
    traj_t, traj_d, traj_reg = [], [], []
    last_wrong = np.full(Sd, -1.0)

    do_behav = theta_inf is not None
    TI = np.asarray(theta_inf) if do_behav else None
    win_hit = np.zeros(Sd)          # region rounds with latent regret>=Delta
    win_tot = np.zeros(Sd)
    traj_behav, traj_behav_n = [], []

    for t in range(1, T + 1):
        X = contexts(rng, Sd * N).reshape(Sd, N, D_C)
        sc = np.einsum('snd,snkd->snk', X, TH)
        bon = np.sqrt(np.maximum(
            np.einsum('snd,snkde,sne->snk', X, Ainv, X), 0.0))
        act = (sc + _alpha(t) * bon).argmax(2)
        lat = np.einsum('snd,kd->snk', X, TH_STAR)
        sel_lat = np.take_along_axis(lat, act[:, :, None], 2)[:, :, 0]
        inst_reg = lat.max(2) - sel_lat
        reg += inst_reg.sum(1)
        reg_clean += inst_reg[:, cmask].sum(1)
        y = (sel_lat + bias_v[None, :] * (act == A1)
             + SIGMA * rng.standard_normal((Sd, N)))

        if do_behav:
            Sb = np.einsum('snd,kd->snk', X, TI)
            atil = Sb.argmax(2)
            reg_all = lat.max(2, keepdims=True) - lat
            gap = np.take_along_axis(Sb, atil[:, :, None], 2) - Sb
            gap_min = np.where(reg_all < Delta_dec, gap, np.inf).min(2)
            in_reg = gap_min >= gamma_dec
            win_tot += in_reg.sum(1)
            win_hit += (in_reg & (inst_reg >= Delta_dec)).sum(1)

        if gate_obj is not None:
            fire = (rng.random((Sd, N, N)) < pbar).astype(float)
            flip = rng.random((Sd, N, N)) < q
            labels = np.abs(clean[None, None, :] - flip)
            f = None
            if feat is not None:
                P = np.zeros((Sd, N, N))
                for a in range(K):
                    Pa = np.einsum('sjd,sid->sij', X, TH[:, :, a])
                    P += Pa * (act == a)[:, None, :]
                feat.update(y[:, None, :] - P)
                f = feat.feats()
            gate_obj.update(fire, labels, f)
            wrong_now = gate_obj.tally(clean)
            last_wrong = np.where(wrong_now > 0, float(t), last_wrong)

        dA = np.zeros((Sd, N, K, D_C, D_C))
        db = np.zeros((Sd, N, K, D_C))
        for a in range(K):
            m = (act == a)
            xa = X * m[:, :, None]
            dAj = np.einsum('snd,sne->snde', xa, xa)
            dbj = xa * (y * m)[:, :, None]
            if agg in ("trim", "median"):
                if agg == "trim":
                    sA = np.sort(dAj, axis=1)[:, 1:-1].mean(1)
                    sB = np.sort(dbj, axis=1)[:, 1:-1].mean(1)
                else:
                    sA = np.median(dAj, axis=1)
                    sB = np.median(dbj, axis=1)
                dA[:, :, a] = sA[:, None]
                db[:, :, a] = sB[:, None]
            else:
                dA[:, :, a] = dAj
                db[:, :, a] = dbj

        Wt = None
        if mix_kind == "gated":
            g = gate_obj.trusted().astype(float)
            Wg = Wb[None] * g
            if gate_mix == "selfcomp":
                idx = np.arange(N)
                off = Wg.sum(2) - Wg[:, idx, idx]
                Wt = Wg.copy()
                Wt[:, idx, idx] = 1.0 - off
            elif gate_mix == "renorm":
                Wt = Wg / np.maximum(Wg.sum(2, keepdims=True), 1e-12)
            else:
                raise ValueError(gate_mix)

        A = mix(A + dA, Wt)
        b = mix(b + db, Wt)
        if t % inv_every == 0 or t < 50:
            Ainv = np.linalg.inv(A)
        TH = np.einsum('snkij,snkj->snki', Ainv, b)
        if t % record_every == 0:
            traj_t.append(t)
            traj_d.append(float(np.linalg.norm(
                TH[:, 0] - TH_STAR[None], axis=(1, 2)).mean()))
            traj_reg.append(float(reg.mean()))
            if do_behav:
                frac = win_hit / np.maximum(win_tot, 1.0)
                traj_behav.append(float(frac.mean()))
                traj_behav_n.append(float(win_tot.sum()))
                win_hit[:] = 0.0
                win_tot[:] = 0.0

    out = dict(reg=reg.tolist(), reg_mean=float(reg.mean()),
               reg_sem=_sem(reg), reg_clean_mean=float(reg_clean.mean()),
               reg_clean_sem=_sem(reg_clean),
               d_star=float(np.linalg.norm(
                   TH[:, 0] - TH_STAR[None], axis=(1, 2)).mean()),
               traj_t=traj_t, traj_d=traj_d, traj_reg=traj_reg,
               bonus_T=float(2 * np.sqrt(2) * _alpha(T) / np.sqrt(T * 0.02)))
    if gate_obj is not None:
        out["gate_mistaken_rounds"] = float(gate_obj.mist.mean())
        out["gate_mistaken_sem"] = _sem(gate_obj.mist)
        onset = np.where(last_wrong < 0, 0.0, last_wrong + 1.0)
        out["gate_onset"] = float(np.median(onset))
        out["gate_onset_mean"] = float(onset.mean())
        out["gate_onset_sem"] = _sem(onset)
        out["gate_onset_censored"] = float((onset >= T).mean())
    if do_behav:
        out["traj_behav"] = traj_behav
        out["traj_behav_n"] = traj_behav_n
        out["t_behav"] = behavioural_collapse_time(traj_t, traj_behav,
                                                   counts=traj_behav_n)
    out["_TH0"] = TH[:, 0].tolist()          # for fixed-point comparison
    return out


def behavioural_collapse_time(ts, fracs, thresh=0.9, counts=None,
                              min_count=20, min_informative=3):
    """First recorded time after which EVERY later window has at least
    `thresh` of decoupling-region rounds playing a >=Delta-regret action.
    Windows with fewer than `min_count` region rounds carry no evidence
    and are skipped rather than counted as failures.  None if no such
    sustained entry occurs before the horizon."""
    if not fracs:
        return None
    if counts is None:
        counts = [min_count] * len(fracs)
    t_star, informative = None, 0
    for i in range(len(fracs) - 1, -1, -1):
        if counts[i] < min_count:          # no evidence either way
            if t_star is not None:
                t_star = ts[i]
            continue
        if fracs[i] >= thresh:
            informative += 1
            t_star = ts[i]
        else:
            break
    return t_star if informative >= min_informative else None


def loglog_slope(xs, ys, tail=0.5):
    """Late-window log-log slope over the last `tail` fraction of points."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = (xs > 0) & (ys > 0)
    xs, ys = xs[m], ys[m]
    if len(xs) < 3:
        return None
    j = int(len(xs) * (1 - tail))
    return float(np.polyfit(np.log(xs[j:]), np.log(ys[j:]), 1)[0])


def run_epoch(c_beta, n_agents, t1, theta_inf, collapse_thr, seed,
              max_epochs=15, chunk=2_000_000):
    rng = np.random.default_rng(seed)
    TH_frozen = TH_STAR.copy()
    hist = []
    for k in range(1, max_epochs + 1):
        m = n_agents * t1 * (2 ** (k - 1))
        Ab = [LAMB * np.eye(D_C) for _ in range(K)]
        bb = [np.zeros(D_C) for _ in range(K)]
        left = m
        while left > 0:
            c = min(left, chunk)
            left -= c
            X = contexts(rng, c)
            pi = (X @ TH_frozen.T).argmax(1)
            for a in range(K):
                Xa = X[pi == a]
                yv = (Xa @ TH_STAR[a] + (c_beta if a == A1 else 0.0)
                      + SIGMA * rng.standard_normal(len(Xa)))
                Ab[a] += Xa.T @ Xa
                bb[a] += yv @ Xa
        TH = np.stack([np.linalg.solve(Ab[a], bb[a]) for a in range(K)])
        dev = np.linalg.norm(TH - TH_STAR)
        if dev > R_BASIN:
            TH = TH_STAR + (TH - TH_STAR) * (R_BASIN / dev)
        TH_frozen = TH
        r = float(np.linalg.norm(TH_frozen - theta_inf))
        hist.append(r)
        if r <= collapse_thr:
            return t1 * (2 ** k), k, hist
    return None, None, hist


# =================================================================== DRIVERS
def exp_certify(quick, full=False):
    grid = ([0.15, 0.4] if quick else
            [0.05, 0.1, 0.15, 0.18, 0.4, 0.7, 1.0])
    n_mc = 120_000 if quick else 400_000
    rows = []
    for cb in grid:
        cached = ckpt_load("certify", f"cb{cb}")
        if cached is not None:
            rows.append(cached)
            log(f"  certify cb={cb}: cached")
            continue
        TI, res, ok = picard(cb, n_mc=n_mc)
        sm = sigma_min_at(TI, n_mc=n_mc)
        # kappa~ divides by the BASIN-UNIFORM excitation constant of
        # Prop. tracking: min over greedy policies of theta in
        # B_R(theta*) (paper constant ~0.0200).  Neither the
        # lambda_min at theta_inf (~0.031) nor at theta* alone is the
        # right denominator; both overstate it.
        rng_u = np.random.default_rng(5)
        thetas = [TH_STAR, TI] + [
            TH_STAR + (lambda U: U / np.linalg.norm(U))(
                rng_u.standard_normal(TH_STAR.shape))
            * rng_u.uniform(0, R_BASIN) for _ in range(60)]
        sm_uniform = min(sigma_min_at(Tt, n_mc=n_mc, seed=6 + i)
                         for i, Tt in enumerate(thetas))
        rho_loc = contraction_modulus(cb, TI, n_mc=n_mc)
        # the modulus the epoch theorem actually consumes (C4)
        rho_glob, rho_wit = global_contraction_modulus(
            cb, n_pairs=80 if quick else 400, n_base=20 if quick else 80,
            n_mc=n_mc)
        # The sampled sup is one-sided, so the MAX over independent batches
        # is the point estimate.  But the paper's claim at c_beta=0.05 is that
        # kappa~ STRADDLES 1, and a single number cannot show that -- so keep
        # the whole per-batch spread, not just the max (C6).
        n_batch = 3 if quick else 8
        cc_batches = []
        for batch in range(n_batch):
            c_b, _ = tracking_modulus(cb, TI, sm_uniform, n_dirs=100,
                                      eps_list=(0.005, 0.01, 0.03, 0.08),
                                      radial=30, n_mc=n_mc,
                                      seed=100 + batch)
            cc_batches.append(float(c_b))
        cc = max(cc_batches)
        kt_batches = [2.0 * c / sm_uniform for c in cc_batches]
        kt = 2.0 * cc / sm_uniform
        md, mc = region_masses(TI)
        # Appendix D tabulates three (gamma, Delta) settings; compute all of
        # them so every cell of that table comes from this run.
        mu_grid = {}
        for gm, dl in ((0.02, 0.05), (0.02, 0.10), (0.05, 0.05)):
            _, m_ = region_masses(TI, gamma=gm, Delta=dl)
            mu_grid[f"g{gm}_D{dl}"] = float(m_)
        row = dict(c_beta=cb, delta_norm=float(np.linalg.norm(TI - TH_STAR)),
                   picard_res=res, sigma_min=sm,
                   sigma_min_uniform=float(sm_uniform),
                   rho_loc=float(rho_loc), rho_global=float(rho_glob),
                   rho_global_witness=rho_wit,
                   rho=float(rho_loc),   # back-compat alias for rho_loc
                   c_circ=float(cc), kappa_tilde=float(kt),
                   n_batches=n_batch,
                   kappa_tilde_batches=[float(x) for x in kt_batches],
                   kappa_tilde_min=float(min(kt_batches)),
                   kappa_tilde_max=float(max(kt_batches)),
                   kappa_straddles_one=bool(min(kt_batches) < 1.0
                                            < max(kt_batches)),
                   mu_dis=md,
                   mu_dec=mc, mu_dec_grid=mu_grid,
                   # the theorem's hypothesis is on the GLOBAL modulus
                   contraction_ok=bool(rho_glob < 1),
                   contraction_ok_loc=bool(rho_loc < 1),
                   tracking_ok=bool(kt < 1))
        ckpt_save("certify", f"cb{cb}", row)
        rows.append(row)
        log(f"  certify cb={cb}: rho_loc={rho_loc:.2f} rho_BR={rho_glob:.2f} "
            f"kappa~={kt:.2f} [{min(kt_batches):.2f},{max(kt_batches):.2f}] "
            f"mu_dec={mc:.4f}")
    return dict(rows=rows)


def exp_epoch(quick, full=False):
    cb, gamma = 0.15, 0.05
    TI, _, _ = picard(cb, n_mc=200_000)
    thr = gamma / 8.0
    Ns = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16]
    seeds = 3 if quick else 6
    m0 = 512 if quick else 2048
    per_N = {}
    for N in Ns:
        t1 = int(np.ceil(m0 / N))
        Ts = []
        for s in range(seeds):
            key = f"N{N}_s{s}"
            cached = ckpt_load("epoch", key)
            if cached is not None:
                Ts.append(cached["T_star"])
                continue
            T, k, hist = run_epoch(cb, N, t1, TI, thr, seed=100 + s,
                                   max_epochs=14)
            ckpt_save("epoch", key, dict(T_star=T, k_star=k))
            Ts.append(T)
            log(f"  epoch N={N} seed={s}: T*={T} k*={k}")
        med = [t for t in Ts if t]
        per_N[N] = dict(t1=t1, T_star=Ts,
                        median=float(np.median(med)) if med else None)
    med = {n: v["median"] for n, v in per_N.items() if v["median"]}
    slope = (float(np.polyfit(np.log(list(med)), np.log(list(med.values())),
                              1)[0]) if len(med) >= 2 else None)
    return dict(per_N=per_N, slope=slope, prediction=-1.0)


def exp_certified(quick, full=False):
    T = 6_000 if quick else 30_000
    seeds = 4 if quick else 6
    grid = [0.15] if quick else [0.15, 0.18]
    out = {}
    for cb in grid:
        cached = ckpt_load("certified", f"cb{cb}")
        if cached is not None:
            out[str(cb)] = cached
            log(f"  certified cb={cb}: cached")
            continue
        TI, _, _ = picard(cb, n_mc=200_000)
        r = run_anytime([cb] * 8, 8, T, seeds, 11, agg="mean")
        TH0 = np.array(r.pop("_TH0"))
        d_star = np.linalg.norm(TH0 - TH_STAR[None], axis=(1, 2))
        d_inf = np.linalg.norm(TH0 - TI[None], axis=(1, 2))
        rec = dict(picard=float(np.linalg.norm(TI - TH_STAR)),
                   d_star=float(d_star.mean()), d_star_sem=_sem(d_star),
                   d_inf=float(d_inf.mean()), d_inf_sem=_sem(d_inf),
                   bonus_T=r["bonus_T"], traj_t=r["traj_t"],
                   traj_d=r["traj_d"])
        ckpt_save("certified", f"cb{cb}", rec)
        out[str(cb)] = rec
        log(f"  certified cb={cb}: ||th-th*||={rec['d_star']:.3f} "
            f"vs Picard {rec['picard']:.3f}, ||th-th_inf||={rec['d_inf']:.4f}")
    return out


def exp_baselines(quick, full=False):
    T = 5_000 if quick else 15_000
    seeds = 4 if quick else 6
    N = 8
    adv = np.zeros(N); adv[0] = 1.0
    res = {"adversary": {}, "floor": {}}
    plan_adv = [("oracle", dict(agg="oracle")), ("mean", dict(agg="mean")),
                ("trim", dict(agg="trim")), ("median", dict(agg="median")),
                ("dcesa_mv", dict(agg="gate", gate="mv", pbar=0.1, q=0.2))]
    for name, kw in plan_adv:
        cached = ckpt_load("baselines", f"adv_{name}")
        if cached is not None:
            res["adversary"][name] = cached
        else:
            r = run_anytime(adv, N, T, seeds, 21, **kw)
            rec = dict(reg=r["reg_mean"], sem=r["reg_sem"])
            ckpt_save("baselines", f"adv_{name}", rec)
            res["adversary"][name] = rec
            log(f"  baselines adv {name}: {rec['reg']:.0f}+-{rec['sem']:.0f}")
    floor = np.full(N, 0.04)
    plan_floor = [("oracle", dict(agg="oracle")), ("mean", dict(agg="mean")),
                  ("trim", dict(agg="trim")),
                  ("dcesa_mv", dict(agg="gate", gate="mv", pbar=0.1, q=0.0))]
    for name, kw in plan_floor:
        cached = ckpt_load("baselines", f"floor_{name}")
        if cached is not None:
            res["floor"][name] = cached
        else:
            r = run_anytime(floor, N, T, seeds, 41, **kw)
            rec = dict(reg=r["reg_mean"], sem=r["reg_sem"])
            ckpt_save("baselines", f"floor_{name}", rec)
            res["floor"][name] = rec
            log(f"  baselines floor {name}: {rec['reg']:.0f}+-{rec['sem']:.0f}")
    return res


def exp_gate_exponent(quick, full=False):
    T = 5_000 if quick else 15_000
    seeds = 4 if quick else 6
    N = 8
    adv = np.zeros(N); adv[0] = 1.0
    orc = ckpt_load("gate_exponent", "oracle_ref")
    if orc is None:
        r = run_anytime(adv, N, T, seeds, 21, agg="oracle")
        orc = dict(reg_clean=r["reg_clean_mean"])
        ckpt_save("gate_exponent", "oracle_ref", orc)
    pbars = [0.005, 0.02, 0.1] if quick else [0.002, 0.005, 0.02, 0.05, 0.1]
    res = {}
    for gname in ["mv", "perceptron", "ogd"]:
        for q in [0.0, 0.2]:
            pts = []
            for pb in pbars:
                key = f"{gname}_q{q}_p{pb}"
                cached = ckpt_load("gate_exponent", key)
                if cached is not None:
                    pts.append([pb, cached["excess"]])
                    continue
                r = run_anytime(adv, N, T, seeds, 31, agg="gate",
                                gate=gname, pbar=pb, q=q)
                ex = r["reg_clean_mean"] - orc["reg_clean"]
                ckpt_save("gate_exponent", key,
                          dict(excess=float(ex), sem=r["reg_clean_sem"],
                               onset=r["gate_onset"],
                               censored=r["gate_onset_censored"]))
                pts.append([pb, float(ex)])
                log(f"  gate {gname} q={q} p={pb}: clean-excess {ex:.1f}")
            # NO clamp: fitting log of a floored quantity biases the
            # exponent toward 0 exactly where the excess is small.
            pos = [(p, e) for p, e in pts if e > 0]
            exponent = None
            if len(pos) >= 2:
                exponent = float(np.polyfit(
                    np.log([p for p, _ in pos]),
                    np.log([e for _, e in pos]), 1)[0])
            res[f"{gname}_q{q}"] = dict(points=pts, n_positive=len(pos),
                                        n_points=len(pts), exponent=exponent)
    return res


def exp_mistakes(quick, full=False):
    T = 4_000 if quick else 15_000
    seeds = 4 if quick else 6
    N = 8
    adv = np.zeros(N); adv[0] = 1.0
    res = {}
    for gname in ["mv", "perceptron", "ogd"]:
        for q in ([0.0] if gname == "perceptron" else [0.0, 0.2]):
            key = f"{gname}_q{q}"
            cached = ckpt_load("mistakes", key)
            if cached is not None:
                res[key] = cached
                continue
            r = run_anytime(adv, N, T, seeds, 51, agg="gate", gate=gname,
                            pbar=0.05, q=q)
            rec = dict(mistaken_rounds=r["gate_mistaken_rounds"],
                       sem=r["gate_mistaken_sem"])
            ckpt_save("mistakes", key, rec)
            res[key] = rec
            log(f"  mistakes {key}: {rec['mistaken_rounds']:.0f} rounds")
    return res


def _scale(quick, full, a, b, c):
    """Pick (quick, paper, full) value."""
    return a if quick else (c if full else b)


# ------------------------------ Exp. 1 (fixed-point plateau against N)
def exp_fixedpoint(quick, full=False):
    T = _scale(quick, full, 4_000, 20_000, 20_000)
    seeds = _scale(quick, full, 3, 8, 30)
    Ns = [1, 4, 8] if quick else [1, 4, 8, 16]
    grid = [0.15] if quick else [0.15, 0.18]
    out = {}
    for cb in grid:
        TI, _, _ = picard(cb, n_mc=200_000)
        pic = float(np.linalg.norm(TI - TH_STAR))
        per_N = {}
        for N in Ns:
            key = f"cb{cb}_N{N}"
            cached = ckpt_load("fixedpoint", key)
            if cached is not None:
                per_N[N] = cached
                continue
            r = run_anytime([cb] * N, N, T, seeds, 61, agg="mean",
                            theta_inf=TI)
            TH0 = np.array(r.pop("_TH0"))
            d_star = np.linalg.norm(TH0 - TH_STAR[None], axis=(1, 2))
            d_inf = np.linalg.norm(TH0 - TI[None], axis=(1, 2))
            rec = dict(d_star=float(d_star.mean()), d_star_sem=_sem(d_star),
                       d_inf=float(d_inf.mean()), d_inf_sem=_sem(d_inf),
                       reg_mean=r["reg_mean"], reg_sem=r["reg_sem"],
                       reg_slope=loglog_slope(r["traj_t"], r["traj_reg"]),
                       t_behav=r.get("t_behav"),
                       traj_t=r["traj_t"], traj_d=r["traj_d"])
            ckpt_save("fixedpoint", key, rec)
            per_N[N] = rec
            log(f"  fixedpoint cb={cb} N={N}: ||th-th*||={rec['d_star']:.3f} "
                f"(Picard {pic:.3f}) ||th-th_inf||={rec['d_inf']:.4f} "
                f"reg slope {rec['reg_slope']}")
        out[str(cb)] = dict(picard=pic, per_N=per_N)
    return out


# ------------ Exp. 5 (behavioural collapse not reachable; was old Exp. 2)
def exp_collapse(quick, full=False, gamma_dec=0.01, Delta_dec=0.01):
    cb = 0.15
    T = _scale(quick, full, 3_000, 15_000, 20_000)
    seeds = _scale(quick, full, 3, 6, 30)
    Ns = ([2, 4, 8] if quick else
          ([2, 4, 8, 16, 32] if not full else [2, 4, 8, 16, 32, 64, 128]))
    TI, _, _ = picard(cb, n_mc=200_000)
    per_N = {}
    for N in Ns:
        key = f"N{N}"
        cached = ckpt_load("collapse", key)
        if cached is not None:
            per_N[N] = cached
            continue
        r = run_anytime([cb] * N, N, T, seeds, 71, agg="mean", theta_inf=TI,
                        gamma_dec=gamma_dec, Delta_dec=Delta_dec,
                        record_every=max(50, T // 60))
        TH0 = np.array(r.pop("_TH0"))
        d_inf = np.linalg.norm(TH0 - TI[None], axis=(1, 2))
        # estimation-side collapse: first recorded t within 25% of ||delta||
        thr = 0.25 * float(np.linalg.norm(TI - TH_STAR))
        t_est = None
        for tt, dd in zip(r["traj_t"], r["traj_d"]):
            if abs(dd - np.linalg.norm(TI - TH_STAR)) <= thr:
                t_est = tt
                break
        rec = dict(err_T=float(d_inf.mean()), err_T_sem=_sem(d_inf),
                   t_est=t_est, t_behav=r.get("t_behav"),
                   behav_frac_final=float(np.mean(r["traj_behav"][-5:]))
                   if r.get("traj_behav") else None,
                   behav_censored=bool(r.get("t_behav") is None),
                   bonus_T=r["bonus_T"], gamma_over_3=gamma_dec / 3.0,
                   reg_mean=r["reg_mean"])
        ckpt_save("collapse", key, rec)
        per_N[N] = rec
        log(f"  collapse N={N}: err={rec['err_T']:.4f} t_est={t_est} "
            f"t_behav={rec['t_behav']} frac={rec['behav_frac_final']} "
            f"bonus_T={rec['bonus_T']:.3f} vs gamma/3={gamma_dec/3:.4f}")

    def _slope(field):
        pts = [(n, v[field]) for n, v in per_N.items() if v.get(field)]
        if len(pts) < 2:
            return None
        return float(np.polyfit(np.log([p[0] for p in pts]),
                                np.log([p[1] for p in pts]), 1)[0])
    n_cens = sum(1 for v in per_N.values() if v.get("behav_censored"))
    return dict(per_N=per_N, slope_err=_slope("err_T"),
                slope_t_est=_slope("t_est"),
                slope_t_behav=(None if n_cens else _slope("t_behav")),
                # partial fit over the N where collapse WAS reached: the
                # anytime collapse time is near-flat in N, against the epoch
                # variant's exact -1, which is the Remark-noN separation
                slope_t_behav_uncensored=_slope("t_behav"),
                behav_censored_count=n_cens,
                behav_note=("behavioural collapse is censored at small N and "
                            "reached at large N; where it is reached its time "
                            "is near-flat in N, because the anytime bonus "
                            "carries no 1/N speedup (Remark noN).  No slope "
                            "is fitted over all N; see "
                            "slope_t_behav_uncensored"),
                prediction=dict(err="-0.5", t_est="-1 to 0",
                                t_behav="0 (Remark noN: no 1/N for anytime)"))


# --------- Exp. 2 (regret vs spectral gap; was old Exps. 3 and 7, N = 20)
def exp_gap(quick, full=False):
    N = 8 if quick else 20
    T = _scale(quick, full, 3_000, 20_000, 50_000)
    seeds = _scale(quick, full, 3, 6, 30)
    cb = 0.4                       # matches the original Exp. 3 bias level
    cbc = 0.15                     # certified band
    fam = circulant_family(N)
    if quick:
        fam = [fam[0], fam[-1]]
    res = {"N": N, "T": T, "c_beta": cb, "points": []}
    for (k, W, gam) in fam:
        key = f"k{k}"
        cached = ckpt_load("gap", key)
        if cached is not None and "naive_adv" in cached:
            res["points"].append(cached)
            continue
        if cached is not None:
            # top up an older checkpoint with the like-for-like naive run
            adv0 = np.full(N, 0.04)
            adv0[:max(1, N // 8)] = 1.0
            rna = run_anytime(adv0, N, T, seeds, 85, agg="gossip", W=W)
            cached["naive_adv"] = rna["reg_mean"]
            cached["naive_adv_sem"] = rna["reg_sem"]
            cached["naive_adv_clean"] = rna["reg_clean_mean"]
            ckpt_save("gap", key, cached)
            res["points"].append(cached)
            log(f"  gap k={k}: naive_adv {cached['naive_adv']:.0f}")
            continue
        row = dict(k=k, **spectral_data(W))
        r = run_anytime([cb] * N, N, T, seeds, 81, agg="gossip", W=W)
        row["naive"] = r["reg_mean"]
        row["naive_sem"] = r["reg_sem"]
        row["naive_d_star"] = r["d_star"]
        rc = run_anytime([cbc] * N, N, T, seeds, 82, agg="gossip", W=W)
        row["naive_certified"] = rc["reg_mean"]
        row["naive_certified_sem"] = rc["reg_sem"]
        # within-tolerance majority plus a large-bias minority: the regime
        # where trust gating can help at all.  A baseline bias above
        # EPS_TOL would mark EVERY agent dirty and degenerate the gate.
        adv = np.full(N, 0.04)
        adv[:max(1, N // 8)] = 1.0            # large-bias minority
        rg = run_anytime(adv, N, T, seeds, 83, agg="gate", gate="mv",
                         pbar=0.1, q=0.2, W=W)
        row["dcesa"] = rg["reg_mean"]
        row["dcesa_sem"] = rg["reg_sem"]
        row["dcesa_clean"] = rg["reg_clean_mean"]
        ri = run_anytime([cb] * N, N, T, seeds, 84, agg="isolated")
        row["isolated"] = ri["reg_mean"]
        row["isolated_sem"] = ri["reg_sem"]
        # like-for-like naive baseline on the SAME adversarial bias vector
        rna = run_anytime(adv, N, T, seeds, 85, agg="gossip", W=W)
        row["naive_adv"] = rna["reg_mean"]
        row["naive_adv_sem"] = rna["reg_sem"]
        row["naive_adv_clean"] = rna["reg_clean_mean"]
        ckpt_save("gap", key, row)
        res["points"].append(row)
        log(f"  gap k={k} gamma={gam:.3f}: naive {row['naive']:.0f} "
            f"iso {row['isolated']:.0f} dcesa {row['dcesa']:.0f}")

    g = np.array([p["gamma"] for p in res["points"]])
    fits = {}
    for name, x in [("inv_gamma", 1.0 / g), ("inv_sqrt_gamma", 1.0 / np.sqrt(g)),
                    ("log_inv_gamma", np.log(1.0 / g))]:
        for tgt in ("naive", "dcesa"):
            y = np.array([p[tgt] for p in res["points"]])
            if len(g) < 3:
                continue
            Amat = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
            pred = Amat @ coef
            ss = 1.0 - ((y - pred) ** 2).sum() / max(
                ((y - y.mean()) ** 2).sum(), 1e-12)
            fits[f"{tgt}_{name}"] = dict(R2=float(ss), coef=coef.tolist())
    res["fits"] = fits
    if len(g) >= 3:
        for tgt in ("naive", "dcesa"):
            y = np.array([p[tgt] for p in res["points"]])
            res[f"{tgt}_flatness"] = float(y.std(ddof=1) / max(y.mean(), 1e-9))
    return res


# ------------- Exp. 3 (certification propagation; was old Exp. 5)
def exp_prop(quick, full=False):
    N = 20
    W = gossip_matrix(N, "cycle")
    pbar, S = 0.1, [0]
    Ks = [1, 2, 5, 10, 20, 50, 100, 200]
    src = np.zeros(N)
    src[S] = pbar
    rows = []
    Wk = np.eye(N)
    prev = 0
    for Kr in Ks:
        Wk = Wk @ np.linalg.matrix_power(W, Kr - prev)
        prev = Kr
        eff = Wk @ src
        rows.append(dict(K=Kr, p_eff_antipodal=float(eff[N // 2]),
                         p_eff_min=float(eff.min()),
                         p_eff_max=float(eff.max())))
    lam = spectral_data(W)
    return dict(N=N, pbar=pbar, uniform_limit=pbar * len(S) / N,
                lemma_bound=[dict(K=r["K"],
                                  lower=float(max(pbar * (len(S) / N -
                                                  np.sqrt(N) *
                                                  lam["lam_star"] ** r["K"]),
                                                  0.0)))
                             for r in rows],
                spectral=lam, rows=rows)


# ------------- Exp. 4 (measured gate onset; replaces old Exp. 6)
def exp_gate_onset(quick, full=False):
    T = _scale(quick, full, 4_000, 15_000, 30_000)
    seeds = _scale(quick, full, 3, 6, 30)
    N = 8
    adv = np.zeros(N)
    adv[0] = 1.0
    pbars = [0.005, 0.05] if quick else [0.002, 0.005, 0.02, 0.05, 0.1]
    res = {}
    for gname in ["mv", "ogd"] if quick else ["mv", "perceptron", "ogd"]:
        for q in [0.0, 0.2]:
            pts, cens = [], []
            for pb in pbars:
                key = f"{gname}_q{q}_p{pb}"
                cached = ckpt_load("gate_onset", key)
                if cached is None:
                    r = run_anytime(adv, N, T, seeds, 91, agg="gate",
                                    gate=gname, pbar=pb, q=q)
                    cached = dict(onset=r["gate_onset"],
                                  onset_mean=r["gate_onset_mean"],
                                  sem=r["gate_onset_sem"],
                                  censored=r["gate_onset_censored"],
                                  mistakes=r["gate_mistaken_rounds"])
                    ckpt_save("gate_onset", key, cached)
                    log(f"  onset {gname} q={q} p={pb}: "
                        f"t={cached['onset']:.0f} "
                        f"censored={cached['censored']:.2f}")
                pts.append([pb, cached["onset"]])
                cens.append(cached["censored"])
            ok = [(p, o) for (p, o), c in zip(pts, cens) if c == 0.0 and o > 0]
            slope = (float(np.polyfit(np.log([p for p, _ in ok]),
                                      np.log([o for _, o in ok]), 1)[0])
                     if len(ok) >= 2 else None)
            res[f"{gname}_q{q}"] = dict(points=pts, censored=cens,
                                        slope=slope, n_uncensored=len(ok))
    res["prediction"] = -1.0        # t_gate ~ 1/(pbar (1-2q)^2)
    return res


DRIVERS = dict(certify=exp_certify, epoch=exp_epoch, certified=exp_certified,
               baselines=exp_baselines, gate_exponent=exp_gate_exponent,
               mistakes=exp_mistakes,
               # ports of the original Exps. 1-7 onto this generator
               fixedpoint=exp_fixedpoint, collapse=exp_collapse,
               gap=exp_gap, prop=exp_prop, gate_onset=exp_gate_onset)

# Diagnostics: runnable on demand via --only, but not part of a default or
# --paper run, because they are not reported results in the paper.
DIAGNOSTIC = ("gate_exponent",)
DEFAULT_DRIVERS = [n for n in DRIVERS if n not in DIAGNOSTIC]


# ===================================================================== FIGURES
def make_figures(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  (figures skipped: {e})")
        return
    os.makedirs(_p("figures", "diagnostic"), exist_ok=True)

    if "certify" in summary:
        rows = summary["certify"]["rows"]
        cb = [r["c_beta"] for r in rows]
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.plot(cb, [r["rho"] for r in rows], "o-", c=WONG[5],
                label=r"$\varrho$")
        ax.plot(cb, [r["kappa_tilde"] for r in rows], "s-", c=WONG[6],
                label=r"$\tilde\kappa$")
        ax.axhline(1, ls=":", c="gray"); ax.set_yscale("log")
        ax.set_xlabel(r"$c_\beta$"); ax.legend()
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "certify.pdf"))
        plt.close(fig)

    if "epoch" in summary and summary["epoch"]["slope"] is not None:
        med = {int(n): v["median"] for n, v in
               summary["epoch"]["per_N"].items() if v["median"]}
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.loglog(list(med), list(med.values()), "o-", c=WONG[5])
        ax.set_xlabel("N"); ax.set_ylabel(r"$T^\star$ (median)")
        ax.set_title(f"slope {summary['epoch']['slope']:.2f} (pred $-1$)")
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "epoch.pdf"))
        plt.close(fig)

    if "certified" in summary:
        fig, ax = plt.subplots(figsize=(4.2, 3))
        for i, (cb, v) in enumerate(summary["certified"].items()):
            ax.plot(v["traj_t"], v["traj_d"], c=WONG[5 + i % 3],
                    label=rf"$c_\beta={cb}$")
            ax.axhline(v["picard"], ls="--", c=WONG[5 + i % 3], lw=1)
        ax.set_xlabel("t"); ax.set_ylabel(r"$\|\hat\theta-\theta^*\|$")
        ax.legend()
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "certified.pdf"))
        plt.close(fig)

    if "baselines" in summary:
        adv = summary["baselines"]["adversary"]
        ks = list(adv)
        fig, ax = plt.subplots(figsize=(4.6, 3))
        ax.bar(range(len(ks)), [adv[k]["reg"] for k in ks],
               yerr=[adv[k]["sem"] for k in ks], color=WONG[1:1 + len(ks)])
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=20)
        ax.set_ylabel("latent regret")
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "baselines.pdf"))
        plt.close(fig)

    if "gate_exponent" in summary:
        fig, ax = plt.subplots(figsize=(4.6, 3))
        for i, (k, v) in enumerate(summary["gate_exponent"].items()):
            pp = [(p, e) for p, e in v["points"] if e > 0]
            if not pp:
                continue
            lab = ("n/a" if v.get("exponent") is None
                   else f"{v['exponent']:.2f}")
            ax.loglog(*zip(*pp), "o-", c=WONG[i % 8],
                      label=f"{k}: {lab} ({v['n_positive']}/{v['n_points']})")
        ax.set_xlabel(r"$\bar p$"); ax.set_ylabel("clean-agent excess")
        ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "gate_exponent.pdf"))
        plt.close(fig)

    if "fixedpoint" in summary:
        fig, ax = plt.subplots(figsize=(4.6, 3))
        for i, (cb, v) in enumerate(summary["fixedpoint"].items()):
            ns = sorted(int(n) for n in v["per_N"])
            ax.errorbar(ns, [v["per_N"][str(n)]["d_star"] if str(n) in v["per_N"]
                             else v["per_N"][n]["d_star"] for n in ns],
                        yerr=[v["per_N"][str(n)]["d_star_sem"] if str(n) in v["per_N"]
                              else v["per_N"][n]["d_star_sem"] for n in ns],
                        fmt="o-", c=WONG[5 + i % 3], label=rf"$c_\beta={cb}$")
            ax.axhline(v["picard"], ls="--", lw=1, c=WONG[5 + i % 3])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N"); ax.set_ylabel(r"$\|\hat\theta-\theta^*\|$")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "fixedpoint.pdf"))
        plt.close(fig)

    if "collapse" in summary:
        per = summary["collapse"]["per_N"]
        ns = sorted(int(n) for n in per)
        get = lambda n, f: (per[str(n)] if str(n) in per else per[n]).get(f)
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
        ax[0].loglog(ns, [get(n, "err_T") for n in ns], "o-", c=WONG[5])
        ax[0].set_xlabel("N"); ax[0].set_ylabel(r"$\|\hat\theta-\theta_\infty\|$")
        ax[0].set_title(f"slope {summary['collapse']['slope_err']}")
        for f, c, lab in [("t_est", WONG[1], "estimation"),
                          ("t_behav", WONG[6], "behavioural")]:
            pts = [(n, get(n, f)) for n in ns if get(n, f)]
            if pts:
                ax[1].loglog(*zip(*pts), "o-", c=c, label=lab)
        ax[1].set_xlabel("N"); ax[1].set_ylabel("collapse time")
        ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "collapse.pdf"))
        plt.close(fig)

    if "gap" in summary:
        pts = summary["gap"]["points"]
        g = [p["gamma"] for p in pts]
        fig, ax = plt.subplots(figsize=(4.6, 3))
        for f, c, lab in [("naive", WONG[6], "naive coop (common bias)"),
                          ("isolated", WONG[2], "isolated (common bias)"),
                          ("naive_adv", WONG[1], "naive coop (adversary)"),
                          ("dcesa", WONG[5], "D-CESA mv (adversary)")]:
            if f not in pts[0]:
                continue
            ax.errorbar(g, [p[f] for p in pts],
                        yerr=[p.get(f + "_sem", 0) for p in pts],
                        fmt="o-", c=c, label=lab)
        ax.set_xlabel(r"$\gamma(W)$"); ax.set_ylabel("final latent regret")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "gap.pdf"))
        plt.close(fig)

    if "prop" in summary:
        r = summary["prop"]
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.semilogx([x["K"] for x in r["rows"]],
                    [x["p_eff_antipodal"] for x in r["rows"]], "o-", c=WONG[5],
                    label="antipodal agent")
        ax.axhline(r["uniform_limit"], ls="--", c="gray",
                   label=r"$\bar p|S|/N$")
        ax.set_xlabel("gossip rounds K"); ax.set_ylabel(r"$p^{eff}$")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "prop.pdf"))
        plt.close(fig)

    if "gate_onset" in summary:
        fig, ax = plt.subplots(figsize=(4.6, 3))
        i = 0
        for k, v in summary["gate_onset"].items():
            if not isinstance(v, dict):
                continue
            pp = [(p, o) for p, o in v["points"] if o > 0]
            if pp:
                ax.loglog(*zip(*pp), "o-", c=WONG[i % 8],
                          label=f"{k}: {v['slope']}")
            i += 1
        ax.set_xlabel(r"$\bar p$"); ax.set_ylabel("measured gate onset")
        ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(_p("figures", "diagnostic", "gate_onset.pdf"))
        plt.close(fig)

    log("  diagnostic figures written to figures/diagnostic/")


# ======================================================================= MAIN
def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="Sec. recon scale")
    ap.add_argument("--full", action="store_true",
                    help="original Exps. 1-7 scale (30 seeds, long horizons)")
    ap.add_argument("--quick", action="store_true", help="CI scale (default)")
    ap.add_argument("--only", default="", help="comma list of experiments")
    ap.add_argument("--outdir", default="out_v4")
    ap.add_argument("--force", action="store_true",
                    help="ignore existing results/ and checkpoints/")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    OUTDIR = a.outdir
    quick = not (a.paper or a.full)
    full = a.full
    names = ([n.strip() for n in a.only.split(",") if n.strip()]
             or list(DEFAULT_DRIVERS))
    unknown = [n for n in names if n not in DRIVERS]
    if unknown:
        sys.exit(f"unknown experiments: {unknown}; choices {list(DRIVERS)}")

    if a.force:
        import shutil
        for sub in ("results", "checkpoints"):
            shutil.rmtree(_p(sub), ignore_errors=True)

    os.makedirs(_p("results"), exist_ok=True)
    log(f"=== reproduce_v35  mode={'quick' if quick else ('full' if full else 'paper')}  "
        f"outdir={OUTDIR}  experiments={names} ===")
    t_all = time.time()
    summary = {}
    for n in names:
        if result_done(n) and not a.force:
            summary[n] = result_load(n)
            log(f"[{n}] already complete -> loaded (use --force to redo)")
            continue
        t0 = time.time()
        log(f"[{n}] running")
        res = DRIVERS[n](quick, full)
        result_save(n, res)
        summary[n] = res
        log(f"[{n}] done in {time.time() - t0:.0f}s -> results/{n}.json")

    # merge into any existing summary: a partial run (--only) must not drop
    # the experiments it did not touch
    merged = {}
    sp = _p("results", "summary.json")
    if os.path.exists(sp) and not a.force:
        try:
            with open(sp) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(summary)
    _atomic_json(sp, merged)
    summary = merged
    if not a.no_figures:
        make_figures(summary)                 # quick diagnostics
        try:                                  # publication figures (C7)
            import make_figures as _pub
            _pub.build_all(OUTDIR)
        except Exception as e:
            log(f"  (publication figures skipped: {e})")
    log(f"=== ALL DONE in {time.time() - t_all:.0f}s ; "
        f"results/summary.json, figures/, checkpoints/ under {OUTDIR} ===")

    # one-line headline table
    def g(path, default="?"):
        cur = summary
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur if cur != {} else default
    log("HEADLINE:")
    if "epoch" in summary:
        log(f"  epoch slope         = {summary['epoch']['slope']} (pred -1)")
    if "baselines" in summary:
        adv = summary["baselines"]["adversary"]
        log("  adversary regret    = " + ", ".join(
            f"{k}:{v['reg']:.0f}" for k, v in adv.items()))
        log("  tolerance floor     = " + ", ".join(
            f"{k}:{v['reg']:.0f}+-{v['sem']:.0f}"
            for k, v in summary["baselines"]["floor"].items()))
    if "gate_exponent" in summary:
        log("  gate exponents      = " + ", ".join(
            f"{k}:{v['exponent']} [{v['n_positive']}/{v['n_points']} "
            f"points positive]"
            for k, v in summary["gate_exponent"].items()))
    if "gate_onset" in summary:
        log("  gate onset slopes   = " + ", ".join(
            f"{k}:{v['slope']}" for k, v in summary["gate_onset"].items()
            if isinstance(v, dict)))
    if "collapse" in summary:
        c = summary["collapse"]
        log(f"  collapse slopes     = err {c['slope_err']}, "
            f"t_est {c['slope_t_est']}, t_behav {c['slope_t_behav']} "
            f"(behavioural prediction 0)")
    if "gap" in summary:
        f = summary["gap"]["fits"]
        log("  gap fits (R2)       = " + ", ".join(
            f"{k}:{v['R2']:.3f}" for k, v in f.items()))
        log(f"  naive flatness      = {summary['gap'].get('naive_flatness')}")
    if "fixedpoint" in summary:
        for cb, v in summary["fixedpoint"].items():
            log(f"  fixedpoint cb={cb}   Picard {v['picard']:.3f} vs " + ", ".join(
                f"N={n}:{r['d_star']:.3f}" for n, r in v["per_N"].items()))


if __name__ == "__main__":
    main()