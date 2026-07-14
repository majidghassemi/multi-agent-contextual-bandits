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
        self.mist += w.sum((1, 2))


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
                gate="mv", pbar=0.1, q=0.0, record_every=250, inv_every=5):
    """Anytime cooperative LinUCB, complete graph, 1/N convention.
    agg in {isolated, mean, oracle, trim, median, gate}."""
    rng = np.random.default_rng(seed0)
    Sd, N = seeds, n_agents
    bias_v = np.asarray(bias_mag, float)
    clean = (np.abs(bias_v) <= EPS_TOL).astype(float)
    cmask = clean > 0.5

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
    traj_t, traj_d = [], []

    for t in range(1, T + 1):
        X = contexts(rng, Sd * N).reshape(Sd, N, D_C)
        sc = np.einsum('snd,snkd->snk', X, TH)
        bon = np.sqrt(np.maximum(
            np.einsum('snd,snkde,sne->snk', X, Ainv, X), 0.0))
        act = (sc + _alpha(t) * bon).argmax(2)
        lat = np.einsum('snd,kd->snk', X, TH_STAR)
        inst_reg = (lat.max(2)
                    - np.take_along_axis(lat, act[:, :, None], 2)[:, :, 0])
        reg += inst_reg.sum(1)
        reg_clean += inst_reg[:, cmask].sum(1)
        y = (np.take_along_axis(lat, act[:, :, None], 2)[:, :, 0]
             + bias_v[None, :] * (act == A1)
             + SIGMA * rng.standard_normal((Sd, N)))

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
            gate_obj.tally(clean)

        dA = np.zeros((Sd, N, K, D_C, D_C))
        db = np.zeros((Sd, N, K, D_C))
        for a in range(K):
            m = (act == a)
            xa = X * m[:, :, None]
            dAj = np.einsum('snd,sne->snde', xa, xa)
            dbj = xa * (y * m)[:, :, None]
            if agg == "isolated":
                dA[:, :, a] = dAj
                db[:, :, a] = dbj
                continue
            if agg in ("trim", "median"):
                if agg == "trim":
                    sA = np.sort(dAj, axis=1)[:, 1:-1].mean(1)
                    sB = np.sort(dbj, axis=1)[:, 1:-1].mean(1)
                else:
                    sA = np.median(dAj, axis=1)
                    sB = np.median(dbj, axis=1)
                dA[:, :, a] = sA[:, None]
                db[:, :, a] = sB[:, None]
                continue
            if agg == "mean":
                w = np.full((Sd, N, N), 1.0 / N)
            elif agg == "oracle":
                w = np.tile(clean / clean.sum(), (Sd, N, 1))
            elif agg == "gate":
                g = gate_obj.trusted().astype(float)
                w = g / g.sum(2, keepdims=True)
            else:
                raise ValueError(agg)
            dA[:, :, a] = np.einsum('sij,sjde->side', w, dAj)
            db[:, :, a] = np.einsum('sij,sjd->sid', w, dbj)
        A += dA
        b += db
        if t % inv_every == 0 or t < 50:
            Ainv = np.linalg.inv(A)
        TH = np.einsum('snkij,snkj->snki', Ainv, b)
        if t % record_every == 0:
            traj_t.append(t)
            traj_d.append(float(np.linalg.norm(
                TH[:, 0] - TH_STAR[None], axis=(1, 2)).mean()))

    out = dict(reg=reg.tolist(), reg_mean=float(reg.mean()),
               reg_sem=_sem(reg), reg_clean_mean=float(reg_clean.mean()),
               reg_clean_sem=_sem(reg_clean),
               d_star=float(np.linalg.norm(
                   TH[:, 0] - TH_STAR[None], axis=(1, 2)).mean()),
               traj_t=traj_t, traj_d=traj_d,
               bonus_T=float(2 * np.sqrt(2) * _alpha(T) / np.sqrt(T * 0.02)))
    if gate_obj is not None:
        out["gate_mistaken_rounds"] = float(gate_obj.mist.mean())
        out["gate_mistaken_sem"] = _sem(gate_obj.mist)
    out["_TH0"] = TH[:, 0].tolist()          # for fixed-point comparison
    return out


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
def exp_certify(quick):
    grid = [0.15, 0.4] if quick else [0.05, 0.1, 0.15, 0.18, 0.4]
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
        rho = contraction_modulus(cb, TI, n_mc=n_mc)
        # sampled sup is one-sided: take the MAX over independent batches
        cc = 0.0
        for batch in range(3):
            c_b, _ = tracking_modulus(cb, TI, sm_uniform, n_dirs=100,
                                      eps_list=(0.005, 0.01, 0.03, 0.08),
                                      radial=30, n_mc=n_mc,
                                      seed=100 + batch)
            cc = max(cc, c_b)
        kt = 2.0 * cc / sm_uniform
        md, mc = region_masses(TI)
        row = dict(c_beta=cb, delta_norm=float(np.linalg.norm(TI - TH_STAR)),
                   picard_res=res, sigma_min=sm,
                   sigma_min_uniform=float(sm_uniform), rho=float(rho),
                   c_circ=float(cc), kappa_tilde=float(kt), mu_dis=md,
                   mu_dec=mc, contraction_ok=bool(rho < 1),
                   tracking_ok=bool(kt < 1))
        ckpt_save("certify", f"cb{cb}", row)
        rows.append(row)
        log(f"  certify cb={cb}: rho={rho:.2f} kappa~={kt:.2f} "
            f"mu_dec={mc:.4f}")
    return dict(rows=rows)


def exp_epoch(quick):
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


def exp_certified(quick):
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


def exp_baselines(quick):
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
            res["floor"][name] = cached["reg"]
        else:
            r = run_anytime(floor, N, T, seeds, 41, **kw)
            ckpt_save("baselines", f"floor_{name}", dict(reg=r["reg_mean"]))
            res["floor"][name] = r["reg_mean"]
            log(f"  baselines floor {name}: {r['reg_mean']:.0f}")
    return res


def exp_gate_exponent(quick):
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
                ex = max(r["reg_clean_mean"] - orc["reg_clean"], 0.5)
                ckpt_save("gate_exponent", key, dict(excess=ex))
                pts.append([pb, ex])
                log(f"  gate {gname} q={q} p={pb}: clean-excess {ex:.1f}")
            lp = np.log([p for p, _ in pts])
            le = np.log([e for _, e in pts])
            res[f"{gname}_q{q}"] = dict(points=pts,
                                        exponent=float(np.polyfit(lp, le, 1)[0]))
    return res


def exp_random(quick):
    n_draws = 40 if quick else 300
    cb = 0.4
    vals = []
    for s in range(n_draws):
        cached = ckpt_load("random", f"draw{s}")
        if cached is not None:
            if cached["ok"]:
                vals.append((cached["mu_dis"], cached["mu_dec"]))
            continue
        ths = theta_star_random(s)
        try:
            TI, res_, ok = picard(cb, th_star=ths, iters=50, n_mc=100_000,
                                  seed=s)
            if not ok:
                ckpt_save("random", f"draw{s}", dict(ok=False))
                continue
            md, mc = region_masses(TI, th_star=ths, n_mc=100_000, seed=s)
        except np.linalg.LinAlgError:
            ckpt_save("random", f"draw{s}", dict(ok=False))
            continue
        ckpt_save("random", f"draw{s}", dict(ok=True, mu_dis=md, mu_dec=mc))
        vals.append((md, mc))
    vd = np.array([v[0] for v in vals])
    vc = np.array([v[1] for v in vals])
    return dict(n=len(vd),
                mu_dis=dict(median=float(np.median(vd)),
                            q90=float(np.quantile(vd, .9)),
                            max=float(vd.max())),
                mu_dec=dict(median=float(np.median(vc)),
                            q90=float(np.quantile(vc, .9)),
                            max=float(vc.max())),
                frac_ge_0p4=float((vd >= 0.4).mean()),
                pickled_value=0.430)


def exp_waiting(quick):
    rng = np.random.default_rng(7)
    n_edges = 56
    seeds = 50 if quick else 400
    res = {}
    for pb in [0.002, 0.005, 0.02, 0.05, 0.1]:
        cached = ckpt_load("waiting", f"p{pb}")
        if cached is not None:
            res[str(pb)] = cached
            continue
        tmax = [np.ceil(np.log(rng.random(n_edges)) / np.log(1 - pb)).max()
                for _ in range(seeds)]
        rec = dict(median=float(np.median(tmax)),
                   pred=float(np.log(n_edges) / pb))
        ckpt_save("waiting", f"p{pb}", rec)
        res[str(pb)] = rec
    lp = np.log([float(k) for k in res])
    lm = np.log([v["median"] for v in res.values()])
    return dict(per_pbar=res, slope=float(np.polyfit(lp, lm, 1)[0]),
                prediction=-1.0)


def exp_mistakes(quick):
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


DRIVERS = dict(certify=exp_certify, epoch=exp_epoch, certified=exp_certified,
               baselines=exp_baselines, gate_exponent=exp_gate_exponent,
               random=exp_random, waiting=exp_waiting, mistakes=exp_mistakes)


# ===================================================================== FIGURES
def make_figures(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  (figures skipped: {e})")
        return
    os.makedirs(_p("figures"), exist_ok=True)

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
        fig.tight_layout(); fig.savefig(_p("figures", "certify.pdf"))
        plt.close(fig)

    if "epoch" in summary and summary["epoch"]["slope"] is not None:
        med = {int(n): v["median"] for n, v in
               summary["epoch"]["per_N"].items() if v["median"]}
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.loglog(list(med), list(med.values()), "o-", c=WONG[5])
        ax.set_xlabel("N"); ax.set_ylabel(r"$T^\star$ (median)")
        ax.set_title(f"slope {summary['epoch']['slope']:.2f} (pred $-1$)")
        fig.tight_layout(); fig.savefig(_p("figures", "epoch.pdf"))
        plt.close(fig)

    if "certified" in summary:
        fig, ax = plt.subplots(figsize=(4.2, 3))
        for i, (cb, v) in enumerate(summary["certified"].items()):
            ax.plot(v["traj_t"], v["traj_d"], c=WONG[5 + i % 3],
                    label=rf"$c_\beta={cb}$")
            ax.axhline(v["picard"], ls="--", c=WONG[5 + i % 3], lw=1)
        ax.set_xlabel("t"); ax.set_ylabel(r"$\|\hat\theta-\theta^*\|$")
        ax.legend()
        fig.tight_layout(); fig.savefig(_p("figures", "certified.pdf"))
        plt.close(fig)

    if "baselines" in summary:
        adv = summary["baselines"]["adversary"]
        ks = list(adv)
        fig, ax = plt.subplots(figsize=(4.6, 3))
        ax.bar(range(len(ks)), [adv[k]["reg"] for k in ks],
               yerr=[adv[k]["sem"] for k in ks], color=WONG[1:1 + len(ks)])
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=20)
        ax.set_ylabel("latent regret")
        fig.tight_layout(); fig.savefig(_p("figures", "baselines.pdf"))
        plt.close(fig)

    if "gate_exponent" in summary:
        fig, ax = plt.subplots(figsize=(4.6, 3))
        for i, (k, v) in enumerate(summary["gate_exponent"].items()):
            ax.loglog(*zip(*v["points"]), "o-", c=WONG[i % 8],
                      label=f"{k}: {v['exponent']:.2f}")
        ax.set_xlabel(r"$\bar p$"); ax.set_ylabel("clean-agent excess")
        ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(_p("figures", "gate_exponent.pdf"))
        plt.close(fig)

    if "random" in summary:
        r = summary["random"]
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.axvline(r["pickled_value"], c=WONG[6], ls="--",
                   label=f"pickled {r['pickled_value']}")
        ax.axvline(r["mu_dis"]["max"], c=WONG[2],
                   label=f"observed max {r['mu_dis']['max']:.3f}")
        ax.set_xlabel(r"$\mu(\tilde a\neq a^*)$"); ax.legend()
        fig.tight_layout(); fig.savefig(_p("figures", "random.pdf"))
        plt.close(fig)
    log("  figures written")


# ======================================================================= MAIN
def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="Sec. recon scale")
    ap.add_argument("--quick", action="store_true", help="CI scale (default)")
    ap.add_argument("--only", default="", help="comma list of experiments")
    ap.add_argument("--outdir", default="out_v34")
    ap.add_argument("--force", action="store_true",
                    help="ignore existing results/ and checkpoints/")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    OUTDIR = a.outdir
    quick = not a.paper
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(DRIVERS)
    unknown = [n for n in names if n not in DRIVERS]
    if unknown:
        sys.exit(f"unknown experiments: {unknown}; choices {list(DRIVERS)}")

    if a.force:
        import shutil
        for sub in ("results", "checkpoints"):
            shutil.rmtree(_p(sub), ignore_errors=True)

    os.makedirs(_p("results"), exist_ok=True)
    log(f"=== reproduce_v34  mode={'quick' if quick else 'paper'}  "
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
        res = DRIVERS[n](quick)
        result_save(n, res)
        summary[n] = res
        log(f"[{n}] done in {time.time() - t0:.0f}s -> results/{n}.json")

    _atomic_json(_p("results", "summary.json"), summary)
    if not a.no_figures:
        make_figures(summary)
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
            f"{k}:{v:.0f}" for k, v in summary["baselines"]["floor"].items()))
    if "gate_exponent" in summary:
        log("  gate exponents      = " + ", ".join(
            f"{k}:{v['exponent']:.2f}"
            for k, v in summary["gate_exponent"].items()))
    if "random" in summary:
        log(f"  random mu_dis max   = {summary['random']['mu_dis']['max']:.3f}"
            f" (pickled {summary['random']['pickled_value']})")
    if "waiting" in summary:
        log(f"  waiting-time slope  = {summary['waiting']['slope']:.2f}")


if __name__ == "__main__":
    main()