# When Feedback Fails and Audits Suffice

Reproduction package for the CEC / D-CESA paper (`paper_v4.tex`).
Pure numpy + matplotlib. No GPU, no environment, no training.

## Reproducing every number in the paper

```bash
python3 reproduce_v35.py --paper --outdir out_v4
```

That single command runs every experiment the paper reports, writes one
JSON per experiment plus a combined `summary.json` under
`out_v4/results/`, and renders every figure under `out_v4/figures/`.
**Every numeral in Sections 8, 9 and Appendix D traces to a value in
`out_v4/results/summary.json`, which is committed to this repository**,
so the paper is checkable without re-running anything.

Other scales:

```bash
python3 reproduce_v35.py --quick    # CI scale, ~30 s, 2-point sweeps
python3 reproduce_v35.py --full     # 30 seeds, long horizons
python3 reproduce_v35.py --only gap,prop        # a subset
python3 reproduce_v35.py --force    # ignore cached checkpoints
```

Runs are checkpointed per experiment under `out_v4/checkpoints/`, so an
interrupted run resumes where it stopped. Use `--force` after changing
any code that affects a recorded value, or the stale checkpoint is
served instead.

Figures alone, from results already on disk:

```bash
python3 make_figures.py --outdir out_v4
```

## Driver -> experiment map

| Driver | Paper | What it establishes |
|---|---|---|
| `fixedpoint` | Exp. 1 | estimator converges to the biased fixed point at every `N` |
| `gap` | Exp. 2 | regret against the spectral gap `gamma(W)`; flatness |
| `prop` | Exp. 3 | certification propagation, and the lemma bound's looseness |
| `gate_onset` | Exp. 4 | measured gate onset against audit rate `pbar` |
| `collapse` | Exp. 5 | behavioural collapse not reachable at simulable scale |
| `epoch` | Exp. R1 | the epoch variant's `1/N` prediction |
| `certified` | Exp. R2 | certified-regime run against the Picard prediction |
| `baselines` | Exp. R3 | robust-aggregation baseline, and the tolerance floor |
| `mistakes` | Exp. R4 | gate comparison under label noise |
| `certify` | Appendix D | the certification table: moduli, masses, `kappa~` |

`gate_exponent` is a **diagnostic**, not a reported result. It is not in
the default driver list and runs only on request:

```bash
python3 reproduce_v35.py --only gate_exponent
```

## The two contraction moduli

`contraction_modulus` perturbs only around `theta_inf` and therefore
reports the *local* modulus `rho_loc`. `global_contraction_modulus`
samples pairs across `B_R(theta*)` and reports `rho_BR`, which is the
modulus Thm. `cec_epoch` actually consumes and the one Appendix D
tabulates. Both are recorded per bias level; they are not
interchangeable, and `rho_loc <= rho_BR` always. Both are sampled
suprema, hence one-sided: each is a lower bound on the true value.

## Layout

```
reproduce_v35.py    every experiment; the code of record
make_figures.py     every publication figure, from results/*.json
paper_style.py      legacy style module, retained from new_version/;
                    make_figures.py carries the current AISTATS style
                    inline and does not import it (see note below)
paper_v4.tex        the paper
out_v4/results/     committed JSON behind every reported number
out_v4/figures/     committed PDF + PNG figures
```

Earlier trees (`v2/`, `new_version/`, `v3/reproduce_v34.py`) were removed;
they contained superseded generators whose numbers appear nowhere in the
paper. They remain retrievable at the `pre-v4-cleanup` tag.

## Note on `paper_style.py`

`paper_style.py` styled the old `new_version/fig_exp*` figures. The
`fig_*` figures the paper actually uses were produced by the style block
now living inside `make_figures.py`, which is tuned for AISTATS column
width and is kept as the single style of record. `paper_style.py` is
retained but unused.
