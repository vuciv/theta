# theta

**The fastest IRT library.** JAX-accelerated Marginal Maximum Likelihood
(MMLE-EM, the Bock–Aitkin algorithm) for the 1PL, 2PL, 3PL, and 4PL logistic
item-response models — item calibration, ability scoring, and standard errors,
all JIT-compiled and GPU-ready.

```python
import theta

# fit item parameters from a 0/1 response matrix (N persons x J items)
model = theta.fit(responses, "2PL")
print(model.a, model.b)          # discrimination, difficulty

# score people (ability estimates + standard errors)
ability, se = model.score(responses, method="EAP")

# item-parameter standard errors
errs = model.standard_errors(responses)
```

## Benchmark

`theta` and [`girth`](https://github.com/eribean/girth) fit the **same** 2PL
model by the **same** method — Marginal Maximum Likelihood on a 61-point
Gauss–Hermite grid — so this is apples-to-apples. They recover **identical
parameters** (Pearson r = 1.000 on both discrimination and difficulty at every
size); `theta` is **15–50× faster**, and the gap widens with scale.

![2PL calibration: theta vs girth](https://raw.githubusercontent.com/vuciv/theta/main/bench/benchmark.png)

> Calibrating a 2PL model. CPU (Apple M-series), `Q = 61` quadrature nodes,
> warm (steady-state — JIT already compiled). Log scale; lower is better.

| size (persons × items) | theta | girth | speedup | param agreement (a, b) |
|---|--:|--:|--:|--:|
| 1,000 × 50    | **61 ms**  | 923 ms  | **15×** | 1.000, 1.000 |
| 5,000 × 50    | **88 ms**  | 2.3 s   | **26×** | 1.000, 1.000 |
| 20,000 × 100  | **0.44 s** | 22.9 s  | **52×** | 1.000, 1.000 |
| 50,000 × 100  | **1.0 s**  | 47.6 s  | **47×** | 1.000, 1.000 |

Reproduce with `uv run --group bench python bench/compare.py`.

**Scaling on its own** (theta, 2PL, 50 items, warm): **100k** persons in 0.9 s,
**500k** in 4.8 s — roughly **5M responses/s** on a laptop CPU, and the identical
code runs on GPU through the `cuda` extra. The first fit of a given shape pays a
one-time XLA compile (~0.5–1 s); every fit after is warm.

## Why it's fast

Calibration is **Marginal Maximum Likelihood via EM**, the same algorithm used
by `mirt`, BILOG and MULTILOG, but the hot path is written so XLA can fuse it:

- **E-step = two matrix multiplies.** The latent ability is integrated out on a
  Gauss–Hermite grid of `Q` nodes. The per-person likelihood over the grid is
  `X @ logP + (1−X) @ log(1−P)`, and the expected sufficient statistics are
  `posteriorᵀ @ X`. Pure BLAS/GEMM — it vectorizes and runs on GPU.
- **M-step decouples across items.** Each item is fit independently by a damped
  (Levenberg) Newton step with a **backtracking line search**, `vmap`-ed over
  items. The line search makes every M-step monotonically improve the penalized
  EM objective (`model.objective_history`, non-decreasing), which is what tames
  the weakly-identified 3PL/4PL asymptotes. Note the *unpenalized* marginal
  log-likelihood (`model.loglik_history`) can dip when asymptote priors are
  active — under priors it is the penalized objective, not the raw likelihood,
  that the EM is guaranteed to ascend.
- **Everything but the convergence check is JIT-compiled**, so the iteration
  runs as fused kernels with no Python overhead. `float64` throughout (required
  for trustworthy standard errors).

Derivatives for the 3PL/4PL M-step and for the standard errors come from JAX
autodiff — no hand-coded Hessians.

## Install

```bash
uv add theta-irt              # CPU
uv add "theta-irt[cuda]"      # + NVIDIA GPU (CUDA 12)
```

Prefer pip? `pip install theta-irt` (or `pip install "theta-irt[cuda]"`).

## Develop, build & publish (uv)

The project is fully `uv`-managed — one tool for the venv, tests, benchmarks,
the wheel, and the PyPI upload:

```bash
git clone https://github.com/vuciv/theta && cd theta
uv sync                                        # create .venv + install theta and deps
uv run pytest                                  # run the test suite
uv run python bench/benchmark.py               # scaling benchmark
uv run --group bench python bench/compare.py   # head-to-head vs girth (+ chart)

uv build                                       # wheel + sdist into dist/
uv publish                                     # upload to PyPI (set UV_PUBLISH_TOKEN)
```

## Models

P(correct | θ) for ability θ and item parameters:

| Model | Formula | Free parameters |
|------|---------|-----------------|
| 1PL  | `σ(a(θ − b))` | one shared `a`, per-item `b` |
| 2PL  | `σ(a(θ − b))` | per-item `a`, `b` |
| 3PL  | `c + (1−c)·σ(a(θ − b))` | `a`, `b`, `c` (guessing) |
| 4PL  | `c + (d−c)·σ(a(θ − b))` | `a`, `b`, `c`, `d` (slip) |

where `a` = discrimination, `b` = difficulty, `c` = lower asymptote
(pseudo-guessing), `d` = upper asymptote (1 − slip).

**Estimation is pure MMLE for the slope and difficulty** (no penalty by
default), so 1PL/2PL are plain Marginal Maximum Likelihood. The 3PL/4PL
asymptotes are weakly identified — a property of the model, not the estimator —
so `theta` puts light Beta priors on `c`/`d` by default (making those two
parameters MAP / penalized-MML). Slope and difficulty recover cleanly;
asymptotes need a lot of data and show large standard errors when the data
can't pin them down. Disable the asymptote priors for a fully unpenalized fit:

```python
flat = theta.Priors(c_a=None, c_b=None, slip_a=None, slip_b=None)
model = theta.fit(responses, "3PL", priors=flat)
```

Item-parameter standard errors are computed from the **likelihood** information
(not the penalized objective), the conventional frequentist quantity.

## API

```python
import theta

# --- fit ---------------------------------------------------------------
model = theta.fit(
    responses,            # [N, J] of 0/1, NaN = missing/not-reached
    "3PL",                # "1PL" | "2PL" | "3PL" | "4PL"
    n_points=61,          # Gauss-Hermite quadrature nodes
    max_iter=500, tol=1e-4,
    priors=theta.Priors(c_a=2, c_b=8),   # optional MAP priors
)

model.a, model.b, model.c, model.d      # item parameters (np arrays)
model.params()                          # structured array of estimated params
model.loglik, model.converged, model.n_iter
model.objective_history                 # penalized EM objective (monotone)
model.aic(); model.bic(n_persons)
model.prob(theta_values)                # P matrix [len(theta), J]

# --- score people ------------------------------------------------------
ability, se = model.score(responses, method="EAP")   # "EAP" | "MAP" | "ML"

# --- item standard errors ---------------------------------------------
errs = model.standard_errors(responses)              # dict: a, b, c, d, cov

# --- simulate data -----------------------------------------------------
sim = theta.simulate(n_persons=5000, n_items=40, model="3PL", seed=0)
sim.responses, sim.theta, sim.a, sim.b, sim.c, sim.d
```

**Ability estimators:** `EAP` (posterior mean, vectorized, always finite — the
default), `MAP` (posterior mode, follows the scoring prior), `ML` (likelihood
mode; bounded to ±6 and NaN for all-correct / all-incorrect patterns).

## Status

v0.1 — dichotomous unidimensional models (1/2/3/4PL). Planned: polytomous
models (GRM, GPCM), multidimensional IRT, and fixed-`θ` / online scoring.

MIT licensed. Issues and PRs welcome.
