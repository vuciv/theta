# theta

Fast item-response theory in Python. `theta` calibrates the 1PL, 2PL, 3PL, and
4PL logistic models by Marginal Maximum Likelihood (the Bock-Aitkin EM
algorithm), and it does ability scoring and standard errors too. The estimator
is written in JAX, so the hot loop is JIT-compiled and runs on GPU without code
changes.

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

## Install

```bash
uv add theta-irt              # CPU
uv add "theta-irt[cuda]"      # + NVIDIA GPU (CUDA 12)
```

Or with pip: `pip install theta-irt` (or `pip install "theta-irt[cuda]"`).

## Benchmarks

`theta`, [`girth`](https://github.com/eribean/girth) (numpy/scipy), and
[`mirt`](https://github.com/philchalmers/mirt) (R, C++) fit the same model the
same way: Marginal Maximum Likelihood on a 61-point Gauss-Hermite grid. All
three converge to the same parameters (Pearson r = 1.000 against both), so the
comparison is like-for-like. On 2PL, `theta` is 4-18x faster than the mirt C++
reference.

Measured on CPU (Apple M-series), Q = 61, warm (XLA already compiled), log
scale, lower is better. Reproduce with `uv run --group bench python
bench/compare.py`.

### 2PL: 4-18x faster than mirt, 15-46x faster than girth

![2PL calibration: theta vs girth vs mirt](https://raw.githubusercontent.com/vuciv/theta/main/bench/benchmark.png)

| size (persons x items) | theta | mirt (C++) | girth | vs mirt | vs girth |
|---|--:|--:|--:|--:|--:|
| 1,000 x 50    | **59 ms**  | 261 ms  | 888 ms  | 4.4x  | 15x |
| 5,000 x 50    | **92 ms**  | 875 ms  | 2.3 s   | 9.5x  | 25x |
| 20,000 x 100  | **0.51 s** | 6.4 s   | 23.4 s  | 12.5x | 46x |
| 50,000 x 100  | **1.1 s**  | 20.5 s  | 49.7 s  | 18.3x | 44x |

### 1PL: 3-13x faster than mirt

`theta`'s 1PL fits one shared discrimination plus a difficulty per item (mirt
is fit with an equal-slopes constraint to match). Every fit converges in 32-50
EM iterations with parameters identical to both references. girth's Rasch
routine is light enough to edge `theta` on the smallest problem; `theta` pulls
ahead as the data grows.

![1PL calibration: theta vs girth vs mirt](https://raw.githubusercontent.com/vuciv/theta/main/bench/benchmark_1pl.png)

| size (persons x items) | theta | mirt (C++) | girth | vs mirt | converged |
|---|--:|--:|--:|--:|:--:|
| 1,000 x 50    | 85 ms      | 232 ms  | **61 ms** | 2.7x  | yes (37 it) |
| 5,000 x 50    | **103 ms** | 430 ms  | 181 ms    | 4.2x  | yes (32 it) |
| 20,000 x 100  | **0.50 s** | 4.2 s   | 1.6 s     | 8.4x  | yes (49 it) |
| 50,000 x 100  | **1.05 s** | 13.8 s  | 8.3 s     | 13.1x | yes (50 it) |

On its own, `theta` fits 100k persons (2PL, 50 items, warm) in 0.9 s and 500k
in 4.8 s, about 5M responses/s on a laptop CPU. The same code runs on GPU
through the `cuda` extra. The first fit of a given shape pays a one-time XLA
compile (~0.5-1 s); every fit after that is warm.

## How it works

Calibration is Marginal Maximum Likelihood by EM, the algorithm behind `mirt`,
BILOG, and MULTILOG. The work is arranged so XLA can fuse the inner loop:

- **E-step is two matrix multiplies.** Ability is integrated out on a
  Gauss-Hermite grid of `Q` nodes. The per-person likelihood over the grid is
  `X @ logP + (1-X) @ log(1-P)`, and the expected sufficient statistics are
  `posteriorᵀ @ X`. That is plain GEMM, so it vectorizes and runs on GPU.
- **M-step decouples across items.** Each item is fit on its own by a damped
  (Levenberg) Newton step with a backtracking line search, `vmap`-ed over
  items. The line search makes every M-step improve the penalized EM objective
  (`model.objective_history` is non-decreasing), which is what stabilizes the
  weakly identified 3PL/4PL asymptotes. Note that the unpenalized marginal
  log-likelihood (`model.loglik_history`) can dip when asymptote priors are
  active; under priors it is the penalized objective, not the raw likelihood,
  that EM is guaranteed to ascend.
- **The whole iteration is JIT-compiled** except the convergence check, so it
  runs as fused kernels with no Python overhead. Computation is `float64`
  throughout, which standard errors need.

Derivatives for the 3PL/4PL M-step and for the standard errors come from JAX
autodiff, so there are no hand-coded Hessians.

## Models

P(correct | θ) for ability θ and item parameters:

| Model | Formula | Free parameters |
|------|---------|-----------------|
| 1PL  | `σ(a(θ - b))` | one shared `a`, per-item `b` |
| 2PL  | `σ(a(θ - b))` | per-item `a`, `b` |
| 3PL  | `c + (1-c)·σ(a(θ - b))` | `a`, `b`, `c` (guessing) |
| 4PL  | `c + (d-c)·σ(a(θ - b))` | `a`, `b`, `c`, `d` (slip) |

where `a` is discrimination, `b` is difficulty, `c` is the lower asymptote
(pseudo-guessing), and `d` is the upper asymptote (1 - slip).

Slope and difficulty are estimated by pure MMLE with no penalty, so 1PL/2PL are
plain Marginal Maximum Likelihood. The 3PL/4PL asymptotes are weakly identified
(a property of the model, not the estimator), so `theta` puts light Beta priors
on `c` and `d` by default, which makes those two parameters MAP estimates.
Slope and difficulty recover cleanly; the asymptotes need a lot of data and
carry large standard errors when the data can't pin them down. Turn the
asymptote priors off for a fully unpenalized fit:

```python
flat = theta.Priors(c_a=None, c_b=None, slip_a=None, slip_b=None)
model = theta.fit(responses, "3PL", priors=flat)
```

Item-parameter standard errors are computed from the likelihood information,
not the penalized objective, which is the conventional frequentist quantity.

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

Ability estimators: `EAP` (posterior mean, vectorized, always finite, the
default), `MAP` (posterior mode, follows the scoring prior), and `ML`
(likelihood mode, bounded to ±6 and NaN for all-correct or all-incorrect
patterns).

## Development

The project is managed with `uv`, which handles the venv, tests, benchmarks,
the wheel, and the PyPI upload:

```bash
git clone https://github.com/vuciv/theta && cd theta
uv sync                                        # create .venv + install theta and deps
uv run pytest                                  # run the test suite
uv run python bench/benchmark.py               # scaling benchmark
uv run --group bench python bench/compare.py   # vs girth (+ mirt if R/mirt installed)

uv build                                       # wheel + sdist into dist/
uv publish                                     # upload to PyPI (set UV_PUBLISH_TOKEN)
```

## Status

v0.1, dichotomous unidimensional models (1/2/3/4PL). On the roadmap: polytomous
models (GRM, GPCM), multidimensional IRT, and fixed-θ / online scoring.

MIT licensed. Issues and PRs welcome.
</content>
</invoke>
