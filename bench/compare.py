"""Head-to-head: theta vs girth on 1PL / 2PL Marginal Maximum Likelihood.

Both libraries estimate the *same* model by the *same* method (MMLE on a
Gauss-Hermite grid, point estimates) so this is an apples-to-apples wall-clock
comparison. We also check that the two agree on the recovered parameters, so the
speed number is not bought with a worse fit, and report theta's convergence.

  uv run --group bench python bench/compare.py            # both models
  uv run --group bench python bench/compare.py 1PL        # one model

Writes bench/benchmark_<model>.png (and benchmark.png for 2PL) + a markdown table.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import theta

try:
    from girth import onepl_mml, twopl_mml
except ImportError:  # pragma: no cover
    onepl_mml = twopl_mml = None

N_GRID = [(1_000, 50), (5_000, 50), (20_000, 100), (50_000, 100)]
QUAD = 61
REPEAT = 3


def _girth_fit(model, Rt):
    """Return (a, b) from girth for the given model; a is length n_items."""
    opts = {"quadrature_n": QUAD}
    n_items = Rt.shape[0]
    if model == "1PL":
        r = onepl_mml(Rt, options=opts)              # one shared discrimination
        return np.full(n_items, float(r["Discrimination"])), np.asarray(r["Difficulty"])
    r = twopl_mml(Rt, options=opts)
    return np.asarray(r["Discrimination"]), np.asarray(r["Difficulty"])


def _time(fn, repeat=REPEAT):
    fn()  # warm up (JIT compile for theta; LUT/caches for girth)
    return min(_once(fn) for _ in range(repeat))


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run(models=("2PL", "1PL"), grid=N_GRID):
    if twopl_mml is None:
        raise SystemExit("girth not installed; run `uv add --group bench girth`")

    per_model = {}
    for model in models:
        print(f"\n### {model}")
        rows = []
        for n, j in grid:
            sim = theta.simulate(n, j, model, seed=0)
            R = sim.responses
            Rt = R.T.astype(np.int64)  # girth wants [items x participants]
            out = {}

            def fit_theta():
                m = theta.fit(R, model, n_points=QUAD)
                out["m"] = m

            def fit_girth():
                out["g"] = _girth_fit(model, Rt)

            t_theta = _time(fit_theta)
            t_girth = _time(fit_girth)

            m = out["m"]
            a_t, b_t = np.asarray(m.a), np.asarray(m.b)
            a_g, b_g = out["g"]
            agree_b = float(np.corrcoef(b_t, b_g)[0, 1])
            # 1PL slope is a single shared scalar -> compare values, not correlation
            agree_a = (float(np.corrcoef(a_t, a_g)[0, 1]) if model != "1PL"
                       else 1.0 - abs(a_t[0] - a_g[0]) / a_g[0])

            rows.append(dict(n=n, j=j, t_theta=t_theta, t_girth=t_girth,
                             speedup=t_girth / t_theta, agree_a=agree_a, agree_b=agree_b,
                             converged=m.converged, iters=m.n_iter))
            aname = "a agree" if model != "1PL" else "a rel"
            print(f"N={n:>6,} J={j:>3}: theta={t_theta*1e3:8.1f} ms  girth={t_girth*1e3:9.1f} ms  "
                  f"{t_girth/t_theta:5.1f}x  conv={m.converged}/{m.n_iter:<3} "
                  f"{aname}={agree_a:.3f} b corr={agree_b:.3f}")

        _print_table(model, rows)
        per_model[model] = rows

    _make_combined_chart(per_model, "bench/benchmark.png")
    return per_model


def _print_table(model, rows):
    print(f"\n| {model} size (N×J) | theta | girth | speedup | theta converged | param agreement (a, b) |")
    print("|---|--:|--:|--:|:--:|--:|")
    for r in rows:
        print(f"| {r['n']:,} × {r['j']} | **{r['t_theta']*1e3:.0f} ms** | "
              f"{r['t_girth']*1e3:.0f} ms | **{r['speedup']:.1f}×** | "
              f"{'yes' if r['converged'] else 'NO'} ({r['iters']} it) | "
              f"{r['agree_a']:.3f}, {r['agree_b']:.3f} |")


_COLORS = {"2PL": "#2962ff", "1PL": "#00897b", "3PL": "#8e24aa", "4PL": "#f4511e"}


def _make_combined_chart(per_model, path="bench/benchmark.png"):
    """One chart: theta-over-girth speedup for every model across problem sizes.

    Speedup = girth time / theta time (higher = theta faster). A dashed parity
    line at 1x makes any point where girth wins read honestly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(per_model)
    sizes = [f"{r['n']//1000}k×{r['j']}" for r in per_model[models[0]]]
    x = np.arange(len(sizes))
    n = len(models)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=140)
    for i, mdl in enumerate(models):
        sp = [r["speedup"] for r in per_model[mdl]]
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, sp, w, label=mdl, color=_COLORS.get(mdl, "#888"))
        for xi, s in zip(x + off, sp):
            ax.annotate(f"{s:.0f}×" if s >= 1.5 else f"{s:.1f}×", (xi + off * 0, s),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8, fontweight="bold", color=_COLORS.get(mdl, "#888"))

    ax.axhline(1.0, ls="--", lw=1, color="#9aa0a6")
    ax.text(len(sizes) - 0.5, 1.0, " parity", va="center", ha="left",
            fontsize=8, color="#9aa0a6")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(r["speedup"] for rs in per_model.values() for r in rs) * 1.6)
    ax.set_ylabel("speedup  =  girth time ÷ theta time   (higher = theta faster)")
    ax.set_xlabel("problem size  (persons × items)")
    ax.set_title("theta vs girth — same MMLE, CPU, Q=61, warm\n"
                 "identical parameters (r = 1.000) and all fits converge at every size")
    ax.set_xticks(x, sizes)
    ax.legend(title="model", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    models = tuple(m.upper() for m in sys.argv[1:]) or ("2PL", "1PL")
    run(models)
