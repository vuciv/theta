"""Head-to-head: theta vs girth on 2PL Marginal Maximum Likelihood.

Both libraries estimate the *same* model by the *same* method (MMLE on a
Gauss-Hermite grid, point estimates) so this is an apples-to-apples wall-clock
comparison. We also check that the two agree on the recovered parameters, so the
speed number is not bought with a worse fit.

  uv run --group bench python bench/compare.py

Writes a grouped bar chart to bench/benchmark.png and prints a markdown table.
"""

from __future__ import annotations

import time

import numpy as np

import theta

try:
    from girth import twopl_mml
except ImportError:  # pragma: no cover
    twopl_mml = None

N_GRID = [(1_000, 50), (5_000, 50), (20_000, 100), (50_000, 100)]
QUAD = 61


def _time(fn, repeat=3):
    fn()  # warm up (JIT compile for theta; LUT warm for girth)
    best = min(_once(fn) for _ in range(repeat))
    return best


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run():
    if twopl_mml is None:
        raise SystemExit("girth not installed; run `uv add --group bench girth`")

    rows = []
    for n, j in N_GRID:
        sim = theta.simulate(n, j, "2PL", seed=0)
        R = sim.responses
        Rt = R.T.astype(np.int64)  # girth wants [items x participants], integer-coded
        opts = {"quadrature_n": QUAD}

        out = {}

        def fit_theta():
            m = theta.fit(R, "2PL", n_points=QUAD)
            out["theta"] = (np.asarray(m.a), np.asarray(m.b))

        def fit_girth():
            res = twopl_mml(Rt, options=opts)
            out["girth"] = (np.asarray(res["Discrimination"]), np.asarray(res["Difficulty"]))

        t_theta = _time(fit_theta)
        t_girth = _time(fit_girth)

        (a_t, b_t), (a_g, b_g) = out["theta"], out["girth"]
        agree_a = float(np.corrcoef(a_t, a_g)[0, 1])
        agree_b = float(np.corrcoef(b_t, b_g)[0, 1])
        rec_t = float(np.corrcoef(b_t, sim.b)[0, 1])
        rec_g = float(np.corrcoef(b_g, sim.b)[0, 1])

        rows.append(dict(n=n, j=j, t_theta=t_theta, t_girth=t_girth,
                         speedup=t_girth / t_theta,
                         agree_a=agree_a, agree_b=agree_b,
                         rec_theta=rec_t, rec_girth=rec_g))
        print(f"N={n:>6,} J={j:>3}: theta={t_theta*1e3:8.1f} ms  girth={t_girth*1e3:9.1f} ms  "
              f"speedup={t_girth/t_theta:5.1f}x  agree(a,b)=({agree_a:.3f},{agree_b:.3f})")

    _print_table(rows)
    _make_chart(rows)
    return rows


def _print_table(rows):
    print("\n| size (N×J) | theta | girth | speedup | param agreement (a, b) |")
    print("|---|--:|--:|--:|--:|")
    for r in rows:
        print(f"| {r['n']:,} × {r['j']} | **{r['t_theta']*1e3:.0f} ms** | "
              f"{r['t_girth']*1e3:.0f} ms | **{r['speedup']:.1f}×** | "
              f"{r['agree_a']:.3f}, {r['agree_b']:.3f} |")


def _make_chart(rows, path="bench/benchmark.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['n']//1000}k×{r['j']}" for r in rows]
    theta_ms = [r["t_theta"] * 1e3 for r in rows]
    girth_ms = [r["t_girth"] * 1e3 for r in rows]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    b1 = ax.bar(x - w / 2, girth_ms, w, label="girth (numpy/scipy MMLE)", color="#9aa0a6")
    b2 = ax.bar(x + w / 2, theta_ms, w, label="theta (JAX MMLE)", color="#2962ff")
    ax.set_yscale("log")
    ax.set_ylabel("2PL fit time — ms (log scale, lower is better)")
    ax.set_xlabel("problem size  (persons × items)")
    ax.set_title("2PL calibration: theta vs girth  (same MMLE, CPU, Q=61, warm)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for r, xi, t in zip(rows, x, theta_ms):
        ax.annotate(f"{r['speedup']:.0f}× faster", (xi + w / 2, t),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color="#2962ff", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
