"""Head-to-head: theta vs girth vs mirt on 1PL / 2PL Marginal Maximum Likelihood.

All three estimate the *same* model by the *same* method (MMLE-EM on a
Gauss-Hermite grid, point estimates), so this is apples-to-apples:

  * theta  — JAX, this package
  * girth  — numpy/scipy (Python)
  * mirt   — R, C++ (Rcpp); the field's reference implementation

mirt runs out-of-process through bench/mirt_fit.R, which times the fit only
(R startup and CSV IO are excluded, just as theta/girth exclude data loading).
If Rscript or the mirt package is missing, mirt is skipped and the comparison
falls back to theta vs girth.

  uv run --group bench python bench/compare.py            # both models
  uv run --group bench python bench/compare.py 1PL        # one model

Writes bench/benchmark_<model>.png (+ benchmark.png for 2PL) and a markdown table.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
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
_MIRT_R = os.path.join(os.path.dirname(__file__), "mirt_fit.R")


# --- competitors -----------------------------------------------------------

def _girth_fit(model, Rt):
    opts = {"quadrature_n": QUAD}
    n_items = Rt.shape[0]
    if model == "1PL":
        r = onepl_mml(Rt, options=opts)
        return np.full(n_items, float(r["Discrimination"])), np.asarray(r["Difficulty"])
    r = twopl_mml(Rt, options=opts)
    return np.asarray(r["Discrimination"]), np.asarray(r["Difficulty"])


def _mirt_available():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", 'cat(requireNamespace("mirt", quietly=TRUE))'],
                       capture_output=True, text=True)
    return "TRUE" in r.stdout


def _mirt_fit(model, R):
    """Returns (fit_time_seconds, a, b, converged) via the R harness."""
    with tempfile.TemporaryDirectory() as d:
        data_csv, out_csv = os.path.join(d, "data.csv"), os.path.join(d, "params.csv")
        np.savetxt(data_csv, R.astype(int), fmt="%d", delimiter=",")
        r = subprocess.run(
            ["Rscript", _MIRT_R, data_csv, model, out_csv, str(QUAD), str(REPEAT)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mirt failed:\n" + r.stderr[-1500:])
        t = conv = None
        for line in r.stdout.splitlines():
            if line.startswith("TIME "):
                t = float(line.split()[1])
            elif line.startswith("CONVERGED "):
                conv = line.split()[1] == "TRUE"
        p = np.genfromtxt(out_csv, delimiter=",", names=True)
        return t, np.atleast_1d(p["a"]), np.atleast_1d(p["b"]), conv


def _time(fn, repeat=REPEAT):
    fn()  # warm up (JIT compile for theta; LUT/caches for girth)
    return min(_once(fn) for _ in range(repeat))


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# --- driver ----------------------------------------------------------------

def run(models=("1PL", "2PL"), grid=N_GRID):
    if twopl_mml is None:
        raise SystemExit("girth not installed; run `uv add --group bench girth`")
    has_mirt = _mirt_available()
    print(f"competitors: theta, girth{', mirt' if has_mirt else '  (mirt unavailable, skipping)'}")

    for model in models:
        print(f"\n### {model}")
        rows = []
        for n, j in grid:
            sim = theta.simulate(n, j, model, seed=0)
            R = sim.responses
            Rt = R.T.astype(np.int64)
            out = {}

            t_theta = _time(lambda: out.__setitem__("m", theta.fit(R, model, n_points=QUAD)))
            t_girth = _time(lambda: out.__setitem__("g", _girth_fit(model, Rt)))
            m = out["m"]
            b_t = np.asarray(m.b)
            a_g, b_g = out["g"]
            row = dict(n=n, j=j, t_theta=t_theta, t_girth=t_girth,
                       sp_girth=t_girth / t_theta, converged=m.converged, iters=m.n_iter,
                       agree_girth=float(np.corrcoef(b_t, b_g)[0, 1]))

            if has_mirt:
                t_m, a_m, b_m, conv_m = _mirt_fit(model, R)
                row.update(t_mirt=t_m, sp_mirt=t_m / t_theta,
                           agree_mirt=float(np.corrcoef(b_t, b_m)[0, 1]))

            rows.append(row)
            extra = (f"  mirt={row['t_mirt']*1e3:9.1f} ms ({row['sp_mirt']:.1f}x, "
                     f"b corr={row['agree_mirt']:.3f})") if has_mirt else ""
            print(f"N={n:>6,} J={j:>3}: theta={t_theta*1e3:8.1f} ms  "
                  f"girth={t_girth*1e3:9.1f} ms ({row['sp_girth']:.1f}x)"
                  f"{extra}  conv={m.converged}/{m.n_iter}")

        _print_table(model, rows, has_mirt)
        # 2PL is the headline chart (benchmark.png); other models get a suffix
        path = "bench/benchmark.png" if model == "2PL" else f"bench/benchmark_{model.lower()}.png"
        _make_chart(model, rows, path, has_mirt)


def _print_table(model, rows, has_mirt):
    head = "| size (N×J) | theta | girth | "
    head += "mirt | " if has_mirt else ""
    head += "speedup | converged | b agreement |"
    print("\n" + head)
    print("|---|--:|--:|" + ("--:|" if has_mirt else "") + "--:|:--:|--:|")
    for r in rows:
        mirt_cell = f" {r['t_mirt']*1e3:.0f} ms |" if has_mirt else ""
        if has_mirt:
            sp = f"**{r['sp_mirt']:.1f}× vs mirt**, {r['sp_girth']:.0f}× vs girth"
            agree = f"{r['agree_mirt']:.3f} (mirt), {r['agree_girth']:.3f} (girth)"
        else:
            sp = f"**{r['sp_girth']:.1f}×**"
            agree = f"{r['agree_girth']:.3f}"
        print(f"| {r['n']:,} × {r['j']} | **{r['t_theta']*1e3:.0f} ms** | "
              f"{r['t_girth']*1e3:.0f} ms |{mirt_cell} {sp} | "
              f"{'yes' if r['converged'] else 'NO'} ({r['iters']} it) | {agree} |")


def _make_chart(model, rows, path, has_mirt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("girth (numpy/scipy)", "t_girth", "#9aa0a6")]
    if has_mirt:
        series.append(("mirt (R / C++)", "t_mirt", "#f4a000"))
    series.append(("theta (JAX)", "t_theta", "#2962ff"))

    labels = [f"{r['n']//1000}k×{r['j']}" for r in rows]
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=140)
    for i, (label, key, color) in enumerate(series):
        vals = [r[key] * 1e3 for r in rows]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=label, color=color)

    # annotate theta with its speedup over the strongest baseline present (mirt)
    base = "sp_mirt" if has_mirt else "sp_girth"
    base_name = "mirt" if has_mirt else "girth"
    theta_off = ((n - 1) - (n - 1) / 2) * w
    for r, xi in zip(rows, x):
        s = r[base]
        if s >= 1.5:
            txt, col = f"{s:.0f}× vs {base_name}", "#2962ff"
        elif s >= 1.05:
            txt, col = f"{s:.1f}× vs {base_name}", "#2962ff"
        elif s >= 0.9:
            txt, col = f"≈ {base_name}", "#5f6368"
        else:
            txt, col = f"{s:.1f}× ({base_name} wins)", "#5f6368"
        ax.annotate(txt, (xi + theta_off, r["t_theta"] * 1e3), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel(f"{model} fit time — ms (log scale, lower is better)")
    ax.set_xlabel("problem size  (persons × items)")
    title = f"{model} calibration: theta vs girth" + (" vs mirt" if has_mirt else "")
    ax.set_title(f"{title}  (same MMLE, CPU, Q=61, warm)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncol=n)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    models = tuple(m.upper() for m in sys.argv[1:]) or ("1PL", "2PL")
    run(models)
