"""Speed benchmark for theta calibration.

Reports wall-clock for the full EM fit across a grid of problem sizes and
models. The first fit of a given (model, shape) pays XLA compilation; we report
a cold fit and a warm re-fit so you can see steady-state throughput.

Run:  uv run python bench/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

import theta


def timeit(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def bench(model, n_persons, n_items, n_points=61):
    sim = theta.simulate(n_persons, n_items, model, seed=0)
    R = sim.responses

    m, cold = timeit(lambda: theta.fit(R, model, n_points=n_points))
    # warm: same shapes/model -> kernels already compiled
    _, warm = timeit(lambda: theta.fit(R, model, n_points=n_points))

    cells = n_persons * n_items
    print(f"{model}  N={n_persons:>7,} J={n_items:>3}  Q={n_points}  "
          f"iters={m.n_iter:>3}  cold={cold:6.2f}s  warm={warm:6.2f}s  "
          f"({cells / warm / 1e6:6.1f}M responses/s, warm)")
    return warm


def main():
    print("theta calibration benchmark (CPU)\n" + "-" * 72)
    for model in ["1PL", "2PL", "3PL", "4PL"]:
        bench(model, 10_000, 50)
    print("-" * 72)
    print("scaling 2PL with N:")
    for n in [1_000, 10_000, 100_000, 500_000]:
        bench("2PL", n, 50)
    print("-" * 72)
    print("scaling 2PL with J (items):")
    for j in [20, 100, 300]:
        bench("2PL", 50_000, j)


if __name__ == "__main__":
    main()
