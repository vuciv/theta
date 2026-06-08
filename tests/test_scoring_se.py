import numpy as np
import pytest

import theta


def test_eap_map_recover_theta():
    sim = theta.simulate(3000, 30, "2PL", seed=1)
    m = theta.fit(sim.responses, "2PL")
    for method in ["EAP", "MAP"]:
        th, se = m.score(sim.responses, method=method)
        assert np.all(np.isfinite(th))
        assert np.all(se > 0)
        assert float(np.corrcoef(th, sim.theta)[0, 1]) > 0.85


def test_ml_bounded_and_flags_degenerate():
    sim = theta.simulate(1500, 20, "2PL", seed=2)
    m = theta.fit(sim.responses, "2PL")
    th, se = m.score(sim.responses, method="ML")
    finite = np.isfinite(th)
    # bounded within trust region
    assert np.nanmax(np.abs(th)) <= 6.0 + 1e-6
    # all-correct / all-wrong patterns are NaN
    n_obs = sim.responses.shape[1]
    sums = sim.responses.sum(axis=1)
    degenerate = (sums == 0) | (sums == n_obs)
    assert np.all(~finite[degenerate]) if degenerate.any() else True


def test_standard_errors_shape_and_finite():
    sim = theta.simulate(4000, 20, "2PL", seed=5)
    m = theta.fit(sim.responses, "2PL")
    se = m.standard_errors(sim.responses)
    assert se["a"].shape == (20,)
    assert np.all(np.isfinite(se["a"])) and np.all(se["a"] > 0)
    assert np.all(np.isfinite(se["b"])) and np.all(se["b"] > 0)
    # 2PL has no c/d
    assert np.all(np.isnan(se["c"]))


def test_se_rough_calibration():
    # z = (est - true)/se should have sd of order 1 across items
    sim = theta.simulate(8000, 40, "2PL", seed=6)
    m = theta.fit(sim.responses, "2PL")
    se = m.standard_errors(sim.responses)
    za = (m.a - sim.a) / se["a"]
    zb = (m.b - sim.b) / se["b"]
    assert 0.5 < np.std(za) < 2.0
    assert 0.5 < np.std(zb) < 2.0
