"""Regression tests for correctness issues found in review."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import theta
from theta.calibrate import _e_step, calibrate
from theta.models import get_spec
from theta.priors import Priors
from theta.quadrature import gauss_hermite


def test_map_honors_nondefault_prior():
    """MAP/ML must use the requested prior, not a hard-coded N(0,1).

    For an all-missing response pattern the likelihood is flat, so the MAP
    estimate must equal the prior mean with SE equal to the prior SD.
    """
    a = np.array([1.0, 1.2, 0.8]); b = np.array([-0.5, 0.0, 0.5])
    c = np.zeros(3); d = np.ones(3)
    resp = np.full((1, 3), np.nan)
    for method in ("EAP", "MAP"):
        th, se = theta.score(resp, a, b, c, d, method=method,
                             prior_mean=2.0, prior_sd=0.5)
        assert abs(th[0] - 2.0) < 1e-3
        assert abs(se[0] - 0.5) < 1e-3


def test_loglik_matches_returned_params_on_early_stop():
    """`loglik` must be evaluated at the returned parameters, not one M-step behind."""
    sim = theta.simulate(2000, 20, "2PL", seed=3)
    res = calibrate(sim.responses, "2PL", max_iter=1)  # force a non-converged stop
    X = np.nan_to_num(sim.responses, nan=0.0)
    Mo = (~np.isnan(sim.responses)).astype(float)
    Xm, Om, Mobs = jnp.asarray(X * Mo), jnp.asarray((1 - X) * Mo), jnp.asarray(Mo)
    _, _, mll = _e_step(res.psi, Xm, Om, Mobs, jnp.log(res.weights), get_spec("2PL"), res.nodes)
    assert abs(res.loglik - float(mll)) < 1e-6
    assert res.loglik_history[-1] == pytest.approx(res.loglik)


def test_map_not_hard_clipped_to_box():
    """MAP must follow the prior, not a [-6, 6] box. All-missing pattern under a
    far-off prior mean returns that mean; ML stays bounded."""
    a = np.array([1.0, 1.2]); b = np.array([0.0, 0.3])
    c = np.zeros(2); d = np.ones(2)
    resp = np.full((1, 2), np.nan)
    th_map, se_map = theta.score(resp, a, b, c, d, method="MAP",
                                 prior_mean=8.0, prior_sd=1.0)
    assert abs(th_map[0] - 8.0) < 1e-3      # not truncated to 6
    # ML remains bounded (its likelihood is flat/divergent here)
    th_ml, _ = theta.score(resp, a, b, c, d, method="ML")
    assert np.isnan(th_ml[0]) or abs(th_ml[0]) <= 6.0 + 1e-6


def test_objective_monotone_under_strong_prior_even_if_loglik_dips():
    """With a strong asymptote prior the unpenalized loglik may decrease, but the
    penalized objective the EM ascends must not."""
    sim = theta.simulate(3000, 25, "3PL", seed=2)
    strong = Priors(c_a=20.0, c_b=20.0)   # heavily pulls c toward 0.5
    m = theta.fit(sim.responses, "3PL", priors=strong)
    obj = np.array(m.objective_history)
    assert np.all(np.diff(obj) > -1e-3)   # penalized objective monotone


def test_invalid_scoring_method_raises():
    sim = theta.simulate(200, 8, "2PL", seed=0)
    m = theta.fit(sim.responses, "2PL")
    with pytest.raises(ValueError):
        m.score(sim.responses, method="not-a-method")


def test_default_priors_are_pure_mml_for_slope_difficulty():
    """1PL/2PL fits must be unpenalized MML by default (no prior on a, b)."""
    p = Priors()
    assert p.la_sd is None and p.beta_sd is None
    assert not p.has_la and not p.has_beta
    # asymptote priors stay on (they are what 3PL/4PL need)
    assert p.has_c and p.has_d


def test_priors_can_be_fully_disabled():
    flat = Priors(c_a=None, c_b=None, slip_a=None, slip_b=None)
    assert not flat.has_c and not flat.has_d
    sim = theta.simulate(3000, 20, "3PL", seed=1)
    # unpenalized 3PL still runs (may be wobblier, but must not crash / NaN)
    m = theta.fit(sim.responses, "3PL", priors=flat)
    assert np.all(np.isfinite(m.a)) and np.all(np.isfinite(m.c))
