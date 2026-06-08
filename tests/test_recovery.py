import numpy as np
import pytest

import theta


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def test_2pl_recovery():
    sim = theta.simulate(4000, 30, "2PL", seed=1)
    m = theta.fit(sim.responses, "2PL")
    assert m.converged
    assert corr(m.a, sim.a) > 0.95
    assert corr(m.b, sim.b) > 0.97


def test_1pl_shared_discrimination():
    sim = theta.simulate(4000, 30, "1PL", seed=3)
    m = theta.fit(sim.responses, "1PL")
    assert m.converged
    # one shared slope -> all estimates equal, close to the true shared value
    assert np.allclose(m.a, m.a[0])
    assert abs(m.a[0] - sim.a[0]) < 0.15
    assert corr(m.b, sim.b) > 0.97


def test_3pl_recovery_difficulty_discrimination():
    sim = theta.simulate(6000, 30, "3PL", seed=2)
    m = theta.fit(sim.responses, "3PL")
    assert m.converged
    assert corr(m.a, sim.a) > 0.85
    assert corr(m.b, sim.b) > 0.92


def test_4pl_runs_and_converges():
    sim = theta.simulate(6000, 25, "4PL", seed=4)
    m = theta.fit(sim.responses, "4PL")
    assert m.converged
    # asymptotes are weakly identified; only require slope/difficulty to recover
    assert corr(m.b, sim.b) > 0.85


@pytest.mark.parametrize("model", ["1PL", "2PL", "3PL", "4PL"])
def test_objective_monotone(model):
    sim = theta.simulate(3000, 20, model, seed=10)
    m = theta.fit(sim.responses, model)
    # the penalized objective the EM ascends must be non-decreasing
    obj = np.array(m.objective_history)
    assert np.all(np.diff(obj) > -1e-3)


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_unpenalized_loglik_monotone_without_active_priors(model):
    # 1PL/2PL have no active priors by default -> raw marginal LL is monotone too
    sim = theta.simulate(3000, 20, model, seed=10)
    m = theta.fit(sim.responses, model)
    h = np.array(m.loglik_history)
    assert np.all(np.diff(h) > -1e-3)


def test_aic_bic_consistency():
    sim = theta.simulate(2000, 20, "2PL", seed=8)
    m2 = theta.fit(sim.responses, "2PL")
    m1 = theta.fit(sim.responses, "1PL")
    # 2PL is the generating model -> should win on AIC
    assert m2.aic() < m1.aic()
    assert m2.n_params == 2 * 20
    assert m1.n_params == 1 + 20
