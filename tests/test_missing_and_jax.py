import jax
import numpy as np

import theta
from theta.simulate import sample_responses


def test_missing_data_calibration():
    sim = theta.simulate(4000, 25, "2PL", seed=1)
    resp = sim.responses.copy()
    rng = np.random.default_rng(0)
    # knock out 15% of entries
    mask = rng.uniform(size=resp.shape) < 0.15
    resp[mask] = np.nan
    m = theta.fit(resp, "2PL")
    assert m.converged
    assert float(np.corrcoef(m.b, sim.b)[0, 1]) > 0.95
    # scoring tolerates missing entries too
    th, se = m.score(resp, method="EAP")
    assert np.all(np.isfinite(th))


def test_jax_native_sampling():
    key = jax.random.PRNGKey(0)
    theta_vals = jax.random.normal(key, (500,))
    a = np.full(10, 1.2); b = np.linspace(-2, 2, 10)
    c = np.zeros(10); d = np.ones(10)
    R = sample_responses(theta_vals, a, b, c, d, key)
    assert R.shape == (500, 10)
    assert set(np.unique(np.asarray(R))).issubset({0.0, 1.0})
