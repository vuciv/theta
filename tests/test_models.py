import jax.numpy as jnp
import numpy as np

from theta import models as M


def test_param_roundtrip_4pl():
    spec = M.get_spec("4PL")
    a = jnp.array([1.2, 0.8, 1.5])
    b = jnp.array([-0.5, 0.0, 1.0])
    c = jnp.array([0.15, 0.20, 0.05])
    d = jnp.array([0.98, 0.95, 0.99])
    psi = M.report_to_psi(a, b, c, d, spec)
    ar, br, cr, dr = M.natural_to_report(*M.psi_to_natural(psi, spec))
    assert jnp.allclose(ar, a, atol=1e-8)
    assert jnp.allclose(br, b, atol=1e-8)
    assert jnp.allclose(cr, c, atol=1e-8)
    assert jnp.allclose(dr, d, atol=1e-8)


def test_probability_bounds_and_ordering():
    spec = M.get_spec("4PL")
    psi = M.report_to_psi(jnp.array([1.0, 2.0]), jnp.array([0.0, 0.5]),
                          jnp.array([0.1, 0.2]), jnp.array([0.9, 0.95]), spec)
    nodes, *_ = __import__("theta.quadrature", fromlist=["gauss_hermite"]).gauss_hermite(41)
    P = M.prob_matrix(nodes, psi, spec)
    a, beta, c, d = M.psi_to_natural(psi, spec)
    assert bool((P >= c[:, None] - 1e-9).all())
    assert bool((P <= d[:, None] + 1e-9).all())


def test_frozen_asymptotes():
    for name, want_c, want_d in [("1PL", 0.0, 1.0), ("2PL", 0.0, 1.0), ("3PL", None, 1.0)]:
        spec = M.get_spec(name)
        psi = M.report_to_psi(jnp.array([1.0]), jnp.array([0.0]),
                              jnp.array([0.1]), jnp.array([0.9]), spec)
        _, _, c, d = M.psi_to_natural(psi, spec)
        if want_c is not None:
            assert float(c[0]) == want_c
        assert float(d[0]) == want_d


def test_report_matches_matrix():
    spec = M.get_spec("3PL")
    a = jnp.array([1.3, 0.7]); b = jnp.array([0.2, -0.4])
    c = jnp.array([0.15, 0.25]); d = jnp.array([1.0, 1.0])
    psi = M.report_to_psi(a, b, c, d, spec)
    from theta.quadrature import gauss_hermite
    nodes, *_ = gauss_hermite(31)
    P = M.prob_matrix(nodes, psi, spec)
    P2 = M.prob_report(nodes[None, :], a[:, None], b[:, None], c[:, None], d[:, None])
    assert jnp.allclose(P, P2, atol=1e-10)
