"""Marginal Maximum Likelihood calibration via the EM algorithm (Bock-Aitkin).

The expensive E-step is two BLAS matmuls; the M-step decouples across items and
is solved by a damped (Levenberg) Newton step, vmapped over items. For 1PL the
single shared discrimination is updated by an extra global coordinate step.

By default the slope and difficulty are estimated by *pure* MML (no penalty),
so 1PL/2PL are plain MMLE. The 3PL/4PL asymptotes (c, d) carry light Beta priors
by default because their likelihood is nearly flat; that makes those fits MAP /
penalized-MML in c and d unless the priors are disabled (see `theta.priors`).

Everything except the Python convergence driver is JIT-compiled, so the hot
loop runs as fused XLA kernels (CPU or GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from theta import models as M
from theta.models import BETA, GC, GD, LA, ModelSpec
from theta.priors import Priors
from theta.quadrature import gauss_hermite

_EPS = 1e-9


# --------------------------------------------------------------------------
# E-step
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(5,))
def _e_step(psi, Xm, Om, Mobs, log_weights, spec: ModelSpec, nodes):
    """One E-step.

    Parameters
    ----------
    psi   : [J, 4] current unconstrained item params
    Xm    : [N, J] observed-correct indicator   (response * observed-mask)
    Om    : [N, J] observed-incorrect indicator  ((1-response) * observed-mask)
    Mobs  : [N, J] observed mask (1 if the response is present)
    nodes : [Q] ability quadrature nodes
    log_weights : [Q] log prior mass at the nodes

    Returns
    -------
    rT    : [J, Q] expected number correct, per item per node
    NbarT : [J, Q] expected number of attempts, per item per node
    mll   : scalar total marginal log-likelihood
    """
    P = M.prob_matrix(nodes, psi, spec)  # [J, Q]
    P = jnp.clip(P, _EPS, 1.0 - _EPS)
    logP = jnp.log(P)
    log1mP = jnp.log1p(-P)

    # per-person log-likelihood at every node: [N, Q] via two matmuls
    loglik = Xm @ logP + Om @ log1mP
    logpost = loglik + log_weights[None, :]
    lse = jax.scipy.special.logsumexp(logpost, axis=1)  # [N]
    posterior = jnp.exp(logpost - lse[:, None])         # [N, Q]
    mll = jnp.sum(lse)

    # expected sufficient statistics (more matmuls)
    r = posterior.T @ Xm        # [Q, J] expected correct
    Nbar = posterior.T @ Mobs   # [Q, J] expected attempts
    return r.T, Nbar.T, mll


# --------------------------------------------------------------------------
# M-step
# --------------------------------------------------------------------------

def _neg_obj_item(psi_j, r_j, Nbar_j, nodes, spec: ModelSpec, priors: Priors):
    """Negative expected complete-data log-posterior for a single item.

    ``psi_j`` is [4]; ``r_j``/``Nbar_j`` are [Q]. The la (slope) prior is
    omitted when the slope is shared (1PL) so it can be added exactly once in
    the global step.
    """
    a, beta, c, d = M.psi_to_natural(psi_j, spec)
    z = a * nodes + beta
    P = jnp.clip(c + (d - c) * M.sigmoid(z), _EPS, 1.0 - _EPS)
    ll = jnp.sum(r_j * jnp.log(P) + (Nbar_j - r_j) * jnp.log1p(-P))

    logprior = 0.0
    if priors.has_beta:
        logprior = logprior + priors.log_beta(beta)
    # la prior is added once in the global step when the slope is shared (1PL)
    if priors.has_la and not spec.shared_a:
        logprior = logprior + priors.log_la(psi_j[LA])
    if spec.free[GC] and priors.has_c:
        logprior = logprior + priors.log_c(c)
    if spec.free[GD] and priors.has_d:
        logprior = logprior + priors.log_d(d)
    return -(ll + logprior)


_obj_vmap = jax.vmap(_neg_obj_item, in_axes=(0, 0, 0, None, None, None))
_grad_vmap = jax.vmap(jax.grad(_neg_obj_item), in_axes=(0, 0, 0, None, None, None))
_hess_vmap = jax.vmap(jax.hessian(_neg_obj_item), in_axes=(0, 0, 0, None, None, None))

# backtracking step sizes; 0.0 is included so the objective can never worsen
_BACKTRACK = jnp.array([1.0, 0.5, 0.25, 0.1, 0.0])
_LAM = 1e-3


@partial(jax.jit, static_argnums=(4, 5, 6))
def _m_step(psi, rT, NbarT, nodes, spec: ModelSpec, priors: Priors, n_newton: int):
    """Monotone damped-Newton coordinate ascent over the items.

    Each inner step solves a per-item damped Newton system, then a vectorized
    backtracking line search picks, per item, the step fraction that most
    decreases the (negative) objective. Including ``t=0`` makes every step
    non-worsening, which kills the oscillation that weak 3PL/4PL identification
    would otherwise produce.
    """
    # per-item free coords; for 1PL the slope is handled globally, not here
    free = spec.free_mask
    if spec.shared_a:
        free = free.at[LA].set(0.0)
    mask2 = free[:, None] * free[None, :]
    frozen_diag = jnp.diag(1.0 - free)
    damp = _LAM * jnp.diag(free)

    def newton_iter(psi, _):
        g = _grad_vmap(psi, rT, NbarT, nodes, spec, priors)   # [J, 4]
        H = _hess_vmap(psi, rT, NbarT, nodes, spec, priors)   # [J, 4, 4]
        Hm = H * mask2[None] + (frozen_diag + damp)[None]
        step = jnp.linalg.solve(Hm, g[..., None])[..., 0] * free  # [J, 4]
        step = jnp.where(jnp.isfinite(step), step, 0.0)

        # per-item backtracking line search
        def obj_at(t):
            f = _obj_vmap(psi - t * step, rT, NbarT, nodes, spec, priors)
            return jnp.where(jnp.isfinite(f), f, jnp.inf)
        fvals = jnp.stack([obj_at(t) for t in _BACKTRACK], axis=0)  # [T, J]
        t_best = _BACKTRACK[jnp.argmin(fvals, axis=0)]              # [J]
        psi = psi - t_best[:, None] * step

        if spec.shared_a:
            psi = _shared_a_step(psi, rT, NbarT, nodes, spec, priors)
        return psi, None

    psi, _ = jax.lax.scan(newton_iter, psi, None, length=n_newton)
    return psi


def _shared_a_step(psi, rT, NbarT, nodes, spec, priors):
    """One global, monotone Newton step on the single shared slope (1PL)."""
    g = _grad_vmap(psi, rT, NbarT, nodes, spec, priors)   # [J, 4]
    H = _hess_vmap(psi, rT, NbarT, nodes, spec, priors)   # [J, 4, 4]
    la = psi[0, LA]
    dprior, d2prior = priors.dlog_la(la)                  # log-prior enters obj negated
    G = jnp.sum(g[:, LA]) - dprior
    Hh = jnp.sum(H[:, LA, LA]) - d2prior
    step = G / (Hh + _LAM)

    def total_neg_la(la_val):
        psi_t = psi.at[:, LA].set(la_val)
        ll = jnp.sum(_obj_vmap(psi_t, rT, NbarT, nodes, spec, priors))
        if priors.has_la:  # add the single shared-slope prior exactly once
            ll = ll - priors.log_la(la_val)
        return ll

    cands = la - _BACKTRACK * step
    raw = jax.vmap(total_neg_la)(cands)
    fvals = jnp.where(jnp.isfinite(raw), raw, jnp.inf)
    la_new = cands[jnp.argmin(fvals)]
    return psi.at[:, LA].set(la_new)


@partial(jax.jit, static_argnums=(1, 2))
def _prior_sum(psi, spec: ModelSpec, priors: Priors):
    """Total log-prior over all item parameters at ``psi``.

    Added to the marginal log-likelihood it gives the penalized objective that
    MAP-EM actually ascends (and which the monotone M-step keeps non-decreasing).
    """
    a, beta, c, d = M.psi_to_natural(psi, spec)
    s = 0.0
    if priors.has_beta:
        s = s + jnp.sum(priors.log_beta(beta))
    if priors.has_la:
        s = s + (priors.log_la(psi[0, LA]) if spec.shared_a
                 else jnp.sum(priors.log_la(psi[:, LA])))
    if spec.free[GC] and priors.has_c:
        s = s + jnp.sum(priors.log_c(c))
    if spec.free[GD] and priors.has_d:
        s = s + jnp.sum(priors.log_d(d))
    return s


# --------------------------------------------------------------------------
# initialization + driver
# --------------------------------------------------------------------------

def _init_psi(X, Mobs, spec: ModelSpec):
    """Sensible warm start: difficulty from item p-values, a=1, small c, high d.

    ``X`` here is already NaN-filled to 0, so the masked mean is a plain ratio.
    """
    pbar = (X * Mobs).sum(axis=0) / np.maximum(Mobs.sum(axis=0), 1.0)
    pbar = np.clip(pbar, 0.02, 0.98)
    J = X.shape[1]
    a0 = np.ones(J)
    c0 = np.full(J, 0.10 if spec.free[GC] else spec.c_fixed)
    d0 = np.full(J, 0.95 if spec.free[GD] else spec.d_fixed)
    # difficulty from the guessing-corrected p-value so easy/guessable items
    # don't start with a wildly off intercept
    p_eff = np.clip((pbar - c0) / (d0 - c0), 0.05, 0.95)
    b0 = -(np.log(p_eff) - np.log1p(-p_eff)) / a0   # intercept = logit(p_eff)
    return M.report_to_psi(jnp.asarray(a0), jnp.asarray(b0), jnp.asarray(c0), jnp.asarray(d0), spec)


@dataclass
class CalibrateResult:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    psi: jnp.ndarray
    spec: ModelSpec
    nodes: jnp.ndarray
    weights: jnp.ndarray
    n_iter: int
    converged: bool
    loglik: float
    loglik_history: list       # unpenalized marginal LL; may dip under priors
    objective_history: list    # penalized objective EM ascends; non-decreasing
    priors: Priors


def calibrate(
    responses,
    model: str = "2PL",
    *,
    n_points: int = 61,
    max_iter: int = 500,
    tol: float = 1e-4,
    n_newton: int = 3,
    priors: Priors | None = None,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> CalibrateResult:
    """Calibrate item parameters from a 0/1 response matrix via MMLE-EM.

    ``responses`` is [N persons, J items]; use NaN for missing/not-reached.
    Returns item parameters in the natural (a, b, c, d) IRT difficulty form.
    """
    spec = M.get_spec(model)
    priors = priors or Priors()

    X = np.asarray(responses, dtype=np.float64)
    Mobs_np = (~np.isnan(X)).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    Xm = jnp.asarray(X * Mobs_np)
    Om = jnp.asarray((1.0 - X) * Mobs_np)
    Mobs = jnp.asarray(Mobs_np)
    nodes, weights, log_weights = gauss_hermite(n_points, prior_mean, prior_sd)

    psi = _init_psi(X, Mobs_np, spec)

    prev = M.natural_to_report(*M.psi_to_natural(psi, spec))
    prev_report = jnp.stack(prev, axis=0)
    history = []
    obj_history = []
    converged = False
    n_done = max_iter
    for it in range(max_iter):
        rT, NbarT, mll = _e_step(psi, Xm, Om, Mobs, log_weights, spec, nodes)
        # record both at the *current* psi (before this iteration's M-step)
        history.append(float(mll))
        obj_history.append(float(mll) + float(_prior_sum(psi, spec, priors)))
        psi = _m_step(psi, rT, NbarT, nodes, spec, priors, n_newton)

        report = jnp.stack(M.natural_to_report(*M.psi_to_natural(psi, spec)), axis=0)
        delta = float(jnp.max(jnp.abs(report - prev_report)))
        prev_report = report
        if delta < tol:
            converged = True
            n_done = it + 1
            break

    # The per-iteration mll above is evaluated *before* that iteration's M-step,
    # so it lags the returned psi. Evaluate the marginal log-likelihood at the
    # final parameters so `loglik` (and AIC/BIC) match what we return.
    _, _, final_mll = _e_step(psi, Xm, Om, Mobs, log_weights, spec, nodes)
    history.append(float(final_mll))
    obj_history.append(float(final_mll) + float(_prior_sum(psi, spec, priors)))

    a, b, c, d = M.natural_to_report(*M.psi_to_natural(psi, spec))
    return CalibrateResult(
        a=np.asarray(a), b=np.asarray(b), c=np.asarray(c), d=np.asarray(d),
        psi=psi, spec=spec, nodes=nodes, weights=weights,
        n_iter=n_done, converged=converged,
        loglik=float(final_mll),
        loglik_history=history, objective_history=obj_history, priors=priors,
    )
