"""High-level user-facing API: ``fit`` and the :class:`IRTModel` result."""

from __future__ import annotations

import numpy as np

from theta import models as M
from theta import score as _score
from theta.calibrate import CalibrateResult, calibrate
from theta.info import standard_errors
from theta.priors import Priors


class IRTModel:
    """A fitted IRT model. Carries item parameters and provides scoring,
    standard errors, and probability prediction."""

    def __init__(self, result: CalibrateResult):
        self._r = result

    # --- item parameters -------------------------------------------------
    @property
    def a(self):
        return self._r.a

    @property
    def b(self):
        return self._r.b

    @property
    def c(self):
        return self._r.c

    @property
    def d(self):
        return self._r.d

    @property
    def model(self):
        return self._r.spec.name

    @property
    def loglik(self):
        return self._r.loglik

    @property
    def loglik_history(self):
        """Unpenalized marginal log-likelihood per EM iteration. Not guaranteed
        monotone when asymptote priors are active (see ``objective_history``)."""
        return self._r.loglik_history

    @property
    def objective_history(self):
        """Penalized objective (marginal LL + log-priors) the EM ascends;
        non-decreasing up to line-search/numerical tolerance."""
        return self._r.objective_history

    @property
    def converged(self):
        return self._r.converged

    @property
    def n_iter(self):
        return self._r.n_iter

    @property
    def n_items(self):
        return len(self._r.b)

    @property
    def n_params(self):
        """Number of free parameters (for AIC/BIC)."""
        spec = self._r.spec
        per = len(spec.report_names)
        if spec.shared_a:  # one shared 'a' instead of one per item
            return 1 + (per - 1) * self.n_items
        return per * self.n_items

    def aic(self):
        return 2 * self.n_params - 2 * self.loglik

    def bic(self, n_persons):
        return self.n_params * np.log(n_persons) - 2 * self.loglik

    # --- prediction / scoring -------------------------------------------
    def prob(self, theta):
        """Item-response probabilities ``P[len(theta), n_items]``."""
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        return np.asarray(
            M.prob_report(theta[:, None], self.a[None], self.b[None],
                          self.c[None], self.d[None])
        )

    def score(self, responses, method="EAP", **kw):
        """Estimate abilities for response patterns. Returns ``(theta, se)``."""
        return _score.score(responses, self.a, self.b, self.c, self.d,
                            method=method, **kw)

    def standard_errors(self, responses):
        """Item-parameter standard errors (see :mod:`theta.info`)."""
        return standard_errors(self._r, responses)

    def params(self):
        """Item parameters as a structured numpy array."""
        names = self._r.spec.report_names
        cols = {"a": self.a, "b": self.b, "c": self.c, "d": self.d}
        dtype = [(n, "f8") for n in names]
        arr = np.empty(self.n_items, dtype=dtype)
        for n in names:
            arr[n] = cols[n]
        return arr

    def __repr__(self):
        status = "converged" if self.converged else f"NOT converged ({self.n_iter} it)"
        return (f"<IRTModel {self.model}: {self.n_items} items, "
                f"loglik={self.loglik:.1f}, {status}>")


def fit(responses, model="2PL", *, n_points=61, max_iter=500, tol=1e-4,
        n_newton=3, priors: Priors | None = None,
        prior_mean=0.0, prior_sd=1.0) -> IRTModel:
    """Fit a 1PL/2PL/3PL/4PL model to a 0/1 response matrix via MMLE-EM.

    Parameters
    ----------
    responses : array [N persons, J items], 0/1, NaN for missing.
    model : ``"1PL"``, ``"2PL"``, ``"3PL"``, or ``"4PL"``.
    n_points : number of Gauss-Hermite quadrature nodes.
    max_iter, tol : EM stopping controls.
    priors : :class:`theta.priors.Priors` (defaults regularize c/d).

    Returns
    -------
    :class:`IRTModel`
    """
    res = calibrate(responses, model, n_points=n_points, max_iter=max_iter,
                    tol=tol, n_newton=n_newton, priors=priors,
                    prior_mean=prior_mean, prior_sd=prior_sd)
    return IRTModel(res)
