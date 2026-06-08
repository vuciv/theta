"""theta: the fastest IRT library (JAX-accelerated MMLE-EM for 1/2/3/4PL)."""

from theta import _config as _config  # noqa: F401  (enables float64 on import)

from theta.api import IRTModel, fit
from theta.priors import Priors
from theta.score import score
from theta.simulate import simulate

__all__ = ["fit", "score", "simulate", "IRTModel", "Priors", "__version__"]

__version__ = "0.1.0"
