"""Reproducible symmetric alpha-stable sampling for the major revision.

The submitted generator is retained as an explicitly named legacy transform so
the revision audit can reproduce its data-generating rule. New S_alpha S draws
must use `cms_symmetric`.
"""

from __future__ import annotations

import numpy as np


def draw_cms_variates(shape, rng: np.random.RandomState):
    """Draw the CMS base variables U ~ Uniform(-pi/2,pi/2), W ~ Exp(1)."""
    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=shape)
    w = rng.exponential(1.0, size=shape)
    return u, w


def from_cms_variates(alpha: float, u, w, scale: float = 1.0):
    """Symmetric S_alpha S draw with characteristic function exp(-|scale*t|^alpha).

    This is the beta=0 Chambers-Mallows-Stuck transform for 0 < alpha < 2.
    At alpha=2, the same characteristic-function convention is N(0, 2*scale^2).
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0, 2]")
    if scale < 0.0:
        raise ValueError("scale must be nonnegative")
    if scale == 0.0:
        return np.zeros(np.broadcast_shapes(np.shape(u), np.shape(w)), dtype=np.float64)
    if alpha == 2.0:
        raise ValueError("alpha=2 requires Gaussian variates; use gaussian_symmetric")

    u = np.asarray(u, dtype=np.float64)
    w = np.maximum(np.asarray(w, dtype=np.float64), np.finfo(np.float64).tiny)
    cos_u = np.cos(u)
    cos_inner = np.cos((1.0 - alpha) * u)
    if np.any(cos_u <= 0.0) or np.any(cos_inner <= 0.0):
        raise ValueError("CMS trigonometric factors must be positive for symmetric sampling")
    x = (
        np.sin(alpha * u)
        / np.power(cos_u, 1.0 / alpha)
        * np.power(cos_inner / w, (1.0 - alpha) / alpha)
    )
    return scale * x


def gaussian_symmetric(shape, rng: np.random.RandomState, scale: float = 1.0):
    """Gaussian endpoint with characteristic function exp(-|scale*t|^2)."""
    return rng.normal(0.0, np.sqrt(2.0) * scale, size=shape)


def from_submitted_legacy_variates(alpha: float, u, w, scale: float = 1.0):
    """Reproduce the pre-review transform exactly; do not label it S_alpha S."""
    u = np.asarray(u, dtype=np.float64)
    w = np.maximum(np.asarray(w, dtype=np.float64), np.finfo(np.float64).tiny)
    return scale * (
        np.sin(alpha * u)
        * np.power(np.cos(u) / w, (1.0 - alpha) / alpha)
        / np.power(np.cos(u), 1.0 / alpha)
    )
