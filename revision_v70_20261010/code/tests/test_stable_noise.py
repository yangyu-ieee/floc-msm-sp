"""Sanity tests for the corrected symmetric CMS sampler.

Run from the revision root with:
    python -m unittest discover -s code/tests -v

These tests validate distribution-level invariants, not a proof of the CMS
algorithm or a replacement for the experiment source-hash audit.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_noise import (  # noqa: E402
    draw_cms_variates,
    from_cms_variates,
    from_submitted_legacy_variates,
    gaussian_symmetric,
)


class SymmetricCMSSamplerTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        first = from_cms_variates(
            1.5, *draw_cms_variates((4096,), np.random.RandomState(20261007)), scale=0.7
        )
        second = from_cms_variates(
            1.5, *draw_cms_variates((4096,), np.random.RandomState(20261007)), scale=0.7
        )
        np.testing.assert_array_equal(first, second)

    def test_symmetric_sign_balance_and_quantiles(self):
        x = from_cms_variates(
            1.5, *draw_cms_variates((100_000,), np.random.RandomState(731)), scale=1.0
        )
        positive_fraction = np.mean(x > 0)
        self.assertAlmostEqual(positive_fraction, 0.5, delta=0.01)
        self.assertLess(abs(np.median(x)), 0.03)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        self.assertAlmostEqual(q25, -q75, delta=0.12)

    def test_alpha_one_is_cauchy_with_requested_scale(self):
        scale = 0.7
        x = from_cms_variates(
            1.0, *draw_cms_variates((120_000,), np.random.RandomState(911)), scale=scale
        )
        q25, q75 = np.quantile(x, [0.25, 0.75])
        self.assertAlmostEqual(q25, -scale, delta=0.035)
        self.assertAlmostEqual(q75, scale, delta=0.035)

    def test_empirical_characteristic_function_matches_symmetric_stable_target(self):
        alpha, scale = 1.5, 0.8
        x = from_cms_variates(
            alpha,
            *draw_cms_variates((180_000,), np.random.RandomState(1201)),
            scale=scale,
        )
        for frequency in (0.25, 0.5, 1.0):
            observed = np.mean(np.exp(1j * frequency * x))
            expected = np.exp(-abs(scale * frequency) ** alpha)
            self.assertAlmostEqual(observed.real, expected, delta=0.012)
            self.assertAlmostEqual(observed.imag, 0.0, delta=0.012)

    def test_gaussian_endpoint_has_matching_characteristic_function_scale(self):
        scale = 0.6
        rng = np.random.RandomState(1601)
        x = gaussian_symmetric((120_000,), rng, scale=scale)
        self.assertAlmostEqual(np.std(x), np.sqrt(2.0) * scale, delta=0.012)
        for frequency in (0.25, 0.75, 1.0):
            observed = np.mean(np.exp(1j * frequency * x))
            expected = np.exp(-(scale * frequency) ** 2)
            self.assertAlmostEqual(observed.real, expected, delta=0.012)
            self.assertAlmostEqual(observed.imag, 0.0, delta=0.012)

    def test_zero_scale_and_invalid_endpoints(self):
        u, w = draw_cms_variates((32,), np.random.RandomState(5))
        np.testing.assert_array_equal(from_cms_variates(1.5, u, w, scale=0.0), 0.0)
        with self.assertRaises(ValueError):
            from_cms_variates(0.0, u, w)
        with self.assertRaises(ValueError):
            from_cms_variates(2.0, u, w)
        with self.assertRaises(ValueError):
            from_cms_variates(1.5, u, w, scale=-1.0)

    def test_legacy_transform_is_not_silently_the_corrected_transform(self):
        rng = np.random.RandomState(2026)
        u, w = draw_cms_variates((4096,), rng)
        corrected = from_cms_variates(1.5, u, w)
        legacy = from_submitted_legacy_variates(1.5, u, w)
        self.assertFalse(np.array_equal(corrected, legacy))


if __name__ == "__main__":
    unittest.main()
