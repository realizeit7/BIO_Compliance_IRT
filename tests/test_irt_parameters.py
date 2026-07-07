"""
Unit tests for the core Rasch (1PL) math in irt_parameters.py.

These are deterministic and offline — no API calls, no DB. They lock the
mathematical contract of the difficulty estimator, the ability MLE, and the
pass-rate retention classifier.

Execute via:
    python3 -m unittest tests.test_irt_parameters
    # or
    pytest tests/test_irt_parameters.py
"""

from __future__ import annotations

import math
import unittest

import irt_parameters as irt


class TestComputeBFromPassRate(unittest.TestCase):
    def test_half_pass_rate_gives_zero_difficulty(self):
        # P = 0.5 → b = ln(0.5/0.5) = 0
        self.assertAlmostEqual(irt.compute_b_from_pass_rate(0.5), 0.0, places=6)

    def test_monotonic_harder_items_have_higher_b(self):
        # lower pass rate → harder item → larger positive b
        b_hard = irt.compute_b_from_pass_rate(0.1)
        b_med = irt.compute_b_from_pass_rate(0.5)
        b_easy = irt.compute_b_from_pass_rate(0.9)
        self.assertGreater(b_hard, b_med)
        self.assertGreater(b_med, b_easy)

    def test_matches_negative_logit_formula(self):
        # b = ln((1 - P) / P) for an interior pass rate
        p = 0.3
        expected = round(math.log((1.0 - p) / p), 4)
        self.assertAlmostEqual(irt.compute_b_from_pass_rate(p), expected, places=6)

    def test_clipping_keeps_b_finite_at_extremes(self):
        # P = 0 and P = 1 would blow up without clipping to [eps, 1 - eps]
        b_all_fail = irt.compute_b_from_pass_rate(0.0)
        b_all_pass = irt.compute_b_from_pass_rate(1.0)
        self.assertTrue(math.isfinite(b_all_fail))
        self.assertTrue(math.isfinite(b_all_pass))
        # symmetric around 0 at the default epsilon = 0.01
        self.assertAlmostEqual(b_all_fail, -b_all_pass, places=6)
        self.assertGreater(b_all_fail, 4.0)  # ≈ +4.60


class TestPCorrect(unittest.TestCase):
    def test_theta_equals_b_gives_half(self):
        # By definition b is the θ where P(correct) = 0.5
        self.assertAlmostEqual(irt.p_correct(theta=1.3, a=1.0, b=1.3), 0.5, places=6)

    def test_probability_increases_with_ability(self):
        low = irt.p_correct(theta=-1.0, a=1.0, b=0.0)
        high = irt.p_correct(theta=2.0, a=1.0, b=0.0)
        self.assertLess(low, high)
        self.assertTrue(0.0 < low < 0.5 < high < 1.0)


class TestEstimateThetaMLE(unittest.TestCase):
    def test_empty_responses_returns_prior(self):
        self.assertEqual(irt.estimate_theta_mle([], theta_init=0.37), 0.37)

    def test_all_fail_returns_lower_boundary(self):
        responses = [(0.0, False), (1.0, False), (2.0, False)]
        self.assertEqual(irt.estimate_theta_mle(responses), -4.0)

    def test_all_pass_returns_upper_boundary(self):
        responses = [(0.0, True), (1.0, True), (2.0, True)]
        self.assertEqual(irt.estimate_theta_mle(responses), 6.0)

    def test_symmetric_pattern_recovers_center(self):
        # Pass every item below b=0, fail every item above → θ should sit near 0.
        responses = [(-2.0, True), (-1.0, True), (1.0, False), (2.0, False)]
        theta = irt.estimate_theta_mle(responses)
        self.assertAlmostEqual(theta, 0.0, places=2)

    def test_mle_satisfies_score_equation(self):
        # At the MLE, Σ (r_i - P(θ, b_i)) ≈ 0 for a mixed response vector.
        responses = [(-1.0, True), (0.0, True), (0.5, False), (1.5, False), (2.0, True)]
        theta = irt.estimate_theta_mle(responses)
        score = sum((1.0 if r else 0.0) - irt.p_correct(theta, irt.RASCH_A, b)
                    for b, r in responses)
        self.assertAlmostEqual(score, 0.0, places=3)


class TestPassRateClassification(unittest.TestCase):
    def test_too_hard_below_threshold(self):
        # P < 0.05 → discarded_too_hard
        self.assertEqual(
            irt.classify_item_from_pass_rate(0.04),
            irt.RetentionOutcome.TOO_HARD,
        )

    def test_easy_above_threshold(self):
        # P > 0.95 → retained_easy
        self.assertEqual(
            irt.classify_item_from_pass_rate(0.96),
            irt.RetentionOutcome.RETAINED_EASY,
        )

    def test_informative_band_retained(self):
        for p in (0.05, 0.5, 0.95):
            self.assertEqual(
                irt.classify_item_from_pass_rate(p),
                irt.RetentionOutcome.RETAINED,
                msg=f"pass_rate={p} should be retained",
            )


if __name__ == "__main__":
    unittest.main()
