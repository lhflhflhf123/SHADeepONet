import unittest

import numpy as np

from frequency_study.physics import GridConfig, exact_zoh, hard_indicators, net_disturbance_pu, positive_load_step
from frequency_study.losses_numpy import discrete_defects


class FirstOrderTests(unittest.TestCase):
    def test_step_matches_independent_closed_form_and_sign(self):
        cfg = GridConfig()
        u = np.full(cfg.n_points, net_disturbance_pu(30.0))
        result = exact_zoh(u, 4.0, 20.0)
        np.testing.assert_allclose(result, positive_load_step(30.0, 4.0, 20.0), atol=2e-15)
        self.assertEqual(result[0], 0)
        self.assertLess(result[-1], 0)
        self.assertTrue(np.all(np.diff(result) <= 0))

    def test_initial_rocof_and_h_d_dependence(self):
        cfg = GridConfig()
        u = np.full(81, net_disturbance_pu(20.0))
        y = exact_zoh(u, 2.0, 15.0)
        post_step_derivative = 50.0 * u[0] / 4.0
        self.assertAlmostEqual(post_step_derivative, -1.25)
        # Forward-difference RoCoF is a discrete metric, not the event derivative.
        self.assertGreater(abs((y[1] - y[0]) / cfg.dt), 0)
        self.assertLess(abs((y[1] - y[0]) / cfg.dt), abs(post_step_derivative))
        self.assertGreater(abs(exact_zoh(u, 2, 15)[-1]), abs(exact_zoh(u, 2, 35)[-1]))
        self.assertGreater(abs(exact_zoh(u, 2, 15)[1]), abs(exact_zoh(u, 8, 15)[1]))

    def test_amplitude_linearity_and_zero_input(self):
        rng = np.random.default_rng(17)
        u = rng.normal(size=81) * .02
        for factor in (-1.8, 0.0, 2.4):
            np.testing.assert_allclose(exact_zoh(factor*u, 3.3, 26.0), factor*exact_zoh(u, 3.3, 26.0), atol=1e-15)

    def test_strict_or_rule(self):
        y = np.zeros(81)
        self.assertFalse(hard_indicators(y)[2])
        y[3] = -.5
        self.assertTrue(hard_indicators(y)[2])  # equal f limit, RoCoF violation
        y = np.full(81, -.5)
        self.assertFalse(hard_indicators(y)[2])  # equality alone is safe

    def test_reference_defect_is_not_zero_for_trapezoid(self):
        u = np.zeros(81)
        u[20:40] = -.12
        y = exact_zoh(u, 4, 20)
        strong, integral = discrete_defects(y, u, 4, 20)
        self.assertGreater(np.max(np.abs(strong[20:35])), 0)
        self.assertGreater(abs(integral[19]), 0)


if __name__ == "__main__":
    unittest.main()
