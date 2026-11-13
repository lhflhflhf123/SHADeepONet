import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ArchitectureTests(unittest.TestCase):
    def test_parameter_count_and_exact_symmetry(self):
        from frequency_study.models import PIDeepONet
        model = PIDeepONet("full_ema")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 27745)
        u = torch.randn(3, 81) * .08
        eta = torch.randn(3, 2)
        times = torch.linspace(0, 10, 81)
        y = model(u, eta, times)
        self.assertTrue(torch.equal(y[:, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(model(-2*u, eta, times), -2*y, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(model(torch.zeros_like(u), eta, times), torch.zeros_like(y)))

    def test_extrema_bounds(self):
        from frequency_study.losses import smooth_indicators, hard_indicators_torch, extrema_gap_bounds
        y = torch.randn(4,81) * .1
        f, r = smooth_indicators(y)
        hf, hr, _ = hard_indicators_torch(y)
        bf, br = extrema_gap_bounds()
        self.assertTrue(torch.all(f <= hf + 1e-5))
        self.assertTrue(torch.all(r >= hr - 1e-6))
        self.assertTrue(torch.all(hf - f <= bf + 1e-5))
        self.assertTrue(torch.all(r - hr <= br + 1e-6))


if __name__ == "__main__":
    unittest.main()
