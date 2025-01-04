import unittest
import torch

from usrapprox.usrapprox.models import calculate_logits, probabilistic_model_torch


class TestProbabilistic(unittest.TestCase):
    def test_calculate_logits_linear(self):
        n = torch.tensor([0.1, 0.5, 0.9])
        result = calculate_logits(n, True)
        self.assertEqual(result.shape, (3, 6))
        self.assertTrue(torch.eq(result.sum(), 6.0))

    def test_calculate_logits_nonlinear(self):
        n = torch.linspace(0, 1, 100)
        result = calculate_logits(n)
        self.assertEqual(result.shape, (100, 6))
        # self.assertTrue(torch.eq(result.sum(), 169.7142))

    def test_probabilistic_model_torch_linear(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        n = torch.tensor([0.1, 0.5, 0.9])
        probs = probabilistic_model_torch(n, linear=True)
        self.assertEqual(probs.shape, (3,))
        self.assertTrue(torch.eq(probs.sum(), 11))

    def test_probabilistic_model_torch_nonlinear(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        input_values = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
        probs = probabilistic_model_torch(input_values, linear=False)
        self.assertEqual(probs.shape, (100,))
        self.assertTrue(torch.eq(probs.sum(), 275))
