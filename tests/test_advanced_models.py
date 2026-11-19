"""Protect sequence inputs against future leakage and model shape errors."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

class AdvancedModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.advanced_models'),
                             'advanced models have not been implemented')

    def test_sequence_representation_is_invariant_to_future_scores(self):
        script = '''
from dataclasses import replace
import numpy as np
from analysis.backtest import Curve, snapshot
from analysis.advanced_models import prefix_sequence
times = np.arange(201) * 3600000 + 1
curve = Curve(300, 2000, 'live_try', 1, 200 * 3600000 + 1, times, np.arange(201) * 100.)
cutoff = int(times[100])
changed = curve.scores.copy();changed[101:] *= 1000
first = prefix_sequence(snapshot(curve, cutoff))
second = prefix_sequence(snapshot(replace(curve, scores=changed), cutoff))
np.testing.assert_array_equal(first, second)
assert first.shape == (64, 8) and np.isfinite(first).all()
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_architectures_have_finite_predictions_and_gradients(self):
        # macOS wheels load distinct OpenMP runtimes for torch and XGBoost.
        # Keep CPU backward kernels in their own process, as native libraries
        # can otherwise segfault later XGBoost tests (pytorch/pytorch#155201).
        script = '''
import torch
from analysis.advanced_models import ARCHITECTURES, Forecaster
torch.set_num_threads(2)
for name in ARCHITECTURES:
    model = Forecaster(name, tab_features=12, width=32)
    result = model(torch.randn(4, 64, 8), torch.randn(4, 12))
    assert result.shape == (4,) and torch.isfinite(result).all(), name
    result.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), name
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
