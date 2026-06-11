"""Core invariants of the spatial-delay-RNN: distance→delay mapping and the geometry.

Run: python -m pytest tests/ -q   (or: python tests/test_core.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from sdrnn.delays import integer_delays


def test_integer_delays_round_clip_symmetric():
    """tau_ij = round(d_ij / v), clipped to [1, max_delay]; symmetric when d is."""
    d = torch.tensor([[0.0, 0.8, 1.6],
                      [0.8, 0.0, 0.4],
                      [1.6, 0.4, 0.0]])
    tau = integer_delays(d, velocity=0.08, max_delay=14)
    assert tau.shape == d.shape
    assert (tau >= 1).all(), "minimum delay is 1 step"
    assert (tau <= 14).all(), "delay is clipped at max_delay"
    assert torch.equal(tau, tau.t()), "symmetric distance -> symmetric delay"


def test_integer_delays_monotone_in_distance():
    """Longer physical distance => no shorter conduction delay (the core premise)."""
    d = torch.tensor([[0.0, 0.20, 1.00, 2.00]])
    tau = integer_delays(d, velocity=0.10, max_delay=30)
    taus = tau[0, 1:]
    assert torch.all(taus[1:] >= taus[:-1]), "delay is non-decreasing in distance"


def test_integer_delays_clip_saturates():
    """Very long edges saturate at max_delay rather than running unbounded."""
    d = torch.tensor([[0.0, 100.0]])
    tau = integer_delays(d, velocity=0.08, max_delay=14)
    assert tau[0, 1].item() == 14


if __name__ == "__main__":
    test_integer_delays_round_clip_symmetric()
    test_integer_delays_monotone_in_distance()
    test_integer_delays_clip_saturates()
    print("all core tests passed")
