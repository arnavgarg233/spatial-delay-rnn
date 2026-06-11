"""Spatially embedded, delay-coupled RNNs.

seRNN baseline (Achterberg et al., Nat. Mach. Intell. 2023) plus two ablatable
mechanisms: distance-proportional transmission delays (``delays``) and learnable
neuron coordinates (``geometry``). Configured through
:class:`~sdrnn.model.SDRNNConfig`, so ablations are toggles rather than forks.
"""

from sdrnn.geometry import NeuronGeometry
from sdrnn.model import SDRNN, SDRNNConfig

__all__ = ["SDRNN", "SDRNNConfig", "NeuronGeometry"]
