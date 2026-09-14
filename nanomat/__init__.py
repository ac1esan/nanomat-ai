"""NanoMatAI: band-gap prediction for 2D semiconductors from crystal structure.

Shared code for training (train_cgcnn.py) and inference (screen_bandgap.py),
so that the graph construction and the model architecture can never drift apart.
"""

from .graph import DEFAULT_CUTOFF, DEFAULT_N_RBF, layer_info, ensure_vacuum, to_graph
from .model import CGCNN, CGCNNcls
from .predict import Predictor, corrected_gap, verdict

__all__ = [
    "DEFAULT_CUTOFF", "DEFAULT_N_RBF", "layer_info", "ensure_vacuum", "to_graph",
    "CGCNN", "CGCNNcls", "Predictor", "corrected_gap", "verdict",
]
__version__ = "0.2.0"
