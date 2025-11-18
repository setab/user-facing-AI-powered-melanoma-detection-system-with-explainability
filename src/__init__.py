"""
Melanoma Detection with Explainability (XAI)
==============================================

A PyTorch-based melanoma detection system with:
- Temperature calibration for reliable probabilities
- Grad-CAM visualizations for explainability
- Interactive Q&A for uncertain diagnoses
- Multi-architecture model comparison
- CLI and web UI interfaces

For documentation, see: docs/
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# Core components
from . import config
from . import inference
from . import training

__all__ = ["config", "inference", "training", "__version__"]
