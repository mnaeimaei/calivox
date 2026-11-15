"""CaliVox — calibrated 3D image classification.

From a manifest of volumes (NIfTI / NumPy) to cross-validated, calibrated and ensembled
class probabilities, with a frozen runtime bundle for new scans.
"""

__version__ = "0.1.0"

from .config import ExperimentConfig  # noqa: E402

__all__ = ["ExperimentConfig", "__version__"]
