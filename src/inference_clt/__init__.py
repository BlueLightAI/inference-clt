"""inference_clt — standalone inference and weight loading for pretrained models."""

__version__ = "0.1.0"

from inference_clt.inference_clt import InferenceCLT
from inference_clt.activation_loader import ActivationLoader

__all__ = ["InferenceCLT", "ActivationLoader"]
