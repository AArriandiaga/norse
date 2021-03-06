"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import benchmark, task
from .torch import functional, models, module, util

__all__ = ["task", "benchmark", "functional", "models", "module", "util"]
