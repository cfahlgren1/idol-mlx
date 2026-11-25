"""Photo to 3D Gaussian splat on Apple Silicon with Idol + Idol Turbo (unofficial)."""

from .client import GenerationError, Idol, Progress
from .splat import Splat

__all__ = ["GenerationError", "Idol", "Progress", "Splat"]
__version__ = "0.1.1"
