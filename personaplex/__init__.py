"""PersonaPlex: Real-time AI persona voice conversion and synthesis.

A fork of NVIDIA/personaplex providing tools for voice-driven persona
conversion using neural audio processing pipelines.
"""

__version__ = "0.1.0"
__author__ = "PersonaPlex Contributors"
__license__ = "MIT"

from personaplex.pipeline import PersonaPlexPipeline
from personaplex.config import PersonaPlexConfig

__all__ = [
    "PersonaPlexPipeline",
    "PersonaPlexConfig",
    "__version__",
]
