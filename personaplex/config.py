"""Configuration management for PersonaPlex.

This module handles loading and validating configuration settings
for audio processing, persona management, and model inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "wav"
    # Maximum duration for a single audio segment in seconds
    max_duration: float = 30.0
    # Silence threshold for voice activity detection (0.0 - 1.0)
    silence_threshold: float = 0.02
    silence_duration: float = 0.5  # seconds of silence before segment split


@dataclass
class ModelConfig:
    """Model inference configuration."""

    model_name: str = "nvidia/persona-base"
    device: str = "cuda"
    dtype: str = "float16"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    # Path to local model weights (overrides model_name if set)
    local_model_path: Optional[str] = None
    # Number of parallel inference workers
    num_workers: int = 1


@dataclass
class PersonaConfig:
    """Persona definition and behaviour configuration."""

    name: str = "Assistant"
    voice_id: str = "default"
    # System prompt that defines the persona's behaviour
    system_prompt: str = (
        "You are a helpful, friendly, and concise voice assistant. "
        "Respond naturally as if speaking aloud."
    )
    # Language / locale code
    language: str = "en-US"
    # Speaking rate multiplier (1.0 = normal)
    speaking_rate: float = 1.0


@dataclass
class ServerConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class PersonaPlexConfig:
    """Top-level configuration aggregating all sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Directory used for temporary files and caches
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PERSONAPLEX_CACHE_DIR", "/tmp/personaplex"))
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("PERSONAPLEX_DEBUG", "0") == "1"
    )

    def __post_init__(self) -> None:
        """Ensure required directories exist and apply env-var overrides."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Allow quick device override via environment variable
        env_device = os.environ.get("PERSONAPLEX_DEVICE")
        if env_device:
            self.model.device = env_device

        # Allow model override via environment variable
        env_model = os.environ.get("PERSONAPLEX_MODEL")
        if env_model:
            self.model.model_name = env_model


def load_config(**overrides) -> PersonaPlexConfig:
    """Create a :class:`PersonaPlexConfig` with optional field overrides.

    Args:
        **overrides: Keyword arguments that map to top-level
            :class:`PersonaPlexConfig` fields.

    Returns:
        A fully initialised configuration object.
    """
    return PersonaPlexConfig(**overrides)


# Module-level default config instance — importable directly.
default_config: PersonaPlexConfig = load_config()
