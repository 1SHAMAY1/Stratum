"""
Stratum Configuration Module
==============================

Centralized application settings using pydantic-settings.
All environment variables and constants are managed here as a single
source of truth for the entire application.

Usage:
    from config.settings import settings
    print(settings.LLM_MODEL)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Application-wide settings for Stratum.

    Reads from environment variables or a .env file automatically.
    All fields with defaults can be overridden via environment variables.

    Attributes:
        GEMINI_API_KEY: Google Gemini API key (optional, defaults to None to avoid import crashes).
        EMBEDDING_MODEL: Gemini model used for generating text embeddings.
        LLM_MODEL: Gemini model used for generating natural language answers.
        MAX_CLUSTERS: Upper bound on K-Means clusters per RAPTOR tree layer.
        MIN_CLUSTER_SIZE: Minimum number of nodes to continue RAPTOR recursion.
        CHUNK_SIZE: Character count per document chunk during ingestion.
        CHUNK_OVERLAP: Overlapping character count between adjacent chunks.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ────────────────────────────────────────────────────────────────
    # Defaulting to None allows the app to import and load cleanly, raising a friendly UI warning instead of crashing on boot.
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="Google Gemini API key. Get one free at https://aistudio.google.com",
    )

    # ── Models ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="models/gemini-embedding-2",
        description="Gemini embedding model identifier used for vectorising text.",
    )
    LLM_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="LLM model identifier used for answer generation and summarisation. Defaults to gemini-2.5-flash for speed and stability.",
    )
    SUMMARIZATION_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Fast LLM model identifier used for background RAPTOR tree summarization to avoid rate limits.",
    )

    # ── RAPTOR Clustering ──────────────────────────────────────────────────
    MAX_CLUSTERS: int = Field(
        default=5,
        ge=2,
        description="Maximum number of K-Means clusters to form per RAPTOR tree layer.",
    )
    MIN_CLUSTER_SIZE: int = Field(
        default=2,
        ge=2,
        description="Minimum number of nodes required to continue RAPTOR recursion upward.",
    )

    # ── Document Chunking ──────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(
        default=1500,
        ge=100,
        description="Number of characters per document chunk during ingestion.",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between consecutive chunks.",
    )


# ── Global singleton ───────────────────────────────────────────────────────
# Import this object everywhere; never instantiate AppSettings directly.
settings = AppSettings()