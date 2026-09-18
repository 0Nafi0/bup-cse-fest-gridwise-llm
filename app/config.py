"""Application configuration and environment settings.

Manages LLM credentials, model parameters, server bindings, and solver
tolerances via Pydantic Settings and environment variables.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production configuration settings for the GridWise service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    ENV: str = "production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # LLM settings
    LLM_PROVIDER: str = "gemini"  # Supported: "gemini", "openai", "groq", "mock"
    LLM_MODEL: str = "gemini-1.5-flash"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_TIMEOUT_SECONDS: float = 15.0

    # Optimization and numeric validation
    NUMERIC_TOLERANCE: float = 0.01
    PEAK_GRID_REGULARIZATION: float = 1e-4

    def get_effective_api_key(self) -> Optional[str]:
        """Resolve the active API key based on provider or fallback to LLM_API_KEY."""
        if self.LLM_API_KEY:
            return self.LLM_API_KEY
        if self.LLM_PROVIDER.lower() == "gemini":
            return self.GEMINI_API_KEY
        if self.LLM_PROVIDER.lower() in ("openai", "groq"):
            return self.OPENAI_API_KEY
        return None


settings = Settings()
