"""Configuration management using pydantic."""

from typing import Optional, List
from pydantic import BaseSettings, Field
import os


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug_mode: bool = Field(default=False, env="DEBUG")
    
    # LLM Provider Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: Optional[str] = Field(default=None, env="AZURE_OPENAI_DEPLOYMENT")
    
    # Research Intelligence
    pubmed_email: Optional[str] = Field(default="researcher@example.com", env="PUBMED_EMAIL")
    pubmed_api_key: Optional[str] = Field(default=None, env="PUBMED_API_KEY")
    
    # Statistical Configuration
    default_alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    default_power: float = Field(default=0.8, ge=0.5, le=0.99)
    default_n_simulations: int = Field(default=10000, ge=100, le=100000)
    
    # Visualization Settings
    plot_theme: str = Field(default="plotly_white", env="PLOT_THEME")
    default_color_scheme: List[str] = Field(
        default=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )
    
    # Performance Settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_available_llm_providers(self) -> List[str]:
        """Return list of configured LLM providers."""
        providers = []
        if self.openai_api_key:
            providers.append("OPENAI")
        if self.gemini_api_key:
            providers.append("GEMINI")
        if self.anthropic_api_key:
            providers.append("ANTHROPIC")
        if self.azure_openai_api_key:
            providers.append("AZURE_OPENAI")
        return providers if providers else ["Default"]


# Singleton instance
settings = Settings()