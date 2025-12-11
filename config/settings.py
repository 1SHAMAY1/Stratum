from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    GEMINI_API_KEY: str = Field(...)
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")
    LLM_MODEL: str = Field(default="gemini-1.5-flash")
    MAX_CLUSTERS: int = Field(default=5, ge=2)
    MIN_CLUSTER_SIZE: int = Field(default=2, ge=2)
    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)

settings = AppSettings()
