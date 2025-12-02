from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    GEMINI_API_KEY: str = Field(...)
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")
    LLM_MODEL: str = Field(default="gemini-1.5-flash")

settings = AppSettings()
