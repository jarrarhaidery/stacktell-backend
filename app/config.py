from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq (free LLM)
    groq_api_key: str

    # App
    app_env: str = "development"
    allowed_origins: str = "http://localhost:5173"

    # GitHub
    github_token: str = ""

    # Storage
    storage_path: str = "./storage"

    # LLM
    groq_model: str = "llama3-8b-8192"
    max_tokens: int = 1500
    temperature: float = 0.2

    # Embeddings (local HuggingFace, no API key)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_results: int = 6

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
