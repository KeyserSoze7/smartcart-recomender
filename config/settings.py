from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017/smartcart"
    mongo_db: str = "smartcart"

    # Redis
    redis_uri: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 600  # 10-minute cache TTL

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Recommendation settings
    candidate_pool_size: int = 100   # CF + CB each generate this many candidates
    llm_input_size: int = 20         # Top-N candidates fed to LLM re-ranker
    final_output_size: int = 10      # Final recommendations returned

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
