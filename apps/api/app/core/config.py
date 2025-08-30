"""Application configuration."""

from typing import Any

from pydantic import Field, PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "AI Financial Analyst"
    
    # CORS
    ALLOWED_HOSTS: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "ai_financial_analyst"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: PostgresDsn | None = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: str | None, values: dict[str, Any]) -> Any:
        """Assemble database URL from components."""
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # MinIO/S3
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "ai-financial-analyst"
    S3_REGION: str = "us-east-1"
    
    # OpenAI
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    
    # Feature Flags
    ENABLE_AUDIO_PROCESSING: bool = True
    ENABLE_XBRL_PROCESSING: bool = True
    ENABLE_PDF_PROCESSING: bool = True
    
    # Observability
    ENABLE_TRACING: bool = False
    JAEGER_ENDPOINT: str = "http://localhost:14268/api/traces"
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


settings = Settings()
