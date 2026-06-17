"""Application configuration via Pydantic Settings (env-driven).

Analogi Laravel: ini setara dengan `config/app.php` + `config/database.php`
yang baca dari `.env`. Pydantic Settings otomatis cast type (mis. CORS_ORIGINS
JSON array → Python list).
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    environment: str = "development"
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tki_kos"

    # CORS — list domain frontend yang boleh akses
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # IR config
    indexes_dir: str = "../data/indexes"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    default_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Neural (IndoBERT) di-OFF di production (hemat RAM Render free 512MB).
    # Hidup di notebook/Colab untuk eval. Set ENABLE_NEURAL=true untuk aktif.
    enable_neural: bool = False

    # Rate limit per-IP untuk /api/search + /api/preprocess (request/menit).
    # 0 = nonaktif. In-memory, cukup untuk single worker free tier.
    rate_limit_per_minute: int = 60

    @field_validator("database_url", mode="after")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        """Normalize URL Postgres ke `postgresql+asyncpg://` (SQLAlchemy 2.0 async).

        Render kasih env var DATABASE_URL dengan scheme `postgres://`, sementara
        SQLAlchemy 2.0 minta `postgresql://` minimum, dan untuk async pakai
        `postgresql+asyncpg://`. Validator ini handle 3 case:

        - `postgres://...`            -> `postgresql+asyncpg://...`  (Render default)
        - `postgresql://...`          -> `postgresql+asyncpg://...`  (bare)
        - `postgresql+asyncpg://...`  -> unchanged                   (sudah benar)
        - `postgresql+<other>://...`  -> unchanged                   (explicit driver)
        """
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        # Match bare `postgresql://` (no driver suffix)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


settings = Settings()
