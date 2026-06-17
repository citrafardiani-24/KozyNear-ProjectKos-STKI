"""Async SQLAlchemy engine + session factory.

Pattern: yield AsyncSession per-request via FastAPI dependency.

Analogi Laravel: ini setara dengan Database Manager + DB facade.
- Eloquent ORM = SQLAlchemy ORM
- DB::transaction = async with session.begin()
- DB::table('x') = select(X)
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base untuk semua ORM models.

    Subclass ini di setiap model file:
        from app.core.db import Base
        class Listing(Base): ...
    """


engine = create_async_engine(
    settings.database_url,
    echo=settings.environment == "development",
    pool_pre_ping=True,  # validate connection before use (avoid stale conns)
)


async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # objects accessible after commit
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield session per request, auto cleanup + rollback.

    Usage di route:
        @router.get("/x")
        async def handler(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables dari metadata. Untuk DEV ONLY.

    Production: pakai Alembic migrations (lihat backend/alembic/).
    Tim TODO: setup Alembic Week 2:
        cd backend
        alembic init alembic
        # edit alembic/env.py untuk import Base
        alembic revision --autogenerate -m "initial schema"
        alembic upgrade head
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
