"""Database connection pool adapter.

Wraps SQLAlchemy's async engine and provides a context-managed
session factory used by all repository modules.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.exceptions import ShipStationError
from app.settings import DATABASE_URL, get_setting

logger = logging.getLogger(__name__)

_POOL_SIZE: int = get_setting("database.pool_size", 20)
_MAX_OVERFLOW: int = get_setting("database.max_overflow", 10)
_ECHO: bool = get_setting("database.echo", False)


class DatabasePool:
    """Manages the async SQLAlchemy engine and session factory."""

    def __init__(self, url: str = DATABASE_URL) -> None:
        self._url = url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Initialise the engine and session factory."""
        logger.info("Connecting to database at %s", self._url.split("@")[-1])
        self._engine = create_async_engine(
            self._url,
            pool_size=_POOL_SIZE,
            max_overflow=_MAX_OVERFLOW,
            echo=_ECHO,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    async def disconnect(self) -> None:
        """Dispose of the engine and release connections."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional session, rolling back on error."""
        if self._session_factory is None:
            raise ShipStationError("DatabasePool not connected")
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise
