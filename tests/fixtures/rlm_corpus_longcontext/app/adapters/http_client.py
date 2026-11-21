"""HTTP client adapter with automatic retry logic.

Used by notification channels and external API integrations to make
outbound HTTP calls with configurable backoff and timeout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.exceptions import ShipStationError
from app.settings import get_setting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for HTTP retry behaviour."""

    max_retries: int = 3
    backoff_factor: float = 0.5
    retryable_status_codes: tuple[int, ...] = (429, 502, 503, 504)


DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=get_setting("stripe.max_retries", 3),
    backoff_factor=0.5,
)


class HttpClient:
    """Async HTTP client with built-in retry and timeout support."""

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 30.0,
        retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._retry_config = retry_config
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the underlying ``httpx.AsyncClient``."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )

    async def close(self) -> None:
        """Shut down the HTTP client gracefully."""
        if self._client is not None:
            await self._client.aclose()

    async def request(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute an HTTP request with retry logic.

        Raises :class:`~app.exceptions.ShipStationError` after exhausting
        all retry attempts.
        """
        assert self._client is not None, "HttpClient not started"
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_config.max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
                if response.status_code not in self._retry_config.retryable_status_codes:
                    return response
                logger.warning(
                    "Retryable status %d on attempt %d for %s %s",
                    response.status_code, attempt, method, path,
                )
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning("Transport error on attempt %d: %s", attempt, exc)
        raise ShipStationError(
            f"HTTP {method} {path} failed after {self._retry_config.max_retries} attempts",
            code="HTTP_RETRY_EXHAUSTED",
        )
