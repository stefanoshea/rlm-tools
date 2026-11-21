"""Elasticsearch adapter for full-text product search.

Provides indexing and query helpers used by the catalog service
to power storefront search and admin product lookup.
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.exceptions import CatalogError
from app.settings import get_setting

logger = logging.getLogger(__name__)

_ES_HOSTS: list[str] = get_setting("elasticsearch.hosts", ["http://localhost:9200"])
_ES_INDEX: str = get_setting("elasticsearch.default_index", "products")


class ElasticsearchAdapter:
    """Thin async wrapper around the Elasticsearch client."""

    def __init__(
        self, hosts: list[str] | None = None, index: str = _ES_INDEX
    ) -> None:
        self._hosts = hosts or _ES_HOSTS
        self._index = index
        self._client: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        """Initialise the Elasticsearch client."""
        self._client = AsyncElasticsearch(hosts=self._hosts)
        logger.info("Connected to Elasticsearch at %s", self._hosts)

    async def disconnect(self) -> None:
        """Close the Elasticsearch transport."""
        if self._client is not None:
            await self._client.close()

    async def index_document(self, doc_id: str, body: dict[str, Any]) -> None:
        """Index or update a single document."""
        if self._client is None:
            raise CatalogError("ElasticsearchAdapter not connected")
        await self._client.index(index=self._index, id=doc_id, document=body)

    async def search(self, query: dict[str, Any], *, size: int = 20) -> list[dict[str, Any]]:
        """Run a search query and return the hit sources."""
        if self._client is None:
            raise CatalogError("ElasticsearchAdapter not connected")
        result = await self._client.search(index=self._index, query=query, size=size)
        return [hit["_source"] for hit in result["hits"]["hits"]]

    async def delete_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if self._client is None:
            raise CatalogError("ElasticsearchAdapter not connected")
        await self._client.delete(index=self._index, id=doc_id, ignore=[404])
