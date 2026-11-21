"""Product search powered by Elasticsearch.

Wraps the low-level ES adapter with domain-aware query building
and result mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from app.catalog.models import Product

logger = logging.getLogger(__name__)

_ES_INDEX = "products"


class ElasticsearchAdapter:
    """Thin wrapper around the Elasticsearch client."""

    def __init__(self, hosts: list[str] | None = None) -> None:
        self._hosts = hosts or ["http://localhost:9200"]

    def search(self, index: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError("ES client not configured")


class SearchService:
    """High-level product search interface."""

    def __init__(self, adapter: ElasticsearchAdapter | None = None) -> None:
        self._adapter = adapter or ElasticsearchAdapter()

    def search_products(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[Product]:
        """Search for products matching *query* with optional facet filters."""
        body = self._build_query(query, filters or {})
        logger.info("Searching ES index=%s query=%s", _ES_INDEX, query)
        hits = self._adapter.search(_ES_INDEX, body)
        return [Product(**hit["_source"]) for hit in hits]

    @staticmethod
    def _build_query(query: str, filters: dict[str, Any]) -> dict[str, Any]:
        must = [{"multi_match": {"query": query, "fields": ["name^2", "description"]}}]
        filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
        return {"query": {"bool": {"must": must, "filter": filter_clauses}}}
