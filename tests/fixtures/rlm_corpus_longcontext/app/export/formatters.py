"""Output formatters for export pipelines.

Each formatter converts an in-memory dataset to a specific serialisation
format.  All formatters implement the same ``format`` interface.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any


class CsvFormatter:
    """Formats data as comma-separated values."""

    def format(self, data: list[dict[str, Any]]) -> str:
        """Serialise *data* as a CSV string with a header row."""
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()


class JsonFormatter:
    """Formats data as a JSON array."""

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def format(self, data: list[dict[str, Any]]) -> str:
        """Serialise *data* as a pretty-printed JSON string."""
        return json.dumps(data, indent=self._indent, default=str)


class XmlFormatter:
    """Formats data as simple XML elements."""

    ROOT_TAG = "records"
    ROW_TAG = "record"

    def format(self, data: list[dict[str, Any]]) -> str:
        """Serialise *data* as an XML document string."""
        lines: list[str] = [f"<{self.ROOT_TAG}>"]
        for row in data:
            lines.append(f"  <{self.ROW_TAG}>")
            for key, value in row.items():
                lines.append(f"    <{key}>{value}</{key}>")
            lines.append(f"  </{self.ROW_TAG}>")
        lines.append(f"</{self.ROOT_TAG}>")
        return "\n".join(lines)
