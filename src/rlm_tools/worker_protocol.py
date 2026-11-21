import json
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any


def send_message(conn: Connection, message: dict[str, Any]) -> None:
    conn.send_bytes(json.dumps(message).encode("utf-8"))


def receive_message(conn: Connection) -> dict[str, Any]:
    message = json.loads(conn.recv_bytes())
    if not isinstance(message, dict):
        raise ValueError("Expected a worker message object")
    return message


@dataclass
class ExecuteRequest:
    code: str


@dataclass
class ExecuteResponse:
    stdout: str
    error: str | None
    variables: list[str]
    final: dict[str, Any] | None = None
    memory_bytes_used: int = 0


@dataclass
class HelperCall:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class HelperResult:
    value: Any = None
    error: str | None = None
