import json
import socket
import struct
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any


MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_HEADER = struct.Struct("!I")
_ENCODING_CHUNK_CHARS = 16 * 1024
_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))


class WorkerProtocolError(ValueError):
    pass


class WorkerChannel:
    """A duplex stream with bounded frames and an optional absolute deadline."""

    def __init__(self, sock: socket.socket):
        self._socket = sock
        self.closed = False

    def close(self) -> None:
        self._socket.close()
        self.closed = True

    def poll(self, timeout: float) -> bool:
        return bool(wait([self._socket], timeout))

    def _set_deadline(self, deadline: float | None) -> None:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Worker IPC deadline exceeded")
            self._socket.settimeout(remaining)

    def _receive_exactly(self, length: int, deadline: float | None) -> bytearray:
        buffer = bytearray(length)
        view = memoryview(buffer)
        while view:
            self._set_deadline(deadline)
            length_read = self._socket.recv_into(view)
            if not length_read:
                raise EOFError("Worker IPC channel closed")
            view = view[length_read:]
        return buffer

    def receive(self, deadline: float | None) -> dict[str, Any]:
        length, = _HEADER.unpack(self._receive_exactly(_HEADER.size, deadline))
        if length > MAX_MESSAGE_BYTES:
            raise WorkerProtocolError("Worker IPC message exceeds the size limit")
        try:
            message = json.loads(self._receive_exactly(length, deadline))
        except (ValueError, RecursionError) as exc:
            raise WorkerProtocolError("Invalid worker IPC message") from exc
        if not isinstance(message, dict):
            raise WorkerProtocolError("Expected a worker message object")
        return message

    def send(self, message: dict[str, Any], deadline: float | None) -> None:
        payload = bytearray()
        for chunk in _ENCODER.iterencode(message):
            if len(chunk) > MAX_MESSAGE_BYTES - len(payload):
                raise WorkerProtocolError("Worker IPC message exceeds the size limit")
            for offset in range(0, len(chunk), _ENCODING_CHUNK_CHARS):
                piece = chunk[offset:offset + _ENCODING_CHUNK_CHARS]
                # JSON permits escaped lone surrogates, though UTF-8 does not.
                encoded = piece.encode("utf-8", errors="backslashreplace")
                if len(encoded) > MAX_MESSAGE_BYTES - len(payload):
                    raise WorkerProtocolError("Worker IPC message exceeds the size limit")
                payload.extend(encoded)
        self._set_deadline(deadline)
        self._socket.sendall(_HEADER.pack(len(payload)))
        self._set_deadline(deadline)
        self._socket.sendall(payload)


def send_message(conn: WorkerChannel, message: dict[str, Any], deadline: float | None = None) -> None:
    conn.send(message, deadline)


def receive_message(conn: WorkerChannel, deadline: float | None = None) -> dict[str, Any]:
    return conn.receive(deadline)


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
