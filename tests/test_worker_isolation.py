import socket
import struct
import subprocess
import sys
import time

import pytest

from rlm_tools import worker_protocol
from rlm_tools.worker_protocol import MAX_MESSAGE_BYTES, WorkerChannel, WorkerProtocolError


def test_ipc_rejects_oversized_frame_before_reading_body():
    parent, peer = socket.socketpair()
    channel = WorkerChannel(parent)
    try:
        peer.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        with pytest.raises(WorkerProtocolError, match="size limit"):
            channel.receive(time.monotonic() + 1)
    finally:
        channel.close()
        peer.close()


def test_ipc_partial_frame_obeys_deadline():
    parent, peer = socket.socketpair()
    channel = WorkerChannel(parent)
    try:
        peer.sendall(struct.pack("!I", 4) + b"{")
        with pytest.raises(TimeoutError):
            channel.receive(time.monotonic() + 0.05)
    finally:
        channel.close()
        peer.close()


def test_ipc_encodes_unicode_without_expanding_valid_characters(monkeypatch):
    monkeypatch.setattr(worker_protocol, "MAX_MESSAGE_BYTES", 256)
    left, right = socket.socketpair()
    sender, receiver = WorkerChannel(left), WorkerChannel(right)
    try:
        original = {"value": "é" * 100 + "\udc80\n\t"}
        sender.send(original, time.monotonic() + 1)
        assert receiver.receive(time.monotonic() + 1) == original
    finally:
        sender.close()
        receiver.close()


def test_ipc_rejects_oversized_output_without_publishing_incomplete_frame(monkeypatch):
    monkeypatch.setattr(worker_protocol, "MAX_MESSAGE_BYTES", 256)
    left, right = socket.socketpair()
    sender, receiver = WorkerChannel(left), WorkerChannel(right)
    try:
        with pytest.raises(WorkerProtocolError, match="size limit"):
            sender.send({"value": "é" * 200}, time.monotonic() + 1)
        sender.send({"value": "valid"}, time.monotonic() + 1)
        assert receiver.receive(time.monotonic() + 1) == {"value": "valid"}
    finally:
        sender.close()
        receiver.close()


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires OS worker confinement")
def test_os_confines_files_network_and_process_creation(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("sample")

    # Exercise the OS boundary directly in a disposable trusted subprocess.
    # Python's REPL import and AST restrictions cannot affect these probes.
    proc = subprocess.run(
        [
            sys.executable, "-c",
            """
import errno
import os
import socket
import sys
from rlm_tools.worker_isolation import confine_worker

def read():
    os.close(os.open(sys.argv[1], os.O_RDONLY))

def write():
    os.close(os.open(sys.argv[1], os.O_WRONLY))

def bind():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))

def fork():
    pid = os.fork()
    if not pid:
        os._exit(0)
    os.waitpid(pid, 0)

fd, peer = socket.socketpair()
confine_worker(fd.fileno())
for probe in (read, write, bind, fork):
    try:
        probe()
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            os._exit(2)
    else:
        os._exit(3)
os._exit(0)
""",
            str(sample),
        ],
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or f"Confinement probe exit status {proc.returncode}"
