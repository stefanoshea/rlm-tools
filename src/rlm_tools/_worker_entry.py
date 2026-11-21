"""Isolated subprocess entry point. Run with Python's -I -S flags."""

import os
import socket
import sys
from pathlib import Path

# Isolated startup skips site initialization and omits this script's path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm_tools.worker_protocol import WorkerChannel
from rlm_tools.worker_runtime import worker_main


if __name__ == "__main__":
    # CPython may set LC_CTYPE while starting with an empty environment.
    os.environ.clear()
    fd, output_limit, timeout, memory_limit, address_limit, cpu_limit = map(int, sys.argv[1:])
    with socket.socket(fileno=fd) as sock:
        worker_main(
            WorkerChannel(sock),
            output_limit,
            timeout,
            memory_limit,
            address_limit,
            cpu_limit,
            channel_fd=fd,
        )
