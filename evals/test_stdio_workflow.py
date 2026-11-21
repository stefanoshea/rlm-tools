"""Exercise the compact exploration contract from a separate MCP client."""

import asyncio
import itertools
import json
import sys
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp.types import LATEST_PROTOCOL_VERSION


@asynccontextmanager
async def stdio_tools(workdir):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "rlm_tools", cwd=workdir,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
    )
    ids = itertools.count(1)

    async def send(method, params, notification=False):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            message["id"] = next(ids)
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()
        if not notification:
            reply = json.loads(await process.stdout.readline())
            assert reply["id"] == message["id"] and "error" not in reply, reply
            return reply["result"]

    async def call(tool, **args):
        result = await send("tools/call", {"name": tool, "arguments": args})
        assert not result.get("isError"), result
        return json.loads(result["content"][0]["text"])

    try:
        with anyio.fail_after(30):
            await send("initialize", {"protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
                                      "clientInfo": {"name": "offline-eval", "version": "1"}})
            await send("notifications/initialized", {}, notification=True)
            tools = await send("tools/list", {})
            assert {"rlm_start", "rlm_execute", "rlm_end"} <= {tool["name"] for tool in tools["tools"]}
        yield call
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.communicate(), timeout=5)
            assert process.returncode == 0
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_interleaved_sessions_return_only_the_requested_summary(tmp_path):
    roots = [tmp_path / "east", tmp_path / "west"]
    counts = [17, 31]
    for root, count in zip(roots, counts):
        root.mkdir()
        (root / "jobs.txt").write_text("".join(f"queued job-{i}\n" for i in range(count)))

    with anyio.fail_after(30):
        async with stdio_tools(tmp_path) as call:
            sessions = []
            for root in roots:
                started = await call("rlm_start", path=str(root), query="Count queued jobs")
                sessions.append(started["session_id"])
            east, west = sessions
            assert east != west
            for session in sessions:
                stored = await call("rlm_execute", session_id=session, code="jobs = grep('^queued ', 'jobs.txt')")
                assert stored == {"stdout": "", "error": None}

            missing = await call("rlm_execute", session_id=east, code="read_file('missing.txt')")
            assert missing["error"]
            for session, count in zip(sessions, counts):
                summary = await call("rlm_execute", session_id=session, code="print(len(jobs))")
                assert summary == {"stdout": f"{count}\n", "error": None}
                assert (await call("rlm_end", session_id=session))["success"]
                closed = await call("rlm_execute", session_id=session, code="print(len(jobs))")
                assert closed["error"]
