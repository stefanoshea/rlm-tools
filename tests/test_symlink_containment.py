"""Symlinked files inside the sandbox root must not leak content from outside it."""

import os
import tempfile

from rlm_tools.helpers import make_helpers


def _setup(tmpdir: str) -> str:
    """Create a secret file outside the root and a symlink to it inside."""
    outside = os.path.join(tmpdir, "outside")
    inside = os.path.join(tmpdir, "root")
    os.makedirs(outside)
    os.makedirs(inside)
    secret = os.path.join(outside, "secret.txt")
    with open(secret, "w") as f:
        f.write("TOP_SECRET_TOKEN supersecret\n")
    os.symlink(secret, os.path.join(inside, "link.txt"))
    return inside


def test_grep_does_not_follow_symlink_outside_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _setup(tmpdir)
        helpers, _ = make_helpers(root)
        results = helpers["grep"]("TOP_SECRET_TOKEN")
        assert results == []


def test_read_file_does_not_follow_symlink_outside_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _setup(tmpdir)
        helpers, _ = make_helpers(root)
        try:
            helpers["read_file"]("link.txt")
            assert False, "Should have raised PermissionError"
        except PermissionError:
            pass


def test_grep_still_matches_real_files_in_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _setup(tmpdir)
        with open(os.path.join(root, "real.py"), "w") as f:
            f.write("VISIBLE_MARKER = 1\n")
        helpers, _ = make_helpers(root)
        results = helpers["grep"]("VISIBLE_MARKER")
        assert len(results) == 1
        assert results[0]["file"] == "real.py"
