import os
import tempfile
import pytest
from rlm_tools.helpers import make_helpers


def test_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "hello.txt")
        with open(test_file, "w") as f:
            f.write("hello world")

        helpers, _ = make_helpers(tmpdir)
        content = helpers["read_file"]("hello.txt")
        assert content == "hello world"


def test_read_file_blocks_path_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        helpers, _ = make_helpers(tmpdir)
        try:
            helpers["read_file"]("../../etc/passwd")
            assert False, "Should have raised"
        except PermissionError:
            pass


def test_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "code.py")
        with open(test_file, "w") as f:
            f.write("def hello():\n    pass\ndef world():\n    pass\n")

        helpers, _ = make_helpers(tmpdir)
        results = helpers["grep"]("def.*hello")
        assert len(results) > 0
        assert "hello" in results[0]["line"]


def test_grep_respects_max_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(4):
            with open(os.path.join(tmpdir, f"code{i}.py"), "w") as f:
                f.write("def shared():\n    pass\n")

        helpers, _ = make_helpers(tmpdir)
        results = helpers["grep"]("def", max_matches=2)
        assert len(results) == 2


def test_glob_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()

        helpers, _ = make_helpers(tmpdir)
        py_files = helpers["glob_files"]("**/*.py")
        assert len(py_files) == 2


def test_tree():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        open(os.path.join(tmpdir, "src", "main.py"), "w").close()

        helpers, _ = make_helpers(tmpdir)
        output = helpers["tree"]()
        assert "src" in output
        assert "main.py" in output


def test_read_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("content a")
        with open(os.path.join(tmpdir, "b.txt"), "w") as f:
            f.write("content b")

        helpers, _ = make_helpers(tmpdir)
        result = helpers["read_files"](["a.txt", "b.txt"])
        assert result["a.txt"] == "content a"
        assert result["b.txt"] == "content b"


def test_read_files_handles_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "exists.txt"), "w") as f:
            f.write("hello")

        helpers, _ = make_helpers(tmpdir)
        result = helpers["read_files"](["exists.txt", "missing.txt"])
        assert result["exists.txt"] == "hello"
        assert "[error:" in result["missing.txt"]


def test_read_file_cache_serves_unchanged_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "cached.txt")
        with open(test_file, "w") as f:
            f.write("original")
        st = os.stat(test_file)

        helpers, _ = make_helpers(tmpdir)
        first = helpers["read_file"]("cached.txt")
        # Pin mtime so the file looks unchanged; cached value should be reused.
        os.utime(test_file, (st.st_atime, st.st_mtime))
        second = helpers["read_file"]("cached.txt")
        assert first == "original"
        assert second == "original"


def test_read_file_cache_invalidates_on_modification():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "cached.txt")
        with open(test_file, "w") as f:
            f.write("original")

        helpers, _ = make_helpers(tmpdir)
        first = helpers["read_file"]("cached.txt")
        assert first == "original"

        with open(test_file, "w") as f:
            f.write("modified")
        # Force a strictly newer mtime so the change is detectable regardless of
        # filesystem timestamp resolution.
        st = os.stat(test_file)
        os.utime(test_file, (st.st_atime, st.st_mtime + 10))

        second = helpers["read_file"]("cached.txt")
        assert second == "modified"


def test_read_file_respects_max_bytes_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "large.txt")
        with open(test_file, "w") as f:
            f.write("a" * 200)

        helpers, _ = make_helpers(tmpdir)
        content = helpers["read_file"]("large.txt", max_bytes_read=20)
        assert "... [read truncated]" in content


def test_truncated_read_does_not_poison_later_full_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "large.txt")
        original = "abcdefghij" * 20
        with open(test_file, "w") as f:
            f.write(original)

        helpers, _ = make_helpers(tmpdir)
        truncated = helpers["read_file"]("large.txt", max_bytes_read=20)
        full = helpers["read_file"]("large.txt")

        assert "... [read truncated]" in truncated
        assert full == original


def test_bounded_read_only_caches_complete_content(tmp_path):
    (tmp_path / "large.txt").write_text("é" * 1000, encoding="utf-8")
    cache_sizes = []
    helpers, _ = make_helpers(str(tmp_path), memory_tracker=cache_sizes.append)

    assert helpers["read_file"]("large.txt", max_bytes_read=3) == "é\n... [read truncated]"
    assert cache_sizes == []
    assert helpers["read_file"]("large.txt") == "é" * 1000
    assert cache_sizes == [2000]


def test_cache_charges_only_retained_bytes_after_edits(tmp_path):
    path = tmp_path / "notes.txt"
    charges = []
    helpers, _ = make_helpers(str(tmp_path), memory_tracker=charges.append)
    for content in ("12345", "hello world", "é"):
        path.write_text(content, encoding="utf-8")
        assert helpers["read_file"]("notes.txt") == content
        assert sum(charges) == len(content.encode("utf-8"))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_read_file_skips_named_pipe(tmp_path):
    os.mkfifo(tmp_path / "events")
    helpers, _ = make_helpers(str(tmp_path))
    assert "regular file" in helpers["read_files"](["events"])["events"]


def test_grep_preserves_splitlines_and_end_anchors(tmp_path):
    (tmp_path / "notes.txt").write_text("a\nb\u2028c\n\nd\n")
    helpers, _ = make_helpers(str(tmp_path))
    results = helpers["grep"](r"c\Z|^$")
    assert [(m["line_number"], m["line"]) for m in results] == [(3, "c"), (4, "")]


def test_read_file_does_not_cache_content_when_memory_tracking_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "large.txt")
        with open(test_file, "w") as f:
            f.write("a" * 5000)

        def fail_if_large(size: int) -> None:
            if size > 1024:
                raise MemoryError("too large")

        helpers, _ = make_helpers(tmpdir, memory_tracker=fail_if_large)

        try:
            helpers["read_file"]("large.txt")
            assert False, "Expected MemoryError on first read"
        except MemoryError:
            pass

        try:
            helpers["read_file"]("large.txt")
            assert False, "Expected MemoryError on second read too"
        except MemoryError:
            pass


def test_grep_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write("def hello():\n    pass\ndef world():\n    pass\n")

        helpers, _ = make_helpers(tmpdir)
        output = helpers["grep_summary"]("def")
        assert "2 matches in 1 files:" in output
        assert "code.py" in output
        assert "L1:" in output
        assert "L3:" in output


def test_grep_summary_no_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write("hello world\n")

        helpers, _ = make_helpers(tmpdir)
        output = helpers["grep_summary"]("zzz_nonexistent")
        assert output == "No matches found."


def test_grep_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "reducer.swift"), "w") as f:
            f.write("struct MyReducer: Reducer {\n    var body: some ReducerOf<Self> {\n    }\n}\n")
        with open(os.path.join(tmpdir, "model.swift"), "w") as f:
            f.write("struct Model {\n    var name: String\n}\n")

        helpers, _ = make_helpers(tmpdir)
        result = helpers["grep_read"]("Reducer")
        assert "reducer.swift" in result["matches"]
        assert "model.swift" not in result["matches"]
        assert "struct MyReducer" in result["files"]["reducer.swift"]
        assert "matches in 1 files" in result["summary"]


def test_grep_read_with_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write("line1\nline2\ntarget\nline4\nline5\n")

        helpers, _ = make_helpers(tmpdir)
        result = helpers["grep_read"]("target", context_lines=1)
        content = result["files"]["code.py"]
        assert "L2:" in content
        assert "L3:" in content
        assert "L4:" in content
        assert "L1:" not in content


def test_grep_read_max_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            with open(os.path.join(tmpdir, f"file{i}.py"), "w") as f:
                f.write(f"def func{i}():\n    pass\n")

        helpers, _ = make_helpers(tmpdir)
        result = helpers["grep_read"]("def", max_files=2)
        assert len(result["files"]) == 2
        assert "more" in result["summary"]


def test_read_files_rejects_single_string(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    helpers, _ = make_helpers(str(tmp_path))
    with pytest.raises(TypeError, match="list of file paths"):
        helpers["read_files"]("a.txt")


def test_grep_read_rejects_negative_limits(tmp_path):
    helpers, _ = make_helpers(str(tmp_path))
    with pytest.raises(ValueError, match="max_files"):
        helpers["grep_read"]("x", max_files=-1)
    with pytest.raises(ValueError, match="context_lines"):
        helpers["grep_read"]("x", context_lines=-1)


def test_glob_files_validation_determinism_and_symlink_safety(tmp_path):
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")
    os.symlink(tmp_path / "a.py", tmp_path / "a_alias.py")
    os.symlink(tmp_path / "loop_b", tmp_path / "loop_a")
    os.symlink(tmp_path / "loop_a", tmp_path / "loop_b")

    helpers, _ = make_helpers(str(tmp_path))
    with pytest.raises(PermissionError, match="absolute"):
        helpers["glob_files"]("/etc/*")

    results = helpers["glob_files"]("**/*.py")
    assert results == ["a.py", "b.py"]


def test_grep_skips_internal_symlinks_and_is_sorted(tmp_path):
    (tmp_path / "z.py").write_text("needle\n")
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "hidden.bin").write_text("needle\n")
    os.symlink(tmp_path / "a.py", tmp_path / "0_alias.bin")
    os.symlink(tmp_path / "a.py", tmp_path / "alias_to_a.py")
    os.symlink(tmp_path / "hidden.bin", tmp_path / "alias_to_binary.py")

    helpers, _ = make_helpers(str(tmp_path))
    results = helpers["grep"]("needle")
    assert [match["file"] for match in results] == ["a.py", "z.py"]


def test_helpers_keep_open_workspace_across_rename(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("original")
    helpers, workspace = make_helpers(str(root))
    try:
        root.rename(tmp_path / "moved")
        root.mkdir()
        (root / "notes.txt").write_text("replacement")

        assert helpers["read_file"]("notes.txt") == "original"
        assert helpers["grep"]("original")[0]["file"] == "notes.txt"
        assert helpers["glob_files"]("*.txt") == ["notes.txt"]
    finally:
        workspace.close()
    with pytest.raises(OSError, match="closed"):
        helpers["read_file"]("notes.txt")


def test_helpers_omit_private_paths_and_keep_templates(tmp_path):
    from rlm_tools.helpers import scan_metadata

    (tmp_path / ".env").write_text("example")
    (tmp_path / ".mcp.local.json").write_text("{}")
    (tmp_path / ".env.example").write_text("example")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("example")
    (tmp_path / "notes.txt").write_text("example")
    helpers, _ = make_helpers(str(tmp_path))

    with pytest.raises(PermissionError, match="private"):
        helpers["read_file"](".env")
    with pytest.raises(PermissionError, match="private"):
        helpers["tree"](".git")
    assert sorted(helpers["glob_files"]("*")) == [".env.example", "notes.txt"]
    assert helpers["read_file"](".env.example") == "example"
    assert scan_metadata(str(tmp_path.resolve()))["total_files"] == 1


def test_helpers_skip_symbolic_links(tmp_path):
    (tmp_path / "notes.txt").write_text("example")
    (tmp_path / "alias.txt").symlink_to("notes.txt")
    helpers, _ = make_helpers(str(tmp_path))

    with pytest.raises(PermissionError, match="Symbolic links"):
        helpers["read_file"]("alias.txt")
    assert helpers["glob_files"]("*.txt") == ["notes.txt"]
    assert [result["file"] for result in helpers["grep"]("example")] == ["notes.txt"]


def test_read_limits_apply_to_default_reads_and_batches(tmp_path, monkeypatch):
    from rlm_tools import helpers as helper_module

    monkeypatch.setattr(helper_module, "MAX_FILE_BYTES", 40)
    monkeypatch.setattr(helper_module, "MAX_BATCH_BYTES", 95)
    for name in ("one", "two", "three"):
        (tmp_path / name).write_text("é" * 30, encoding="utf-8")
    helpers, _ = make_helpers(str(tmp_path))

    content = helpers["read_file"]("one")
    assert content == "é" * 20 + "\n... [read truncated]"
    batch = helpers["read_files"](["one", "two", "three"])
    assert batch["one"] == content
    assert "three" not in batch
    assert sum(len(k.encode()) + len(v.encode()) for k, v in batch.items()) <= 95


def test_grep_ignores_incomplete_line_when_file_is_truncated(tmp_path, monkeypatch):
    from rlm_tools import helpers as helper_module

    monkeypatch.setattr(helper_module, "MAX_FILE_BYTES", 10)
    (tmp_path / "notes.txt").write_text("match\nmatching")
    helpers, _ = make_helpers(str(tmp_path))

    pattern = r"^match$|^matc$"
    with pytest.raises(RuntimeError, match="scan truncated"):
        helpers["grep"](pattern)
    summary = helpers["grep_summary"](pattern)
    assert "L1: match" in summary
    assert "L2:" not in summary
    assert "scan truncated" in summary
    assert helpers["grep_summary"]("absent") == "[scan truncated: workspace limit reached]"
    assert "scan truncated" in helpers["grep_read"]("absent")["summary"]


def test_grep_rejects_unsupported_pattern_without_a_fallback(tmp_path):
    (tmp_path / "notes.txt").write_text("project notes")
    helpers, workspace = make_helpers(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="unsupported RE2 pattern"):
            helpers["grep"]("project(?= notes)")
        assert helpers["grep"]("(?i)NOTES")[0]["line"] == "project notes"
    finally:
        workspace.close()


def test_metadata_marks_bounded_line_counts(tmp_path, monkeypatch):
    from rlm_tools import helpers as helper_module

    monkeypatch.setattr(helper_module, "MAX_FILE_BYTES", 5)
    (tmp_path / "notes.txt").write_text("line\nnext\n")
    metadata = helper_module.scan_metadata(str(tmp_path.resolve()))

    assert metadata["total_files"] == 1
    assert metadata["total_lines"] == 1
    assert metadata["truncated"] is True


def test_helpers_skip_regular_files_with_multiple_hard_links(tmp_path):
    from rlm_tools.helpers import scan_metadata

    (tmp_path / "draft.txt").write_text("example")
    os.link(tmp_path / "draft.txt", tmp_path / "draft-copy.txt")
    helpers, _ = make_helpers(str(tmp_path))

    with pytest.raises(PermissionError, match="multiple hard links"):
        helpers["read_file"]("draft.txt")
    assert helpers["grep"]("example") == []
    assert helpers["glob_files"]("*.txt") == []
    assert scan_metadata(str(tmp_path.resolve()))["total_files"] == 0


def test_automatic_result_limit_reports_truncation(tmp_path, monkeypatch):
    from rlm_tools import helpers as helper_module

    monkeypatch.setattr(helper_module, "MAX_RESULTS", 2)
    for name in ("one", "two", "three"):
        (tmp_path / name).write_text("example")
    helpers, _ = make_helpers(str(tmp_path))

    assert len(helpers["grep"]("example", max_matches=2)) == 2
    with pytest.raises(RuntimeError, match="scan truncated"):
        helpers["grep"]("example")
    with pytest.raises(RuntimeError, match="scan truncated"):
        helpers["glob_files"]("*")
    assert "scan truncated" in helpers["tree"]()
