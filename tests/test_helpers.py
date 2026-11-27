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
    assert summary.startswith("TRUNCATED (")
    assert "L1: match" in summary
    assert "L2:" not in summary
    assert "scan truncated" in summary
    absent_summary = helpers["grep_summary"]("absent")
    assert absent_summary.startswith("TRUNCATED (")
    assert "scan truncated" in absent_summary
    read_res = helpers["grep_read"]("absent")
    assert read_res["summary"].startswith("TRUNCATED (")
    assert read_res["truncated"] is True
    assert read_res["limit_hit"] == "file_bytes"


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
    tree_out = helpers["tree"]()
    assert tree_out.startswith("TRUNCATED (")
    assert "scan truncated" in tree_out


def test_gitignore_filtering_and_nested_rules(tmp_path):
    (tmp_path / ".gitignore").write_text(
        "# comment\n"
        "*.log\n"
        "!important.log\n"
        "ignored_dir/\n"
        "/root_only.txt\n"
        "**/generated/*.py\n"
    )
    (tmp_path / "app.py").write_text("needle\n")
    (tmp_path / "debug.log").write_text("needle\n")
    (tmp_path / "important.log").write_text("needle\n")
    (tmp_path / "root_only.txt").write_text("needle\n")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "secret.py").write_text("needle\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "root_only.txt").write_text("needle\n")
    (tmp_path / "sub" / ".gitignore").write_text("local_ignore.py\n")
    (tmp_path / "sub" / "local_ignore.py").write_text("needle\n")
    (tmp_path / "sub" / "generated").mkdir()
    (tmp_path / "sub" / "generated" / "gen.py").write_text("needle\n")

    helpers, _ = make_helpers(str(tmp_path))
    files = [m["file"] for m in helpers["grep"]("needle")]
    assert files == ["app.py", "important.log", "sub/root_only.txt"]
    tree_out = helpers["tree"](max_depth=4)
    assert "ignored_dir" not in tree_out
    assert "local_ignore.py" not in tree_out


def test_gitignore_symlink_is_ignored_safely(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("*.py\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").symlink_to(outside)
    (repo / "keep.py").write_text("needle\n")

    helpers, _ = make_helpers(str(repo))
    assert [m["file"] for m in helpers["grep"]("needle")] == ["keep.py"]


def test_binary_nul_and_minified_files_skipped_before_byte_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_MAX_SCAN_BYTES", "200")
    (tmp_path / "nul_blob.dat").write_bytes(b"needle\x00" + b"x" * 5000)
    (tmp_path / "big.min.json").write_bytes(b"{" + b'"a":1,' * 250_000 + b'"needle":1}')
    (tmp_path / "real.py").write_text("needle = 1\n")

    from rlm_tools import helpers as helper_module
    monkeypatch.setattr(helper_module, "_MINIFIED_SIZE_CAP", 1024)

    helpers, _ = make_helpers(str(tmp_path))
    res = helpers["grep_read"]("needle")
    assert res["truncated"] is False
    assert res["files_scanned"] == 1
    assert res["files_skipped"] == 2
    assert list(res["matches"]) == ["real.py"]


def test_scan_limit_env_overrides_and_partial_grep(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_MAX_SCAN_ENTRIES", "2")
    for idx in range(5):
        (tmp_path / f"f{idx}.py").write_text("hit = 1\n")

    helpers, _ = make_helpers(str(tmp_path))
    with pytest.raises(RuntimeError, match="TRUNCATED \\(2 entry scan limit"):
        helpers["grep"]("hit")

    partial = helpers["grep"]("hit", allow_partial=True)
    assert len(partial) <= 2

    summary = helpers["grep_summary"]("hit")
    assert summary.splitlines()[0].startswith("TRUNCATED (2 entry scan limit;")


def test_grep_clips_long_lines_around_match(tmp_path):
    long_line = ("a" * 600) + "NEEDLE_TOKEN" + ("b" * 600)
    (tmp_path / "big.txt").write_text(long_line + "\n")

    helpers, _ = make_helpers(str(tmp_path))
    matches = helpers["grep"]("NEEDLE_TOKEN")
    assert len(matches) == 1
    assert "NEEDLE_TOKEN" in matches[0]["line"]
    assert len(matches[0]["line"]) <= 500

    summary = helpers["grep_summary"]("NEEDLE_TOKEN")
    assert "NEEDLE_TOKEN" in summary
    match_lines = [ln for ln in summary.splitlines() if "L1:" in ln]
    assert len(match_lines) == 1
    assert len(match_lines[0]) <= 320


def test_dir_cache_invalidates_when_file_added(tmp_path):
    (tmp_path / "a.py").write_text("token = 1\n")
    helpers, _ = make_helpers(str(tmp_path))
    assert [m["file"] for m in helpers["grep"]("token")] == ["a.py"]

    (tmp_path / "b.py").write_text("token = 2\n")
    st = os.stat(tmp_path)
    os.utime(tmp_path, (st.st_atime, st.st_mtime + 10))
    assert [m["file"] for m in helpers["grep"]("token")] == ["a.py", "b.py"]


def test_ast_enclosing_and_find_assignments(tmp_path):
    sample = tmp_path / "flow.py"
    sample.write_text(
        "TOP = 1\n"
        "\n"
        "class RunFlow:\n"
        "    def __init__(self):\n"
        "        self.table_envelopes: list[str] = []\n"
        "\n"
        "    async def run(self):\n"
        "        def _nested_helper():\n"
        "            return 1\n"
        "\n"
        "        self.table_envelopes += ['a']\n"
        "        if (x := len(self.table_envelopes)) > 0:\n"
        "            self.table_envelopes = ['b']\n"
        "        return x\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text("def oops(:\n", encoding="utf-8")

    helpers, _ = make_helpers(str(tmp_path))
    assert helpers["enclosing"]("flow.py", 1)["qualified"] == "<module>"
    assert helpers["enclosing"]("flow.py", 9)["qualified"] == "RunFlow.run._nested_helper"
    # Line 11 is after _nested_helper closed: must resolve to RunFlow.run, not _nested_helper
    enc = helpers["enclosing"]("flow.py", 11)
    assert enc["qualified"] == "RunFlow.run"
    assert enc["chain"] == ["RunFlow", "run"]

    hits = helpers["find_assignments"]("flow.py", "self.table_envelopes")
    assert [(h["line_number"], h["kind"], h["enclosing"]) for h in hits] == [
        (5, "ann_assign", "RunFlow.__init__"),
        (11, "aug_assign", "RunFlow.run"),
        (13, "assign", "RunFlow.run"),
    ]
    named = helpers["find_assignments"]("flow.py", "x")
    assert [(h["line_number"], h["kind"], h["enclosing"]) for h in named] == [
        (12, "named_expr", "RunFlow.run"),
    ]

    with pytest.raises(ValueError, match="parse Python AST"):
        helpers["enclosing"]("broken.py", 1)


def test_gitignore_char_classes_escapes_reinclusion_redos_and_explicit_subdirs(tmp_path):
    import time as _time
    from rlm_tools._workspace import _compile_gitignore_glob

    # ReDoS regression: adversarial repeated wildcards must finish in <50ms under RE2.
    adversarial = _compile_gitignore_glob("*a*a*a*a*a*a*a*a*a*a*a*a*a*b")
    t0 = _time.monotonic()
    assert adversarial.match("a" * 29 + "c") is None
    assert (_time.monotonic() - t0) < 0.05

    # Character classes: negated [!a-z], leading ] in negated [!]], range spanning '/' [.-0], empty [/].
    assert _compile_gitignore_glob("[!a-z].txt").match("1.txt") is not None
    assert _compile_gitignore_glob("[!a-z].txt").match("a.txt") is None
    assert _compile_gitignore_glob("[!]].txt").match("].txt") is None
    assert _compile_gitignore_glob("[!]].txt").match("a.txt") is not None
    assert _compile_gitignore_glob("[!-a].txt").match("-.txt") is None
    assert _compile_gitignore_glob("[!-a].txt").match("a.txt") is None
    assert _compile_gitignore_glob("[!-a].txt").match("0.txt") is not None
    r_span = _compile_gitignore_glob("[.-0].txt")
    assert r_span.match("..txt") is not None
    assert r_span.match("0.txt") is not None
    assert r_span.match("/.txt") is None
    assert _compile_gitignore_glob("[/]").match("/") is None

    # Escaped leading !, #, trailing space, and explicit subdir targets.
    (tmp_path / ".gitignore").write_text(
        "\\!important.txt\n"
        "\\#hash.txt\n"
        "trailing\\ \n"
        "ignored_dir/\n"
        "/anchored_ignored\n"
        "bare_ignored\n",
        encoding="utf-8",
    )
    (tmp_path / "!important.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "#hash.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "trailing ").write_text("needle\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("needle\n", encoding="utf-8")
    for dirname in ("ignored_dir", "anchored_ignored", "bare_ignored"):
        sub = tmp_path / dirname / "sub"
        sub.mkdir(parents=True)
        (sub / "secret.py").write_text("needle\n", encoding="utf-8")

    helpers, _ = make_helpers(str(tmp_path))
    assert [m["file"] for m in helpers["grep"]("needle", ".")] == ["keep.py"]
    for dirname in ("ignored_dir", "anchored_ignored", "bare_ignored"):
        assert helpers["grep"]("needle", dirname) == []
        assert helpers["grep"]("needle", f"{dirname}/sub") == []
        assert helpers["tree"](dirname) == dirname
        assert helpers["tree"](f"{dirname}/sub") == f"{dirname}/sub"

    # Directory re-inclusion: * + !*/ + !*.py and foo/** + !foo/keep.py.
    sub_repo = tmp_path / "reincl"
    sub_repo.mkdir()
    (sub_repo / ".gitignore").write_text(
        "*\n!*/\n!*.py\nfoo/**\n!foo/keep.py\n",
        encoding="utf-8",
    )
    (sub_repo / "pkg").mkdir()
    (sub_repo / "pkg" / "ok.py").write_text("hit\n", encoding="utf-8")
    (sub_repo / "pkg" / "drop.txt").write_text("hit\n", encoding="utf-8")
    (sub_repo / "foo").mkdir()
    (sub_repo / "foo" / "keep.py").write_text("hit\n", encoding="utf-8")
    (sub_repo / "foo" / "drop.py").write_text("hit\n", encoding="utf-8")
    sub_helpers, _ = make_helpers(str(sub_repo))
    assert [m["file"] for m in sub_helpers["grep"]("hit", ".")] == ["foo/keep.py", "pkg/ok.py"]


def test_multibyte_utf8_line_clipping_preserves_match(tmp_path):
    from rlm_tools.helpers import MAX_GREP_LINE_BYTES, MAX_SUMMARY_LINE_CHARS

    (tmp_path / "utf8.txt").write_text("あ" * 300 + "NEEDLE_TOKEN" + "あ" * 300 + "\n", encoding="utf-8")
    helpers, _ = make_helpers(str(tmp_path))

    hits = helpers["grep"]("NEEDLE_TOKEN", ".")
    assert len(hits) == 1
    assert "NEEDLE_TOKEN" in hits[0]["line"]
    assert len(hits[0]["line"].encode("utf-8")) <= MAX_GREP_LINE_BYTES

    summary = helpers["grep_summary"]("NEEDLE_TOKEN", ".")
    assert "NEEDLE_TOKEN" in summary
    match_line = [ln for ln in summary.splitlines() if "L1:" in ln][0]
    clipped_part = match_line.split("L1: ", 1)[1]
    assert len(clipped_part.encode("utf-8")) <= MAX_SUMMARY_LINE_CHARS


def test_ast_helpers_deep_expr_formfeed_dotted_receiver_and_cache_eviction(tmp_path):
    # >1500 deep binary op and attribute chain must not raise RecursionError.
    deep_expr = "x = " + " + ".join(["1"] * 1500) + "\n"
    deep_attr = ".".join(["a"] * 1200) + " = 1\n"
    (tmp_path / "deep.py").write_text(deep_expr + deep_attr, encoding="utf-8")

    # Form feed (\f) and U+2028 inside comments must not desync AST lineno.
    (tmp_path / "ff.py").write_text(
        "a = 1\n# \x0c page break \u2028 line\nb = 2\n",
        encoding="utf-8",
    )

    # Call/subscript receivers, decorators, and unpacking targets.
    (tmp_path / "complex.py").write_text(
        "@dec\n"
        "def fn():\n"
        "    other().config = 1\n"
        "    get_obj().self.items = 2\n"
        "    (first, *rest), mapping[0] = [1, 2, 3], 4\n"
        "    return first\n",
        encoding="utf-8",
    )

    helpers, workspace = make_helpers(str(tmp_path))
    assert helpers["enclosing"]("deep.py", 1)["qualified"] == "<module>"
    deep_hits = helpers["find_assignments"]("deep.py", "x")
    assert len(deep_hits) == 1 and deep_hits[0]["line_number"] == 1

    ff_hits = helpers["find_assignments"]("ff.py", "b")
    assert len(ff_hits) == 1
    assert ff_hits[0]["line_number"] == 3
    assert ff_hits[0]["line"] == "b = 2"

    # Decorator line (line 1) resolves to fn.
    assert helpers["enclosing"]("complex.py", 1)["qualified"] == "fn"
    # Call receiver must not match bare or self-prefixed names.
    assert helpers["find_assignments"]("complex.py", "config") == []
    assert helpers["find_assignments"]("complex.py", "self.items") == []
    # Tuple, starred, and subscript unpacking targets must match.
    assert [h["target"] for h in helpers["find_assignments"]("complex.py", "first")] == ["first"]
    assert [h["target"] for h in helpers["find_assignments"]("complex.py", "rest")] == ["rest"]
    assert [h["target"] for h in helpers["find_assignments"]("complex.py", "mapping")] == ["mapping"]

    # evict_cache clears both file/ast caches and workspace._dir_cache.
    helpers["tree"](".")
    assert len(workspace._dir_cache) > 0
    workspace._evict_file_cache()
    assert len(workspace._dir_cache) == 0


def test_exact_scan_byte_limit_and_lru_file_cache_and_tree_truncation_banner(tmp_path, monkeypatch):
    # Exact RLM_MAX_SCAN_BYTES == st_size without trailing newline must be complete=True.
    (tmp_path / "exact.txt").write_text("0123456789", encoding="utf-8")
    monkeypatch.setenv("RLM_MAX_SCAN_BYTES", "10")
    helpers, _ = make_helpers(str(tmp_path))
    hits = helpers["grep"]("0123456789", ".")
    assert len(hits) == 1 and hits[0]["line"] == "0123456789"
    assert helpers["read_file"]("exact.txt") == "0123456789"

    # tree() truncation banner reports entries visited instead of '0 files scanned'.
    monkeypatch.delenv("RLM_MAX_SCAN_BYTES", raising=False)
    monkeypatch.setenv("RLM_MAX_SCAN_ENTRIES", "2")
    for idx in range(5):
        (tmp_path / f"f{idx}.txt").write_text("x\n", encoding="utf-8")
    tree_out = helpers["tree"](".")
    assert tree_out.startswith("TRUNCATED (2 entry scan limit;")
    assert "0 files scanned" not in tree_out
    assert "entries visited" in tree_out

    # LRU file_cache evicts oldest entries under memory_tracker cap.
    monkeypatch.delenv("RLM_MAX_SCAN_ENTRIES", raising=False)
    tracked = {"bytes": 0}

    def tracker(delta: int) -> None:
        if delta > 0 and tracked["bytes"] + delta > 15:
            raise MemoryError("cap")
        tracked["bytes"] = max(0, tracked["bytes"] + delta)

    (tmp_path / "a.txt").write_text("a" * 10, encoding="utf-8")
    (tmp_path / "b.txt").write_text("b" * 10, encoding="utf-8")
    lru_helpers, _ = make_helpers(str(tmp_path), memory_tracker=tracker)
    assert lru_helpers["read_file"]("a.txt") == "a" * 10
    assert tracked["bytes"] == 10
    assert lru_helpers["read_file"]("b.txt") == "b" * 10
    assert tracked["bytes"] == 10


def test_private_config_files_blocked_while_committed_mcp_json_readable(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".npmrc").write_text("//registry.npmjs.org/:_authToken=secret\n")
    (tmp_path / ".pypirc").write_text("[pypi]\npassword = secret\n")
    (tmp_path / ".netrc").write_text("machine example.com password secret\n")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text('{"env":{"SECRET":"1"}}\n')
    (tmp_path / ".mcp.json").write_text('{"mcpServers":{}}\n')

    helpers, _ = make_helpers(str(tmp_path))
    for private_path in ("sub/.npmrc", ".pypirc", ".netrc", ".claude/settings.local.json"):
        with pytest.raises(PermissionError, match="private"):
            helpers["read_file"](private_path)
    assert helpers["grep"]("secret") == []
    assert ".mcp.json" in helpers["glob_files"]("*")
    assert '"mcpServers"' in helpers["read_file"](".mcp.json")
    assert "settings.local.json" not in helpers["glob_files"](".claude/*")


def test_glob_files_supports_explicit_dot_folders_and_trailing_globstar(tmp_path):
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: test\n")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text("{}\n")
    (tmp_path / ".x" / "nested").mkdir(parents=True)
    (tmp_path / ".x" / "nested" / "file.txt").write_text("hi\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode" / "ignored.py").write_text("x = 1\n")
    (tmp_path / "main.py").write_text("x = 1\n")

    helpers, _ = make_helpers(str(tmp_path))
    assert helpers["glob_files"](".github/workflows/*.yml") == [".github/workflows/test.yml"]
    assert helpers["glob_files"](".claude/*") == [".claude/settings.json"]
    assert helpers["glob_files"](".x/**") == [".x/nested/file.txt"]
    assert helpers["glob_files"]("**/*.py") == ["main.py"]
    assert helpers["glob_files"](".git/*") == []


def test_oversized_file_truncation_names_exact_file(tmp_path, monkeypatch):
    from rlm_tools import helpers as helper_module

    monkeypatch.setattr(helper_module, "MAX_FILE_BYTES", 50)
    (tmp_path / "small.py").write_text("hit = 1\n")
    (tmp_path / "huge.py").write_text("x" * 120)

    helpers, _ = make_helpers(str(tmp_path))
    with pytest.raises(RuntimeError, match="huge\\.py"):
        helpers["grep"]("hit")

    summary = helpers["grep_summary"]("hit")
    assert summary.startswith("TRUNCATED (")
    assert "huge.py" in summary
    assert "exceeded 4 MiB" in summary

    res = helpers["grep_read"]("hit")
    assert res["truncated"] is True
    assert res["limit_hit"] == "file_bytes"
    assert any("huge.py" in item for item in res["oversized_files"])


def test_read_file_line_slicing_and_validation(tmp_path):
    (tmp_path / "lines.py").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    helpers, _ = make_helpers(str(tmp_path))

    assert helpers["read_file"]("lines.py", start_line=2, end_line=4) == "l2\nl3\nl4"
    assert helpers["read_file"]("lines.py", start_line=4) == "l4\nl5"
    assert helpers["read_file"]("lines.py", end_line=2) == "l1\nl2"

    with pytest.raises(ValueError, match="start_line"):
        helpers["read_file"]("lines.py", start_line=0)
    with pytest.raises(ValueError, match="end_line"):
        helpers["read_file"]("lines.py", start_line=4, end_line=2)


def test_outline_extracts_classes_fields_decorators_and_multiline_defs(tmp_path):
    src = (
        "@tool(name='search', readonly=True)\n"
        "async def multi_line_fn(\n"
        "    a: int,\n"
        "    b: str,\n"
        ") -> None:\n"
        "    def inner():\n"
        "        return 1\n"
        "    return None\n"
        "\n"
        "@dataclass\n"
        "class Widget(BaseWidget):\n"
        "    title: str\n"
        "    count: int = 0\n"
        "    def __init__(self, extra: int):\n"
        "        self.extra = extra\n"
        "        self.items: list[str] = []\n"
        "    def render(\n"
        "        self,\n"
        "    ) -> str:\n"
        "        return self.title\n"
    )
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")
    # Deeply nested expression inside decorator must not raise RecursionError.
    (tmp_path / "deep_dec.py").write_text(
        "@dec(" + " + ".join(["1"] * 1200) + ")\n"
        "def f():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    helpers, _ = make_helpers(str(tmp_path))
    out = helpers["outline"]("mod.py")
    assert out["path"] == "mod.py"
    assert out["total_lines"] == 20
    assert [f["qualified"] for f in out["functions"]] == ["multi_line_fn", "multi_line_fn.inner"]
    top_fn = out["functions"][0]
    assert top_fn["kind"] == "async_function"
    assert top_fn["decorator_line"] == 1
    assert top_fn["start_line"] == 2
    assert top_fn["end_line"] == 8
    assert top_fn["lines"] == 7
    assert top_fn["decorators"][0]["name"] == "tool"
    assert top_fn["decorators"][0]["kwargs"] == {"name": "'search'", "readonly": "True"}

    assert len(out["classes"]) == 1
    cls = out["classes"][0]
    assert cls["name"] == "Widget"
    assert cls["bases"] == ["BaseWidget"]
    assert cls["decorators"] == [{"name": "dataclass"}]
    assert cls["fields"] == ["title", "count"]
    assert cls["field_count"] == 2
    assert cls["init_attributes"] == ["extra", "items"]
    assert [m["name"] for m in cls["methods"]] == ["__init__", "render"]
    assert cls["methods"][1]["lines"] == 4

    deep_out = helpers["outline"]("deep_dec.py")
    assert deep_out["functions"][0]["name"] == "f"
