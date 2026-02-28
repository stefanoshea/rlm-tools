import os
import tempfile
import time

import pytest

from rlm_tools.cache import DirectoryCache, CACHE_DIR


def _make_dir_with_files(tmpdir: str, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        full = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


class TestDirectoryCacheMetadata:
    def test_returns_none_when_no_cache(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        assert cache.get_metadata() is None

    def test_save_and_load_metadata(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        meta = {"total_files": 42, "total_lines": 1000}
        cache.save_metadata(meta)
        loaded = cache.get_metadata()
        assert loaded == meta

    def test_cache_dir_created(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        cache.save_metadata({"total_files": 1})
        assert (tmp_path / CACHE_DIR).is_dir()

    def test_cache_valid_after_save(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        cache.save_metadata({"x": 1})
        assert cache.get_metadata() is not None

    def test_invalidate_clears_cache(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        cache.save_metadata({"total_files": 3})
        cache.invalidate()
        assert cache.get_metadata() is None


class TestDirectoryCacheFileIndex:
    def test_returns_none_when_no_index(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        assert cache.get_file_index() is None

    def test_save_and_load_file_index(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        files = ["src/a.py", "src/b.py", "README.md"]
        cache.save_file_index(files)
        loaded = cache.get_file_index()
        assert loaded == files

    def test_invalidate_clears_file_index(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        cache.save_file_index(["a.py", "b.py"])
        cache.invalidate()
        assert cache.get_file_index() is None

    def test_both_metadata_and_index_saved(self, tmp_path):
        cache = DirectoryCache(str(tmp_path))
        cache.save_metadata({"total_files": 2})
        cache.save_file_index(["a.py", "b.py"])
        assert cache.get_metadata() == {"total_files": 2}
        assert cache.get_file_index() == ["a.py", "b.py"]


class TestHelpersWithFileIndex:
    def test_grep_uses_index(self, tmp_path):
        from rlm_tools.helpers import make_helpers

        _make_dir_with_files(str(tmp_path), {
            "src/main.py": "def hello():\n    pass\n",
            "src/util.py": "def helper():\n    pass\n",
        })
        file_index = ["src/main.py", "src/util.py"]
        helpers, _ = make_helpers(str(tmp_path), file_index=file_index)

        results = helpers["grep"]("def hello")
        assert len(results) == 1
        assert "main.py" in results[0]["file"]

    def test_grep_without_index_still_works(self, tmp_path):
        from rlm_tools.helpers import make_helpers

        _make_dir_with_files(str(tmp_path), {
            "a.py": "def foo():\n    pass\n",
        })
        helpers, _ = make_helpers(str(tmp_path), file_index=None)
        results = helpers["grep"]("def foo")
        assert len(results) == 1

    def test_glob_uses_index(self, tmp_path):
        from rlm_tools.helpers import make_helpers

        _make_dir_with_files(str(tmp_path), {
            "src/main.py": "",
            "src/util.py": "",
            "README.md": "",
        })
        file_index = ["src/main.py", "src/util.py", "README.md"]
        helpers, _ = make_helpers(str(tmp_path), file_index=file_index)

        py_files = helpers["glob_files"]("**/*.py")
        assert len(py_files) == 2
        assert all(f.endswith(".py") for f in py_files)

    def test_glob_without_index_still_works(self, tmp_path):
        from rlm_tools.helpers import make_helpers

        _make_dir_with_files(str(tmp_path), {
            "a.py": "",
            "b.txt": "",
        })
        helpers, _ = make_helpers(str(tmp_path), file_index=None)
        py_files = helpers["glob_files"]("**/*.py")
        assert len(py_files) == 1

    def test_index_does_not_expose_cache_dir_in_grep(self, tmp_path):
        from rlm_tools.helpers import make_helpers

        _make_dir_with_files(str(tmp_path), {
            "src/main.py": "hello = 1\n",
        })
        cache_file = tmp_path / ".rlm_cache" / "metadata.json"
        cache_file.parent.mkdir()
        cache_file.write_text('{"hello": 1}')

        file_index = ["src/main.py"]
        helpers, _ = make_helpers(str(tmp_path), file_index=file_index)

        results = helpers["grep"]("hello")
        files_found = [r["file"] for r in results]
        assert not any(".rlm_cache" in f for f in files_found)


class TestSandboxWithFileIndex:
    def test_sandbox_passes_index_to_helpers(self, tmp_path):
        from rlm_tools.sandbox import Sandbox

        _make_dir_with_files(str(tmp_path), {
            "a.py": "x = 1\n",
            "b.py": "x = 2\n",
        })
        sandbox = Sandbox(
            base_path=str(tmp_path),
            file_index=["a.py", "b.py"],
        )
        result = sandbox.execute("results = grep('x = 1')\nprint(len(results))")
        assert result.error is None
        assert "1" in result.stdout

    def test_sandbox_without_index_works(self, tmp_path):
        from rlm_tools.sandbox import Sandbox

        _make_dir_with_files(str(tmp_path), {
            "a.py": "y = 42\n",
        })
        sandbox = Sandbox(base_path=str(tmp_path))
        result = sandbox.execute("results = grep('y = 42')\nprint(len(results))")
        assert result.error is None
        assert "1" in result.stdout
