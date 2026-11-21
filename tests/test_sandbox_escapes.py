"""Regression tests for sandbox escape vectors that bypass the AST dot-attribute guard.

The AST validator blocks `x.__class__` style access, but several stdlib primitives
reach attributes by *string name* at runtime, sidestepping the AST check entirely.
"""

from rlm_tools.sandbox import Sandbox


def _run(tmp_path, code: str):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    try:
        return sandbox.execute(code)
    finally:
        sandbox.close()


def test_str_format_attribute_traversal_blocked(tmp_path):
    # "{0.__class__}".format(obj) reaches __class__ via C-level getattr, no AST Attribute node.
    result = _run(tmp_path, "print('{0.__class__}'.format(()))")
    assert result.error is not None


def test_str_format_map_attribute_traversal_blocked(tmp_path):
    result = _run(tmp_path, "print('{x.__class__}'.format_map({'x': ()}))")
    assert result.error is not None


def test_operator_attrgetter_escape_blocked(tmp_path):
    # attrgetter('__class__')(obj) returns the real type object, enabling __subclasses__ walk -> RCE.
    code = (
        "import operator\n"
        "cls = operator.attrgetter('__class__')(())\n"
        "print(cls)\n"
    )
    result = _run(tmp_path, code)
    assert result.error is not None


def test_operator_attrgetter_alias_escape_blocked(tmp_path):
    # Aliasing defeats any name-based AST check; the primitive itself must be gone.
    code = (
        "import operator\n"
        "ag = operator.attrgetter\n"
        "print(ag('__class__')(()))\n"
    )
    result = _run(tmp_path, code)
    assert result.error is not None


def test_operator_methodcaller_escape_blocked(tmp_path):
    code = (
        "import operator\n"
        "mc = operator.methodcaller('__reduce__')\n"
        "print(mc(()))\n"
    )
    result = _run(tmp_path, code)
    assert result.error is not None


def test_operator_itemgetter_still_available(tmp_path):
    # itemgetter is harmless and useful; sanitisation must not remove it.
    code = (
        "import operator\n"
        "second = operator.itemgetter(1)\n"
        "print(second([10, 20, 30]))\n"
    )
    result = _run(tmp_path, code)
    assert result.error is None
    assert result.stdout.strip() == "20"


def test_string_formatter_attribute_traversal_blocked(tmp_path):
    code = (
        "import string\n"
        "print(string.Formatter().vformat('{0.__class__}', [()], {}))\n"
    )
    result = _run(tmp_path, code)
    assert result.error is not None


def test_fstrings_remain_available_as_format_alternative(tmp_path):
    # f-strings are the supported formatting path; dunder access inside them is
    # still caught by the AST attribute guard, so they are safe to allow.
    result = _run(tmp_path, "name = 'world'\nprint(f'hello {name}')")
    assert result.error is None
    assert result.stdout.strip() == "hello world"


def test_leaked_stdlib_submodules_are_blocked(tmp_path):
    probes = [
        "import fnmatch\nprint(fnmatch.os)",
        "from fnmatch import os\nprint(os)",
        "import fnmatch.os",
        "import statistics\nprint(statistics.sys)",
        "import json\nprint(json.codecs)",
        "import re\nprint(re.enum)",
        "import textwrap\nprint(textwrap.re)",
        "import collections\nprint(collections.abc)",
    ]
    for code in probes:
        result = _run(tmp_path, code)
        assert result.error is not None, f"Expected error for: {code}"


def test_match_class_dunder_escape_blocked(tmp_path):
    code = "match ():\n    case tuple(__class__=c):\n        print(c)\n"
    result = _run(tmp_path, code)
    assert result.error is not None
    assert "PermissionError" in result.error


def test_functools_wrapper_helpers_blocked_while_safe_helpers_work(tmp_path):
    for attr in ("update_wrapper", "wraps"):
        result = _run(tmp_path, f"import functools\nprint(functools.{attr})")
        assert result.error is not None

    safe_code = (
        "import functools\n"
        "add_one = functools.partial(lambda x, y: x + y, 1)\n"
        "@functools.lru_cache(maxsize=8)\n"
        "def sq(n):\n"
        "    return n * n\n"
        "print(add_one(2), sq(4))\n"
    )
    result = _run(tmp_path, safe_code)
    assert result.error is None
    assert result.stdout.strip() == "3 16"


def test_blocked_introspection_and_loader_builtins(tmp_path):
    for name in ("__loader__", "__spec__", "help", "license"):
        result = _run(tmp_path, f"print({name})")
        assert result.error is not None, f"Expected error for builtin {name}"


def test_relative_and_private_imports_blocked(tmp_path):
    assert _run(tmp_path, "from . import re").error is not None
    assert _run(tmp_path, "from re import _compile").error is not None


def test_sanitized_import_does_not_expose_star_imports_or_mutate_host_modules():
    import operator
    from rlm_tools.worker_runtime import ALLOWED_MODULES, _make_restricted_import

    restricted_import = _make_restricted_import(ALLOWED_MODULES)
    proxy = restricted_import("operator", {}, {}, ("*",), 0)
    assert not hasattr(proxy, "attrgetter")
    assert not hasattr(proxy, "methodcaller")
    assert "attrgetter" not in proxy.__all__
    assert "methodcaller" not in proxy.__all__
    assert hasattr(operator, "attrgetter")
    assert hasattr(operator, "methodcaller")


def test_hasattr_hides_helper_proxy_internals(tmp_path):
    result = _run(tmp_path, "print(hasattr(read_file, '_conn'), hasattr(rlm_finalize, '_callback'))")
    assert result.error is None
    assert result.stdout.strip() == "False False"


def test_base_exception_in_sandbox_does_not_kill_worker(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    try:
        err_result = sandbox.execute("raise BaseException('boom')")
        assert err_result.error == "BaseException: boom"
        alive = sandbox.execute("print('alive')")
        assert alive.error is None
        assert alive.stdout.strip() == "alive"
    finally:
        sandbox.close()
