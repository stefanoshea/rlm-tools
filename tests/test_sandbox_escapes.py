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
