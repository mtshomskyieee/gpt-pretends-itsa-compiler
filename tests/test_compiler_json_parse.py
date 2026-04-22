"""JSON extraction for compiler fallback."""

from pretend_compiler.agents.compiler import _parse_json_object_from_text


def test_parse_raw_object() -> None:
    assert _parse_json_object_from_text('{"compile_ok":true,"diagnostics":[],"ir_ops":[]}') == {
        "compile_ok": True,
        "diagnostics": [],
        "ir_ops": [],
    }


def test_parse_fenced_json() -> None:
    text = """Here:
```json
{"a": 1}
```
"""
    assert _parse_json_object_from_text(text) == {"a": 1}


def test_parse_trailing_junk_ignored() -> None:
    """raw_decode stops after first JSON value."""
    data = _parse_json_object_from_text('{"x": 2} trailing')
    assert data == {"x": 2}
