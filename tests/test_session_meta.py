"""Tests for kimi_code_acp.session_meta.

The Kimi Code ACP adapter does **not`` interpret a ``_meta`` field in the
ACP ``session/new`` request.  ``build_session_meta()`` therefore takes no
arguments and returns an **empty dict** -- the shape is a seam for future
Kimi-specific options.
"""

from __future__ import annotations

from kimi_code_acp.session_meta import build_session_meta, session_meta_is_safe


class TestBuildSessionMeta:
    def test_returns_empty_dict(self):
        """``build_session_meta()`` returns an empty dict (no args)."""
        meta = build_session_meta()
        assert meta == {}

    def test_returns_dict(self):
        meta = build_session_meta()
        assert isinstance(meta, dict)

    def test_returns_independent_dict(self):
        """Each call must return a fresh dict -- no shared state."""
        meta1 = build_session_meta()
        meta2 = build_session_meta()
        assert meta1 is not meta2

    def test_returns_mutable_dict(self):
        meta = build_session_meta()
        meta["new_key"] = "value"
        assert meta["new_key"] == "value"


class TestSessionMetaIsSafe:
    def test_empty_dict_is_safe(self):
        assert session_meta_is_safe({}) is True

    def test_dict_is_safe(self):
        assert session_meta_is_safe({"any_key": "any_value"}) is True

    def test_non_dict_is_not_safe(self):
        assert session_meta_is_safe("not a dict") is False
        assert session_meta_is_safe(None) is False
        assert session_meta_is_safe(123) is False
        assert session_meta_is_safe(["a", "b"]) is False
        assert session_meta_is_safe(object()) is False
