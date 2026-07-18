"""Tests for the backend -> core generic ACP approval bridge wiring.

These tests verify the **minimal** boundary contract of
``kimi_code_acp.backend._get_approval_callback``:

  1. It calls the Hermes core factory
     ``agent.transports.acp_approval.make_acp_approval_callback()`` and
     returns whatever that factory returns (forwarded verbatim).
  2. When the core factory import fails, it returns ``None`` (the
     session then uses its built-in fail-closed default policy).
  3. When the core factory raises at call time, it returns ``None``.

The plugin owns **no** approval module and **no** approval policy.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from kimi_code_acp import backend


# --------------------------------------------------------------------------- #
# 1. _get_approval_callback forwards the core factory's result verbatim
# --------------------------------------------------------------------------- #

class TestGetApprovalCallbackForwardsCoreFactory:
    def test_returns_core_factory_result(self):
        sentinel_cb = MagicMock(name="core_callback")
        fake_factory = MagicMock(name="make_acp_approval_callback",
                                 return_value=sentinel_cb)
        with patch(
            "agent.transports.acp_approval.make_acp_approval_callback",
            fake_factory,
        ):
            result = backend._get_approval_callback()
        assert result is sentinel_cb
        fake_factory.assert_called_once_with()

    def test_returns_none_when_core_factory_raises(self):
        fake_factory = MagicMock(
            name="make_acp_approval_callback",
            side_effect=RuntimeError("core boom"),
        )
        with patch(
            "agent.transports.acp_approval.make_acp_approval_callback",
            fake_factory,
        ):
            result = backend._get_approval_callback()
        assert result is None
        fake_factory.assert_called_once_with()


# --------------------------------------------------------------------------- #
# 2. _get_approval_callback returns None when the core import fails
# --------------------------------------------------------------------------- #

class TestGetApprovalCallbackImportFailure:
    def test_returns_none_when_core_module_unavailable(self):
        real_import = __import__

        def blocking_import(name, *args, **kwargs):
            if name == "agent.transports.acp_approval":
                raise ImportError("core module unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocking_import):
            result = backend._get_approval_callback()
        assert result is None


# --------------------------------------------------------------------------- #
# 3. auto_approve_permissions is always False (regression guard)
# --------------------------------------------------------------------------- #

class TestAutoApproveIsAlwaysFalse:
    def test_backend_has_no_bypass_helper(self):
        assert not hasattr(backend, "_is_approval_bypass_active")

    def test_auto_approve_constant_is_false_in_source(self):
        """The literal ``auto_approve = False`` is present in
        ``run_task`` and no ``auto_approve = True`` assignment exists."""
        source = inspect.getsource(backend.run_task)
        assert "auto_approve = False" in source
        assert "auto_approve = True" not in source
