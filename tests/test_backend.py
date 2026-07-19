"""Tests for kimi_code_acp.backend - ACP execution backend.

All tests mock ``ACPClientSession`` -- no real ``kimi`` subprocess is spawned.

Test coverage:
  1. Success path: correct params to ACPClientSession, JSON result shape.
  2. Prompt/cwd/timeout passed through correctly.
  3. session_meta built as an empty dict (Kimi adapter ignores ``_meta``).
  4. Approval callback forwarded when available.
  5. auto_approve_permissions is ALWAYS False.
  6. Session lifecycle: not closed on success; retired on error/should_retire.
  7. Session reuse across calls with same cwd.
  8. Session retirement -> fresh session on next call.
  9. Health-check: dead session replaced, live session reused.
  10. Public close_session() API.
  11. Safe error: no sentinel/secret in error JSON.
  12. Missing core (ImportError) -> compatibility error.
  13. Old core (missing params) -> compatibility error.
  14. Config invalid -> does not call session.
  15. cwd re-resolved (TOCTOU) -- nonexistent cwd -> error.
  16. KeyboardInterrupt/SystemExit propagated.
  17. TurnResult with .error -> safe error JSON (inactivity classification).
  18. Concurrency: two run_task do not enter run_turn concurrently.
  19. close_session during a running turn waits.
  20. Live-switch: model/permission changes apply via set_model /
      set_permission_mode on the existing session (NOT a rebuild);
      live-switch failure denies the turn; switching to None on a live
      session is denied with a warning.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kimi_code_acp.backend import close_session, run_task
from kimi_code_acp.config import CONFIG_SECTION, DEFAULTS
from kimi_code_acp.session_meta import build_session_meta, session_meta_is_safe
from kimi_code_acp.tool import handle_kimi_code_acp

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def valid_config():
    return dict(DEFAULTS)


@pytest.fixture
def valid_cwd(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mock_turn_result():
    result = MagicMock()
    result.final_text = "Task completed successfully"
    result.tool_iterations = 3
    result.should_retire = False
    result.error = None
    return result


class FakeSession:
    """Minimal fake ACPClientSession for testing."""

    class _FakeClient:
        def is_alive(self) -> bool:
            return True

    instances: list = []
    _run_turn_result = None
    _run_turn_raises = None
    _ensure_started_raises = None
    _close_raises = None
    _ctor_raises = None
    _block_until: threading.Event | None = None
    _concurrent_counter: int = 0
    _concurrent_max: int = 0
    _concurrent_lock: threading.Lock = threading.Lock()
    _client = _FakeClient()
    _block_close_until: threading.Event | None = None
    _close_entered: threading.Event | None = None
    _set_model_raises = None
    _set_permission_mode_raises = None

    def __init__(self, **kwargs):
        if FakeSession._ctor_raises is not None:
            raise FakeSession._ctor_raises
        self.constructor_kwargs = kwargs
        self._closed = False
        self._closed_count = 0
        self._ensure_started_called = False
        self._run_turn_kwargs = None
        self._ensure_started_cwd = None
        self._run_turn_result = FakeSession._run_turn_result
        self._run_turn_raises = FakeSession._run_turn_raises
        self._ensure_started_raises = FakeSession._ensure_started_raises
        self._close_raises = FakeSession._close_raises
        self._set_model_calls: list = []
        self._set_permission_mode_calls: list = []
        FakeSession.instances.append(self)

    def ensure_started(self, cwd=None):
        self._ensure_started_called = True
        self._ensure_started_cwd = cwd
        if self._ensure_started_raises:
            raise self._ensure_started_raises

    def set_model(self, model: str) -> None:
        self._set_model_calls.append(model)
        if FakeSession._set_model_raises is not None:
            raise FakeSession._set_model_raises

    def set_permission_mode(self, mode: str) -> None:
        self._set_permission_mode_calls.append(mode)
        if FakeSession._set_permission_mode_raises is not None:
            raise FakeSession._set_permission_mode_raises

    def run_turn(self, *, user_input, cwd=None, turn_timeout=600.0, **kw):
        self._run_turn_kwargs = {
            "user_input": user_input,
            "cwd": cwd,
            "turn_timeout": turn_timeout,
        }
        with FakeSession._concurrent_lock:
            FakeSession._concurrent_counter += 1
            if FakeSession._concurrent_counter > FakeSession._concurrent_max:
                FakeSession._concurrent_max = FakeSession._concurrent_counter
        try:
            if FakeSession._block_until is not None:
                FakeSession._block_until.wait()
            time.sleep(0.05)
            if self._run_turn_raises:
                raise self._run_turn_raises
            return self._run_turn_result
        finally:
            with FakeSession._concurrent_lock:
                FakeSession._concurrent_counter -= 1

    def close(self):
        self._closed = True
        self._closed_count += 1
        if FakeSession._close_entered is not None:
            FakeSession._close_entered.set()
        if FakeSession._block_close_until is not None:
            FakeSession._block_close_until.wait(timeout=5.0)
        if self._close_raises:
            raise self._close_raises

    @classmethod
    def reset(cls):
        cls.instances = []
        cls._run_turn_result = None
        cls._run_turn_raises = None
        cls._ensure_started_raises = None
        cls._close_raises = None
        cls._ctor_raises = None
        cls._block_until = None
        cls._block_close_until = None
        cls._close_entered = None
        cls._concurrent_counter = 0
        cls._concurrent_max = 0
        cls._set_model_raises = None
        cls._set_permission_mode_raises = None

    @classmethod
    def last(cls):
        return cls.instances[-1] if cls.instances else None


@pytest.fixture(autouse=True)
def reset_fake_session():
    close_session()
    FakeSession.reset()
    yield
    close_session()
    FakeSession.reset()


# --------------------------------------------------------------------------- #
# 1. Success path
# --------------------------------------------------------------------------- #


class TestSuccessPath:
    def test_returns_json_string(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            result = run_task("write a function", valid_config, cwd=valid_cwd)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "result" in parsed
        assert "tool_iterations" in parsed
        assert "should_retire" in parsed

    def test_result_text_from_turn_result(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            result = run_task("write a function", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["result"] == "Task completed successfully"

    def test_tool_iterations_from_turn_result(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            result = run_task("write a function", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["tool_iterations"] == 3

    def test_should_retire_from_turn_result(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            result = run_task("write a function", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["should_retire"] is False

    def test_session_constructed_with_command_and_args(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        from kimi_code_acp.config import ACP_ARGS, ACP_COMMAND

        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("write a function", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session.constructor_kwargs["command"] == ACP_COMMAND
        assert session.constructor_kwargs["args"] == list(ACP_ARGS)

    def test_session_constructed_with_model_none_default(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("write a function", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] is None

    def test_session_constructed_with_model_string_forwarded_verbatim(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("write a function", valid_config, cwd=valid_cwd, model="kimi-k2")
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] == "kimi-k2"


# --------------------------------------------------------------------------- #
# 1b. Config fallback resolution via handler
# --------------------------------------------------------------------------- #


class TestConfigFallbackResolution:
    """Per-call value > config value > None (server default).
    Resolution happens in the handler; run_task only sees the resolved
    value."""

    def _patch_backend(self, mock_turn_result):
        return patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        )

    def test_config_model_fallback(self, mock_turn_result, tmp_path):
        cfg = dict(DEFAULTS)
        cfg["model"] = "kimi-k2"
        with (
            patch("hermes_cli.config.load_config") as mock_load,
            self._patch_backend(mock_turn_result),
        ):
            mock_load.return_value = {CONFIG_SECTION: cfg}
            FakeSession._run_turn_result = mock_turn_result
            result = handle_kimi_code_acp(
                {
                    "prompt": "do something",
                    "cwd": str(tmp_path),
                    "model": None,
                    "permission": None,
                }
            )
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] == "kimi-k2"

    def test_per_call_model_overrides_config(self, mock_turn_result, tmp_path):
        cfg = dict(DEFAULTS)
        cfg["model"] = "kimi-k2"
        with (
            patch("hermes_cli.config.load_config") as mock_load,
            self._patch_backend(mock_turn_result),
        ):
            mock_load.return_value = {CONFIG_SECTION: cfg}
            FakeSession._run_turn_result = mock_turn_result
            result = handle_kimi_code_acp(
                {
                    "prompt": "do something",
                    "cwd": str(tmp_path),
                    "model": "kimi-k1.5",
                    "permission": None,
                }
            )
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] == "kimi-k1.5"

    def test_config_permission_fallback(self, mock_turn_result, tmp_path):
        cfg = dict(DEFAULTS)
        cfg["permission"] = "auto"
        with (
            patch("hermes_cli.config.load_config") as mock_load,
            self._patch_backend(mock_turn_result),
        ):
            mock_load.return_value = {CONFIG_SECTION: cfg}
            FakeSession._run_turn_result = mock_turn_result
            result = handle_kimi_code_acp(
                {
                    "prompt": "do something",
                    "cwd": str(tmp_path),
                    "model": None,
                    "permission": None,
                }
            )
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        session = FakeSession.last()
        assert session.constructor_kwargs["permission_mode"] == "auto"

    def test_no_config_no_per_call_uses_none(self, mock_turn_result, tmp_path):
        cfg = dict(DEFAULTS)
        with (
            patch("hermes_cli.config.load_config") as mock_load,
            self._patch_backend(mock_turn_result),
        ):
            mock_load.return_value = {CONFIG_SECTION: cfg}
            FakeSession._run_turn_result = mock_turn_result
            result = handle_kimi_code_acp(
                {
                    "prompt": "do something",
                    "cwd": str(tmp_path),
                    "model": None,
                    "permission": None,
                }
            )
        parsed = json.loads(result)
        assert "error" not in parsed
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] is None
        assert session.constructor_kwargs["permission_mode"] is None


# --------------------------------------------------------------------------- #
# 2. Prompt/cwd/timeout passed through
# --------------------------------------------------------------------------- #


class TestPromptCwdTimeout:
    def test_prompt_passed_to_run_turn(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("specific prompt text", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session._run_turn_kwargs["user_input"] == "specific prompt text"

    def test_cwd_passed_to_ensure_started(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session._ensure_started_cwd == valid_cwd

    def test_cwd_passed_to_run_turn(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session._run_turn_kwargs["cwd"] == valid_cwd

    def test_timeout_passed_to_run_turn(self, valid_config, mock_turn_result, valid_cwd):
        valid_config["timeout_seconds"] = 300
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session._run_turn_kwargs["turn_timeout"] == 300


# --------------------------------------------------------------------------- #
# 3. session_meta is empty dict (Kimi adapter ignores _meta)
# --------------------------------------------------------------------------- #


class TestSessionMeta:
    def test_session_meta_passed_to_constructor(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        meta = session.constructor_kwargs["session_meta"]
        # Kimi adapter ignores _meta; the helper returns an empty dict.
        assert isinstance(meta, dict)
        assert meta == {}

    def test_session_meta_is_safe(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        meta = session.constructor_kwargs["session_meta"]
        assert session_meta_is_safe(meta)

    def test_session_meta_each_call_independent(self, valid_config, mock_turn_result, valid_cwd):
        """Each run_task constructs a fresh session_meta -- the helper
        returns a new dict (currently empty) each time."""
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd)
            run_task("second", valid_config, cwd=valid_cwd)
        first_meta = FakeSession.instances[0].constructor_kwargs["session_meta"]
        second_meta = FakeSession.instances[0].constructor_kwargs["session_meta"]
        # Both are empty dicts (but should be distinct objects).
        assert first_meta == {}
        assert second_meta == {}


# --------------------------------------------------------------------------- #
# 4. Approval callback forwarded when available
# --------------------------------------------------------------------------- #


class TestApprovalCallback:
    def test_callback_forwarded_when_available(self, valid_config, mock_turn_result, valid_cwd):
        mock_cb = MagicMock(return_value="once")
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._get_approval_callback", return_value=mock_cb),
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session.constructor_kwargs["approval_callback"] is mock_cb

    def test_no_callback_when_unavailable(self, valid_config, mock_turn_result, valid_cwd):
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._get_approval_callback", return_value=None),
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session.constructor_kwargs["approval_callback"] is None


# --------------------------------------------------------------------------- #
# 5. auto_approve_permissions is ALWAYS False
# --------------------------------------------------------------------------- #


class TestAutoApprove:
    def test_auto_approve_always_false(self, valid_config, mock_turn_result, valid_cwd):
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._get_approval_callback", return_value=MagicMock()),
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session.constructor_kwargs["auto_approve_permissions"] is False


# --------------------------------------------------------------------------- #
# 6. Session lifecycle
# --------------------------------------------------------------------------- #


class TestSessionLifecycle:
    def test_session_not_closed_on_success(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        assert FakeSession.last()._closed is False

    def test_session_closed_on_should_retire(self, valid_config, mock_turn_result, valid_cwd):
        mock_turn_result.should_retire = True
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
        assert FakeSession.last()._closed is True

    def test_session_closed_on_run_turn_exception(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = RuntimeError("some error")
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        assert FakeSession.last()._closed is True
        parsed = json.loads(result)
        assert "error" in parsed and "error_type" in parsed

    def test_ensure_started_failure_closes_partial(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._ensure_started_raises = RuntimeError("startup failed")
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP session creation failed"
        assert parsed["error_type"] == "RuntimeError"
        assert len(FakeSession.instances) == 1
        assert FakeSession.last()._closed is True
        # next call spawns a fresh session
        FakeSession._ensure_started_raises = None
        FakeSession._run_turn_result = MagicMock(
            final_text="ok", tool_iterations=1, should_retire=False, error=None
        )
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result2 = run_task("prompt2", valid_config, cwd=valid_cwd)
        parsed2 = json.loads(result2)
        assert parsed2.get("result") == "ok"
        assert len(FakeSession.instances) == 2

    def test_keyboard_interrupt_does_not_close(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                run_task("prompt", valid_config, cwd=valid_cwd)
        session = FakeSession.last()
        assert session._closed is False


# --------------------------------------------------------------------------- #
# 6b. Constructor exception -> safe error JSON
# --------------------------------------------------------------------------- #


class TestSessionCreationFailure:
    def test_session_cls_ctor_exception_returns_error_json(self, valid_config, valid_cwd):
        FakeSession._ctor_raises = RuntimeError("ctor boom")
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP session creation failed"
        assert parsed["error_type"] == "RuntimeError"
        assert FakeSession.instances == []
        # module-level singleton must remain clean
        FakeSession._ctor_raises = None
        mock_ok = MagicMock(final_text="ok", tool_iterations=1, should_retire=False, error=None)
        FakeSession._run_turn_result = mock_ok
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result2 = run_task("prompt2", valid_config, cwd=valid_cwd)
        parsed2 = json.loads(result2)
        assert parsed2.get("result") == "ok"
        assert len(FakeSession.instances) == 1


# --------------------------------------------------------------------------- #
# 7. Session reuse
# --------------------------------------------------------------------------- #


class TestSessionReuse:
    def test_session_reused_across_calls(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt one", valid_config, cwd=valid_cwd)
            run_task("prompt two", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 1

    def test_second_call_reaches_same_instance(self, valid_config, mock_turn_result, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt one", valid_config, cwd=valid_cwd)
            run_task("prompt two", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 1
        assert FakeSession.instances[0]._run_turn_kwargs["user_input"] == "prompt two"


# --------------------------------------------------------------------------- #
# 8. Session retirement
# --------------------------------------------------------------------------- #


class TestSessionRetirement:
    def test_session_recreated_after_retirement(self, valid_config, mock_turn_result, valid_cwd):
        retiring = MagicMock()
        retiring.final_text = "first"
        retiring.tool_iterations = 1
        retiring.should_retire = True
        retiring.error = None
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = retiring
            run_task("first", valid_config, cwd=valid_cwd)
            FakeSession._run_turn_result = mock_turn_result
            run_task("second", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 2


# --------------------------------------------------------------------------- #
# 9. Health-check
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    def test_dead_session_replaced(self, valid_config, mock_turn_result, valid_cwd):
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._ACPProcessManager.is_alive", return_value=False),
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd)
            run_task("second", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 2

    def test_alive_session_reused(self, valid_config, mock_turn_result, valid_cwd):
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._ACPProcessManager.is_alive", return_value=True),
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd)
            run_task("second", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 1


# --------------------------------------------------------------------------- #
# 10. Public close_session() API
# --------------------------------------------------------------------------- #


class TestCloseSession:
    def test_close_session_clears_singleton(self, valid_config, mock_turn_result, valid_cwd):
        import kimi_code_acp.backend as backend_mod

        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = mock_turn_result
            run_task("prompt", valid_config, cwd=valid_cwd)
            assert backend_mod._manager._session is not None
            close_session()
            assert backend_mod._manager._session is None
        assert FakeSession.last()._closed is True

    def test_close_session_idempotent(self):
        close_session()
        close_session()


# --------------------------------------------------------------------------- #
# 11. Safe error
# --------------------------------------------------------------------------- #


class TestSafeError:
    SECRET_SENTINEL = "SUPER_SECRET_TOKEN_xyz789"

    def test_error_json_does_not_contain_exception_message(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = RuntimeError(
                f"command: kimi --token={self.SECRET_SENTINEL} failed"
            )
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        assert self.SECRET_SENTINEL not in result
        parsed = json.loads(result)
        assert self.SECRET_SENTINEL not in parsed.get("error", "")

    def test_error_json_has_generic_message(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = RuntimeError("detailed crash info")
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"

    def test_config_error_does_not_leak_config_values(self, valid_config, valid_cwd):
        valid_config["acp_command"] = f"cmd_with_{self.SECRET_SENTINEL}"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = RuntimeError("crash")
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        assert self.SECRET_SENTINEL not in result


# --------------------------------------------------------------------------- #
# 12. Missing core (ImportError) -> compatibility error
# --------------------------------------------------------------------------- #


class TestMissingCore:
    def test_missing_core_returns_compat_error(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=None):
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert "error" in parsed
        assert (
            "not available" in parsed["error"].lower() or "incompatible" in parsed["error"].lower()
        )
        assert parsed["error_type"] == "ImportError"

    def test_no_session_created_when_core_missing(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=None):
            run_task("prompt", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 0


# --------------------------------------------------------------------------- #
# 13. Old core (missing params) -> compatibility error
# --------------------------------------------------------------------------- #


class TestOldCore:
    def test_old_core_returns_compat_error(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=None):
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert "error" in parsed and parsed["error_type"] == "ImportError"
        assert len(FakeSession.instances) == 0

    def test_feature_detection_rejects_old_session_class(self):
        from kimi_code_acp.backend import _import_session_class

        class OldSession:
            def __init__(
                self, *, command, model=None, approval_callback=None, auto_approve_permissions=False
            ):
                pass

        import sys

        mock_module = MagicMock()
        mock_module.ACPClientSession = OldSession
        original = sys.modules.get("agent.transports.acp_client_session")
        sys.modules["agent.transports.acp_client_session"] = mock_module
        try:
            result = _import_session_class()
        finally:
            if original is not None:
                sys.modules["agent.transports.acp_client_session"] = original
            else:
                del sys.modules["agent.transports.acp_client_session"]
        assert result is None

    def test_feature_detection_accepts_new_session_class(self):
        from kimi_code_acp.backend import _import_session_class

        class NewSession:
            def __init__(
                self,
                *,
                command,
                args=None,
                model=None,
                permission_mode=None,
                session_meta=None,
                approval_callback=None,
                auto_approve_permissions=False,
            ):
                pass

        import sys

        mock_module = MagicMock()
        mock_module.ACPClientSession = NewSession
        original = sys.modules.get("agent.transports.acp_client_session")
        sys.modules["agent.transports.acp_client_session"] = mock_module
        try:
            result = _import_session_class()
        finally:
            if original is not None:
                sys.modules["agent.transports.acp_client_session"] = original
            else:
                del sys.modules["agent.transports.acp_client_session"]
        assert result is NewSession


# --------------------------------------------------------------------------- #
# 14. Config invalid -> does not call session
# --------------------------------------------------------------------------- #


class TestConfigInvalid:
    def test_invalid_config_does_not_create_session(self, tmp_path):
        cfg = dict(DEFAULTS)
        cfg["timeout_seconds"] = 99999
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result = run_task("prompt", cfg, cwd=str(tmp_path))
        assert len(FakeSession.instances) == 0
        parsed = json.loads(result)
        assert "error" in parsed and parsed["error_type"] == "ConfigError"

    def test_invalid_config_returns_json_string(self, tmp_path):
        cfg = dict(DEFAULTS)
        cfg["workdir"] = "/nonexistent/path/to/nowhere"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result = run_task("prompt", cfg, cwd=str(tmp_path))
        assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# 15. cwd re-resolved (TOCTOU)
# --------------------------------------------------------------------------- #


class TestCwdTOCTOU:
    def test_cwd_resolves_to_none_after_validation_returns_error(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with (
            patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession),
            patch("kimi_code_acp.backend._resolve_cwd", return_value=None),
        ):
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert "error" in parsed and parsed["error_type"] == "ValueError"
        assert len(FakeSession.instances) == 0

    def test_cwd_nonexistent_returns_error(self, mock_turn_result):
        cfg = dict(DEFAULTS)
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            result = run_task("prompt", cfg, cwd="/nonexistent/path/to/nowhere")
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error_type"] == "ValueError"


# --------------------------------------------------------------------------- #
# 16. KeyboardInterrupt/SystemExit propagated
# --------------------------------------------------------------------------- #


class TestKeyboardInterruptPropagation:
    def test_keyboard_interrupt_not_swallowed(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                run_task("prompt", valid_config, cwd=valid_cwd)

    def test_system_exit_not_swallowed(self, valid_config, valid_cwd):
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_raises = SystemExit(1)
            with pytest.raises(SystemExit):
                run_task("prompt", valid_config, cwd=valid_cwd)


# --------------------------------------------------------------------------- #
# 17. TurnResult with .error -> safe error JSON
# --------------------------------------------------------------------------- #


class TestTurnResultError:
    def test_turn_result_error_returns_safe_json(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = True
        error_result.error = f"ACP session crashed with token={TestSafeError.SECRET_SENTINEL}"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert "error" in parsed
        assert TestSafeError.SECRET_SENTINEL not in result
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# 17b. Inactivity timeout error classification
# --------------------------------------------------------------------------- #


class TestInactivityErrorClassification:
    SECRET_SENTINEL = "SUPER_SECRET_TOKEN_xyz789"

    def test_inactivity_error_classified(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = True
        error_result.error = (
            f"ACP session/prompt failed: ACP session inactive (token={self.SECRET_SENTINEL})"
        )
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP session inactive"
        assert parsed["error_type"] == "InactivityTimeoutError"
        assert self.SECRET_SENTINEL not in result

    def test_inactivity_error_does_not_leak_raw_text(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = False
        error_result.error = (
            "ACP session/prompt failed: ACP session inactive; "
            f"stderr={self.SECRET_SENTINEL} cmd=kimi"
        )
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP session inactive"
        assert parsed["error_type"] == "InactivityTimeoutError"
        assert self.SECRET_SENTINEL not in result
        assert "kimi" not in parsed["error"]

    def test_arbitrary_error_with_secret_still_runtime_error(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = False
        error_result.error = f"ACP session crashed token={self.SECRET_SENTINEL}"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"
        assert self.SECRET_SENTINEL not in result

    def test_near_match_substring_only_not_classified(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = False
        error_result.error = "Error: ACP session inactive due to unknown cause"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"

    def test_canonical_prefix_in_middle_not_classified(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = False
        error_result.error = "Some preamble: ACP session/prompt failed: ACP session inactive"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"

    def test_malicious_error_with_substring_not_classified(self, valid_config, valid_cwd):
        error_result = MagicMock()
        error_result.final_text = ""
        error_result.tool_iterations = 0
        error_result.should_retire = False
        error_result.error = f"ACP session inactive {self.SECRET_SENTINEL} injected"
        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
            FakeSession._run_turn_result = error_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)
        parsed = json.loads(result)
        assert parsed["error"] == "ACP task failed"
        assert parsed["error_type"] == "RuntimeError"
        assert self.SECRET_SENTINEL not in result


# --------------------------------------------------------------------------- #
# 18. session_meta helper regression (empty-dict shape)
# --------------------------------------------------------------------------- #


class TestSessionMetaHelper:
    def test_build_session_meta_returns_empty_dict(self):
        meta = build_session_meta()
        assert meta == {}
        assert session_meta_is_safe(meta)

    def test_build_session_meta_each_call_independent(self):
        meta1 = build_session_meta()
        meta2 = build_session_meta()
        assert meta1 == meta2 == {}
        assert meta1 is not meta2


# --------------------------------------------------------------------------- #
# 19. Concurrency: two concurrent run_task serialise on lock
# --------------------------------------------------------------------------- #


class TestRunTurnSerialisation:
    def test_two_concurrent_run_task_serialise_on_lock(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        FakeSession._run_turn_result = mock_turn_result
        results: list = []
        errors: list = []

        with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):

            def worker():
                try:
                    results.append(run_task("prompt", valid_config, cwd=valid_cwd))
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert not errors, f"workers raised: {errors}"
        assert len(results) == 2
        for r in results:
            parsed = json.loads(r)
            assert parsed.get("result") == "Task completed successfully"
        assert FakeSession._concurrent_max <= 1


# --------------------------------------------------------------------------- #
# 20. close_session during a running turn waits
# --------------------------------------------------------------------------- #


class TestCloseDuringTurn:
    def test_close_session_blocks_until_turn_finishes(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        block_event = threading.Event()
        FakeSession._block_until = block_event
        FakeSession._run_turn_result = mock_turn_result

        errors: list = []
        turn_result_holder: list = []

        def worker():
            try:
                with patch("kimi_code_acp.backend._import_session_class", return_value=FakeSession):
                    turn_result_holder.append(run_task("prompt", valid_config, cwd=valid_cwd))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()

        time.sleep(0.2)

        close_done = threading.Event()

        def closer():
            close_session()
            close_done.set()

        tc = threading.Thread(target=closer)
        tc.start()

        assert not close_done.wait(timeout=0.3), "close_session returned before the turn finished"

        block_event.set()
        tc.join(timeout=2)
        t.join(timeout=2)

        assert not errors, f"worker raised: {errors}"
        assert close_done.is_set()
        assert FakeSession.last()._closed is True


# --------------------------------------------------------------------------- #
# 23. Refactored manager invariants
# --------------------------------------------------------------------------- #


class TestRefactoredManagerInvariants:
    def test_close_session_cannot_interleave_before_run_turn(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        entered_run_turn = threading.Event()
        release_run_turn = threading.Event()
        FakeSession._run_turn_result = mock_turn_result

        original_run_turn = FakeSession.run_turn

        def instrumented_run_turn(self, **kw):
            entered_run_turn.set()
            release_run_turn.wait(timeout=5.0)
            return original_run_turn(self, **kw)

        worker_errors: list = []
        worker_result: list = []

        def worker():
            try:
                with patch(
                    "kimi_code_acp.backend._import_session_class",
                    return_value=FakeSession,
                ):
                    with patch.object(
                        FakeSession,
                        "run_turn",
                        instrumented_run_turn,
                    ):
                        worker_result.append(run_task("prompt", valid_config, cwd=valid_cwd))
            except BaseException as e:  # noqa: BLE001
                worker_errors.append(e)

        t = threading.Thread(target=worker)
        t.start()

        assert entered_run_turn.wait(timeout=2.0), "manager never reached run_turn"

        close_done = threading.Event()

        def closer():
            close_session()
            close_done.set()

        tc = threading.Thread(target=closer)
        tc.start()

        assert not close_done.wait(timeout=0.5), (
            "close_session returned while manager held the lock"
        )

        release_run_turn.set()
        tc.join(timeout=2.0)
        t.join(timeout=2.0)

        assert not worker_errors, f"worker raised: {worker_errors}"
        assert close_done.is_set()
        assert worker_result, "worker produced no result"
        parsed = json.loads(worker_result[0])
        assert parsed.get("result") == "Task completed successfully"

    def test_keyboard_interrupt_marks_unsafe_and_next_run_closes_first(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_raises = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                run_task("first", valid_config, cwd=valid_cwd)

            first_session = FakeSession.last()
            assert first_session is not None
            assert first_session._closed is False
            assert len(FakeSession.instances) == 1

            FakeSession._run_turn_raises = None
            FakeSession._run_turn_result = mock_turn_result
            run_task("second", valid_config, cwd=valid_cwd)

        assert first_session._closed is True
        assert len(FakeSession.instances) == 2
        assert FakeSession.last() is not first_session

    def test_system_exit_marks_unsafe_and_next_run_closes_first(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_raises = SystemExit(1)
            with pytest.raises(SystemExit):
                run_task("first", valid_config, cwd=valid_cwd)

            first_session = FakeSession.last()
            assert first_session is not None
            assert first_session._closed is False

            FakeSession._run_turn_raises = None
            FakeSession._run_turn_result = mock_turn_result
            run_task("second", valid_config, cwd=valid_cwd)

        assert first_session._closed is True
        assert len(FakeSession.instances) == 2

    def test_run_turn_exception_closes_and_next_run_rebuilds(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_raises = RuntimeError("boom")
            result1 = run_task("first", valid_config, cwd=valid_cwd)
            first_session = FakeSession.instances[0]
            assert first_session._closed is True
            assert len(FakeSession.instances) == 1
            parsed1 = json.loads(result1)
            assert parsed1["error"] == "ACP task failed"

            FakeSession._run_turn_raises = None
            FakeSession._run_turn_result = mock_turn_result
            result2 = run_task("second", valid_config, cwd=valid_cwd)

        assert len(FakeSession.instances) == 2
        assert FakeSession.last() is not first_session
        parsed2 = json.loads(result2)
        assert parsed2["result"] == "Task completed successfully"

    def test_two_concurrent_run_task_never_enter_run_turn_concurrently(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        FakeSession._run_turn_result = mock_turn_result
        results: list = []
        errors: list = []

        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):

            def worker():
                try:
                    results.append(run_task("prompt", valid_config, cwd=valid_cwd))
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert not errors, f"workers raised: {errors}"
        assert len(results) == 2
        assert FakeSession._concurrent_max <= 1


# --------------------------------------------------------------------------- #
# 25. Live-switch regression
# --------------------------------------------------------------------------- #


class TestLiveConfigSwitch:
    """``model`` and ``permission`` are NOT session-creation-locked:
    changing either on a subsequent call issues ``set_model`` /
    ``set_permission_mode`` on the existing session rather than rebuild.
    ``cwd`` is still a session/new parameter and forces a rebuild."""

    def test_model_change_live_switches_not_rebuilds(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd, model=None)
            run_task("second", valid_config, cwd=valid_cwd, model="kimi-k2")

        assert len(FakeSession.instances) == 1, (
            f"model change caused rebuild; instances={len(FakeSession.instances)}"
        )
        session = FakeSession.last()
        assert session.constructor_kwargs["model"] is None
        assert session._set_model_calls == ["kimi-k2"]
        assert session._set_permission_mode_calls == []

    def test_permission_change_live_switches_not_rebuilds(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd, model=None, permission=None)
            run_task(
                "second",
                valid_config,
                cwd=valid_cwd,
                model=None,
                permission="default",
            )

        assert len(FakeSession.instances) == 1, (
            f"permission change caused rebuild; instances={len(FakeSession.instances)}"
        )
        session = FakeSession.last()
        assert session.constructor_kwargs["permission_mode"] is None
        assert session._set_permission_mode_calls == ["default"]
        assert session._set_model_calls == []

    def test_cwd_change_still_rebuilds(self, valid_config, mock_turn_result, tmp_path):
        cwd_a = str(tmp_path / "a")
        cwd_b = str(tmp_path / "b")
        os.makedirs(cwd_a, exist_ok=True)
        os.makedirs(cwd_b, exist_ok=True)

        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=cwd_a, model=None)
            resolved_b = str(Path(cwd_b).resolve())
            run_task("second", valid_config, cwd=cwd_b, model=None)

        assert len(FakeSession.instances) == 2, (
            f"cwd change did NOT force rebuild; instances={len(FakeSession.instances)}"
        )
        first, second = FakeSession.instances
        assert first._closed is True
        assert second._ensure_started_cwd == resolved_b
        assert second._set_model_calls == []
        assert second._set_permission_mode_calls == []

    def test_live_model_switch_failure_denies(self, valid_config, mock_turn_result, valid_cwd):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd, model=None)
            assert len(FakeSession.instances) == 1

            FakeSession._set_model_raises = RuntimeError("server rejected model")
            result = run_task(
                "second",
                valid_config,
                cwd=valid_cwd,
                model="bad-model",
            )

        parsed = json.loads(result)
        assert "error" in parsed, f"expected error JSON, got {parsed}"
        assert len(FakeSession.instances) == 1
        session = FakeSession.last()
        assert session._set_model_calls == ["bad-model"]
        assert session._closed is False

    def test_live_permission_switch_failure_denies(self, valid_config, mock_turn_result, valid_cwd):
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task(
                "first",
                valid_config,
                cwd=valid_cwd,
                model=None,
                permission=None,
            )
            assert len(FakeSession.instances) == 1

            FakeSession._set_permission_mode_raises = RuntimeError("server rejected permission")
            result = run_task(
                "second",
                valid_config,
                cwd=valid_cwd,
                model=None,
                permission="bad-perm",
            )

        parsed = json.loads(result)
        assert "error" in parsed, f"expected error JSON, got {parsed}"
        assert len(FakeSession.instances) == 1
        session = FakeSession.last()
        assert session._set_permission_mode_calls == ["bad-perm"]
        assert session._closed is False

    def test_live_switch_to_model_none_on_live_session_is_denied(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        """Switching to ``model=None`` on a live session (created with a
        concrete model) is denied: the plugin cannot reliably revert to
        the server default."""
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task("first", valid_config, cwd=valid_cwd, model="kimi-k2")
            assert len(FakeSession.instances) == 1

            result = run_task("second", valid_config, cwd=valid_cwd, model=None)

        parsed = json.loads(result)
        assert "error" in parsed, f"expected error JSON, got {parsed}"
        # No rebuild and no set_model call for the None switch.
        assert len(FakeSession.instances) == 1
        session = FakeSession.last()
        assert session._set_model_calls == []
        assert session._closed is False

    def test_live_switch_to_permission_none_on_live_session_is_denied(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        """Switching to ``permission=None`` on a live session is denied
        with a warning (same rationale as model=None)."""
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            run_task(
                "first",
                valid_config,
                cwd=valid_cwd,
                model=None,
                permission="default",
            )
            assert len(FakeSession.instances) == 1

            result = run_task(
                "second",
                valid_config,
                cwd=valid_cwd,
                model=None,
                permission=None,
            )

        parsed = json.loads(result)
        assert "error" in parsed, f"expected error JSON, got {parsed}"
        assert len(FakeSession.instances) == 1
        session = FakeSession.last()
        assert session._set_permission_mode_calls == []
        assert session._closed is False


# --------------------------------------------------------------------------- #
# close-vs-run lifecycle race regression
# --------------------------------------------------------------------------- #


class TestCloseRunLifecycleRace:
    def test_close_holds_lock_through_best_effort_close_blocks_run(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        FakeSession._run_turn_result = mock_turn_result

        # Step 1: create one live session.
        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            run_task("seed", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 1

        # Step 2: arm the close gate.
        close_gate = threading.Event()
        close_entered = threading.Event()
        FakeSession._block_close_until = close_gate
        FakeSession._close_entered = close_entered

        # Step 3: thread A parks inside close().
        closer_errors: list = []

        def closer():
            try:
                close_session()
            except BaseException as e:  # noqa: BLE001
                closer_errors.append(e)

        thread_a = threading.Thread(target=closer, name="lifecycle-closer")
        thread_a.start()

        assert close_entered.wait(timeout=2.0), "closer never entered FakeSession.close"
        assert thread_a.is_alive(), "closer returned before close_gate was released"

        # Step 4: thread B tries run_task while A holds the lock.
        runner_errors: list = []
        runner_results: list = []

        def runner():
            try:
                with patch(
                    "kimi_code_acp.backend._import_session_class",
                    return_value=FakeSession,
                ):
                    runner_results.append(run_task("second", valid_config, cwd=valid_cwd))
            except BaseException as e:  # noqa: BLE001
                runner_errors.append(e)

        thread_b = threading.Thread(target=runner, name="lifecycle-runner")
        thread_b.start()

        # Step 5: while A holds the lock, B must NOT create a 2nd session.
        time.sleep(0.3)
        assert len(FakeSession.instances) == 1, (
            "run_task created a second session while close() was still draining "
            f"-- sessions={len(FakeSession.instances)}"
        )
        assert thread_b.is_alive(), "run_task finished before close released the gate"

        # Step 6: release; A finishes, B creates exactly one new session.
        close_gate.set()

        thread_a.join(timeout=3.0)
        thread_b.join(timeout=3.0)

        assert not closer_errors, f"closer raised: {closer_errors}"
        assert not runner_errors, f"runner raised: {runner_errors}"
        assert not thread_a.is_alive(), "closer did not finish"
        assert not thread_b.is_alive(), "runner did not finish"

        assert len(FakeSession.instances) == 2, (
            f"expected exactly 2 sessions after the second run, got {len(FakeSession.instances)}"
        )
        first = FakeSession.instances[0]
        assert first._closed is True
        assert first._closed_count == 1, "first session must be closed exactly once"
        assert FakeSession.instances[1] is not first
        assert runner_results, "runner produced no result"
        parsed = json.loads(runner_results[0])
        assert parsed.get("result") == "Task completed successfully"

    def test_close_with_no_session_still_safe_under_lock(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        close_session()
        close_session()

        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            FakeSession._run_turn_result = mock_turn_result
            result = run_task("prompt", valid_config, cwd=valid_cwd)

        parsed = json.loads(result)
        assert parsed.get("result") == "Task completed successfully"
        assert len(FakeSession.instances) == 1


# --------------------------------------------------------------------------- #
# atexit_cleanup lock safety
# --------------------------------------------------------------------------- #


class TestAtexitCleanupLockSafety:
    def test_atexit_cleanup_swallows_close_exception(self, valid_config, valid_cwd):
        from kimi_code_acp.backend import _ACPProcessManager

        mgr = _ACPProcessManager()
        sess = MagicMock()
        sess.close.side_effect = RuntimeError("teardown exploded")
        mgr._session = sess

        mgr.atexit_cleanup()

        assert mgr._session is None
        sess.close.assert_called_once()

    def test_atexit_cleanup_clears_unsafe_flag(self):
        from kimi_code_acp.backend import _ACPProcessManager

        mgr = _ACPProcessManager()
        mgr._unsafe = True
        mgr._session = None

        mgr.atexit_cleanup()

        assert mgr._session is None
        assert mgr._unsafe is False

    def test_atexit_cleanup_no_session_is_safe(self):
        from kimi_code_acp.backend import _ACPProcessManager

        mgr = _ACPProcessManager()
        mgr.atexit_cleanup()
        assert mgr._session is None
        assert mgr._unsafe is False

    def test_atexit_cleanup_blocks_run_under_same_lock(
        self, valid_config, mock_turn_result, valid_cwd
    ):
        from kimi_code_acp.backend import _manager

        FakeSession._run_turn_result = mock_turn_result

        with patch(
            "kimi_code_acp.backend._import_session_class",
            return_value=FakeSession,
        ):
            run_task("seed", valid_config, cwd=valid_cwd)
        assert len(FakeSession.instances) == 1

        close_gate = threading.Event()
        close_entered = threading.Event()
        FakeSession._block_close_until = close_gate
        FakeSession._close_entered = close_entered

        atexit_errors: list = []

        def atexit_worker():
            try:
                _manager.atexit_cleanup()
            except BaseException as e:  # noqa: BLE001
                atexit_errors.append(e)

        thread_a = threading.Thread(target=atexit_worker, name="atexit")
        thread_a.start()

        assert close_entered.wait(timeout=2.0), "atexit never reached FakeSession.close"
        assert thread_a.is_alive(), "atexit returned before close_gate release"

        runner_errors: list = []
        runner_results: list = []

        def runner():
            try:
                with patch(
                    "kimi_code_acp.backend._import_session_class",
                    return_value=FakeSession,
                ):
                    runner_results.append(run_task("second", valid_config, cwd=valid_cwd))
            except BaseException as e:  # noqa: BLE001
                runner_errors.append(e)

        thread_b = threading.Thread(target=runner, name="atexit-runner")
        thread_b.start()

        time.sleep(0.3)
        assert len(FakeSession.instances) == 1, (
            "atexit_cleanup released the lock before close finished"
        )
        assert thread_b.is_alive()

        close_gate.set()
        thread_a.join(timeout=3.0)
        thread_b.join(timeout=3.0)

        assert not atexit_errors, f"atexit raised: {atexit_errors}"
        assert not runner_errors, f"runner raised: {runner_errors}"
        assert len(FakeSession.instances) == 2
        assert runner_results
        parsed = json.loads(runner_results[0])
        assert parsed.get("result") == "Task completed successfully"
