"""Tests for plugin registration and tool handler.

Tests verify:
  1. The tool schema exposes exactly ``prompt`` + ``cwd`` + ``model``
     + ``permission`` and no forbidden params.
  2. The auxiliary task is registered with the correct key and defaults.
  3. The tool handler returns a JSON string for every path.
  4. register() registers all four surfaces: auxiliary task, coding tool,
     delegation provider, and two ACP runtime providers (kimi-agent-acp
     + kimi-code-acp alias).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kimi_code_acp.config import CONFIG_SECTION, DEFAULTS
from kimi_code_acp.tool import (
    FORBIDDEN_PARAMS,
    KIMI_CODE_ACP_SCHEMA,
    REQUIRED_PARAMS,
    handle_kimi_code_acp,
    run_task,
    validate_permission,
)


# --------------------------------------------------------------------------- #
# Schema surface
# --------------------------------------------------------------------------- #

class TestSchemaSurface:
    def test_schema_name(self):
        assert KIMI_CODE_ACP_SCHEMA["name"] == "kimi_code_acp"

    def test_schema_has_prompt_property(self):
        props = KIMI_CODE_ACP_SCHEMA["parameters"]["properties"]
        assert "prompt" in props
        assert props["prompt"]["type"] == "string"

    def test_schema_required_is_prompt_cwd_model_permission(self):
        required = KIMI_CODE_ACP_SCHEMA["parameters"]["required"]
        assert required == ["prompt", "cwd", "model", "permission"]

    def test_schema_has_no_forbidden_params(self):
        props = KIMI_CODE_ACP_SCHEMA["parameters"]["properties"]
        exposed = set(props.keys())
        forbidden_found = exposed & FORBIDDEN_PARAMS
        assert forbidden_found == set(), (
            f"Forbidden params exposed in schema: {forbidden_found}"
        )

    def test_schema_required_matches_expected(self):
        assert REQUIRED_PARAMS == ("prompt", "cwd", "model", "permission")

    def test_schema_properties_are_prompt_cwd_model_permission(self):
        props = KIMI_CODE_ACP_SCHEMA["parameters"]["properties"]
        assert set(props.keys()) == {"prompt", "cwd", "model", "permission"}

    def test_schema_model_property_is_nullable_with_none_default(self):
        props = KIMI_CODE_ACP_SCHEMA["parameters"]["properties"]
        model_prop = props["model"]
        assert model_prop["type"] == ["string", "null"]
        assert model_prop["default"] is None

    def test_schema_permission_property_is_nullable_with_none_default(self):
        props = KIMI_CODE_ACP_SCHEMA["parameters"]["properties"]
        perm_prop = props["permission"]
        assert perm_prop["type"] == ["string", "null"]
        assert perm_prop["default"] is None


# --------------------------------------------------------------------------- #
# validate_permission — per-call permission argument validation
# --------------------------------------------------------------------------- #

class TestValidatePermission:
    def test_none_passes(self):
        assert validate_permission(None) is None

    def test_non_empty_string_passes(self):
        assert validate_permission("default") == "default"
        assert validate_permission("plan") == "plan"
        assert validate_permission("auto") == "auto"

    def test_string_is_stripped(self):
        assert validate_permission("  default  ") == "default"
        assert validate_permission("\tplan\n") == "plan"

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            validate_permission("")

    def test_whitespace_only_string_rejected(self):
        with pytest.raises(ValueError):
            validate_permission("   ")
        with pytest.raises(ValueError):
            validate_permission("\t\n")

    def test_non_string_rejected(self):
        for bad in (0, 1, True, False, 3.14, ["default"], {"k": "v"}, object()):
            with pytest.raises(ValueError):
                validate_permission(bad)

    def test_error_message_does_not_echo_value(self):
        sentinel = "SUPER_SECRET_PERM_xyz789"
        with pytest.raises(ValueError) as exc_info:
            validate_permission([sentinel])
        assert sentinel not in str(exc_info.value)
        assert sentinel not in repr(exc_info.value)


# --------------------------------------------------------------------------- #
# Plugin registration
# --------------------------------------------------------------------------- #

def _load_plugin_module():
    """Load plugin __init__.py fresh (avoid global singleton state)."""
    plugin_dir = Path(__file__).resolve().parent.parent
    init_file = plugin_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        f"plugin_init_reg_{id(init_file)}",
        str(init_file),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPluginRegistration:
    def test_register_does_not_call_register_auxiliary_task(self):
        """Regression guard: the plugin is a tool + process transport,
        NOT an LLM side-task, so it must never register as auxiliary.
        See ``register()`` docstring for the architectural rationale."""
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        ctx.register_auxiliary_task.assert_not_called()

    def test_register_surface_is_tool_delegation_and_two_runtimes(self):
        """Lock in the contract: ``register(ctx)`` calls exactly these
        four registration surfaces and nothing else."""
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        ctx.register_tool.assert_called_once()
        ctx.register_delegation_provider.assert_called_once()
        assert ctx.register_acp_runtime_provider.call_count == 2

    def test_register_calls_register_tool(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        ctx.register_tool.assert_called_once()
        call_kwargs = ctx.register_tool.call_args
        assert call_kwargs.kwargs["name"] == "kimi_code_acp"
        assert call_kwargs.kwargs["schema"] == KIMI_CODE_ACP_SCHEMA
        assert callable(call_kwargs.kwargs["handler"])

    def test_register_calls_register_delegation_provider(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        ctx.register_delegation_provider.assert_called_once()

    def test_register_calls_register_acp_runtime_provider_twice(self):
        """Two runtime provider keys: ``kimi-agent-acp`` + the
        ``kimi-code-acp`` alias."""
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        assert ctx.register_acp_runtime_provider.call_count == 2

    def test_runtime_provider_keys(self):
        """The two runtime provider registrations cover both
        ``kimi-agent-acp`` and ``kimi-code-acp``."""
        from kimi_code_acp.runtime import RUNTIME_PROVIDER_KEY
        from kimi_code_acp.delegation import DELEGATION_PROVIDER_KEY
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        keys = [
            call.args[0]
            for call in ctx.register_acp_runtime_provider.call_args_list
        ]
        assert RUNTIME_PROVIDER_KEY in keys
        assert DELEGATION_PROVIDER_KEY in keys


# --------------------------------------------------------------------------- #
# Handler return type
# --------------------------------------------------------------------------- #

class TestHandlerReturnType:
    def test_handler_returns_string(self, tmp_path):
        cfg = dict(DEFAULTS)
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {CONFIG_SECTION: cfg}
            result = handle_kimi_code_acp({
                "prompt": "do something",
                "cwd": str(tmp_path),
                "model": None,
                "permission": None,
            })
            assert isinstance(result, str)

    def test_handler_empty_prompt_returns_error_string(self, tmp_path):
        result = handle_kimi_code_acp({"prompt": "", "cwd": str(tmp_path)})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_handler_missing_prompt_returns_error_string(self, tmp_path):
        result = handle_kimi_code_acp({"cwd": str(tmp_path)})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_handler_bad_cwd_returns_error_string(self):
        result = handle_kimi_code_acp({
            "prompt": "do something",
            "cwd": "/nonexistent/path/to/nowhere",
            "model": None,
            "permission": None,
        })
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed.get("error_type") == "ValueError"

    def test_handler_bad_model_returns_error_string(self, tmp_path):
        """Non-string non-null ``model`` is rejected with ValueError JSON."""
        result = handle_kimi_code_acp({
            "prompt": "do something",
            "cwd": str(tmp_path),
            "model": 123,
            "permission": None,
        })
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed.get("error_type") == "ValueError"

    def test_handler_bad_permission_returns_error_string(self, tmp_path):
        """Non-string non-null ``permission`` is rejected with ValueError JSON."""
        result = handle_kimi_code_acp({
            "prompt": "do something",
            "cwd": str(tmp_path),
            "model": None,
            "permission": 123,
        })
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed.get("error_type") == "ValueError"


# --------------------------------------------------------------------------- #
# Handler ConfigError: no exception text leaked
# --------------------------------------------------------------------------- #

class TestHandlerConfigErrorNoLeak:
    SECRET_SENTINEL = "SUPER_SECRET_TOKEN_xyz789"

    def test_config_error_message_not_leaked(self, tmp_path):
        from kimi_code_acp.config import ConfigError

        sentinel_msg = f"acp_command references {self.SECRET_SENTINEL}"
        with patch("kimi_code_acp.config.merge_config") as mock_merge, \
             patch("kimi_code_acp.config.validate_config",
                   side_effect=ConfigError(sentinel_msg)):
            mock_merge.return_value = {}
            result = handle_kimi_code_acp({
                "prompt": "do something",
                "cwd": str(tmp_path),
                "model": None,
                "permission": None,
            })

        parsed = json.loads(result)
        assert parsed["error"] == "ACP configuration validation failed"
        assert parsed["error_type"] == "ConfigError"
        assert self.SECRET_SENTINEL not in result

    def test_config_error_stable_json_shape(self, tmp_path):
        from kimi_code_acp.config import ConfigError

        with patch("kimi_code_acp.config.merge_config") as mock_merge, \
             patch("kimi_code_acp.config.validate_config",
                   side_effect=ConfigError("detailed info with path /secret")):
            mock_merge.return_value = {}
            result = handle_kimi_code_acp({
                "prompt": "do something",
                "cwd": str(tmp_path),
                "model": None,
                "permission": None,
            })

        parsed = json.loads(result)
        assert set(parsed.keys()) == {"error", "error_type"}
        assert parsed["error"] == "ACP configuration validation failed"
        assert parsed["error_type"] == "ConfigError"


# --------------------------------------------------------------------------- #
# Backend delegation (run_task delegates to backend.run_task)
# --------------------------------------------------------------------------- #

class TestBackendDelegation:
    def test_run_task_returns_json_string(self, tmp_path):
        cfg = dict(DEFAULTS)
        with patch("kimi_code_acp.backend._import_session_class", return_value=None):
            result = run_task("test prompt", cfg, cwd=str(tmp_path))
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_run_task_does_not_fake_success_without_backend(self, tmp_path):
        cfg = dict(DEFAULTS)
        with patch("kimi_code_acp.backend._import_session_class", return_value=None):
            result = run_task("test prompt", cfg, cwd=str(tmp_path))
        parsed = json.loads(result)
        assert "error" in parsed


# --------------------------------------------------------------------------- #
# register() surface isolation — no aux-task, exactly four surfaces
# --------------------------------------------------------------------------- #

class TestRegisterSurfaceIsolation:
    def test_register_does_not_touch_other_surfaces(self):
        """``register(ctx)`` must ONLY touch the four documented surfaces.
        This guards against an accidental re-add of auxiliary-task
        registration or some other plugin hook."""
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        # Tool + delegation + 2× runtime = exactly 4 register_* calls.
        total = (
            ctx.register_tool.call_count
            + ctx.register_delegation_provider.call_count
            + ctx.register_acp_runtime_provider.call_count
        )
        assert total == 4
        assert ctx.register_auxiliary_task.call_count == 0
