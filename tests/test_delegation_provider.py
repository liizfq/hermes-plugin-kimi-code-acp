"""Tests for the kimi-code-acp delegation provider.

Tests verify:
  1. The plugin registers a delegation provider during register(ctx).
  2. The resolver returns the correct descriptor shape.
  3. delegation.model overrides every other source.
  4. Model priority: requested_model > auxiliary.model > _DEFAULT_DELEGATION_MODEL
     ("kimi-k2").
  5. The descriptor does NOT carry workdir/workspace/workspaces/cwd.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kimi_code_acp.config import ACP_ARGS, ACP_COMMAND, AUXILIARY_KEY, DEFAULTS
from kimi_code_acp.delegation import (
    DELEGATION_PROVIDER_KEY,
    _DEFAULT_DELEGATION_MODEL,
    resolve_delegation_provider,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

class TestDelegationProviderKey:
    def test_key_is_camel_case_slug(self):
        assert DELEGATION_PROVIDER_KEY == "kimi-code-acp"


# --------------------------------------------------------------------------- #
# Resolver behaviour
# --------------------------------------------------------------------------- #

class TestResolverShape:
    def test_returns_dict_with_required_keys(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert isinstance(result, dict)
        for key in ("provider", "model", "api_mode", "base_url", "api_key",
                     "command", "args", "display_provider"):
            assert key in result

    def test_provider_is_acp_client(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["provider"] == "acp_client"

    def test_display_provider_is_kimi_code_acp(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["display_provider"] == DELEGATION_PROVIDER_KEY

    def test_api_mode_is_acp_client(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["api_mode"] == "acp_client"

    def test_api_key_is_empty(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["api_key"] == ""

    def test_command_and_args_from_config(self):
        """``command`` / ``args`` always come from the fixed launcher
        constants, NOT from operator config."""
        custom = dict(DEFAULTS)
        custom["acp_command"] = "custom-binary"
        custom["acp_args"] = ["--flag", "value"]
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = custom
            result = resolve_delegation_provider(None, {})
        assert result["command"] == ACP_COMMAND
        assert result["args"] == list(ACP_ARGS)

    def test_command_is_kimi(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["command"] == "kimi"

    def test_args_has_acp(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["args"] == ["acp"]


class TestResolverNoWorkdir:
    """The descriptor must NOT carry path keys -- cwd is a per-call
    parameter on the tool, not an operator-config field."""

    def test_no_path_keys_in_descriptor(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        for forbidden in ("workdir", "workspace", "workspaces", "cwd"):
            assert forbidden not in result, (
                f"delegation descriptor carries forbidden key {forbidden!r}"
            )


class TestResolverModelOverride:
    def test_requested_model_overrides_auxiliary_model(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider("my-model", {})
        assert result["model"] == "my-model"

    def test_falls_back_to_fixed_delegation_default(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        assert result["model"] == _DEFAULT_DELEGATION_MODEL

    def test_fixed_default_is_kimi_k2(self):
        assert _DEFAULT_DELEGATION_MODEL == "kimi-k2"

    def test_falls_back_to_fixed_default_when_auxiliary_missing(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = {}
            result = resolve_delegation_provider(None, {})
        assert result["model"] == _DEFAULT_DELEGATION_MODEL

    def test_auxiliary_model_overrides_fixed_default(self):
        custom = dict(DEFAULTS)
        custom["model"] = "my-aux-model"
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = custom
            result = resolve_delegation_provider(None, {})
        assert result["model"] == "my-aux-model"

    def test_requested_model_wins_over_auxiliary_model(self):
        custom = dict(DEFAULTS)
        custom["model"] = "aux-model"
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = custom
            result = resolve_delegation_provider("explicit", {})
        assert result["model"] == "explicit"


class TestResolverConfigReading:
    def test_reads_from_auxiliary_config(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            resolve_delegation_provider(None, {})
        mock_merge.assert_called_once()

    def test_args_are_independent_copy(self):
        with patch("kimi_code_acp.delegation.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_delegation_provider(None, {})
        original_args = list(ACP_ARGS)
        assert isinstance(result["args"], list)
        result["args"].append("INJECTED")
        assert list(ACP_ARGS) == original_args


# --------------------------------------------------------------------------- #
# Plugin registration
# --------------------------------------------------------------------------- #

def _load_plugin_module():
    plugin_dir = Path(__file__).resolve().parent.parent
    init_file = plugin_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        f"plugin_init_del_{id(init_file)}",
        str(init_file),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPluginRegistersDelegationProvider:
    def test_register_calls_register_delegation_provider(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        ctx.register_delegation_provider.assert_called_once()

    def test_registered_with_correct_key(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        call_args = ctx.register_delegation_provider.call_args
        assert call_args.args[0] == DELEGATION_PROVIDER_KEY

    def test_registered_resolver_is_callable(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        call_args = ctx.register_delegation_provider.call_args
        assert callable(call_args.args[1])
