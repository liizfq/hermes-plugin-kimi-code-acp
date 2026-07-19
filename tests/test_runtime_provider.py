"""Tests for the kimi-code-acp ACP runtime provider.

Tests verify:
  1. The plugin registers an ACP runtime provider during register(ctx).
  2. The resolver returns the correct descriptor shape.
  3. Model priority (runtime path):
       requested_model (highest) > runtime_model override >
       kimi_code_acp.model > _DEFAULT_RUNTIME_MODEL ("kimi-k2").
  4. No workdir key in the descriptor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from kimi_code_acp.config import ACP_ARGS, ACP_COMMAND, AUXILIARY_KEY, DEFAULTS
from kimi_code_acp.runtime import (
    _DEFAULT_RUNTIME_MODEL,
    RUNTIME_PROVIDER_KEY,
    resolve_runtime_provider,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


class TestRuntimeProviderKey:
    def test_key_is_kimi_agent_acp(self):
        """Users type ``/acp-client-runtime on kimi-agent-acp``."""
        assert RUNTIME_PROVIDER_KEY == "kimi-agent-acp"


# --------------------------------------------------------------------------- #
# Resolver behaviour
# --------------------------------------------------------------------------- #


class TestResolverShape:
    def test_returns_dict_with_required_keys(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert isinstance(result, dict)
        for key in (
            "provider",
            "api_mode",
            "display_provider",
            "model",
            "command",
            "args",
            "base_url",
            "api_key",
        ):
            assert key in result

    def test_provider_is_acp_client(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["provider"] == "acp_client"

    def test_api_mode_is_acp_client(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["api_mode"] == "acp_client"

    def test_display_provider_is_kimi_code_acp(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["display_provider"] == "kimi-code-acp"

    def test_api_key_is_empty(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["api_key"] == ""

    def test_command_and_args_from_config(self):
        """``command`` / ``args`` always come from the fixed launcher
        constants, NOT from operator config."""
        custom = dict(DEFAULTS)
        custom["acp_command"] = "custom-binary"
        custom["acp_args"] = ["--flag", "value"]
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = custom
            result = resolve_runtime_provider(None, {})
        assert result["command"] == ACP_COMMAND
        assert result["args"] == list(ACP_ARGS)

    def test_default_command_is_kimi(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["command"] == "kimi"

    def test_default_args_has_acp(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        assert result["args"] == ["acp"]


class TestResolverNoWorkdir:
    """The descriptor must NOT carry path keys."""

    def test_no_path_keys_in_descriptor(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
        for forbidden in ("workdir", "workspace", "workspaces", "cwd"):
            assert forbidden not in result


class TestResolverModel:
    def test_default_model_is_kimi_k2(self):
        """When no model is configured anywhere, the runtime default is
        ``kimi-k2``."""
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = {}
                result = resolve_runtime_provider(None, {})
        assert result["model"] == "kimi-k2"

    def test_runtime_default_constant_is_kimi_k2(self):
        assert _DEFAULT_RUNTIME_MODEL == "kimi-k2"

    def test_requested_model_overrides_everything(self):
        """requested_model (priority 1) overrides all config-section fallbacks
        and the runtime default."""
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider("my-requested-model", {})
        assert result["model"] == "my-requested-model"

    def test_runtime_model_overrides_runtime_default(self):
        """A runtime-specific operator override (``runtime_model``) wins
        over the runtime default (``kimi-k2``)."""
        raw_aux = {AUXILIARY_KEY: {"runtime_model": "my-rt-model"}}
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = raw_aux
                result = resolve_runtime_provider(None, {})
        assert result["model"] == "my-rt-model"

    def test_runtime_model_overrides_general_config_model(self):
        """A runtime-specific override (``runtime_model``) wins over the
        general-purpose ``model`` default."""
        raw_aux = {
            AUXILIARY_KEY: {
                "runtime_model": "my-rt-model",
                "model": "general-model",
            }
        }
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = raw_aux
                result = resolve_runtime_provider(None, {})
        assert result["model"] == "my-rt-model"

    def test_config_model_overrides_runtime_default(self):
        """The general-purpose operator-configured
        ``kimi_code_acp.model`` is consulted as a fallback
        BEFORE the runtime default (``kimi-k2``)."""
        raw_aux = {AUXILIARY_KEY: {"model": "my-custom-model"}}
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = raw_aux
                result = resolve_runtime_provider(None, {})
        assert result["model"] == "my-custom-model"

    def test_requested_model_overrides_runtime_and_config(self):
        raw_aux = {
            AUXILIARY_KEY: {
                "runtime_model": "rt-model",
                "model": "general-model",
            }
        }
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = raw_aux
                result = resolve_runtime_provider("explicit-requested", {})
        assert result["model"] == "explicit-requested"

    def test_no_model_uses_runtime_default(self):
        """When the config block has no explicit model key, the runtime
        default (``kimi-k2``) applies."""
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            with patch("hermes_cli.config.load_config") as mock_lc:
                mock_lc.return_value = {AUXILIARY_KEY: {}}
                result = resolve_runtime_provider(None, {})
        assert result["model"] == "kimi-k2"


class TestResolverConfigReading:
    def test_reads_from_config_section(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            resolve_runtime_provider(None, {})
        mock_merge.assert_called_once()

    def test_args_are_independent_copy(self):
        with patch("kimi_code_acp.config.merge_config") as mock_merge:
            mock_merge.return_value = dict(DEFAULTS)
            result = resolve_runtime_provider(None, {})
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
        f"plugin_init_rt_{id(init_file)}",
        str(init_file),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPluginRegistersRuntimeProvider:
    def test_register_calls_register_acp_runtime_provider(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        # Should be called twice: kimi-agent-acp + kimi-code-acp alias
        assert ctx.register_acp_runtime_provider.call_count == 2

    def test_registered_with_correct_key(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        keys = [call.args[0] for call in ctx.register_acp_runtime_provider.call_args_list]
        assert RUNTIME_PROVIDER_KEY in keys

    def test_registered_alias_kimi_code_acp(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        keys = [call.args[0] for call in ctx.register_acp_runtime_provider.call_args_list]
        assert "kimi-code-acp" in keys

    def test_registered_resolver_is_callable(self):
        mod = _load_plugin_module()
        ctx = MagicMock()
        mod.register(ctx)
        for call in ctx.register_acp_runtime_provider.call_args_list:
            assert callable(call.args[1])
