"""Tests for kimi_code_acp.config — defaults, merge, and validation.

Kimi-specific differences from claude-code-acp:
  * ``DEFAULTS`` carries exactly three keys: ``timeout_seconds`` (600),
    ``model`` (None), ``permission`` (None).  **No** ``setting_sources``
    key -- Kimi's auth lives under ``~/.kimi-code/``.
  * ``ACP_COMMAND`` == ``"kimi"``, ``ACP_ARGS`` == ``("acp",)``.
  * ``RETIRED_CONFIG_KEYS`` does NOT exist in the Kimi config module
    (it exists in Claude but not here) — tests must not import it.
  * ``DEPRECATED_PATH_KEYS`` exists with ``{"workdir", "workspace",
    "workspaces"}``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kimi_code_acp.config import (
    ACP_ARGS,
    ACP_COMMAND,
    AUXILIARY_KEY,
    DEFAULTS,
    DEPRECATED_PATH_KEYS,
    ConfigError,
    merge_config,
    validate_config,
)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

class TestDefaults:
    def test_defaults_has_exactly_three_keys(self):
        """Only ``timeout_seconds``, ``model``, and ``permission`` are
        operator-configurable.  Unlike the Claude plugin there is **no**
        ``setting_sources`` key -- Kimi Code CLI carries its own auth."""
        assert set(DEFAULTS.keys()) == {"timeout_seconds", "model", "permission"}

    def test_defaults_model_and_permission_are_none(self):
        """``model`` and ``permission`` default to ``None`` (\"use the
        Kimi ACP server's own default\")."""
        assert DEFAULTS["model"] is None
        assert DEFAULTS["permission"] is None

    def test_defaults_does_not_carry_launcher_keys(self):
        """The ACP launcher is a fixed compatibility constant."""
        assert "acp_command" not in DEFAULTS
        assert "acp_args" not in DEFAULTS

    def test_defaults_does_not_carry_setting_sources(self):
        """Kimi auth lives under ``~/.kimi-code/`` -- there is no
        ``setting_sources`` operator config key."""
        assert "setting_sources" not in DEFAULTS

    def test_acp_command_constant_is_kimi(self):
        """The fixed launcher command constant is ``kimi``."""
        assert ACP_COMMAND == "kimi"

    def test_acp_args_constant_is_acp_subcommand(self):
        """The fixed launcher args constant is the ``acp`` subcommand."""
        assert list(ACP_ARGS) == ["acp"]

    def test_defaults_has_no_workdir(self):
        """The working directory is a per-call ``cwd`` parameter on the
        tool, not an auxiliary config field."""
        assert "workdir" not in DEFAULTS
        assert "workspace" not in DEFAULTS
        assert "workspaces" not in DEFAULTS
        assert "cwd" not in DEFAULTS

    def test_defaults_no_secrets(self):
        for key, val in DEFAULTS.items():
            key_lower = key.lower()
            assert "secret" not in key_lower
            assert "token" not in key_lower
            assert "key" not in key_lower
            assert "password" not in key_lower

    def test_timeout_within_bounds(self):
        assert 1 <= DEFAULTS["timeout_seconds"] <= 3600


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

class TestMerge:
    def test_merge_with_no_overrides_returns_defaults(self):
        merged = merge_config(user_overrides={})
        for key in DEFAULTS:
            assert merged[key] == DEFAULTS[key]

    def test_merge_user_wins_on_conflict(self):
        overrides = {"model": "kimi-k2", "timeout_seconds": 300}
        merged = merge_config(user_overrides=overrides)
        assert merged["model"] == "kimi-k2"
        assert merged["timeout_seconds"] == 300
        # Untouched keys keep defaults
        assert merged["permission"] is None

    def test_merge_none_overrides_returns_defaults(self):
        """When user_overrides is None (runtime path), merge_config reads
        from the Hermes config layer."""
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            merged = merge_config(user_overrides=None)
        for key in DEFAULTS:
            assert merged[key] == DEFAULTS[key]

    def test_merge_none_overrides_with_user_config_wins(self):
        """When user_overrides is None and a real config.yaml has an
        auxiliary override, the operator's value wins over DEFAULTS."""
        from kimi_code_acp.config import AUXILIARY_KEY
        fake_config = {
            "auxiliary": {
                AUXILIARY_KEY: {"model": "kimi-k2", "timeout_seconds": 300}
            }
        }
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = fake_config
            merged = merge_config(user_overrides=None)
        assert merged["model"] == "kimi-k2"
        assert merged["timeout_seconds"] == 300
        # Untouched keys keep defaults
        assert merged["permission"] is None

    def test_merge_adds_new_keys(self):
        # merge_config itself does not validate -- it just merges.
        overrides = {"custom_field": "value"}
        merged = merge_config(user_overrides=overrides)
        assert merged["custom_field"] == "value"


class TestMutableDefaultsIsolation:
    """Ensure merge_config() never shares mutable objects with the
    module-level DEFAULTS."""

    def test_merge_does_not_share_dict(self):
        """Mutating the merge result must not pollute DEFAULTS."""
        merged = merge_config(user_overrides={})
        merged["model"] = "INJECTED"
        assert DEFAULTS["model"] is None


# --------------------------------------------------------------------------- #
# Validation — unknown keys
# --------------------------------------------------------------------------- #

class TestValidateUnknownKeys:
    def test_unknown_key_rejected(self):
        cfg = _valid_config(unknown_key="value")
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    def test_unknown_key_name_not_leaked(self):
        sensitive_key = "SECRET_TOKEN_xyz789"
        cfg = _valid_config()
        cfg[sensitive_key] = "value"
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert sensitive_key not in str(exc_info.value)

    def test_unknown_key_value_not_leaked(self):
        secret_value = "SUPER_SECRET_VALUE_xyz789"
        cfg = _valid_config()
        cfg["unknown_key"] = secret_value
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert secret_value not in str(exc_info.value)

    def test_multiple_unknown_keys_rejected(self):
        cfg = _valid_config(key1="a", key2="b")
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    def test_only_defaults_keys_accepted(self):
        cfg = _valid_config()
        validate_config(cfg)  # should not raise


# --------------------------------------------------------------------------- #
# Validation — path keys are rejected (cwd-per-call design)
# --------------------------------------------------------------------------- #

class TestValidatePathKeysRejected:
    @pytest.mark.parametrize("path_key", sorted(DEPRECATED_PATH_KEYS))
    def test_deprecated_path_key_rejected(self, path_key):
        cfg = _valid_config()
        cfg[path_key] = "/some/path"
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    def test_cwd_key_rejected(self):
        cfg = _valid_config()
        cfg["cwd"] = "/some/path"
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    def test_workdir_value_not_leaked(self):
        secret = "SUPER_SECRET_PATH_xyz789"
        cfg = _valid_config()
        cfg["workdir"] = f"/tmp/{secret}/nonexistent"
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert secret not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Validation — acp_command / acp_args rejected as unsupported keys
# --------------------------------------------------------------------------- #

class TestValidateLauncherKeysRejected:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("acp_command", ""),
            ("acp_command", "   "),
            ("acp_command", 123),
            ("acp_command", "custom-binary"),
            ("acp_args", "not-a-list"),
            ("acp_args", ["-y", 123]),
            ("acp_args", ["--flag", "value"]),
        ],
    )
    def test_launcher_field_rejected_as_unsupported(self, field, value):
        cfg = _valid_config()
        cfg[field] = value
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    @pytest.mark.parametrize("field", ["acp_command", "acp_args"])
    def test_launcher_field_name_not_leaked(self, field):
        cfg = _valid_config()
        cfg[field] = "SUPER_SECRET_VALUE_xyz789"
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert field not in str(exc_info.value)

    @pytest.mark.parametrize("field", ["acp_command", "acp_args"])
    def test_launcher_field_value_not_leaked(self, field):
        secret = "SUPER_SECRET_VALUE_xyz789"
        cfg = _valid_config()
        cfg[field] = secret
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert secret not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Validation — setting_sources is rejected as unknown (Kimi has no
# setting_sources config key; auth lives under ~/.kimi-code/)
# --------------------------------------------------------------------------- #

class TestValidateSettingSourcesRejected:
    def test_setting_sources_rejected_as_unknown_key(self):
        cfg = _valid_config()
        cfg["setting_sources"] = ["user", "project", "local"]
        with pytest.raises(ConfigError, match="unsupported keys"):
            validate_config(cfg)

    def test_setting_sources_value_not_leaked(self):
        sentinel = "SUPER_SECRET_TOKEN_xyz789"
        cfg = _valid_config()
        cfg["setting_sources"] = [sentinel, "project"]
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert sentinel not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Validation — timeout_seconds
# --------------------------------------------------------------------------- #

class TestValidateTimeout:
    def test_timeout_below_min_rejected(self):
        cfg = _valid_config(timeout_seconds=0)
        with pytest.raises(ConfigError, match="timeout"):
            validate_config(cfg)

    def test_timeout_above_max_rejected(self):
        cfg = _valid_config(timeout_seconds=3601)
        with pytest.raises(ConfigError, match="timeout"):
            validate_config(cfg)

    def test_timeout_non_number_rejected(self):
        cfg = _valid_config(timeout_seconds="abc")
        with pytest.raises(ConfigError, match="timeout"):
            validate_config(cfg)

    def test_timeout_boolean_rejected(self):
        cfg = _valid_config(timeout_seconds=True)
        with pytest.raises(ConfigError, match="timeout"):
            validate_config(cfg)

    def test_timeout_at_boundaries_accepted(self):
        validate_config(_valid_config(timeout_seconds=1))
        validate_config(_valid_config(timeout_seconds=3600))


# --------------------------------------------------------------------------- #
# Validation — model
# --------------------------------------------------------------------------- #

class TestValidateModel:
    def test_model_none_passes(self):
        cfg = _valid_config(model=None)
        validate_config(cfg)
        assert cfg["model"] is None

    def test_model_non_empty_string_passes_and_is_stripped(self):
        cfg = _valid_config(model="  kimi-k2  ")
        validate_config(cfg)
        assert cfg["model"] == "kimi-k2"

    def test_model_simple_non_empty_string_passes(self):
        cfg = _valid_config(model="kimi-k2")
        validate_config(cfg)
        assert cfg["model"] == "kimi-k2"

    def test_model_empty_string_rejected(self):
        cfg = _valid_config(model="")
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)

    def test_model_whitespace_only_string_rejected(self):
        cfg = _valid_config(model="   ")
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)
        cfg = _valid_config(model="\t\n")
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)

    def test_model_int_rejected(self):
        cfg = _valid_config(model=123)
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)

    def test_model_bool_rejected(self):
        cfg = _valid_config(model=True)
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)

    def test_model_list_rejected(self):
        cfg = _valid_config(model=["kimi-k2"])
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)


# --------------------------------------------------------------------------- #
# Validation — permission
# --------------------------------------------------------------------------- #

class TestValidatePermission:
    def test_permission_none_passes(self):
        cfg = _valid_config(permission=None)
        validate_config(cfg)
        assert cfg["permission"] is None

    def test_permission_non_empty_string_passes_and_is_stripped(self):
        cfg = _valid_config(permission="  default  ")
        validate_config(cfg)
        assert cfg["permission"] == "default"

    def test_permission_empty_string_rejected(self):
        cfg = _valid_config(permission="")
        with pytest.raises(ConfigError, match="permission"):
            validate_config(cfg)

    def test_permission_whitespace_only_string_rejected(self):
        cfg = _valid_config(permission="   ")
        with pytest.raises(ConfigError, match="permission"):
            validate_config(cfg)

    def test_permission_int_rejected(self):
        cfg = _valid_config(permission=123)
        with pytest.raises(ConfigError, match="permission"):
            validate_config(cfg)

    def test_permission_bool_rejected(self):
        cfg = _valid_config(permission=True)
        with pytest.raises(ConfigError, match="permission"):
            validate_config(cfg)


# --------------------------------------------------------------------------- #
# Validation — model + permission combined
# --------------------------------------------------------------------------- #

class TestValidateModelAndPermissionCombined:
    def test_both_none_passes(self):
        cfg = _valid_config(model=None, permission=None)
        validate_config(cfg)

    def test_both_strings_passes(self):
        cfg = _valid_config(model="kimi-k2", permission="auto")
        validate_config(cfg)
        assert cfg["model"] == "kimi-k2"
        assert cfg["permission"] == "auto"

    def test_model_valid_permission_invalid_rejected(self):
        cfg = _valid_config(model="kimi-k2", permission="")
        with pytest.raises(ConfigError, match="permission"):
            validate_config(cfg)

    def test_model_invalid_permission_valid_rejected(self):
        cfg = _valid_config(model=123, permission="default")
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)


# --------------------------------------------------------------------------- #
# Security: error messages must not leak operator config values
# --------------------------------------------------------------------------- #

class TestNoSecretLeakInErrors:
    SECRET_SENTINEL = "SUPER_SECRET_TOKEN_xyz789"

    def test_command_value_not_leaked(self):
        cfg = _valid_config(acp_command="")
        cfg["acp_args"] = [self.SECRET_SENTINEL]
        cfg["acp_command"] = 123
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert self.SECRET_SENTINEL not in str(exc_info.value)

    def test_args_secret_not_leaked(self):
        cfg = _valid_config()
        cfg["acp_args"] = [self.SECRET_SENTINEL, 123]
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert self.SECRET_SENTINEL not in str(exc_info.value)

    def test_model_secret_not_leaked(self):
        cfg = _valid_config()
        cfg["model"] = 123
        cfg["acp_args"] = [self.SECRET_SENTINEL]
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert self.SECRET_SENTINEL not in str(exc_info.value)

    def test_permission_secret_not_leaked(self):
        cfg = _valid_config()
        cfg["permission"] = 123
        cfg["acp_args"] = [self.SECRET_SENTINEL]
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert self.SECRET_SENTINEL not in str(exc_info.value)

    def test_timeout_value_is_reported(self):
        """timeout_seconds IS safe to report numerically (no secret risk)."""
        cfg = _valid_config(timeout_seconds=99999)
        with pytest.raises(ConfigError) as exc_info:
            validate_config(cfg)
        assert "99999" in str(exc_info.value)

    def test_handler_config_error_no_secret_leak(self, tmp_path):
        """End-to-end: handler error string must not contain secret."""
        import json
        from kimi_code_acp.tool import handle_kimi_code_acp

        cfg = dict(DEFAULTS)
        cfg["acp_args"] = ["acp", self.SECRET_SENTINEL]
        cfg["timeout_seconds"] = 99999
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {"auxiliary": {AUXILIARY_KEY: cfg}}
            result = handle_kimi_code_acp(
                {"prompt": "do something", "cwd": str(tmp_path)}
            )
        parsed = json.loads(result)
        assert "error" in parsed
        assert self.SECRET_SENTINEL not in result

    def test_handler_path_key_secret_no_leak(self, tmp_path):
        import json
        from kimi_code_acp.tool import handle_kimi_code_acp

        cfg = dict(DEFAULTS)
        cfg["workdir"] = f"/nonexistent/{self.SECRET_SENTINEL}/path"
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {"auxiliary": {AUXILIARY_KEY: cfg}}
            result = handle_kimi_code_acp(
                {"prompt": "do something", "cwd": str(tmp_path)}
            )
        assert self.SECRET_SENTINEL not in result
        parsed = json.loads(result)
        assert "error" in parsed


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _valid_config(**overrides) -> dict:
    """Return a valid merged config (DEFAULTS only), plus overrides."""
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg
