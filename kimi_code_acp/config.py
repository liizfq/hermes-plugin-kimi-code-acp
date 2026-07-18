"""Configuration helpers for the kimi-code-acp plugin.

This module provides:
  - ``AUXILIARY_KEY``: the stable auxiliary-task key registered with Hermes.
  - ``ACP_COMMAND`` / ``ACP_ARGS``: the **fixed, non-configurable** ACP
    launcher.  The plugin always spawns ``kimi acp`` (the ACP mode of
    the ``@moonshot-ai/kimi-code`` CLI).  A different launcher is a
    source-level change, NOT an operator config override.
  - ``DEFAULTS``: safe default values for operator auxiliary config
    (``timeout_seconds``, ``model``, ``permission``).  ``model`` and
    ``permission`` are optional operator-supplied defaults that the
    per-call tool parameters fall back to when the caller passes
    ``null``; both default to ``None`` (meaning "use the Kimi ACP
    server's own default").
  - ``merge_config``: merge defaults <- user config, using
    ``hermes_cli.config.load_config`` to read the ``auxiliary.<key>``
    block from config.yaml.
  - ``validate_config``: strict validation of all operator-supplied fields.
  - ``ConfigError``: raised on validation failure.

Design notes
------------
* The auxiliary config slot is ``auxiliary.kimi_code_acp`` in config.yaml.
* Unlike the Claude Code ACP plugin, there is **no** ``setting_sources``
  config key.  Kimi Code CLI carries its own authentication state under
  ``~/.kimi-code/`` (populated by ``kimi`` login); it does not consult
  ``~/.claude/settings.json``.  Operator config covers only the
  inactivity ``timeout_seconds`` and the optional ``model`` /
  ``permission`` fallbacks.
* The ACP launcher (``acp_command`` / ``acp_args``) is **fixed**.  Any
  operator-supplied ``acp_command`` or ``acp_args`` key is rejected as
  an unknown key.
* The working directory is **not** operator-configured.  It is supplied
  per call by the model via the ``cwd`` parameter of the
  ``kimi_code_acp`` tool and validated in the handler at call time.
* ``model`` / ``permission`` are per-call on the tool.  When the caller
  passes ``null`` for either, the handler resolves the value from the
  operator-configured fallback in this auxiliary block, and finally from
  ``None`` (Kimi server default).  For Kimi, ``permission`` is mapped to
  the ACP ``mode`` axis (``default`` / ``plan`` / ``auto`` / ...) via
  ``session/set_config_option``.

Security note
-------------
ConfigError messages are returned to the model via the tool handler.
They must NEVER echo operator-supplied values (model names may be
sensitive).  Only type/rule information and numeric bounds are safe to
report.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

AUXILIARY_KEY = "kimi_code_acp"

#: Fixed ACP launcher -- the only launcher this plugin supports.
#:
#: The plugin always spawns ``kimi acp`` because the plugin's session
#: metadata, approval wiring, and ACP contract assumptions are pinned to
#: the Kimi Code CLI ACP adapter (``@moonshot-ai/acp-adapter`` inside
#: the ``MoonshotAI/kimi-code`` monorepo).  If a different launcher is
#: ever required, it must be a source-level change in this module, NOT
#: an operator config override.
#:
#: ``kimi`` resolves to the ``bin`` entry of the ``@moonshot-ai/kimi-code``
#: npm package.  ``acp`` is the subcommand that switches the CLI into
#: ACP (JSON-RPC over stdio) mode -- see ``docs/{zh,en}/reference/kimi-acp.md``
#: in the kimi-code repo for the capability matrix.
ACP_COMMAND: str = "kimi"
ACP_ARGS: tuple = ("acp",)

#: Safe default values for the auxiliary task.
#: These are layered underneath user config so the operator's explicit
#: settings always win. These same values are passed as ``defaults`` to
#: ``ctx.register_auxiliary_task()`` at plugin register time.
#:
#: Note: there is deliberately **no** ``acp_command`` / ``acp_args`` key
#: here -- the launcher is fixed.  There is also no ``setting_sources``
#: key -- Kimi Code CLI does not consult Claude-style setting scopes;
#: its auth lives under ``~/.kimi-code/``.
#:
#: ``model`` and ``permission`` are optional operator-configured fallbacks
#: for the per-call ``model`` / ``permission`` parameters on the
#: ``kimi_code_acp`` tool.  Both default to ``None`` ("use the Kimi ACP
#: server's own default").  The per-call value always wins; when the
#: caller passes ``null`` for either, the handler resolves the value
#: from config, and finally from ``None`` = server default.  Non-empty
#: strings are stripped; empty / whitespace-only strings and non-string
#: values are rejected by :func:`validate_config`.
DEFAULTS: Dict[str, Any] = {
    "timeout_seconds": 600,  # inactivity timeout (max continuous gap without ACP activity)
    # Optional per-call fallbacks. ``None`` means "use the Kimi ACP
    # server's own default"; a non-empty string is forwarded verbatim
    # to the ACP session constructor when the tool caller passes ``null``.
    "model": None,
    "permission": None,
}

#: Timeout bounds (seconds).
#: ``timeout_seconds`` is an **inactivity timeout**: the maximum continuous
#: period without ACP protocol activity before the turn is aborted.  It
#: is NOT a total task-duration limit.
_TIMEOUT_MIN = 1
_TIMEOUT_MAX = 3600

#: Operator config keys that historically named a workspace / working
#: directory.  Rejected as unknown keys; exported for tests and docs.
DEPRECATED_PATH_KEYS = frozenset({"workdir", "workspace", "workspaces"})


class ConfigError(ValueError):
    """Raised when auxiliary config validation fails."""


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def merge_config(user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge plugin defaults with user config.

    Resolution order (user wins):

        DEFAULTS  <-  user_overrides (if provided)
                     <-  config.yaml ``auxiliary.kimi_code_acp`` (if available)

    When *user_overrides* is ``None`` (the normal runtime path), the
    function reads from the Hermes config layer and applies it on top
    of ``dict(DEFAULTS)``.

    Only ``timeout_seconds``, ``model``, and ``permission`` are
    operator-configurable.  The ACP launcher is **never** read from
    config; it is a fixed code constant.  Any operator-supplied
    ``acp_command`` / ``acp_args`` is surfaced unchanged here but
    rejected by :func:`validate_config`.

    Parameters
    ----------
    user_overrides
        When provided (test path), use this dict directly instead of
        reading from Hermes config. When ``None``, read from Hermes.

    Returns
    -------
    dict
        Merged configuration.  The returned dict and its mutable values
        are independent copies — mutating them never pollutes the
        module-level ``DEFAULTS``.
    """
    merged = deepcopy(DEFAULTS)

    if user_overrides is not None:
        if isinstance(user_overrides, dict):
            for k, v in user_overrides.items():
                merged[k] = v
        return merged

    # --- Runtime path: read from Hermes config layer ------------------- #
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        return merged

    if not isinstance(config, dict):
        return merged

    aux_section = config.get("auxiliary", {})
    if not isinstance(aux_section, dict):
        return merged

    user_cfg = aux_section.get(AUXILIARY_KEY, {})
    if isinstance(user_cfg, dict):
        for k, v in user_cfg.items():
            merged[k] = v

    return merged


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a merged config dict and return it if valid.

    Raises :class:`ConfigError` on any violation.

    Validation rules
    ----------------
    * Only keys declared in :data:`DEFAULTS` (``timeout_seconds``,
      ``model``, ``permission``) are accepted.  All other keys --
      including ``acp_command``, ``acp_args``, ``setting_sources``,
      ``workdir``, ``workspace``, ``workspaces``, ``cwd`` -- are
      rejected as unknown.
    * ``timeout_seconds``: number in [1, 3600].  This is an **inactivity
      timeout**, not a total task-duration limit.
    * ``model`` and ``permission``: each must be ``None`` or a non-empty
      string (after stripping whitespace).  ``None`` means "use the Kimi
      ACP server's default"; a non-empty string is the operator-configured
      fallback for the per-call tool argument of the same name.

    Security: error messages report only type/rule information, never the
    operator-supplied values, to prevent leaking sensitive config to the
    model.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a dict.")

    # -- unknown keys -- #
    allowed_keys = set(DEFAULTS.keys())
    unknown = set(cfg.keys()) - allowed_keys
    if unknown:
        raise ConfigError(
            "Configuration contains unsupported keys. "
            "Only the documented auxiliary.kimi_code_acp keys are accepted."
        )

    # -- timeout_seconds -- #
    timeout = cfg.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ConfigError("timeout_seconds must be a number")
    if timeout < _TIMEOUT_MIN or timeout > _TIMEOUT_MAX:
        raise ConfigError(
            f"timeout_seconds must be in [{_TIMEOUT_MIN}, {_TIMEOUT_MAX}], "
            f"got {timeout}"
        )

    # -- model / permission -- #
    for _field in ("model", "permission"):
        _value = cfg.get(_field)
        if _value is None:
            continue
        if isinstance(_value, str):
            if not _value.strip():
                raise ConfigError(
                    f"{_field} must be null or a non-empty string"
                )
            cfg[_field] = _value.strip()
            continue
        raise ConfigError(
            f"{_field} must be null or a non-empty string"
        )

    return cfg
