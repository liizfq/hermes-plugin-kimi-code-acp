"""Kimi Code ACP tool — ``prompt`` + ``cwd`` + ``model`` + ``permission`` schema and handler seam.

The tool schema exposes exactly **four** parameters, all required:

* ``prompt`` — the coding task (model-controlled).
* ``cwd``    — the absolute, existing-directory working directory in which
  the ACP agent must execute this task (model-controlled, validated at
  call time).
* ``model``  — the model id for this call (model-controlled, **nullable**).
  ``null`` / Python ``None`` means "fall back to the operator-configured
  ``auxiliary.kimi_code_acp.model`` default, and finally to the Kimi ACP
  server's default model"; a non-empty string requests a specific model
  id accepted by the Kimi ACP server (e.g. ``"kimi-k2"``, ``"auto"``).
  Required in the JSON schema; default value is ``null``.
* ``permission`` — the session mode for this call (model-controlled,
  **nullable**).  ``null`` / Python ``None`` means "fall back to the
  operator-configured ``auxiliary.kimi_code_acp.permission`` default,
  and finally to the Kimi ACP server's default mode"; a non-empty
  string requests a specific mode accepted by the Kimi ACP server's
  ``session/set_config_option`` dispatcher: ``"default"``, ``"plan"``,
  ``"auto"``, etc.  Required in the JSON schema; default value is
  ``null``.

Resolution priority for both ``model`` and ``permission`` is:

    per-call value (non-null)  >  config value  >  ``None`` (server default)

The handler merges + validates operator config *before* resolving the
per-call value, so a ``null`` per-call value picks up the operator
default (if any) and only falls back to ``None`` when the operator also
left the config key unset.

No command, args, effort, timeout fields are exposed to the model —
those are either operator-configured via the
``auxiliary.kimi_code_acp`` config block (timeout and the optional
model / permission fallbacks) or **fixed at the code level** (the ACP
launcher — see :data:`kimi_code_acp.config.ACP_COMMAND` / :data:`ACP_ARGS`).

The handler delegates to :func:`run_task` which is the execution backend
implemented in :mod:`kimi_code_acp.backend`.

Security
--------
``cwd`` is validated strictly: it must be a non-empty absolute path that
resolves (symlinks followed) to an existing directory.  It is forwarded
to the backend where it becomes the ACP session's working directory
(via ``ensure_started(cwd=...)``) and a session-identity key for
rebuild-on-mismatch.  It is **not** a sandbox boundary and **not** an
approval root.  The plugin owns no approval policy; all capability
classification and approval routing come from the Hermes core
(:func:`agent.transports.acp_approval.make_acp_approval_callback`).

``model`` and ``permission`` are validated strictly: only ``None`` or a
non-empty string are accepted.  Other types and blank strings return a
safe ``ValueError`` JSON error that does not leak the supplied value.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Tool schema — prompt + cwd + model + permission (model/permission nullable)
# --------------------------------------------------------------------------- #

KIMI_CODE_ACP_SCHEMA: Dict[str, Any] = {
    "name": "kimi_code_acp",
    "description": (
        "Run a Kimi Code ACP coding task with the given prompt in the "
        "given working directory. The Kimi Code CLI (kimi acp) executes "
        "in an operator-configured runtime with operator-pinned "
        "authentication; the caller chooses the cwd and may override the "
        "per-call model and session mode for this task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The coding task prompt to send to the Kimi Code "
                    "ACP agent."
                ),
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Absolute path to an existing directory that the ACP "
                    "agent should use as the working directory for this "
                    "task. Must exist and be a directory."
                ),
            },
            "model": {
                "type": ["string", "null"],
                "description": (
                    "Model id to use for this call. Pass null (None) to "
                    "fall back to the operator-configured "
                    "auxiliary.kimi_code_acp.model default, and finally "
                    "to the Kimi ACP server's default model; pass a "
                    "non-empty string to request a specific model id "
                    "accepted by the Kimi ACP server (e.g. \"kimi-k2\", "
                    "\"auto\"). The plugin does NOT translate or alias "
                    "the value -- whatever the caller passes is "
                    "forwarded verbatim to the ACP session constructor."
                ),
                "default": None,
            },
            "permission": {
                "type": ["string", "null"],
                "description": (
                    "Session mode to use for this call (maps to the "
                    "Kimi ACP server's session/set_config_option mode "
                    "axis). Pass null (None) to fall back to the "
                    "operator-configured "
                    "auxiliary.kimi_code_acp.permission default, and "
                    "finally to the Kimi ACP server's default mode; "
                    "pass one of the accepted non-empty string values "
                    "to request a specific mode: \"default\", \"plan\", "
                    "\"auto\", etc. The plugin does NOT translate or "
                    "alias the value -- whatever the caller passes is "
                    "forwarded verbatim to the ACP session. "
                    "Live-switched on the existing session via "
                    "session/set_config_option when the value differs "
                    "from the live session's mode."
                ),
                "default": None,
            },
        },
        "required": ["prompt", "cwd", "model", "permission"],
    },
}

#: The required parameters — used by tests to verify schema surface.
REQUIRED_PARAMS = ("prompt", "cwd", "model", "permission")

#: Parameters that must NEVER appear in the schema (security boundary).
FORBIDDEN_PARAMS = frozenset({
    "command", "args", "effort", "workdir", "workspace",
    "workspaces", "timeout", "setting_sources",
})


# --------------------------------------------------------------------------- #
# cwd + model validation
# --------------------------------------------------------------------------- #

def validate_cwd(cwd: Any) -> str:
    """Validate the call-time ``cwd`` argument.

    Returns the resolved absolute path string on success.

    Raises :class:`ValueError` (not :class:`ConfigError`) on any failure.

    Rules:
      * must be a non-empty string;
      * must be an absolute path;
      * must resolve (symlinks followed) to an existing directory.

    The error messages intentionally do not echo the supplied path.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd is required and must be a non-empty string")
    if not os.path.isabs(cwd):
        raise ValueError("cwd must be an absolute path")
    try:
        resolved = Path(cwd).resolve()
    except (OSError, ValueError):
        raise ValueError("cwd could not be resolved")
    if not resolved.exists():
        raise ValueError("cwd does not exist after resolving symlinks")
    if not resolved.is_dir():
        raise ValueError("cwd is not a directory after resolving symlinks")
    return str(resolved)


def validate_model(model: Any) -> Optional[str]:
    """Validate the per-call ``model`` argument.

    Returns the normalized value: ``None`` if the caller wants the Kimi
    ACP server's default, or a non-empty stripped string for a specific
    model id.

    Rules:
      * ``None`` is accepted and returned as-is (server default).
      * non-empty ``str`` is accepted, stripped, and returned.
      * any other type, or a blank/whitespace-only string, is rejected.
      * the supplied value is NEVER echoed in error messages.
    """
    if model is None:
        return None
    if isinstance(model, str):
        stripped = model.strip()
        if not stripped:
            raise ValueError("model must be null or a non-empty string")
        return stripped
    raise ValueError("model must be null or a non-empty string")


def validate_permission(permission: Any) -> Optional[str]:
    """Validate the per-call ``permission`` argument.

    Returns the normalized value: ``None`` if the caller wants the Kimi
    ACP server's default mode, or a non-empty stripped string for a
    specific session mode (e.g. ``"default"``, ``"plan"``, ``"auto"``).

    Rules:
      * ``None`` is accepted and returned as-is (server default).
      * non-empty ``str`` is accepted, stripped, and returned.
      * any other type, or a blank/whitespace-only string, is rejected.
      * the supplied value is NEVER echoed in error messages.

    The plugin does NOT enforce the enumerated list of accepted values
    here — the ACP server itself rejects unknown values via its
    ``session/set_config_option`` dispatcher.
    """
    if permission is None:
        return None
    if isinstance(permission, str):
        stripped = permission.strip()
        if not stripped:
            raise ValueError("permission must be null or a non-empty string")
        return stripped
    raise ValueError("permission must be null or a non-empty string")


# --------------------------------------------------------------------------- #
# Execution seam — delegates to backend
# --------------------------------------------------------------------------- #

def run_task(
    prompt: str,
    config: Dict[str, Any],
    *,
    cwd: str,
    model: Optional[str] = None,
    permission: Optional[str] = None,
) -> str:
    """Execute a Kimi Code ACP task.

    Delegates to :func:`kimi_code_acp.backend.run_task` which constructs
    a real ``ACPClientSession``, runs one turn, and returns a JSON string
    with the result, tool iteration count, and should_retire flag.
    """
    from .backend import run_task as _run_task
    return _run_task(
        prompt, config, cwd=cwd, model=model, permission=permission,
    )


# --------------------------------------------------------------------------- #
# Tool handler — called by the Hermes tools registry
# --------------------------------------------------------------------------- #

def handle_kimi_code_acp(args: Dict[str, Any], **_kw: Any) -> str:
    """Handle a ``kimi_code_acp`` tool call.

    Extracts ``prompt``, ``cwd``, ``model``, and ``permission`` from
    *args*, merges + validates config, validates the call-time cwd,
    resolves ``model`` and ``permission`` against config defaults, and
    delegates to :func:`run_task`.

    The handler always returns a **string** (per Hermes tool contract).
    Validation errors are returned as JSON error strings, not raised.
    """
    from .config import merge_config, validate_config, ConfigError

    if not isinstance(args, dict):
        return json.dumps({"error": "args must be an object"})

    prompt = args.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return json.dumps({"error": "prompt is required and must be a non-empty string"})

    try:
        resolved_cwd = validate_cwd(args.get("cwd"))
    except ValueError:
        return json.dumps({
            "error": "cwd is invalid: must be an absolute path to an existing directory",
            "error_type": "ValueError",
        })

    try:
        cfg = merge_config()
        validate_config(cfg)
    except ConfigError:
        return json.dumps({
            "error": "ACP configuration validation failed",
            "error_type": "ConfigError",
        })

    try:
        per_call_model = validate_model(args.get("model"))
    except ValueError:
        return json.dumps({
            "error": "model is invalid: must be null or a non-empty string",
            "error_type": "ValueError",
        })

    try:
        per_call_permission = validate_permission(args.get("permission"))
    except ValueError:
        return json.dumps({
            "error": "permission is invalid: must be null or a non-empty string",
            "error_type": "ValueError",
        })

    # Resolve model and permission with priority:
    #   per-call value (non-null) > config value > None (server default).
    resolved_model = per_call_model if per_call_model is not None else cfg.get("model")
    resolved_permission = (
        per_call_permission if per_call_permission is not None
        else cfg.get("permission")
    )

    return run_task(
        prompt, cfg,
        cwd=resolved_cwd,
        model=resolved_model,
        permission=resolved_permission,
    )
