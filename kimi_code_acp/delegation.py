"""Delegation provider resolver for the kimi-code-acp plugin.

When ``delegation.provider`` is set to ``kimi-code-acp`` in config.yaml,
the generic delegation provider registry calls the resolver registered
here.  It reads the operator-configured ``kimi_code_acp``
block (timeout, model, permission) and produces a generic descriptor
dict that routes the child agent through the ACP transport.

Model resolution policy
-----------------------
  1. ``delegation.model`` (operator-configured in ``config.yaml``) wins
     if set.
  2. Otherwise, the optional operator-configured
     ``kimi_code_acp.model`` default is consulted.
  3. Otherwise, fall back to a fixed module-level default
     :data:`_DEFAULT_DELEGATION_MODEL`.

Other behaviours
----------------
  * Auth is handled by the spawned ``kimi`` binary (its own login state
    under ``~/.kimi-code/``), so ``api_key`` is empty.
  * The descriptor sets ``api_mode = "acp_client"`` and ``provider =
    "acp_client"`` so the core correctly detects the ACP transport.
  * ``command`` / ``args`` come from the **fixed** launcher constants.
  * No ``workdir`` / ``workspace`` / ``workspaces`` key.  The working
    directory is a per-call ``cwd`` parameter on the ``kimi_code_acp``
    tool, not an operator-config field.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import ACP_ARGS, ACP_COMMAND, merge_config

logger = logging.getLogger(__name__)

#: The provider key registered with the generic delegation registry.
DELEGATION_PROVIDER_KEY = "kimi-code-acp"

#: Fixed fallback model for the delegation (subagent) provider path.
#: Operators who want a different model should set ``delegation.model``
#: (highest priority) or ``kimi_code_acp.model`` (second
#: priority).  This is the last-resort fallback so the descriptor always
#: carries a non-empty model id.
_DEFAULT_DELEGATION_MODEL = "kimi-k2"


def resolve_delegation_provider(
    requested_model: Optional[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Produce a descriptor dict for the kimi-code-acp delegation provider.

    Parameters
    ----------
    requested_model
        The ``delegation.model`` value from config.yaml.  When set, it
        wins over every other source.
    cfg
        The full ``delegation`` config block (unused directly — ACP
        settings live under ``kimi_code_acp``).

    Returns
    -------
    dict
        Generic descriptor with keys: provider, display_provider, model,
        api_mode, base_url, api_key, command, args, metadata.
    """
    acp_cfg = merge_config()

    effective_model = requested_model or acp_cfg.get("model") or _DEFAULT_DELEGATION_MODEL

    return {
        "provider": "acp_client",
        "display_provider": DELEGATION_PROVIDER_KEY,
        "model": effective_model,
        "api_mode": "acp_client",
        "base_url": "",
        "api_key": "",
        # Launcher is fixed (code-level compatibility constant), not config.
        "command": ACP_COMMAND,
        "args": list(ACP_ARGS),
        # No working directory: cwd is a per-call parameter on the tool.
        "metadata": {
            "timeout_seconds": acp_cfg.get("timeout_seconds", 600),
        },
    }
