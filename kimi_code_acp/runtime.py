"""ACP runtime provider resolver for the kimi-code-acp plugin.

When the user runs ``/acp-client-runtime on kimi-agent-acp``, the generic
ACP runtime provider registry calls the resolver registered here to
produce a descriptor dict that the switch writes into config.

This path is the **main agent's ACP runtime** provider path -- it is NOT
the per-call ``kimi_code_acp`` tool schema.  The per-call tool schema has
its own ``model`` parameter (nullable-required, ``None`` falls back to
the operator-configured ``auxiliary.kimi_code_acp.model`` default and
finally to the Kimi ACP server default).

The launcher (``command`` / ``args``) is **not** operator-configurable.
``acp_command`` / ``acp_args`` are rejected by config validation as
unknown keys.

Model resolution priority (runtime path — main agent's ACP session):

  1. ``requested_model`` (from command/config — highest priority)
  2. operator-configured runtime-specific override
     (``auxiliary.kimi_code_acp.runtime_model``)
  3. operator-configured general-purpose default
     (``auxiliary.kimi_code_acp.model``)
  4. runtime default (:data:`_DEFAULT_RUNTIME_MODEL`)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import ACP_ARGS, ACP_COMMAND, AUXILIARY_KEY, DEFAULTS

logger = logging.getLogger(__name__)

#: The runtime provider key registered with the generic runtime registry.
#: This is the key users type in ``/acp-client-runtime on <key>``.
RUNTIME_PROVIDER_KEY = "kimi-agent-acp"

#: Default model for the runtime provider.
#: Kimi K2 is a sensible default for the main ACP runtime.
_DEFAULT_RUNTIME_MODEL = "kimi-k2"


def resolve_runtime_provider(
    requested_model: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a descriptor dict for the ACP runtime provider.

    Parameters
    ----------
    requested_model
        Model override from the command/config, or ``None``.
    cfg
        The full config dict from config.yaml (unused directly — ACP
        settings are read via ``merge_config()`` from the auxiliary block).

    Returns
    -------
    dict
        Generic descriptor with keys: provider, api_mode, display_provider,
        model, command, args, base_url, api_key, metadata.
    """
    from .config import merge_config
    acp_cfg = merge_config()

    effective_model = requested_model or _DEFAULT_RUNTIME_MODEL

    if not requested_model:
        try:
            from hermes_cli.config import load_config as _lc
            _raw = _lc()
            _aux = (_raw.get("auxiliary") or {}).get(AUXILIARY_KEY) or {}
            if isinstance(_aux, dict) and _aux.get("runtime_model"):
                effective_model = _aux["runtime_model"]
            elif isinstance(_aux, dict) and _aux.get("model"):
                effective_model = _aux["model"]
        except Exception:
            pass

    return {
        "provider": "acp_client",
        "api_mode": "acp_client",
        "display_provider": "kimi-code-acp",
        "model": effective_model,
        # Launcher is fixed (code-level compatibility constant), not config.
        "command": ACP_COMMAND,
        "args": list(ACP_ARGS),
        "base_url": "",
        "api_key": "",
        "metadata": {
            "timeout_seconds": acp_cfg.get("timeout_seconds", 600),
        },
    }
