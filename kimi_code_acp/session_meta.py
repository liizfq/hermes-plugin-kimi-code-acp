"""Kimi Code session_meta builder.

This is a **pure helper** that constructs the ``session_meta`` dict forwarded
verbatim as ``params["_meta"]`` in the ACP ``session/new`` request.  The core
``ACPClientSession`` does not inspect or rewrap this value -- it deep-copies it
and sends it on the wire.

Unlike the Claude Code ACP adapter, the Kimi Code ACP adapter does **not**
consult a ``settingSources`` field.  Its ``session/new`` handler reads only
``cwd`` and ``mcpServers`` from the standard ACP params; the ``_meta`` field is
not interpreted by the current adapter (see ``packages/acp-adapter/src/server.ts``
``newSession()`` in the ``MoonshotAI/kimi-code`` monorepo).

This builder therefore returns an **empty dict** by default.  The shape exists
as a seam so future Kimi-specific options can be added without touching the
backend or handler.  The plugin owns all Kimi semantics; the Hermes core is
vendor-agnostic.

Security: no secrets flow through this helper.  It only constructs a static
shape with no operator-controlled string content.
"""

from __future__ import annotations

from typing import Any, Dict


def build_session_meta() -> Dict[str, Any]:
    """Build the Kimi-specific ``session_meta`` for ACP ``session/new``.

    Returns
    -------
    dict
        A fresh, independent dict.  Currently empty because the Kimi ACP
        adapter does not interpret ``_meta``.  The shape is preserved as
        a seam for future Kimi-specific options.
    """
    return {}


def session_meta_is_safe(meta: Dict[str, Any]) -> bool:
    """Lightweight shape check for the session_meta dict.

    Verifies the top-level is a dict.  Used by tests to confirm the helper
    produces the expected shape.
    """
    return isinstance(meta, dict)
