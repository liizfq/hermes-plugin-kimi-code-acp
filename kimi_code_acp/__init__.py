"""kimi_code_acp package - configuration, session_meta, backend, and tool logic.

The ACP execution backend is implemented via
:func:`kimi_code_acp.backend.run_task`.  A long-lived singleton
``ACPClientSession`` is created lazily on the first call and reused for
subsequent calls **with the same cwd**; a call with a different cwd
closes the old session under the manager lock and creates a new one.
``model`` and ``permission`` are **live-switched** on the existing
session via ``session/set_config_option`` (the Kimi ACP adapter's
unified dispatcher).  At any instant there is at most one live
subprocess.

The operator provides timeout via ``auxiliary.kimi_code_acp``.  The ACP
launcher (``kimi acp``) is **fixed** (see
:data:`kimi_code_acp.config.ACP_COMMAND` / :data:`ACP_ARGS`) and is not
operator-configurable.  The working directory is supplied **per call**
by the model via the ``cwd`` parameter of the ``kimi_code_acp`` tool and
validated at call time.  ``cwd`` is the ACP session's working directory
and a session-identity key for rebuild-on-mismatch; it is **not** a
sandbox boundary and **not** an approval root — the plugin owns no
approval policy.  All capability classification and approval routing
come from the Hermes core
(:func:`agent.transports.acp_approval.make_acp_approval_callback`).
"""

from .config import (
    AUXILIARY_KEY,
    DEFAULTS,
    merge_config,
    validate_config,
    ConfigError,
)
from .session_meta import build_session_meta, session_meta_is_safe
from .tool import KIMI_CODE_ACP_SCHEMA, handle_kimi_code_acp, run_task

__all__ = [
    "AUXILIARY_KEY",
    "DEFAULTS",
    "merge_config",
    "validate_config",
    "ConfigError",
    "build_session_meta",
    "session_meta_is_safe",
    "KIMI_CODE_ACP_SCHEMA",
    "handle_kimi_code_acp",
    "run_task",
]
