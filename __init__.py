"""Kimi Code ACP plugin - standalone Hermes plugin.

Registers:
  * An auxiliary task key ``kimi_code_acp`` (operator config slot).
  * A coding tool ``kimi_code_acp`` whose model-facing schema is exactly
    ``prompt`` + ``cwd`` + ``model`` + ``permission`` (all four required;
    ``model`` and ``permission`` are nullable with default ``None``
    meaning "use the Kimi ACP server's default").

The ACP execution backend is implemented via
:func:`kimi_code_acp.backend.run_task`.  A long-lived singleton
``ACPClientSession`` is created lazily on the first call and reused for
subsequent calls **with the same cwd**; a call with a different cwd
closes the old session under the manager lock and creates a new one.
``model`` and ``permission`` are live-switched via
``session/set_config_option`` and do NOT trigger a rebuild.

The operator provides timeout via ``auxiliary.kimi_code_acp``.  The ACP
launcher (``kimi acp``) is **fixed** at the code level — see
:data:`kimi_code_acp.config.ACP_COMMAND` / :data:`ACP_ARGS`; it is not
operator-configurable.

The plugin does **not** own any approval policy.  All ACP permission
classification and approval routing are owned by the Hermes core
(:func:`agent.transports.acp_approval.make_acp_approval_callback`).
``cwd`` is not a sandbox boundary and not an approval root; it is only
the ACP session's working directory and a session-identity key for
rebuild-on-mismatch.

Install by symlinking or copying this directory into
``~/.hermes/plugins/kimi-code-acp/`` and enabling it via
``plugins.enabled`` in config.yaml.
"""

from __future__ import annotations

from copy import deepcopy

# The root __init__.py is loaded two different ways:
#
#   1. **Hermes directory loader** loads it as ``hermes_plugins.<slug>``
#      (with ``__package__`` set to ``hermes_plugins.<slug>`` and
#      ``__path__`` pointing at the repo root).  In this context the
#      sub-package is reachable via a relative import: ``.kimi_code_acp``.
#
#   2. **pytest** discovers the repo root as a package because of this
#      ``__init__.py`` and imports it without a parent package.  In that
#      context the sub-package is reachable via an absolute import:
#      ``kimi_code_acp`` (the repo root is on ``sys.path``).
#
# We try the relative import first (correct under Hermes), then fall back
# to the absolute import (correct under pytest).
try:
    from .kimi_code_acp.tool import (
        KIMI_CODE_ACP_SCHEMA,
        handle_kimi_code_acp,
    )
    from .kimi_code_acp.config import AUXILIARY_KEY, DEFAULTS
    from .kimi_code_acp.delegation import (
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )
    from .kimi_code_acp.runtime import (
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )
except ImportError:
    from kimi_code_acp.tool import (
        KIMI_CODE_ACP_SCHEMA,
        handle_kimi_code_acp,
    )
    from kimi_code_acp.config import AUXILIARY_KEY, DEFAULTS
    from kimi_code_acp.delegation import (
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )
    from kimi_code_acp.runtime import (
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )


def register(ctx) -> None:
    """Register the plugin with the Hermes PluginContext."""
    # 1. Register auxiliary task — operator config slot.
    ctx.register_auxiliary_task(
        key=AUXILIARY_KEY,
        display_name="Kimi Code ACP",
        description=(
            "Kimi Code CLI (kimi acp) ACP coding agent.  The launcher "
            "(kimi acp) is fixed and not operator-configurable; operator "
            "config covers timeout only.  The model and permission are "
            "per-call parameters on the kimi_code_acp tool "
            "(nullable-required; null = use the Kimi ACP server default)."
        ),
        defaults=deepcopy(DEFAULTS),
    )

    # 2. Register the coding tool — prompt + cwd + model + permission.
    ctx.register_tool(
        name="kimi_code_acp",
        toolset="kimi_code_acp",
        schema=KIMI_CODE_ACP_SCHEMA,
        handler=handle_kimi_code_acp,
        description=(
            "Run a Kimi Code ACP coding task.  The model supplies the "
            "prompt, the cwd (absolute, existing directory), and the "
            "per-call model and permission mode (nullable: null = use "
            "the Kimi ACP server default, a non-empty string requests "
            "a specific id).  All other execution parameters are "
            "operator-configured via auxiliary.kimi_code_acp."
        ),
        emoji="🚀",
    )

    # 3. Register as a delegation provider.
    ctx.register_delegation_provider(
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )

    # 4. Register as an ACP runtime provider.  Both ``kimi-agent-acp``
    # (the command key) and ``kimi-code-acp`` (the plugin slug alias)
    # map to the same resolver.
    ctx.register_acp_runtime_provider(
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )
    ctx.register_acp_runtime_provider(
        DELEGATION_PROVIDER_KEY,    # alias
        resolve_runtime_provider,
    )
