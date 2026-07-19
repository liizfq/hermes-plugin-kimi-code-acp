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

The operator provides timeout via ``kimi_code_acp``.  The ACP
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
    from .kimi_code_acp.delegation import (
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )
    from .kimi_code_acp.runtime import (
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )
    from .kimi_code_acp.tool import (
        KIMI_CODE_ACP_SCHEMA,
        handle_kimi_code_acp,
    )
except ImportError:
    from kimi_code_acp.delegation import (
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )
    from kimi_code_acp.runtime import (
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )
    from kimi_code_acp.tool import (
        KIMI_CODE_ACP_SCHEMA,
        handle_kimi_code_acp,
    )


def register(ctx) -> None:
    """Register the plugin with the Hermes PluginContext.

    Note: this plugin deliberately does **not** call
    ``ctx.register_auxiliary_task()``.  The Hermes auxiliary system is a
    LLM side-task routing abstraction (vision, compression, web_extract,
    approval, ...) and every auxiliary task is invoked through
    ``auxiliary_client.call_llm()`` carrying the
    ``provider/model/base_url/api_key`` routing quadruple.  This plugin
    is a process transport (subprocess + JSON-RPC over stdio), not an
    LLM call — the ACP server inside the subprocess owns the LLM
    provider.  Registering it as an auxiliary task would pollute
    ``config.yaml`` via the ``hermes model`` menu and trigger spurious
    ``AUXILIARY_KIMI_CODE_ACP_*`` env-var bridging.  Operator config
    lives under the top-level ``kimi_code_acp:`` section instead.
    """
    # 1. Register the coding tool — prompt + cwd + model + permission.
    # This is the plugin's primary identity: a model-invoked tool whose
    # handler spawns the ACP subprocess and runs one turn.
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
            "operator-configured via the top-level kimi_code_acp "
            "section in config.yaml."
        ),
        emoji="🚀",
    )

    # 2. Register as a delegation provider.
    ctx.register_delegation_provider(
        DELEGATION_PROVIDER_KEY,
        resolve_delegation_provider,
    )

    # 3. Register as an ACP runtime provider.  Both ``kimi-agent-acp``
    # (the command key) and ``kimi-code-acp`` (the plugin slug alias)
    # map to the same resolver.
    ctx.register_acp_runtime_provider(
        RUNTIME_PROVIDER_KEY,
        resolve_runtime_provider,
    )
    ctx.register_acp_runtime_provider(
        DELEGATION_PROVIDER_KEY,  # alias
        resolve_runtime_provider,
    )
