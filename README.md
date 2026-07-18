# kimi-code-acp

Kimi Code ACP coding tool plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Runs [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) (`kimi acp`) as a
delegated coding agent over the [Agent Client Protocol](https://agentclientprotocol.com/).

## What This Plugin Does

- Registers an **auxiliary task** `kimi_code_acp` as an operator config slot.
- Registers a **coding tool** `kimi_code_acp` whose model-facing schema is
  exactly `prompt` + `cwd` + `model` + `permission` (all four required;
  `model` and `permission` are nullable with default `null`).
- Registers a **delegation provider** `kimi-code-acp` so `delegate_task`
  can route subagents through the Kimi ACP transport.
- Registers an **ACP runtime provider** `kimi-agent-acp` so
  `/acp-client-runtime on kimi-agent-acp` switches the main agent to Kimi.
- Provides **strict config validation** for all operator-supplied fields.
- Bridges to the Hermes core generic ACP approval factory — the plugin
  owns **no** approval policy.

## Launcher

The ACP launcher is **fixed** at the code level:

| Constant | Value |
|---|---|
| `kimi_code_acp.config.ACP_COMMAND` | `"kimi"` |
| `kimi_code_acp.config.ACP_ARGS` | `("acp",)` |

The `kimi` binary is the `bin` entry of the `@moonshot-ai/kimi-code` npm
package. The `acp` subcommand switches the CLI into ACP mode (JSON-RPC
over stdin/stdout) — see `docs/{zh,en}/reference/kimi-acp.md` in the
`MoonshotAI/kimi-code` repo for the capability matrix.

A different launcher is **not** operator-configurable: it is a source-level
compatibility constant. Any operator-supplied `acp_command` / `acp_args`
key is rejected by strict unknown-key validation.

## Authentication

Unlike Claude Code ACP (which reads `~/.claude/settings.json`), Kimi Code
CLI carries its own authentication state under `~/.kimi-code/`. Complete
the login before first use:

```sh
kimi
# follow the login prompts, then exit
```

The ACP adapter's `session/new` handler checks the auth token and returns
`authRequired (-32000)` if missing. The `kimi acp` process does not drive
the login flow — log in via the interactive CLI first.

## Security Boundary

| Field | Model-controlled? | Operator-controlled? | Fixed (code-level)? |
|---|---|---|---|
| `prompt` | ✅ | - | - |
| `cwd` | ✅ (must be absolute, existing directory) | - | - |
| `model` | ✅ (nullable: `null` = server default) | optional fallback (`auxiliary.kimi_code_acp.model`) | - |
| `permission` | ✅ (nullable: `null` = server default mode) | optional fallback (`auxiliary.kimi_code_acp.permission`) | - |
| `acp_command` | ❌ | ❌ | ✅ `kimi` |
| `acp_args` | ❌ | ❌ | ✅ `["acp"]` |
| `timeout_seconds` | ❌ | ✅ | - |

### Permission Policy

The plugin **does not own any approval policy**. All ACP permission
classification and approval routing are owned by the Hermes core
(`agent.transports.acp_approval.make_acp_approval_callback()`). The
plugin backend bridges to that factory with no arguments and passes the
resulting callback to the `ACPClientSession` constructor.

- **Default: fail-closed.** If no approval callback is available (e.g.
  gateway mode) and approval bypass is not active, permission requests
  from the ACP agent are denied.
- **`auto_approve_permissions` is always `False`** for plugin ACP
  sessions. Yolo / `approvals.mode: off` bypass is applied *inside* the
  core approval bridge, so the callback must run unconditionally.

### `cwd` Is Not a Sandbox

`cwd` sets the ACP session directory and serves as a session-identity
key for rebuild-on-mismatch. It does **not** constrain the agent's
filesystem access. A real probe with `cwd` set to the plugin directory
can create files under `/tmp` via an absolute path. Hard containment
requires an underlying launcher sandbox; do not infer it from `cwd`.

## Configuration

Configure in `~/.hermes/config.yaml` under `auxiliary.kimi_code_acp`:

```yaml
plugins:
  enabled:
    - kimi-code-acp

auxiliary:
  kimi_code_acp:
    timeout_seconds: 600
    # Optional per-call fallbacks (null = Kimi ACP server default)
    model: null
    permission: null
```

### Validation Rules

- Unknown keys are rejected. In particular, `acp_command`, `acp_args`,
  `setting_sources`, `workdir`, `workspace`, `workspaces`, and `cwd` are
  all **not** accepted in the auxiliary block.
- `timeout_seconds` must be between 1 and 3600 (this is an **inactivity
  timeout**, not a total task-duration limit).
- `model` and `permission` must each be `null` or a non-empty string.

## Per-call `cwd`

The working directory is supplied per call by the model via the `cwd`
parameter. The handler validates it strictly:

- must be a non-empty string;
- must be an absolute path;
- must resolve (symlinks followed) to an existing directory.

A call with a different `cwd` than the live session's closes the old
session under the manager lock and creates a new one bound to the new
`cwd` (`cwd` is a `session/new` parameter and cannot be live-switched).

## Per-call `model` and `permission`

Both are supplied per call by the model. Resolution priority:

    per-call value (non-null) > config value > None (server default)

- `null` / `None` = "use the Kimi ACP server's default";
- a non-empty string is forwarded **verbatim** to the ACP session
  constructor (the plugin does NOT translate, alias, or coerce the value);
- any other type, or a blank/whitespace-only string, is rejected.

On the long-lived session, `model` and `permission` are **live-switched**
via `session/set_model` and `session/set_config_option` (the Kimi ACP
adapter's unified dispatcher). They do NOT trigger a session rebuild —
only `cwd` does.

## Example Tool Call

```json
// Model sends (model=null: use Kimi ACP server default):
{"prompt": "Write a Python function that reverses a string", "cwd": "/abs/path/to/repo", "model": null, "permission": null}

// Model sends (model="kimi-k2": request a specific model id):
{"prompt": "Refactor this module", "cwd": "/abs/path/to/repo", "model": "kimi-k2", "permission": "plan"}

// Plugin returns (success):
{"result": "def reverse(s): return s[::-1]", "tool_iterations": 2, "should_retire": false}

// Plugin returns (error):
{"error": "ACP task failed", "error_type": "RuntimeError"}
```

## Differences from `claude-code-acp`

| Concern | claude-code-acp | kimi-code-acp |
|---|---|---|
| Launcher | `npx -y @agentclientprotocol/claude-agent-acp` | `kimi acp` |
| `session/new` `_meta` | `{"claudeCode":{"options":{"settingSources":[...]}}}` | `{}` (Kimi adapter does not interpret `_meta`) |
| Operator `setting_sources` | yes (`user`/`project`/`local`) | **no** (auth lives under `~/.kimi-code/`, not `~/.claude/settings.json`) |
| Model default | `sonnet` / `opus[1m]` | `kimi-k2` |
| Permission axis | Claude `permission_mode` (`default`/`acceptEdits`/...) | Kimi ACP mode (`default`/`plan`/`auto`/...) |
| Approval policy | none (bridges to Hermes core) | **same** — none (bridges to Hermes core) |

The approval architecture is deliberately identical to `claude-code-acp`:
the plugin owns **no** approval module. Every ACP `session/request_permission`
is routed by `agent.transports.acp_approval.make_acp_approval_callback()`.

## Dependencies

Requires:
- The Hermes Agent core (`agent.transports.acp_client_session.ACPClientSession`)
  with `session_meta`, `approval_callback`, and `auto_approve_permissions`
  constructor support.
- The `kimi` binary on `PATH` (install via
  `npm install -g @moonshot-ai/kimi-code`), with login completed.

If the core is missing or too old, the plugin returns a clear
compatibility error without crashing.

## Installation

### Local development (symlink)

```bash
ln -s /absolute/path/to/hermes-plugin-kimi-code-acp ~/.hermes/plugins/kimi-code-acp

# Enable in ~/.hermes/config.yaml:
# plugins:
#   enabled:
#     - kimi-code-acp
```

## License

MIT
