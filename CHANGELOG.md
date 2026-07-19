# Changelog

All notable changes to **hermes-plugin-kimi-code-acp** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-07-19

### Changed
- **Breaking config layout.** Operator config moved from `auxiliary.kimi_code_acp`
  to the **top-level** `kimi_code_acp` section in `~/.hermes/config.yaml`,
  matching the convention used by other plugin-provided tools (`image_gen`,
  `web`, `tts`). Existing `auxiliary.kimi_code_acp` blocks must be moved
  manually:

  ```yaml
  # Before (0.1.x)
  auxiliary:
    kimi_code_acp:
      timeout_seconds: 600

  # After (0.2.0)
  kimi_code_acp:
    timeout_seconds: 600
  ```

- **Plugin no longer registers as an auxiliary task.** The Hermes auxiliary
  system is an LLM side-task routing abstraction (vision, compression,
  web_extract, …) carrying the `provider/model/base_url/api_key` quadruple.
  This plugin is a process transport (spawns `kimi acp`, JSON-RPC over stdio),
  not an LLM call, so registering it as auxiliary was a category error that
  caused `hermes model` to pollute `config.yaml` with unusable routing fields
  (subsequently rejected by `validate_config`) and triggered spurious
  `AUXILIARY_KIMI_CODE_ACP_*` env-var bridging at gateway startup.

### Fixed
- `validate_config` now also rejects the LLM-routing fields `provider`,
  `base_url`, and `api_key` if they appear in the `kimi_code_acp` block
  (residue from a previous `hermes model` write). The error message is
  corrected to point at `kimi_code_acp` instead of `auxiliary.kimi_code_acp`.
- `runtime.py` and `delegation.py` now read the operator-configured
  `runtime_model` / `model` fallback from the top-level `kimi_code_acp`
  section. Previously they read from `auxiliary.kimi_code_acp` and silently
  missed user overrides.

### Added
- Negative regression test (`test_register_does_not_call_register_auxiliary_task`)
  locking in the architectural decision so future contributors don't
  accidentally re-add the auxiliary registration.
- `CONFIG_SECTION` constant in `kimi_code_acp/config.py` (alias of the legacy
  `AUXILIARY_KEY`, kept for import compatibility).
- README section "Kimi Code CLI authentication" documenting both paths:
  Kimi managed service (OAuth) and custom OpenAI-compatible provider
  (LiteLLM / OpenRouter / etc.).

## [0.1.0] — 2026-07-19

### Added
- Initial release.
- `kimi_code_acp` tool — `prompt + cwd + model + permission` schema.
  Launcher is fixed at `kimi acp` (not operator-configurable); operator
  config covers `timeout_seconds` only.
- `register_delegation_provider("kimi-code-acp", …)` so `delegate_task` can
  route to Kimi Code ACP when `delegation.provider: kimi-code-acp` is set.
- `register_acp_runtime_provider("kimi-agent-acp", …)` plus the
  `kimi-code-acp` alias so Hermes can switch its main chat model to a Kimi
  ACP transport via `model.provider: kimi-agent-acp` or `kimi-code-acp`.
- Operator config schema: `timeout_seconds` (1..3600), `model`, `permission`
  (both nullable). Strict validation — unknown keys are rejected with a
  safe error that does not leak supplied values.
- Single-session `_ACPProcessManager` with live-switch support for
  `model` / `permission` (no rebuild), rebuild-on-`cwd`-mismatch, and
  atexit cleanup.
- Approval bridged to the Hermes core via
  `agent.transports.acp_approval.make_acp_approval_callback` — no
  plugin-owned approval policy.
- Session `_meta` is an empty dict (Kimi's ACP adapter does not interpret
  `_meta`, unlike Claude's `settingSources` structure).
- Full pytest suite (235 tests, 93% coverage).
