"""ACP execution backend for the kimi-code-acp plugin.

This module implements ``run_task(prompt, config, *, cwd, model, permission)``
-- the execution seam between the tool handler and the generic Hermes core
``ACPClientSession``.

Design principles
-----------------
* **Per-call ``cwd``, ``model``, ``permission``.**  Every ``run_task``
  invocation receives ``cwd`` (absolute, existing directory), ``model``
  (None or non-empty string), and ``permission`` (None or non-empty
  string).  ``cwd`` is the ACP session's working directory for the turn
  and a session-identity key for rebuild-on-mismatch (cwd is a
  ``session/new`` parameter and cannot be live-switched).  It is **not**
  a sandbox boundary and is **not** an approval root — the plugin does
  not own any approval policy; all ACP permission requests are routed
  by the Hermes core generic bridge
  (:func:`agent.transports.acp_approval.make_acp_approval_callback`).
  ``model`` and ``permission`` are forwarded verbatim to the
  ``ACPClientSession`` constructor: ``None`` means "use the Kimi ACP
  server's default"; a non-empty string requests a specific id.

* **Long-lived singleton session via :class:`_ACPProcessManager`.**  A
  single ``ACPClientSession`` is created lazily on the first call and
  reused for subsequent calls **as long as the cwd matches** the value
  the live session was created with.  When a call arrives with a
  different cwd, the manager closes the old session under the lock and
  creates a new one bound to the new cwd.  ``model`` and ``permission``
  mismatches are **live-switched** via ``session/set_config_option``
  (the Kimi ACP adapter's unified dispatcher) — they do NOT trigger a
  rebuild.  At any instant there is at most one live subprocess.

* **Single turn lock.**  One ``threading.Lock`` serialises the entire
  turn.  ``close_session()`` takes the same lock, so it cannot run
  between two manager operations inside a turn.

* **Creation-time parameter freeze.**  ``command``, ``args``, ``model``,
  ``cwd``, ``session_meta``, ``approval_callback`` and
  ``auto_approve_permissions`` are supplied at session creation and
  never re-sent while the session lives.  Changing ``cwd`` forces a
  rebuild.  ``turn_timeout`` is passed per call.

* **Lazy import.**  The core ``ACPClientSession`` is imported inside
  ``run_task`` so that plugin discovery does not crash when the core
  module is absent.

* **Safe error boundary.**  Errors returned to the model contain only
  generic error type names and messages -- never command, args, stderr,
  or raw exception text that may embed secrets.

* **Fail-closed permissions, always.**  ``auto_approve_permissions`` is
  **always** ``False``.  The approval callback is obtained directly from
  the Hermes core generic factory.  The plugin owns no approval module
  and no approval policy.

* **JSON string result.**  Success or error, the return is always a JSON
  string.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import ACP_ARGS, ACP_COMMAND, validate_config, ConfigError
from .session_meta import build_session_meta

logger = logging.getLogger(__name__)


# Canonical error prefix emitted by the core's ``TurnResult.error`` when an
# inactivity timeout occurs.  We match this as a **stable prefix** to avoid
# misclassifying unrelated or crafted errors.
_INACTIVITY_ERROR_PREFIX = "ACP session/prompt failed: ACP session inactive"


# --------------------------------------------------------------------------- #
# ACP process manager (singleton session owner)
# --------------------------------------------------------------------------- #


class _ACPProcessManager:
    """Owns the module-level singleton ``ACPClientSession``.

    All lifecycle operations (unsafe cleanup, cwd-mismatch replacement,
    health check, (re)creation, live config switch, ``ensure_started``,
    ``run_turn``, retirement) happen inside a single :meth:`run` call
    that holds ``self._lock`` for its entire duration.

    The core ``ACPClientSession`` is NOT thread-safe; the lock is the only
    thing preventing two callers from entering ``session.run_turn``
    concurrently.

    The manager tracks the cwd the live session was created with.  When a
    call arrives with a different cwd, the existing session is closed
    inside the lock *before* a new one is created, preserving the
    at-most-one-subprocess invariant.  ``model`` and ``permission`` are
    tracked separately and **live-switched** on the existing session via
    ``set_model`` / ``set_permission_mode`` (which the Kimi ACP adapter
    maps to ``session/set_model`` and ``session/set_config_option``).
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._session_cwd: Optional[str] = None
        self._session_model: Optional[str] = None
        self._session_permission: Optional[str] = None
        self._lock = threading.Lock()
        self._atexit_registered = False
        # Set when a turn is aborted by KeyboardInterrupt/SystemExit.
        self._unsafe = False

    @staticmethod
    def is_alive(session: Any) -> bool:
        """Return ``True`` iff the session's underlying subprocess is alive."""
        try:
            client = getattr(session, "_client", None)
            if client is None:
                return False
            return bool(client.is_alive())
        except Exception:
            return False

    def _close_quietly(self, session: Any) -> None:
        """Best-effort ``close()`` -- swallows exceptions, logs only type."""
        try:
            session.close()
        except Exception:
            logger.debug(
                "ACP session close raised (error_type=%s)",
                _safe_exc_type_name(),
                exc_info=False,
            )

    def run(
        self,
        *,
        session_cls: Any,
        command: str,
        args: Any,
        model: Optional[str],
        cwd: str,
        permission: Optional[str],
        session_meta: Dict[str, Any],
        approval_callback: Optional[Callable[..., str]],
        auto_approve: bool,
        prompt: str,
        turn_timeout: float,
    ) -> Any:
        """Run one complete turn under ``self._lock``.

        Sequence (all inside the lock):
            1. Register atexit exactly once.
            2. If the prior session was marked unsafe, close + clear it.
            3. **cwd-mismatch replacement**: cwd is a ``session/new``
               parameter and cannot be live-switched, so a different cwd
               forces a full rebuild.  Model and permission are NOT
               rebuild triggers — they are live-switched in step 6.
            4. Health-check the live session; if dead, close + clear.
            5. If no session, create + ``ensure_started``.  On any
               failure close the partial and return ``None``.
            6. **Live config switch**: before ``run_turn``, if the live
               session's model or permission does not match this call's
               request, send the appropriate ``session/set_*`` to update
               the existing session in place.  A live-switch failure
               denies the turn (return ``None``).
            7. Call ``session.run_turn``.
            8. ``should_retire=True`` or any ``Exception`` from
               ``run_turn``: close + clear.
            9. ``KeyboardInterrupt``/``SystemExit``: mark unsafe, re-raise.

        Returns the TurnResult on success, or ``None`` if session
        creation or live config switch failed.
        """
        with self._lock:
            if not self._atexit_registered:
                atexit.register(self.atexit_cleanup)
                self._atexit_registered = True

            # (1) Unsafe cleanup -- prior turn was interrupted.
            if self._unsafe:
                old = self._session
                self._session = None
                self._session_cwd = None
                self._session_model = None
                self._session_permission = None
                self._unsafe = False
                if old is not None:
                    self._close_quietly(old)

            # (2) cwd-mismatch replacement.
            if (
                self._session is not None
                and self._session_cwd is not None
                and self._session_cwd != cwd
            ):
                old = self._session
                self._session = None
                self._session_cwd = None
                self._session_model = None
                self._session_permission = None
                self._close_quietly(old)

            # (3) Health check.
            if self._session is not None and not self.is_alive(self._session):
                self._close_quietly(self._session)
                self._session = None
                self._session_cwd = None
                self._session_model = None
                self._session_permission = None

            # (4) Create + ensure_started if needed.
            if self._session is None:
                session = None
                try:
                    session = session_cls(
                        command=command,
                        args=args,
                        model=model,
                        permission_mode=permission,
                        session_meta=session_meta,
                        approval_callback=approval_callback,
                        auto_approve_permissions=auto_approve,
                    )
                    session.ensure_started(cwd=cwd)
                except Exception:
                    logger.error(
                        "ACP session creation failed (error_type=%s)",
                        _safe_exc_type_name(),
                        exc_info=False,
                    )
                    if session is not None:
                        self._close_quietly(session)
                    return None
                self._session = session
                self._session_cwd = cwd
                self._session_model = model
                self._session_permission = permission

            # (5) Live config switch: model and permission.
            if self._session_model != model:
                try:
                    if model is None:
                        # Cannot reliably switch back to server default.
                        logger.warning(
                            "ACP live switch to model=None is not supported "
                            "(session was created with model=%r) -- denying turn",
                            self._session_model,
                        )
                        return None
                    self._session.set_model(model)
                except Exception:
                    logger.error(
                        "ACP live model switch failed (error_type=%s)",
                        _safe_exc_type_name(),
                        exc_info=False,
                    )
                    return None
                self._session_model = model

            if self._session_permission != permission:
                try:
                    if permission is None:
                        logger.warning(
                            "ACP live switch to permission=None is not "
                            "supported (session was created with "
                            "permission=%r) -- denying turn",
                            self._session_permission,
                        )
                        return None
                    self._session.set_permission_mode(permission)
                except Exception:
                    logger.error(
                        "ACP live permission switch failed (error_type=%s)",
                        _safe_exc_type_name(),
                        exc_info=False,
                    )
                    return None
                self._session_permission = permission

            # (6) Run the turn.
            try:
                turn_result = self._session.run_turn(
                    user_input=prompt,
                    cwd=cwd,
                    turn_timeout=turn_timeout,
                )
            except (KeyboardInterrupt, SystemExit):
                self._unsafe = True
                raise
            except Exception:
                self._close_quietly(self._session)
                self._session = None
                self._session_cwd = None
                self._session_model = None
                self._session_permission = None
                raise

            # (7) Retirement.
            if getattr(turn_result, "should_retire", False):
                self._close_quietly(self._session)
                self._session = None
                self._session_cwd = None
                self._session_model = None
                self._session_permission = None

            return turn_result

    def close(self) -> None:
        """Close + clear the singleton, blocking on any in-flight turn.

        The detach *and* the best-effort ``close()`` both run **inside**
        the lock to preserve the single-long-lived-process invariant.
        Idempotent.
        """
        with self._lock:
            session = self._session
            self._session = None
            self._session_cwd = None
            self._session_model = None
            self._session_permission = None
            self._unsafe = False
            if session is not None:
                self._close_quietly(session)

    def atexit_cleanup(self) -> None:
        """Best-effort close on interpreter shutdown (swallow everything)."""
        try:
            self.close()
        except BaseException:
            pass


_manager = _ACPProcessManager()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def run_task(
    prompt: str,
    config: Dict[str, Any],
    *,
    cwd: str,
    model: Optional[str] = None,
    permission: Optional[str] = None,
) -> str:
    """Execute a Kimi Code ACP coding task and return a JSON result string.

    Uses a long-lived singleton ``ACPClientSession`` managed by
    :class:`_ACPProcessManager`.  The session is created lazily on the
    first call and reused for subsequent calls **with the same cwd**;
    ``model`` and ``permission`` are **live-switched** via
    ``session/set_config_option`` (the Kimi adapter's unified dispatcher).

    Returns
    -------
    str
        JSON string.  On success::

            {"result": "<agent output>",
             "tool_iterations": <int>,
             "should_retire": <bool>}

        On error::

            {"error": "ACP task failed", "error_type": "RuntimeError"}
    """
    # ---- 1. Re-validate config (defence in depth) -------------------- #
    try:
        validate_config(config)
    except ConfigError:
        return _error_json("ACP configuration validation failed", "ConfigError")

    # ---- 2. Re-validate + re-resolve cwd (TOCTOU mitigation) --------- #
    resolved_cwd = _resolve_cwd(cwd)
    if resolved_cwd is None:
        return _error_json(
            "cwd is not an accessible directory", "ValueError",
        )

    # ---- 3. Lazy-import the core session class ----------------------- #
    session_cls = _import_session_class()
    if session_cls is None:
        return _error_json(
            "ACP core (ACPClientSession) is not available or incompatible",
            "ImportError",
        )

    # ---- 4. Build session parameters --------------------------------- #
    session_meta = build_session_meta()
    approval_callback = _get_approval_callback()
    auto_approve = False
    turn_timeout = config["timeout_seconds"]

    # ---- 5. Run one complete turn under the manager's single lock ----- #
    try:
        turn_result = _manager.run(
            session_cls=session_cls,
            command=ACP_COMMAND,
            args=list(ACP_ARGS),
            model=model,
            cwd=resolved_cwd,
            permission=permission,
            session_meta=session_meta,
            approval_callback=approval_callback,
            auto_approve=auto_approve,
            prompt=prompt,
            turn_timeout=turn_timeout,
        )
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception:
        logger.error(
            "ACP task failed (error_type=%s)",
            _safe_exc_type_name(),
            exc_info=False,
        )
        return _error_json("ACP task failed", _safe_exc_type_name())

    if turn_result is None:
        return _error_json("ACP session creation failed", "RuntimeError")

    return _success_json(turn_result)


def close_session() -> None:
    """Close and clear the singleton session (public, idempotent)."""
    _manager.close()


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _resolve_cwd(cwd: Any) -> Optional[str]:
    """Re-resolve and check cwd existence (TOCTOU mitigation)."""
    if not isinstance(cwd, str) or not cwd.strip():
        return None
    if not os.path.isabs(cwd):
        return None
    try:
        resolved = Path(cwd).resolve()
    except (OSError, ValueError):
        return None
    if not resolved.is_dir():
        return None
    return str(resolved)


def _import_session_class():
    """Lazy-import the core ACPClientSession class.

    Returns the class object, or ``None`` if the import fails or the class
    does not support the required constructor parameters.
    """
    try:
        from agent.transports.acp_client_session import ACPClientSession
    except ImportError:
        return None

    import inspect
    try:
        sig = inspect.signature(ACPClientSession.__init__)
        params = set(sig.parameters.keys())
    except (ValueError, TypeError):
        return ACPClientSession

    required_params = {
        "command",
        "args",
        "model",
        "permission_mode",
        "session_meta",
        "approval_callback",
        "auto_approve_permissions",
    }
    if not required_params <= params:
        return None

    return ACPClientSession


def _get_approval_callback() -> Optional[Callable[..., str]]:
    """Obtain the core generic ACP approval bridge callback for the session.

    Returns the callback produced by the Hermes core factory
    :func:`agent.transports.acp_approval.make_acp_approval_callback`,
    which routes every ACP permission request to the best available
    approval channel (CLI callback, gateway notify, or fail-closed).

    The plugin owns **no** approval policy: no kind classification, no
    command-guard mapping, no path extraction, no cwd approval-root
    check, no sensitive-path list.

    Returns the core bridge callback, or ``None`` when the core module
    is unavailable — in that case the session runs with its built-in
    fail-closed default policy.
    """
    try:
        from agent.transports.acp_approval import make_acp_approval_callback
    except Exception:
        logger.debug(
            "agent.transports.acp_approval.make_acp_approval_callback "
            "import failed; returning None (session will use its "
            "built-in fail-closed default)",
            exc_info=True,
        )
        return None

    try:
        return make_acp_approval_callback()
    except Exception:
        logger.debug(
            "make_acp_approval_callback raised; returning None "
            "(session will use its built-in fail-closed default)",
            exc_info=True,
        )
        return None


def _success_json(turn_result: Any) -> str:
    """Build a JSON success string from a TurnResult-like object."""
    result_text = getattr(turn_result, "final_text", "") or ""
    tool_iterations = getattr(turn_result, "tool_iterations", 0) or 0
    should_retire = getattr(turn_result, "should_retire", False)

    turn_error = getattr(turn_result, "error", None)
    if turn_error:
        if isinstance(turn_error, str) and turn_error.startswith(_INACTIVITY_ERROR_PREFIX):
            return _error_json("ACP session inactive", "InactivityTimeoutError")
        return _error_json("ACP task failed", "RuntimeError")

    return json.dumps({
        "result": result_text,
        "tool_iterations": tool_iterations,
        "should_retire": should_retire,
    })


def _error_json(message: str, error_type: str) -> str:
    """Build a JSON error string with a safe generic message and type name."""
    return json.dumps({
        "error": message,
        "error_type": error_type,
    })


def _safe_exc_type_name() -> str:
    """Return the current exception's type name, or 'RuntimeError' as fallback."""
    import sys
    exc = sys.exc_info()[1]
    if exc is not None:
        return type(exc).__name__
    return "RuntimeError"
