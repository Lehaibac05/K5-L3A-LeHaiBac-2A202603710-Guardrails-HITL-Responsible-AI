"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def _mask(text: str) -> str:
    """Mask PII / secrets before they are written to disk.

    The audit log is itself a data store: keeping raw phone numbers, emails or
    leaked credentials in it would create a second leak. Reuses the CP2
    content_filter instead of re-implementing the patterns.
    """
    if not text:
        return text or ""
    try:
        from guardrails.output_guardrails import content_filter
    except Exception:  # pragma: no cover - keep logging even if import fails
        return text
    return content_filter(text)["redacted"]


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request key -> {"start": perf_counter, "timestamp", "user_id", "input"}
        self._open: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the (masked) input and its start time; return the request key.

        Without ``request_id`` the key is ``user:<user_id>`` (one in-flight
        request per user), so ``record_output`` with the same user pairs up.
        """
        key = self._key(user_id, request_id)
        self._open[key] = {
            "start": time.perf_counter(),
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "input": _mask(text),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the request: output, block decision, layer, latency."""
        key = self._key(user_id, request_id)
        opened = self._open.pop(key, None)
        latency_ms = (
            round((time.perf_counter() - opened["start"]) * 1000, 1)
            if opened else None
        )
        entry = {
            "request_id": key,
            "timestamp": opened["timestamp"] if opened else utc_now_iso(),
            "user_id": user_id,
            "input": opened["input"] if opened else None,
            "output": _mask(text),
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
