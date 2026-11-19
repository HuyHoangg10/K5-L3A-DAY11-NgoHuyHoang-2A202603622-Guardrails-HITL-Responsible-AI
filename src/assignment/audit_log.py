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


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    @staticmethod
    def _scrub(text: str) -> str:
        """Không để secret/PII nằm nguyên văn trong file log (log cũng có thể bị lộ)."""
        from guardrails.output_guardrails import content_filter

        text = (text or "")[:2000]
        return content_filter(text)["redacted"]

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + mốc thời gian bắt đầu, khóa theo request_id (hoặc user_id)."""
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": self._scrub(text),
            "started_at": utc_now_iso(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu câu trả lời, lớp đã chặn, latency; thêm vào self.logs."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": request_id,
            "user_id": user_id,
            "input": None,
            "started_at": None,
        }
        entry.update(
            {
                "output": self._scrub(text),
                "blocked": bool(blocked),
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)
                if started is not None
                else None,
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Ghi log ra đĩa (mảng JSON) dưới <repo>/outputs/ theo mặc định."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
