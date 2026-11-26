"""Append-only, hash-chained event log.

Every event carries prev_hash + hash; hash = sha256(prev_hash || canonical(event without hash)).
This is what makes replay tamper-evident, which is what "verifiable" actually means here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _canonical(obj: dict[str, Any]) -> str:
    """Deterministic JSON. sort_keys + no spaces. Never hash a python object directly."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_event(event_without_hash: dict[str, Any], prev_hash: str) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical(event_without_hash).encode("utf-8"))
    return h.hexdigest()


class Tracer:
    """One per run. Reopening the same path resumes the chain."""

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self._seq = 0
        self._prev_hash = GENESIS

        if self.path.exists():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                ev = json.loads(raw)
                self._seq = ev["seq"] + 1
                self._prev_hash = ev["hash"]

    def emit(self, kind: str, node_id: str | None = None, **payload: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "seq": self._seq,
            "ts": time.time(),
            "run_id": self.run_id,
            "node_id": node_id,
            "kind": kind,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        event["hash"] = _hash_event(event, self._prev_hash)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_canonical(event) + "\n")
        self._seq += 1
        self._prev_hash = event["hash"]
        return event


def verify_chain(path: str | Path) -> tuple[bool, str]:
    """Returns (ok, message). ok=False means someone tampered or truncated."""
    prev = GENESIS
    expected_seq = 0
    p = Path(path)
    if not p.exists():
        return False, "file missing"
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        ev = json.loads(raw)
        if ev.get("seq") != expected_seq:
            return False, f"line {lineno}: seq {ev.get('seq')} != {expected_seq}"
        if ev.get("prev_hash") != prev:
            return False, f"line {lineno}: prev_hash mismatch"
        body = {k: v for k, v in ev.items() if k != "hash"}
        if _hash_event(body, prev) != ev["hash"]:
            return False, f"line {lineno}: hash mismatch"
        prev = ev["hash"]
        expected_seq += 1
    return True, f"ok ({expected_seq} events)"
