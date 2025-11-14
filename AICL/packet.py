# core/aicl/packet.py
# Author: Vansh Bukkarwal ( A localhousellm project )
# Date: 2025-11-05
#
# AICLPacket — canonical inter-module communication object
#
# Purpose:
#   - Single canonical object exchanged between modules using AICL (Artificial
#     Intelligence Communication Language).
#   - Carries compact symbolic payloads, metadata, trace entries, and safety
#     annotations. Designed for auditability, reproducibility, and easy
#     serialization (JSON / msgpack).
#
# Design goals:
#   1. Stable, explicit fields with clear types and validation.
#   2. Append-only trace semantics (every stamp is preserved).
#   3. Lightweight helpers to add symbols, responses, metadata and safety info.
#   4. Robust (defensive) conversion to/from dict/json/msgpack for transport.
#   5. Hooks for provenance, merging, and shallow copy without mutation issues.
#
# Notes:
#   - Timestamps are timezone-aware (UTC) ISO strings.
#   - msgpack support is optional; if not installed, binary methods raise helpful errors.
#   - This file focuses only on packet behavior (no routing or safety logic).
#

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Union, Iterable
from datetime import datetime, timezone, timedelta
from copy import deepcopy

# Optional binary serialization dependency
try:
    import msgpack  # type: ignore
    _HAS_MSGPACK = True
except Exception:
    _HAS_MSGPACK = False

logger = logging.getLogger("aicl.packet")
logger.addHandler(logging.NullHandler())

DEFAULT_VERSION = "AICL/0.1"


class AICLPacketError(Exception):
    """Base exception for AICLPacket-related errors."""


class AICLPacket:
    """
    Canonical packet object for AICL communication.

    Typical usage:
        p = AICLPacket(origin="user", symbols=["MRL:L_hello", "MRL:L_how"])
        p.stamp("translator", "encode")
        p.add_response_symbol("MRL:L_received")
        b = p.to_msgpack()
        p2 = AICLPacket.from_msgpack(b)
    """

    __slots__ = (
        "version",
        "session_id",
        "origin",
        "targets",
        "symbols",
        "intent",
        "confidence",
        "metadata",
        "trace",
        "safety",
        "response_symbols",
    )

    def __init__(
        self,
        session_id: Optional[str] = None,
        origin: str = "user",
        targets: Optional[Iterable[str]] = None,
        symbols: Optional[Iterable[str]] = None,
        intent: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = DEFAULT_VERSION,
    ):
        # Core identity
        self.version: str = version
        self.session_id: str = session_id or self._generate_session_id()
        self.origin: str = origin

        # Routing / audience
        self.targets: List[str] = list(targets) if targets else []

        # Symbolic payload (primary)
        self.symbols: List[str] = list(symbols) if symbols else []

        # Intent + confidence
        self.intent: str = intent or "unknown"
        self.confidence: float = float(confidence)

        # Additional data: arbitrary, must be JSON-serializable for persistence
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

        # Trace: append-only list of stamps (actor/action/timestamp/note)
        self.trace: List[Dict[str, Any]] = []

        # Safety summary structure (status + flags)
        # status: unknown | ok | flagged | rejected
        # flags: list of short strings explaining reasons
        self.safety: Dict[str, Any] = {"status": "unknown", "flags": []}

        # Response symbols produced by handlers (secondary payload)
        self.response_symbols: List[str] = []

    # --------------------
    # Helpers / Constructors
    # --------------------

    @staticmethod
    def _now_iso() -> str:
        """Return timezone-aware UTC ISO timestamp string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_session_id() -> str:
        ts = int(time.time() * 1000)
        uid = uuid.uuid4().hex[:8]
        return f"session-{ts}-{uid}"

    # --------------------
    # Stamp / trace API
    # --------------------

    def stamp(self, actor: str, action: str, note: Optional[str] = None) -> Dict[str, Any]:
        """
        Append a trace entry and return it.

        Trace entry contains:
            - actor: who stamped (e.g., "translator", "orchestrator")
            - action: short verb (e.g., "encode", "route", "safety_alpha")
            - timestamp: UTC ISO string
            - note: optional human note
        """
        entry = {
            "actor": actor,
            "action": action,
            "timestamp": self._now_iso(),
        }
        if note is not None:
            entry["note"] = str(note)
        self.trace.append(entry)
        logger.debug("stamp: %s %s %s", actor, action, note)
        return entry

    def append_trace_entry(self, entry: Dict[str, Any]) -> None:
        """Append a pre-constructed trace entry (defensive copy)."""
        if not isinstance(entry, dict):
            raise AICLPacketError("trace entry must be a dict")
        e = dict(entry)
        # ensure timestamp exists
        if "timestamp" not in e:
            e["timestamp"] = self._now_iso()
        self.trace.append(e)

    # --------------------
    # Payload mgmt
    # --------------------

    def add_symbol(self, symbol: str) -> "AICLPacket":
        """Append a symbol to `symbols` list (primary payload)."""
        if not isinstance(symbol, str):
            raise AICLPacketError("symbol must be a string")
        self.symbols.append(symbol)
        return self

    def extend_symbols(self, symbols: Iterable[str]) -> "AICLPacket":
        """Append multiple symbols (order preserved)."""
        for s in symbols:
            self.add_symbol(s)
        return self

    def add_response_symbol(self, symbol: str) -> "AICLPacket":
        """Append a symbol to `response_symbols` (module response payload)."""
        if not isinstance(symbol, str):
            raise AICLPacketError("response symbol must be a string")
        self.response_symbols.append(symbol)
        return self

    def extend_response_symbols(self, symbols: Iterable[str]) -> "AICLPacket":
        for s in symbols:
            self.add_response_symbol(s)
        return self

    # --------------------
    # Metadata / safety
    # --------------------

    def set_safety(self, status: str, flags: Optional[Iterable[str]] = None) -> None:
        """
        Set safety status. Status should be one of: unknown, ok, flagged, rejected.
        Flags are short strings describing reasons (e.g., "alpha:TOXIC").
        """
        status = str(status)
        if status not in {"unknown", "ok", "flagged", "rejected"}:
            # allow unknown tokens but warn
            logger.warning("set_safety called with non-standard status: %s", status)
        self.safety["status"] = status
        if flags:
            # ensure list of strings
            f = [str(x) for x in flags]
            # extend existing flags (append-only)
            self.safety.setdefault("flags", []).extend(f)
        else:
            self.safety.setdefault("flags", [])

    def add_metadata(self, key: str, value: Any) -> None:
        """Add/overwrite a metadata key (must be JSON-serializable for persistence)."""
        try:
            json.dumps(value)
        except Exception as e:
            raise AICLPacketError(f"metadata value for '{key}' is not JSON-serializable: {e}")
        self.metadata[key] = value

    # --------------------
    # Validation / quick checks
    # --------------------

    def validate(self, raise_on_error: bool = False) -> bool:
        """
        Run basic validation checks. Returns True if OK.

        Checks:
          - session_id present
          - origin non-empty string
          - confidence is finite and 0 <= confidence <= 1
          - symbols are strings
          - trace entries have timestamps
        """
        errs: List[str] = []
        if not self.session_id:
            errs.append("missing session_id")
        if not isinstance(self.origin, str) or not self.origin.strip():
            errs.append("invalid origin")
        try:
            c = float(self.confidence)
            if not (0.0 <= c <= 1.0):
                errs.append("confidence out of range [0,1]")
        except Exception:
            errs.append("confidence not numeric")

        for s in self.symbols:
            if not isinstance(s, str):
                errs.append("non-string symbol present")
                break

        # trace timestamp integrity: each entry must have timestamp
        for t in self.trace:
            if "timestamp" not in t:
                errs.append("trace entry missing timestamp")
                break

        if errs:
            logger.debug("validate errors: %s", errs)
            if raise_on_error:
                raise AICLPacketError("; ".join(errs))
            return False
        return True

    # --------------------
    # Merge / copy utilities
    # --------------------

    def shallow_copy(self) -> "AICLPacket":
        """Return a shallow copy (new lists, but elements same)."""
        p = AICLPacket(
            session_id=self.session_id,
            origin=self.origin,
            targets=list(self.targets),
            symbols=list(self.symbols),
            intent=self.intent,
            confidence=self.confidence,
            metadata=deepcopy(self.metadata),
            version=self.version,
        )
        p.trace = list(self.trace)
        p.safety = dict(self.safety)
        p.response_symbols = list(self.response_symbols)
        return p

    def merge(self, other: "AICLPacket", prefer_self: bool = True) -> "AICLPacket":
        """
        Merge another packet into this one and return a new merged packet.

        Merge logic:
          - session_id kept from self (or other's if self missing)
          - origin preserved as 'self.origin' unless missing
          - targets unioned (order preserved: self then others not present)
          - symbols concatenated (self then other)
          - response_symbols concatenated
          - metadata keys from both (self wins if prefer_self=True)
          - trace concatenated (self then other)
          - safety: if any 'rejected' then result rejected; else aggregate flags
        """
        if not isinstance(other, AICLPacket):
            raise AICLPacketError("merge requires another AICLPacket")

        merged = self.shallow_copy()

        # targets: union preserving order
        for t in other.targets:
            if t not in merged.targets:
                merged.targets.append(t)

        # symbols and response_symbols: simple concatenation
        merged.symbols.extend(other.symbols)
        merged.response_symbols.extend(other.response_symbols)

        # metadata merge
        if prefer_self:
            merged_meta = dict(other.metadata)
            merged_meta.update(merged.metadata)  # self overwrites other
        else:
            merged_meta = dict(merged.metadata)
            merged_meta.update(other.metadata)
        merged.metadata = merged_meta

        # trace
        merged.trace.extend(other.trace)

        # safety aggregation
        flags = list(merged.safety.get("flags", [])) + list(other.safety.get("flags", []))
        status = "ok"
        if merged.safety.get("status") == "rejected" or other.safety.get("status") == "rejected":
            status = "rejected"
        elif merged.safety.get("status") == "flagged" or other.safety.get("status") == "flagged" or flags:
            status = "flagged"
        merged.safety = {"status": status, "flags": flags}

        # confidence: weighted average (simple heuristic)
        try:
            merged.confidence = float((self.confidence + other.confidence) / 2.0)
        except Exception:
            merged.confidence = max(self.confidence, other.confidence)

        return merged

    # --------------------
    # Serialization: dict / json / msgpack
    # --------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to plain dict (safe for json)."""
        return {
            "version": self.version,
            "session_id": self.session_id,
            "origin": self.origin,
            "targets": list(self.targets),
            "symbols": list(self.symbols),
            "intent": self.intent,
            "confidence": float(self.confidence),
            "metadata": self.metadata,
            "trace": list(self.trace),
            "safety": dict(self.safety),
            "response_symbols": list(self.response_symbols),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AICLPacket":
        """Create packet from dict produced by to_dict()."""
        if not isinstance(d, dict):
            raise AICLPacketError("from_dict expects a dict")
        p = cls(
            session_id=d.get("session_id"),
            origin=d.get("origin", "user"),
            targets=d.get("targets", []),
            symbols=d.get("symbols", []),
            intent=d.get("intent", "unknown"),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
            version=d.get("version", DEFAULT_VERSION),
        )
        p.trace = list(d.get("trace", []))
        p.safety = dict(d.get("safety", {"status": "unknown", "flags": []}))
        p.response_symbols = list(d.get("response_symbols", []))
        return p

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Return JSON string for the packet (uses allow_nan=False to be strict)."""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)
        except TypeError as e:
            # Metadata likely contains non-serializable info; raise helpful error
            raise AICLPacketError(f"to_json failed (non-serializable metadata): {e}")

    @classmethod
    def from_json(cls, s: str) -> "AICLPacket":
        """Load packet from JSON string."""
        try:
            d = json.loads(s)
        except Exception as e:
            raise AICLPacketError(f"from_json failed: {e}")
        return cls.from_dict(d)

    def to_msgpack(self) -> bytes:
        """Serialize packet to msgpack bytes. Requires msgpack installed."""
        if not _HAS_MSGPACK:
            raise AICLPacketError("msgpack not available (install msgpack-python)")

        try:
            return msgpack.packb(self.to_dict(), use_bin_type=True)
        except Exception as e:
            raise AICLPacketError(f"to_msgpack failed: {e}")

    @classmethod
    def from_msgpack(cls, b: bytes) -> "AICLPacket":
        """Deserialize msgpack bytes into an AICLPacket."""
        if not _HAS_MSGPACK:
            raise AICLPacketError("msgpack not available (install msgpack-python)")
        try:
            d = msgpack.unpackb(b, raw=False)
        except Exception as e:
            raise AICLPacketError(f"from_msgpack failed: {e}")
        return cls.from_dict(d)

    # --------------------
    # Human-friendly representations
    # --------------------

    def pretty(self, max_symbols: int = 8) -> str:
        """Compact one-line representation for logs and debugging."""
        syms = ", ".join(self.symbols[:max_symbols])
        if len(self.symbols) > max_symbols:
            syms += ", ..."
        resp = ", ".join(self.response_symbols[:max_symbols]) if self.response_symbols else ""
        return (
            f"<AICLPacket session={self.session_id} origin={self.origin} intent={self.intent} "
            f"conf={self.confidence:.3f} symbols=[{syms}] resp=[{resp}] safety={self.safety.get('status')}>"
        )

    def summarize(self) -> Dict[str, Any]:
        """Return a short dict with core info (for dashboards)."""
        return {
            "session_id": self.session_id,
            "origin": self.origin,
            "intent": self.intent,
            "confidence": round(float(self.confidence), 3),
            "symbols_count": len(self.symbols),
            "response_count": len(self.response_symbols),
            "safety": dict(self.safety),
            "trace_length": len(self.trace),
        }

    def export_trace(self) -> List[Dict[str, Any]]:
        """Return a deep copy of the trace list for external analysis."""
        return deepcopy(self.trace)

    # --------------------
    # Bytes helpers (convenience)
    # --------------------

    def as_bytes(self, fmt: str = "msgpack") -> bytes:
        """Return binary representation. fmt in {'msgpack','json'}."""
        fmt = fmt.lower()
        if fmt == "msgpack":
            return self.to_msgpack()
        elif fmt in {"json", "utf8", "utf-8"}:
            return self.to_json().encode("utf-8")
        else:
            raise AICLPacketError(f"unsupported format for as_bytes: {fmt}")

    @classmethod
    def from_bytes(cls, b: bytes, fmt: str = "msgpack") -> "AICLPacket":
        fmt = fmt.lower()
        if fmt == "msgpack":
            return cls.from_msgpack(b)
        elif fmt in {"json", "utf8", "utf-8"}:
            return cls.from_json(b.decode("utf-8"))
        else:
            raise AICLPacketError(f"unsupported format for from_bytes: {fmt}")

    # --------------------
    # Utilities
    # --------------------

    def get_provenance(self) -> Dict[str, Any]:
        """
        Return provenance summary: first & last actors, duration, steps count.
        Useful for dashboards and research logging.
        """
        if not self.trace:
            return {"steps": 0, "first": None, "last": None, "duration_seconds": 0.0}
        first = self.trace[0]
        last = self.trace[-1]
        try:
            t0 = datetime.fromisoformat(first["timestamp"])
            t1 = datetime.fromisoformat(last["timestamp"])
            duration = (t1 - t0).total_seconds()
        except Exception:
            duration = 0.0
        return {"steps": len(self.trace), "first": first, "last": last, "duration_seconds": duration}

    # repr
    def __repr__(self) -> str:
        return self.pretty()

    # equality (useful for tests)
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AICLPacket):
            return False
        return self.to_dict() == other.to_dict()
