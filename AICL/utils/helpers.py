# AICL/utils/helpers.py
# Author: Vansh Bukkarwal (A LocalHouseLLM Project)
# Date: 2025-11-05
#
# Utility helpers for AICL core subsystems.
#
# Purpose:
#   - Shared utilities for packet, router, registry, and safety modules
#   - High-performance timestamp, hashing, randomness, safe merges
#   - Strong typing + predictable, stable behavior
#
# This module is used throughout the entire AICL runtime.
#

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Any, Dict, Optional, List, Tuple, Callable


# ============================================================
# Time
# ============================================================

def now_ts() -> float:
    """High-precision monotonic timestamp for packet stamps."""
    return time.time()


# ============================================================
# UUID / IDs
# ============================================================

def gen_id(prefix: str = "") -> str:
    """Generate a fast UUID4-based id with optional prefix."""
    if prefix:
        return f"{prefix}_{uuid.uuid4().hex}"
    return uuid.uuid4().hex


def short_id() -> str:
    """Generate a short 8-char identifier."""
    return uuid.uuid4().hex[:8]


# ============================================================
# Hashing
# ============================================================

def sha256_hex(data: bytes | str) -> str:
    """Return SHA-256 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def short_hash(text: str) -> str:
    """A short 8-char hash for text."""
    return sha256_hex(text)[:8]


# ============================================================
# Symbol Processing
# ============================================================

def sanitize_symbol(sym: str) -> str:
    """
    Ensure symbol is clean:
    - strip whitespace
    - no newlines/tab
    - no illegal chars
    """
    sym = sym.strip()
    bad = ["\n", "\r", "\t"]
    for b in bad:
        sym = sym.replace(b, " ")
    return sym


# ============================================================
# Dict / Merge Utilities
# ============================================================

def safe_deep_merge(
    a: Dict[str, Any],
    b: Dict[str, Any],
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    - Does NOT merge lists.
    - Recursively merges dicts.
    """
    out = dict(a)
    for k, v in b.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = safe_deep_merge(out[k], v, overwrite)
        else:
            if overwrite or k not in out:
                out[k] = v
    return out


class FrozenDict(dict):
    """Immutable dict used for safe metadata snapshots."""

    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable")

    def update(self, *_args, **_kwargs):
        raise TypeError("FrozenDict is immutable")


# ============================================================
# JSON Encoding
# ============================================================

def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def safe_json(data: Any) -> str:
    """Compact, safe JSON string used inside packets."""
    try:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


# ============================================================
# Async timeout helpers
# ============================================================

def timeout_wrapper(func: Callable, timeout: float) -> Any:
    """
    Execute func() with a time limit.
    Used by AICL safety & router for sandbox calls.
    """
    result = [None]
    exc = [None]

    def runner():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"timeout exceeded ({timeout}s)")

    if exc[0]:
        raise exc[0]

    return result[0]


# ============================================================
# Trace helpers
# ============================================================

def trace_event(
    who: str,
    what: str,
    note: Optional[str] = None
) -> Dict[str, Any]:
    """Standard AICL trace entry."""
    return {
        "ts": now_ts(),
        "who": who,
        "what": what,
        "note": note or "",
    }


# ============================================================
# Packet summarization
# ============================================================

def summarize_packet(pkt) -> Dict[str, Any]:
    """Compact view for logs and debugging."""
    return {
        "id": pkt.id,
        "origin": pkt.origin,
        "targets": pkt.targets,
        "symbols": pkt.symbols[:6],  # Light preview
        "intent": pkt.intent,
        "confidence": pkt.confidence,
        "metadata_keys": list(pkt.metadata.keys())[:10],
        "trace_len": len(pkt.trace),
    }


# ============================================================
# Math / Stats Utilities
# ============================================================

def exp_backoff(base: float, attempt: int, cap: float = 60.0) -> float:
    """
    Exponential backoff calculator used for retry policy.
    """
    out = base * (2 ** attempt)
    return min(out, cap)


def smooth_ratio(a: float, b: float, eps: float = 1e-9) -> float:
    """Return a/(b+eps) to avoid zero-division in metrics."""
    return a / (b + eps)


# ============================================================
# Thread-safe atomic counter
# ============================================================

class AtomicCounter:
    """Fast thread-safe counter."""

    def __init__(self, initial: int = 0):
        self.value = initial
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> int:
        with self._lock:
            self.value += n
            return self.value

    def get(self) -> int:
        return self.value


# ============================================================
# Sliding window
# ============================================================

class SlidingWindow:
    """
    Fixed-size sliding window for stats/rate calculations.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self._items: List[float] = []
        self._lock = threading.Lock()

    def push(self, value: float):
        with self._lock:
            self._items.append(value)
            if len(self._items) > self.size:
                self._items.pop(0)

    def avg(self) -> float:
        with self._lock:
            if not self._items:
                return 0.0
            return sum(self._items) / len(self._items)

    def values(self) -> List[float]:
        with self._lock:
            return list(self._items)
