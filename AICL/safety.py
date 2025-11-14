# AICL/safety.py
# Author: Vansh Bukkarwal ( A localhousellm project )
# Date: 2025-11-05
#
# AICL Safety Layer
#
# Purpose:
#   Production-minded safety / verification layer for AICL packets.
#   Designed to be plugged into Router as:
#       pre_route_hook = safety.pre_route
#       post_route_hook = safety.post_route
#
# Responsibilities:
#   - Packet structural validation & schema checks
#   - Content safety filtering (blacklist / regex / heuristics)
#   - Rate-limiting & throttling (per-origin & per-target)
#   - Replay / loop detection (duplicate suppression)
#   - TTL, size & complexity limits
#   - Confidence / intent gating
#   - Optional ML-based safety classifier (pluggable stub)
#   - Metrics & logging hooks for observability
#
# Design notes:
#   - Pure-Python, zero external dependencies.
#   - Thread-safe (locks for mutating state).
#   - Pluggable policies through constructor args.
#   - Does not perform content transformation unless explicitly allowed.
#
# Usage example:
#   safety = SafetyLayer()
#   router = Router(pre_route_hook=safety.pre_route, post_route_hook=safety.post_route)
#
# (This file focuses on pragmatic, auditable safety checks suitable for
#  an open-source, local-first modular AI runtime.)
#

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

# Correct import path for AICL packet
try:
    from AICL.packet import AICLPacket
except Exception:
    # Best-effort fallback import if package path differs; keep runtime informative.
    raise ImportError("AICL.packet must be importable. Ensure AICL package is on PYTHONPATH.")

logger = logging.getLogger("AICL.safety")
logger.addHandler(logging.NullHandler())


# ---------------------------
# Exceptions
# ---------------------------
class SafetyError(Exception):
    """Raised when a packet fails safety checks."""


# ---------------------------
# Data classes / helpers
# ---------------------------
@dataclass
class RateLimitBucket:
    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_ts: float = field(default_factory=time.time)

    def consume(self, amount: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_ts
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_ts = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


# ---------------------------
# Default policies
# ---------------------------
DEFAULT_BLACKLIST = {
    # coarse-grained banned tokens/phrases (lowercased)
    "bomb",
    "kill",
    "attack",
    "explosive",
    "terrorist",
    "self-harm",
    "suicide",
    # policy authors can extend this set
}

DEFAULT_SENSITIVE_REGEX = [
    # patterns that look like API keys, private keys, JWTs, or SSNs-ish patterns
    r"(?i)api[_-]?key[:=]\s*[A-Za-z0-9\-_]{16,}",
    r"-----BEGIN PRIVATE KEY-----",
    r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",  # naive SSN-like
    # Long base64-ish tokens
    r"[A-Za-z0-9_\-]{40,}",
]


# ---------------------------
# Safety Layer
# ---------------------------
class SafetyLayer:
    """
    SafetyLayer enforces content & routing policies.

    Methods:
      - pre_route(packet) -> AICLPacket
          Called before router dispatch; may block by raising SafetyError
          or return a (possibly annotated) packet.

      - post_route(orig_packet, response_packet) -> Optional[AICLPacket]
          Called after a module returned a response; may block/modify response.

    Configurable via constructor parameters.
    """

    def __init__(
        self,
        *,
        blacklist: Optional[Iterable[str]] = None,
        sensitive_patterns: Optional[Iterable[str]] = None,
        max_symbols: int = 512,
        max_metadata_size: int = 32 * 1024,  # bytes
        max_roundtrips: int = 10,  # detect loops by stamps
        replay_window_seconds: int = 30,
        rate_limit_capacity: float = 10.0,
        rate_limit_refill: float = 1.0,
        per_origin_limit: bool = True,
        per_target_limit: bool = True,
        allow_transformations: bool = False,
        log_dir: Optional[str] = None,
        metrics_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        ml_safety_classifier: Optional[Callable[[AICLPacket], Tuple[bool, float]]] = None,
    ):
        # policies
        self.blacklist = set(w.lower() for w in (blacklist or DEFAULT_BLACKLIST))
        self.sensitive_regex = [re.compile(p) for p in (sensitive_patterns or DEFAULT_SENSITIVE_REGEX)]
        self.max_symbols = int(max_symbols)
        self.max_metadata_size = int(max_metadata_size)
        self.max_roundtrips = int(max_roundtrips)
        self.replay_window_seconds = int(replay_window_seconds)

        # rate limiting
        self.per_origin_limit = bool(per_origin_limit)
        self.per_target_limit = bool(per_target_limit)
        self._capacity = float(rate_limit_capacity)
        self._refill = float(rate_limit_refill)
        self._buckets_lock = threading.RLock()
        self._origin_buckets: Dict[str, RateLimitBucket] = {}
        self._target_buckets: Dict[str, RateLimitBucket] = {}

        # loop/replay detection (store hash -> timestamps)
        self._recent_hashes_lock = threading.RLock()
        self._recent_hashes: Deque[Tuple[str, float]] = deque()  # (hash, ts)
        self._recent_hash_set: Dict[str, float] = {}  # hash -> ts

        # operational
        self.allow_transformations = bool(allow_transformations)
        self.metrics_hook = metrics_hook
        self.ml_safety_classifier = ml_safety_classifier  # returns (is_safe, confidence)

        # logging
        self.log_dir = log_dir or os.path.join(os.getcwd(), "data", "aicl_safety_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_log_path = os.path.join(self.log_dir, "safety_audit.jsonl")
        self._audit_lock = threading.RLock()

        # runtime kill switch
        self._kill_switch = False
        self._kill_lock = threading.RLock()

        logger.info("SafetyLayer initialized | log_dir=%s", self.log_dir)

    # ---------------------------
    # Public control
    # ---------------------------
    def enable_kill_switch(self) -> None:
        with self._kill_lock:
            self._kill_switch = True
            logger.warning("SafetyLayer: KILL SWITCH ENABLED (all packets will be blocked)")

    def disable_kill_switch(self) -> None:
        with self._kill_lock:
            self._kill_switch = False
            logger.info("SafetyLayer: KILL SWITCH DISABLED")

    def is_kill_switch_on(self) -> bool:
        with self._kill_lock:
            return bool(self._kill_switch)

    # ---------------------------
    # Pre-route hook
    # ---------------------------
    def pre_route(self, packet: AICLPacket) -> AICLPacket:
        """
        Called by Router before dispatch. Returns packet (possibly modified)
        or raises SafetyError to block the packet.
        """
        if packet is None:
            raise SafetyError("pre_route received None packet")

        if self.is_kill_switch_on():
            raise SafetyError("system kill switch engaged - blocking packet")

        # Basic structural validation (packet should provide origin & symbols)
        if not getattr(packet, "origin", None):
            raise SafetyError("packet missing origin")

        # Stamp arrival for traceability
        packet.stamp("safety", "pre_route_received", note=f"ts={time.time()}")

        # Replay / duplicate suppression
        pkt_hash = self._packet_hash(packet)
        if self._is_recent_duplicate(pkt_hash):
            # annotate and block duplicates
            packet.stamp("safety", "duplicate_blocked")
            self._audit("duplicate_block", packet, {"hash": pkt_hash})
            raise SafetyError("duplicate or replay packet detected")

        # Rate limiting check (origin-target)
        if self.per_origin_limit:
            origin = str(packet.origin)
            if not self._consume_bucket(self._origin_buckets, origin):
                packet.stamp("safety", "rate_limited_origin")
                self._audit("rate_limited_origin", packet, {"origin": origin})
                raise SafetyError(f"rate limit exceeded for origin '{origin}'")

        if self.per_target_limit and packet.targets:
            # If multiple targets, check all target buckets; block if any target exhausted
            for t in packet.targets:
                if not self._consume_bucket(self._target_buckets, t):
                    packet.stamp("safety", "rate_limited_target", note=f"target={t}")
                    self._audit("rate_limited_target", packet, {"target": t})
                    raise SafetyError(f"rate limit exceeded for target '{t}'")

        # Size & complexity checks
        if len(getattr(packet, "symbols", []) or []) > self.max_symbols:
            packet.stamp("safety", "symbols_over_limit")
            self._audit("symbols_over_limit", packet, {"num_symbols": len(packet.symbols)})
            raise SafetyError(f"packet symbols exceed allowed max ({self.max_symbols})")

        metadata_json = json.dumps(getattr(packet, "metadata", {}), ensure_ascii=False)
        if len(metadata_json.encode("utf-8")) > self.max_metadata_size:
            packet.stamp("safety", "metadata_over_limit")
            self._audit("metadata_over_limit", packet, {"size_bytes": len(metadata_json.encode("utf-8"))})
            raise SafetyError("packet metadata too large")

        # Content checks (simple blacklist)
        content_str = self._packet_content_string(packet)
        if self._contains_blacklist(content_str):
            packet.stamp("safety", "blacklist_blocked")
            self._audit("blacklist_block", packet, {"matched": True})
            raise SafetyError("packet content matched blacklist")

        # Sensitive pattern detection
        matched_sensitive = self._sensitive_match(content_str)
        if matched_sensitive:
            packet.stamp("safety", "sensitive_pattern_blocked")
            self._audit("sensitive_block", packet, {"pattern": matched_sensitive})
            raise SafetyError("packet contains sensitive pattern (possible key/credential)")

        # ML safety classifier (optional)
        if self.ml_safety_classifier:
            try:
                safe, confidence = self.ml_safety_classifier(packet)
                packet.stamp("safety", "ml_check", note=f"safe={safe} conf={confidence:.3f}")
                if not safe:
                    self._audit("ml_block", packet, {"confidence": confidence})
                    raise SafetyError(f"ml safety classifier blocked packet (conf={confidence:.3f})")
            except Exception as e:
                logger.exception("ml_safety_classifier failed: %s", e)
                # fail-safe: allow packet but annotate
                packet.stamp("safety", "ml_check_error", note=str(e))

        # record packet hash in recent window
        self._record_hash(pkt_hash)

        # metrics hook
        if self.metrics_hook:
            try:
                self.metrics_hook("pre_route_ok", {"origin": packet.origin, "targets": list(packet.targets or [])})
            except Exception:
                logger.exception("metrics_hook pre_route_ok failed")

        # allowed to pass through; optionally transform (not done by default)
        if self.allow_transformations:
            # Example: remove PII from metadata - placeholder; keep unchanged by default
            pass

        return packet

    # ---------------------------
    # Post-route hook
    # ---------------------------
    def post_route(self, orig_packet: AICLPacket, response_packet: Optional[AICLPacket]) -> Optional[AICLPacket]:
        """
        Called by Router after a module has returned a response.
        Can block/modify responses by raising SafetyError or returning modified Packet.
        """
        # Response can be None for async handlers or fire-and-forget
        if response_packet is None:
            # nothing to verify, but log event
            if self.metrics_hook:
                try:
                    self.metrics_hook("post_route_none", {"origin": orig_packet.origin})
                except Exception:
                    logger.exception("metrics_hook post_route_none failed")
            return None

        # Basic checks similar to pre_route
        response_packet.stamp("safety", "post_route_received", note=f"ts={time.time()}")

        # Prevent self-looping: if response references original request in stamp chain too often, block
        if self._detect_loop(response_packet):
            response_packet.stamp("safety", "loop_blocked")
            self._audit("loop_block", response_packet, {"origin": orig_packet.origin})
            raise SafetyError("response appears to be part of a loop or excessive roundtrip")

        # Check content again (module might generate unsafe content)
        content_str = self._packet_content_string(response_packet)
        if self._contains_blacklist(content_str):
            response_packet.stamp("safety", "blacklist_blocked_post")
            self._audit("blacklist_block_post", response_packet)
            raise SafetyError("module response contained blacklisted content")

        matched_sensitive = self._sensitive_match(content_str)
        if matched_sensitive:
            response_packet.stamp("safety", "sensitive_block_post")
            self._audit("sensitive_block_post", response_packet, {"pattern": matched_sensitive})
            raise SafetyError("module response contains sensitive pattern")

        # Optionally run ML classifier on response
        if self.ml_safety_classifier:
            try:
                safe, confidence = self.ml_safety_classifier(response_packet)
                response_packet.stamp("safety", "ml_check_response", note=f"safe={safe} conf={confidence:.3f}")
                if not safe:
                    self._audit("ml_block_response", response_packet, {"confidence": confidence})
                    raise SafetyError(f"ml classifier blocked response (conf={confidence:.3f})")
            except Exception:
                # do not completely block on ML errors; annotate
                response_packet.stamp("safety", "ml_check_error_response")

        # metrics
        if self.metrics_hook:
            try:
                self.metrics_hook("post_route_ok", {"origin": orig_packet.origin, "handler_origin": response_packet.origin})
            except Exception:
                logger.exception("metrics_hook post_route_ok failed")

        return response_packet

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _packet_content_string(self, packet: AICLPacket) -> str:
        """
        Convert symbols + metadata/text fields into a single lowercased string for scanning.
        Non-destructive: does not mutate packet.
        """
        parts: List[str] = []
        # symbols are often tokenized MRL symbols; join them but also include metadata text
        symbols = getattr(packet, "symbols", None)
        if symbols:
            if isinstance(symbols, (list, tuple)):
                parts.append(" ".join(str(s) for s in symbols if s is not None))
            else:
                parts.append(str(symbols))
        md = getattr(packet, "metadata", {}) or {}
        # include explicit 'text' or common keys
        for key in ("text", "message", "body"):
            if key in md:
                parts.append(str(md.get(key)))
        # include other metadata values (string-cast)
        for k, v in md.items():
            if v and isinstance(v, (str, int, float)):
                parts.append(str(v))
        joined = " ".join(parts).lower()
        return joined

    def _contains_blacklist(self, content: str) -> bool:
        # whole-word check for blacklist tokens
        if not content:
            return False
        for term in self.blacklist:
            # word boundary checks to avoid substring false positives
            if re.search(rf"\b{re.escape(term)}\b", content):
                logger.debug("blacklist matched term=%s", term)
                return True
        return False

    def _sensitive_match(self, content: str) -> Optional[str]:
        if not content:
            return None
        for pattern in self.sensitive_regex:
            if pattern.search(content):
                logger.debug("sensitive pattern matched: %s", pattern.pattern)
                return pattern.pattern
        return None

    def _packet_hash(self, packet: AICLPacket) -> str:
        """
        Deterministic-ish packet fingerprint for replay detection.
        Uses origin + symbols + a short metadata digest.
        """
        origin = str(getattr(packet, "origin", ""))
        symbols_part = "|".join(map(str, getattr(packet, "symbols", []) or []))
        md = getattr(packet, "metadata", {}) or {}
        md_part = json.dumps({k: md[k] for k in sorted(md) if k in ("text", "message", "body")}, sort_keys=True, ensure_ascii=False)
        raw = f"{origin}:{symbols_part}:{md_part}"
        return sha1(raw.encode("utf-8")).hexdigest()

    def _is_recent_duplicate(self, h: str) -> bool:
        now = time.time()
        with self._recent_hashes_lock:
            # purge old
            cutoff = now - self.replay_window_seconds
            while self._recent_hashes and self._recent_hashes[0][1] < cutoff:
                old_hash, _ = self._recent_hashes.popleft()
                if old_hash in self._recent_hash_set:
                    del self._recent_hash_set[old_hash]
            # check membership
            return h in self._recent_hash_set

    def _record_hash(self, h: str) -> None:
        now = time.time()
        with self._recent_hashes_lock:
            self._recent_hashes.append((h, now))
            self._recent_hash_set[h] = now

    def _consume_bucket(self, bucket_map: Dict[str, RateLimitBucket], key: str) -> bool:
        with self._buckets_lock:
            b = bucket_map.get(key)
            if b is None:
                b = RateLimitBucket(capacity=self._capacity, tokens=self._capacity, refill_rate=self._refill)
                bucket_map[key] = b
        return b.consume()

    def _detect_loop(self, packet: AICLPacket) -> bool:
        """
        Detect loops by counting stamps from the same component in the packet's trace.
        If the number of router/safety/orchestrator stamps exceeds max_roundtrips -> loop.
        """
        # assume packet has a 'stamps' or 'trace' field we can inspect; fallback to metadata
        trace = getattr(packet, "stamps", None) or packet.metadata.get("stamps", []) if getattr(packet, "metadata", None) else []
        if not isinstance(trace, list):
            # try to parse if it's present as a string
            try:
                trace = json.loads(str(trace))
                if not isinstance(trace, list):
                    trace = []
            except Exception:
                trace = []
        # Count occurrences of safety/orchestrator/router stamps
        count = 0
        for s in trace:
            if isinstance(s, dict) and s.get("by") in ("router", "safety", "orchestrator"):
                count += 1
            elif isinstance(s, str) and any(tag in s for tag in ("router", "safety", "orchestrator")):
                count += 1
        return count > self.max_roundtrips

    def _audit(self, event: str, packet: AICLPacket, details: Optional[Dict[str, Any]] = None) -> None:
        """Append an audit entry to safety_audit.jsonl"""
        entry = {
            "ts": time.time(),
            "event": event,
            "packet_origin": getattr(packet, "origin", None),
            "packet_targets": list(packet.targets) if getattr(packet, "targets", None) else [],
            "packet_hash": self._packet_hash(packet),
            "details": details or {},
        }
        with self._audit_lock:
            try:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("failed to write safety audit log")

    # ---------------------------
    # Utility: default ML safety stub
    # ---------------------------
    @staticmethod
    def default_ml_safety_stub(packet: AICLPacket) -> Tuple[bool, float]:
        """
        Default ML safety stub: always returns (True, 1.0).
        Replace with a real model that returns (is_safe, confidence).
        """
        return True, 1.0


# ---------------------------
# Convenience factory
# ---------------------------
def make_default_safety(**kwargs) -> SafetyLayer:
    """
    Create a default safety layer prewired with default blacklist / patterns.
    Accepts overrides for any SafetyLayer constructor param.
    """
    sl = SafetyLayer(
        blacklist=kwargs.pop("blacklist", None),
        sensitive_patterns=kwargs.pop("sensitive_patterns", None),
        ml_safety_classifier=kwargs.pop("ml_safety_classifier", SafetyLayer.default_ml_safety_stub),
        **kwargs,
    )
    return sl


# ---------------------------
# Module-level simple test (self-run)
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    safety = make_default_safety()
    # Simple mock packet object if AICLPacket unavailable; else use AICLPacket
    try:
        pkt = AICLPacket(origin="user", symbols=["hello", "world"], intent="greet", confidence=0.9)
    except Exception:
        # fallback minimal duck-typed object
        class _Pkt:
            def __init__(self):
                self.origin = "user"
                self.symbols = ["hello", "world"]
                self.targets = ["echo"]
                self.metadata = {"text": "hello world"}
                self.stamps = []

            def stamp(self, by, tag, note=None):
                self.stamps.append({"by": by, "tag": tag, "note": note})
        pkt = _Pkt()

    try:
        print("pre_route OK ->", safety.pre_route(pkt).origin)
    except Exception as e:
        print("pre_route failed:", e)

    # simulate response packet
    try:
        resp = AICLPacket(origin="echo", symbols=["RESP:hello"], metadata={"text": "Hello back"})
    except Exception:
        class _Resp:
            def __init__(self):
                self.origin = "echo"
                self.symbols = ["RESP:hello"]
                self.metadata = {"text": "Hello back"}
                self.stamps = []

            def stamp(self, by, tag, note=None):
                self.stamps.append({"by": by, "tag": tag, "note": note})
        resp = _Resp()

    try:
        out = safety.post_route(pkt, resp)
        print("post_route OK ->", getattr(out, "origin", None))
    except Exception as e:
        print("post_route failed:", e)
