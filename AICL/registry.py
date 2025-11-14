# AICL/registry.py
# Author: Vansh Bukkarwal ( A localhousellm project )
# Date: 2025-11-05
#
# AICL Module Registry
#
# Purpose:
#   Production-grade module registry for the AICL Router ecosystem.
#   Responsibilities:
#     - Register / unregister modules (callable or dotted import path)
#     - Maintain metadata, capabilities, tags, versions, and priorities
#     - Durable persistence to a JSON file + optional in-memory snapshot
#     - Heartbeat (TTL) monitoring and health status updates
#     - Query & discovery: by name, capability, tag, priority
#     - Event subscriptions: callbacks for register/unregister/health-change
#     - Dynamic re-loading of handler callables from module path
#
# Notes:
#   - Handlers (callables) are not persisted directly; registry persists a "handler_path"
#     when available (dotted path like "mypkg.mymodule.make_handler"). When a callable
#     is provided directly, it's kept in-memory but cannot be reconstructed on reload
#     unless a handler_path is also supplied.
#   - Persistence file defaults to "./.aicl/registry.json" but can be overridden.
#   - Heartbeat monitor runs in a background thread and marks modules UNHEALTHY when
#     they don't send heartbeat within (ttl) seconds.
#
# Usage:
#   reg = Registry()  # loads persisted state
#   reg.register(name="language", handler=my_handler, capabilities=["nlp"], ttl=30)
#   reg.heartbeat("language")  # update heartbeat
#   choices = reg.find_by_capability("nlp")
#

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger("AICL.registry")
logger.addHandler(logging.NullHandler())

# -------------------------
# Exceptions
# -------------------------
class RegistryError(Exception):
    """Generic registry error."""


class ModuleNotFoundError(RegistryError):
    """Raised when a module record cannot be located."""


class ModuleRegistrationError(RegistryError):
    """Raised when registration fails (duplicate, invalid args, etc.)."""


# -------------------------
# Data classes
# -------------------------
@dataclass
class ModuleRecord:
    """
    Canonical record in the registry.

    handler_callable is only set for in-memory registrations where a Python callable
    was provided. When persisted to disk, handler_path should be provided (dotted
    import path) so the registry can re-import the handler on startup.
    """
    name: str
    handler_path: Optional[str] = None  # dotted import path, if available
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: Optional[str] = None
    priority: int = 0  # higher priority preferred
    is_async: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=lambda: time.time())
    last_heartbeat: float = field(default_factory=lambda: time.time())
    ttl: Optional[float] = 60.0  # seconds; None = no TTL/always healthy
    status: str = "healthy"  # healthy | unhealthy | deregistered
    handler_callable_name: Optional[str] = None

    # runtime-only, not serialized
    handler_callable: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)


# -------------------------
# Registry
# -------------------------
class Registry:
    """
    Thread-safe registry that stores ModuleRecord objects, persists them,
    monitors health via TTL, and provides discovery APIs.

    Key methods:
      - register(...)
      - unregister(name)
      - heartbeat(name)
      - get(name)
      - find_by_capability(cap, min_priority=None, tags=None)
      - persist() / load()
      - subscribe(event, callback)
    """

    DEFAULT_PERSIST_PATH = os.path.join(".aicl", "registry.json")

    def __init__(
        self,
        persist_path: Optional[str] = None,
        start_health_monitor: bool = True,
        health_check_interval: float = 5.0,
    ):
        self._lock = threading.RLock()
        self._records: Dict[str, ModuleRecord] = {}
        self.persist_path = persist_path or self.DEFAULT_PERSIST_PATH
        self._ensure_persist_dir()
        self._event_subscribers: Dict[str, List[Callable[[ModuleRecord], None]]] = {}
        self._health_check_interval = float(health_check_interval)
        self._stop_monitor = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Attempt to load persisted state if available
        self._load_persisted_state()

        # Start background health monitor
        if start_health_monitor:
            self._start_health_monitor()
        logger.info("Registry initialized (persist=%s)", self.persist_path)

    # -------------------------
    # Persistence helpers
    # -------------------------
    def _ensure_persist_dir(self) -> None:
        d = os.path.dirname(self.persist_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _load_persisted_state(self) -> None:
        """Load persisted module records. Handlers are only reloaded if handler_path present."""
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded = 0
            for rec_json in raw.get("modules", []):
                # Create ModuleRecord without handler_callable (must be re-imported)
                rec = ModuleRecord(
                    name=rec_json["name"],
                    handler_path=rec_json.get("handler_path"),
                    capabilities=rec_json.get("capabilities", []),
                    tags=rec_json.get("tags", []),
                    version=rec_json.get("version"),
                    priority=int(rec_json.get("priority", 0)),
                    is_async=bool(rec_json.get("is_async", False)),
                    metadata=rec_json.get("metadata", {}),
                    registered_at=float(rec_json.get("registered_at", time.time())),
                    last_heartbeat=float(rec_json.get("last_heartbeat", time.time())),
                    ttl=(float(rec_json["ttl"]) if rec_json.get("ttl") is not None else None),
                    status=rec_json.get("status", "healthy"),
                    handler_callable_name=rec_json.get("handler_callable_name"),
                )
                # Attempt import of handler if handler_path present
                if rec.handler_path:
                    try:
                        rec.handler_callable = self._import_callable_by_path(rec.handler_path)
                        logger.debug("Imported handler for %s from %s", rec.name, rec.handler_path)
                    except Exception as e:
                        logger.warning("Failed to import handler for %s from %s: %s", rec.name, rec.handler_path, e)
                with self._lock:
                    self._records[rec.name] = rec
                loaded += 1
            logger.info("Registry loaded %d records from %s", loaded, self.persist_path)
        except Exception as e:
            logger.exception("Failed to load registry from %s: %s", self.persist_path, e)

    def persist(self) -> None:
        """
        Persist the registry to disk in a reasonably compact JSON format.
        Note: handler_callable (callable object) is not persisted; prefer passing handler_path.
        """
        with self._lock:
            data = {"modules": []}
            for rec in self._records.values():
                rec_json = {
                    "name": rec.name,
                    "handler_path": rec.handler_path,
                    "capabilities": rec.capabilities,
                    "tags": rec.tags,
                    "version": rec.version,
                    "priority": rec.priority,
                    "is_async": rec.is_async,
                    "metadata": rec.metadata,
                    "registered_at": rec.registered_at,
                    "last_heartbeat": rec.last_heartbeat,
                    "ttl": rec.ttl,
                    "status": rec.status,
                    "handler_callable_name": rec.handler_callable_name,
                }
                data["modules"].append(rec_json)
            tmp = self.persist_path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.persist_path)
                logger.debug("Registry persisted to %s", self.persist_path)
            except Exception as e:
                logger.exception("Failed to persist registry to %s: %s", self.persist_path, e)
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

    # -------------------------
    # Import utilities
    # -------------------------
    def _import_callable_by_path(self, dotted: str) -> Callable[..., Any]:
        """
        Import a callable given a dotted path like:
          package.module:callable_name  OR package.module.callable_name
        Accepts ":" or last dot separator.
        """
        if not dotted or not isinstance(dotted, str):
            raise ModuleRegistrationError("handler_path must be a dotted import path string")
        # allow "pkg.module:attr" syntax
        if ":" in dotted:
            module_path, attr = dotted.split(":", 1)
        else:
            parts = dotted.rsplit(".", 1)
            if len(parts) == 2:
                module_path, attr = parts
            else:
                raise ModuleRegistrationError(f"invalid handler_path '{dotted}'")
        module = importlib.import_module(module_path)
        if not hasattr(module, attr):
            raise ModuleRegistrationError(f"module {module_path} has no attribute {attr}")
        obj = getattr(module, attr)
        if not callable(obj):
            raise ModuleRegistrationError(f"imported object {dotted} is not callable")
        return obj

    # -------------------------
    # Registration / lifecycle
    # -------------------------
    def register(
        self,
        name: str,
        handler: Optional[Union[Callable[..., Any], str]] = None,
        *,
        capabilities: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        priority: int = 0,
        is_async: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = 60.0,
        persist: bool = True,
    ) -> ModuleRecord:
        """
        Register a module.

        handler: either a callable (kept in-memory) or a dotted import path (string).
        ttl: Time-to-live in seconds for heartbeat. None => never expire.
        persist: whether to write registry to disk after registration.
        """
        if not name or not isinstance(name, str):
            raise ModuleRegistrationError("module name must be a non-empty string")
        name = name.strip()

        with self._lock:
            if name in self._records and self._records[name].status != "deregistered":
                raise ModuleRegistrationError(f"module '{name}' already registered")

            handler_path = None
            handler_callable = None
            handler_callable_name = None

            if isinstance(handler, str):
                handler_path = handler
                try:
                    handler_callable = self._import_callable_by_path(handler_path)
                    handler_callable_name = getattr(handler_callable, "__name__", None)
                except Exception as e:
                    # allow register with handler_path even if import fails; warn
                    logger.warning("Failed to import handler at register-time for %s: %s", name, e)
                    handler_callable = None
            elif callable(handler):
                handler_callable = handler
                handler_callable_name = getattr(handler, "__name__", None)
                # try to determine a handler_path if the callable is a function defined in a module
                try:
                    mod = inspect.getmodule(handler_callable)
                    if mod and hasattr(handler_callable, "__name__") and mod.__name__:
                        handler_path = f"{mod.__name__}.{handler_callable.__name__}"
                except Exception:
                    handler_path = None

            rec = ModuleRecord(
                name=name,
                handler_path=handler_path,
                capabilities=list(capabilities) if capabilities else [],
                tags=list(tags) if tags else [],
                version=version,
                priority=int(priority),
                is_async=bool(is_async),
                metadata=metadata.copy() if metadata else {},
                ttl=(float(ttl) if ttl is not None else None),
            )
            rec.handler_callable = handler_callable
            rec.handler_callable_name = handler_callable_name
            rec.registered_at = time.time()
            rec.last_heartbeat = time.time()
            rec.status = "healthy"

            self._records[name] = rec

            logger.info("module registered: %s (caps=%s tags=%s)", name, rec.capabilities, rec.tags)

        # persist and notify outside lock
        if persist:
            self.persist()
        self._notify("register", rec)
        return rec

    def unregister(self, name: str, persist: bool = True) -> ModuleRecord:
        """Unregister module (mark as deregistered and persist)."""
        with self._lock:
            if name not in self._records:
                raise ModuleNotFoundError(f"module '{name}' not found")
            rec = self._records[name]
            rec.status = "deregistered"
            rec.handler_callable = None
            logger.info("module unregistered: %s", name)
        if persist:
            self.persist()
        self._notify("unregister", rec)
        return rec

    def get(self, name: str) -> ModuleRecord:
        with self._lock:
            rec = self._records.get(name)
            if not rec:
                raise ModuleNotFoundError(f"module '{name}' not found")
            return rec

    def update(
        self,
        name: str,
        *,
        capabilities: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        persist: bool = True,
    ) -> ModuleRecord:
        """Update metadata for a registered module."""
        with self._lock:
            rec = self.get(name)
            if capabilities is not None:
                rec.capabilities = list(capabilities)
            if tags is not None:
                rec.tags = list(tags)
            if version is not None:
                rec.version = version
            if priority is not None:
                rec.priority = int(priority)
            if metadata is not None:
                rec.metadata.update(metadata)
            if ttl is not None:
                rec.ttl = float(ttl) if ttl is not None else None
            rec.last_heartbeat = time.time()
            logger.info("module updated: %s", name)
        if persist:
            self.persist()
        self._notify("update", rec)
        return rec

    # -------------------------
    # Heartbeat / health
    # -------------------------
    def heartbeat(self, name: str) -> ModuleRecord:
        """Touch module's last_heartbeat and mark healthy."""
        with self._lock:
            rec = self.get(name)
            rec.last_heartbeat = time.time()
            if rec.status != "healthy":
                old = rec.status
                rec.status = "healthy"
                logger.info("module %s recovered (was %s)", name, old)
                self._notify("health_change", rec)
        self.persist()
        return rec

    def _start_health_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_monitor.clear()
        t = threading.Thread(target=self._health_monitor_loop, name="AICL-Registry-Health", daemon=True)
        self._monitor_thread = t
        t.start()

    def _stop_health_monitor(self) -> None:
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _health_monitor_loop(self) -> None:
        """Background loop marking modules unhealthy when TTL exceeded."""
        logger.debug("health monitor started (interval=%s)", self._health_check_interval)
        while not self._stop_monitor.is_set():
            now = time.time()
            changed = []
            with self._lock:
                for rec in list(self._records.values()):
                    if rec.status == "deregistered":
                        continue
                    if rec.ttl is None:
                        continue
                    age = now - rec.last_heartbeat
                    if rec.status == "healthy" and age > rec.ttl:
                        rec.status = "unhealthy"
                        logger.info("module %s marked UNHEALTHY (age=%.1f ttl=%.1f)", rec.name, age, rec.ttl)
                        changed.append(rec)
                    elif rec.status == "unhealthy" and age <= rec.ttl:
                        rec.status = "healthy"
                        logger.info("module %s marked HEALTHY again (age=%.1f ttl=%.1f)", rec.name, age, rec.ttl)
                        changed.append(rec)
            for rec in changed:
                self._notify("health_change", rec)
            # Small sleep with stoppable wait
            self._stop_monitor.wait(self._health_check_interval)
        logger.debug("health monitor stopped")

    # -------------------------
    # Query / discovery
    # -------------------------
    def list_all(self) -> List[str]:
        with self._lock:
            return list(self._records.keys())

    def find_by_capability(
        self,
        capability: str,
        min_priority: Optional[int] = None,
        tags: Optional[Iterable[str]] = None,
        healthy_only: bool = True,
    ) -> List[ModuleRecord]:
        """
        Return modules that advertise a capability, optionally filtered by tags and priority.
        Results sorted by priority desc, then registered_at asc.
        """
        tags_set = set(tags) if tags else None
        with self._lock:
            candidates = [
                rec
                for rec in self._records.values()
                if capability in rec.capabilities
            ]
            if healthy_only:
                candidates = [r for r in candidates if r.status == "healthy"]
            if min_priority is not None:
                candidates = [r for r in candidates if r.priority >= min_priority]
            if tags_set is not None:
                candidates = [r for r in candidates if tags_set.issubset(set(r.tags))]
            candidates.sort(key=lambda r: (-r.priority, r.registered_at))
            return list(candidates)

    def find_best_match(
        self,
        capability: str,
        tags: Optional[Iterable[str]] = None,
        healthy_only: bool = True,
    ) -> Optional[ModuleRecord]:
        """Return single best module record matching capability (highest priority)."""
        lst = self.find_by_capability(capability, tags=tags, healthy_only=healthy_only)
        return lst[0] if lst else None

    def find_by_tag(self, tag: str, healthy_only: bool = True) -> List[ModuleRecord]:
        with self._lock:
            candidates = [r for r in self._records.values() if tag in r.tags]
            if healthy_only:
                candidates = [r for r in candidates if r.status == "healthy"]
            candidates.sort(key=lambda r: (-r.priority, r.registered_at))
            return candidates

    # -------------------------
    # Event subscriptions
    # -------------------------
    def subscribe(self, event: str, callback: Callable[[ModuleRecord], None]) -> None:
        """
        Subscribe to registry events:
          - 'register', 'unregister', 'update', 'health_change'
        Callback will be called with ModuleRecord (in its current state).
        """
        if not callable(callback):
            raise RegistryError("callback must be callable")
        with self._lock:
            self._event_subscribers.setdefault(event, []).append(callback)
        logger.debug("subscriber added for %s", event)

    def unsubscribe(self, event: str, callback: Callable[[ModuleRecord], None]) -> None:
        with self._lock:
            subs = self._event_subscribers.get(event, [])
            if callback in subs:
                subs.remove(callback)

    def _notify(self, event: str, rec: ModuleRecord) -> None:
        subs = []
        with self._lock:
            subs = list(self._event_subscribers.get(event, []))
        for cb in subs:
            try:
                cb(rec)
            except Exception as e:
                logger.exception("subscriber for %s failed: %s", event, e)

    # -------------------------
    # Utilities & cleanup
    # -------------------------
    def export_snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the registry (for diagnostics)."""
        with self._lock:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(self._records),
                "modules": {name: {k: v for k, v in asdict(rec).items() if k != "handler_callable"} for name, rec in self._records.items()},
            }

    def shutdown(self, persist: bool = True) -> None:
        """Stop background threads and optionally persist state."""
        logger.debug("shutting down registry")
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if persist:
            try:
                self.persist()
            except Exception:
                logger.exception("failed to persist registry during shutdown")

    # -------------------------
    # Context manager
    # -------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    # -------------------------
    # Representation
    # -------------------------
    def __repr__(self) -> str:
        return f"<Registry modules={len(self._records)} persist={self.persist_path}>"

# =========================
# Standalone demo / tests
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    reg = Registry(start_health_monitor=True, health_check_interval=2.0)

    # Example handler (callable)
    def dummy_handler(pkt):
        return f"got:{pkt}"

    # Register a few modules
    reg.register("lang.v1", handler=dummy_handler, capabilities=["nlp", "chat"], tags=["core"], priority=10, ttl=5.0)
    reg.register("math.v1", handler="math:pow", capabilities=["math"], tags=["core"], priority=8, ttl=10.0)

    print("All modules:", reg.list_all())
    print("Find nlp:", [r.name for r in reg.find_by_capability("nlp")])
    print("Find best math:", reg.find_best_match("math"))

    # subscribe to health changes
    def on_health(rec: ModuleRecord):
        print("[EVENT] health change:", rec.name, rec.status)
    reg.subscribe("health_change", on_health)

    # simulate heartbeat
    time.sleep(3)
    reg.heartbeat("lang.v1")
    time.sleep(6)

    # Export snapshot
    print("Snapshot:", json.dumps(reg.export_snapshot(), indent=2))

    reg.shutdown()
