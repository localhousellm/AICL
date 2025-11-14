# AICL/router.py
# Author: Vansh Bukkarwal ( A localhousellm project )
# Date: 2025-11-06
#
# AICL Router / Communicator (production-grade)
#
# Purpose:
#   - Route AICLPacket objects between registered modules (via Registry).
#   - Provide request/response, broadcast, multicast semantics.
#   - Honor module async/sync preferences, timeouts, retries, backoff.
#   - Integrate pre-route/post-route hooks for safety/inspection.
#   - Emit lightweight metrics and events; integrate with Registry health.
#
# Design notes:
#   - Router relies on AICL.packet.AICLPacket and AICL.registry.Registry.
#   - Modules are expected to be registered in Registry. The router will
#     look up ModuleRecord.handler_callable (or import from handler_path).
#   - Router does not accept raw callables at register time; use Registry.register.
#   - Router executes handlers using ThreadPoolExecutor and returns normalized AICLPacket responses.
#
# Usage:
#   from AICL.registry import Registry
#   from AICL.packet import AICLPacket
#   r = Router(registry=my_registry)
#   pkt = AICLPacket(origin="user", symbols=["hello"], metadata={"text":"hi"})
#   r.send(pkt, targets=["lang.v1"])
#
# Notes:
#   - This file intentionally includes comprehensive error handling and logs.
#   - Designed to be testable and to integrate with observability / metrics systems.
#

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Local package imports (AICL)
from AICL.packet import AICLPacket, AICLPacketError
from AICL.registry import Registry, ModuleRecord, RegistryError

logger = logging.getLogger("AICL.router")
logger.addHandler(logging.NullHandler())

# Type aliases
PacketHandler = Callable[[AICLPacket], Union[AICLPacket, List[AICLPacket], None]]
PreRouteHook = Callable[[AICLPacket], AICLPacket]  # may modify packet
PostRouteHook = Callable[[AICLPacket, Optional[AICLPacket]], Optional[AICLPacket]]
MetricsHook = Callable[[str, Dict[str, Any]], None]  # event name, payload


class RouterError(Exception):
    """Router-level error wrapper."""


class Router:
    """
    Router coordinates sending AICLPackets to registered modules discovered from Registry.

    Key features:
      - health-aware target selection (uses registry status)
      - per-request timeout, retries, and exponential backoff
      - synchronous and asynchronous module support
      - ability to broadcast, multicast, and targeted send
      - optional pre/post routing hooks for safety/inspection
      - metrics hook for instrumentation
    """

    def __init__(
        self,
        registry: Optional[Registry] = None,
        max_workers: int = 16,
        default_timeout: float = 5.0,
        default_retries: int = 0,
        default_backoff: float = 0.1,
        pre_route_hook: Optional[PreRouteHook] = None,
        post_route_hook: Optional[PostRouteHook] = None,
        metrics_hook: Optional[MetricsHook] = None,
    ):
        self.registry = registry or Registry()  # registry will start its own health monitor
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self.default_timeout = float(default_timeout)
        self.default_retries = int(default_retries)
        self.default_backoff = float(default_backoff)
        self.pre_route_hook = pre_route_hook
        self.post_route_hook = post_route_hook
        self.metrics_hook = metrics_hook

        # counters
        self._metrics: Dict[str, int] = {"sent": 0, "responses": 0, "errors": 0, "timeouts": 0}
        logger.info("Router initialized (workers=%d timeout=%s)", max_workers, default_timeout)

    # -----------------------
    # Low-level helpers
    # -----------------------
    def _get_handler_for_module(self, module_name: str) -> PacketHandler:
        """
        Return a callable handler for the module. This uses the Registry.ModuleRecord
        to obtain handler_callable or import from handler_path.
        """
        try:
            rec: ModuleRecord = self.registry.get(module_name)
        except Exception as e:
            raise RouterError(f"module '{module_name}' not found in registry: {e}")

        handler = rec.handler_callable
        if handler is None and rec.handler_path:
            try:
                handler = self.registry._import_callable_by_path(rec.handler_path)  # reuse registry import util
                rec.handler_callable = handler  # cache in runtime
            except Exception as e:
                logger.exception("failed to import handler for module %s: %s", module_name, e)
                raise RouterError(f"failed to import handler for module '{module_name}': {e}")
        if not callable(handler):
            raise RouterError(f"module '{module_name}' has no callable handler available")
        return handler  # type: ignore

    def _safe_invoke_handler(self, module_name: str, handler: PacketHandler, packet: AICLPacket) -> Optional[AICLPacket]:
        """
        Calls handler(packet), normalizes result to AICLPacket or None.
        Catches exceptions and returns an error packet (with metadata) for visibility.
        """
        try:
            raw_ret = handler(packet)
            # None -> side-effect, already enqueued or async
            if raw_ret is None:
                return None
            # Single AICLPacket
            if isinstance(raw_ret, AICLPacket):
                raw_ret.stamp(module_name, "handled")
                return raw_ret
            # List of packets -> merge into single packet
            if isinstance(raw_ret, list):
                merged: Optional[AICLPacket] = None
                for item in raw_ret:
                    if not isinstance(item, AICLPacket):
                        # if module returns mixed types, encapsulate non-packets as metadata
                        pkt = AICLPacket(origin=module_name, symbols=[], metadata={"raw": str(item)})
                        pkt.stamp(module_name, "wrapped")
                        item = pkt
                    if merged is None:
                        merged = item.shallow_copy()
                    else:
                        merged = merged.merge(item, prefer_self=True)
                if merged:
                    merged.stamp(module_name, "merged")
                return merged
            # Unexpected return type -> wrap into a packet
            pkt = AICLPacket(origin=module_name, symbols=[], metadata={"raw_result": str(raw_ret)})
            pkt.stamp(module_name, "wrapped_nonpacket")
            return pkt
        except Exception as e:
            logger.exception("module handler exception (%s): %s", module_name, e)
            # error packet for visibility
            err = AICLPacket(origin="router", symbols=[], metadata={"error": str(e)})
            err.stamp("router", "handler_exception", note=f"module={module_name}")
            return err

    # -----------------------
    # Target selection
    # -----------------------
    def _select_targets(self, packet: AICLPacket, explicit_targets: Optional[Iterable[str]] = None, healthy_only: bool = True) -> List[str]:
        """
        Determine modules to route to:
          - explicit_targets (if provided and registered)
          - packet.targets
          - capability-based (if packet.capability provided)
          - otherwise all registered modules
        Preference order: explicit_targets -> packet.targets -> capability -> registry preferred (high priority)
        """
        with self._lock:
            registered = set(self.registry.list_all())

        targets: List[str] = []
        if explicit_targets:
            for t in explicit_targets:
                if t in registered:
                    targets.append(t)
            if targets:
                return targets

        # Use packet.targets if set
        if packet.targets:
            for t in packet.targets:
                if t in registered and t not in targets:
                    targets.append(t)
            if targets:
                return targets

        # If packet requests a capability, consult registry.find_by_capability
        if packet.intent and isinstance(packet.intent, str):
            try:
                candidates = self.registry.find_by_capability(packet.intent, healthy_only=healthy_only)
                targets = [c.name for c in candidates]
                if targets:
                    return targets
            except Exception:
                # ignore and fallback
                logger.debug("capability lookup failed for intent=%s", packet.intent)

        # final fallback: all registered (apply healthy_only filter)
        if healthy_only:
            targets = [n for n in registered if self.registry.get(n).status == "healthy"]
        else:
            targets = list(registered)
        return targets

    # -----------------------
    # Send API
    # -----------------------
    def send(
        self,
        packet: AICLPacket,
        targets: Optional[Iterable[str]] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        backoff: Optional[float] = None,
        healthy_only: bool = True,
    ) -> Dict[str, Union[AICLPacket, Exception, None]]:
        """
        Send a packet to resolved targets. Returns mapping name -> response|Exception|None (async/dropped).
        - packet.validate(raise_on_error=True) will be used by user prior to send; we'll attempt a shallow validation.
        """
        try:
            packet.validate(raise_on_error=True)
        except AICLPacketError as e:
            raise RouterError(f"invalid packet: {e}")

        # pre-route hook
        if self.pre_route_hook:
            try:
                packet = self.pre_route_hook(packet)
            except Exception as e:
                logger.exception("pre_route_hook failed: %s", e)
                self._metrics["errors"] += 1
                raise RouterError(f"pre_route_hook failed: {e}") from e

        # pick targets
        target_list = self._select_targets(packet, explicit_targets=targets, healthy_only=healthy_only)
        if not target_list:
            logger.debug("no targets resolved for packet: %s", packet.pretty())
            return {}

        timeout = float(timeout) if timeout is not None else self.default_timeout
        retries = int(retries) if retries is not None else self.default_retries
        backoff = float(backoff) if backoff is not None else self.default_backoff

        self._metrics["sent"] += 1
        futures: Dict[str, Future] = {}
        results: Dict[str, Union[AICLPacket, Exception, None]] = {}

        # Submit tasks
        for name in target_list:
            try:
                rec = self.registry.get(name)
            except RegistryError as e:
                results[name] = RouterError(str(e))
                continue

            # obtain handler
            try:
                handler = self._get_handler_for_module(name)
            except RouterError as e:
                results[name] = e
                self._metrics["errors"] += 1
                continue

            # shallow copy packet to avoid cross-module side effects
            pkt = packet.shallow_copy()
            pkt.stamp("router", f"dispatch_to:{name}")

            # if the module is marked async by registry, we schedule but may not block
            if rec.is_async and not wait:
                # fire-and-forget: push to inbox (registered in ModuleRecord) if available and schedule
                try:
                    rec.push_inbox(pkt)  # ModuleRegistration-like behaviour expected; Registry may not have inbox. If not, skip.
                except Exception:
                    # If registry record lacks inbox semantics, just log
                    logger.debug("registry record for %s has no inbox; continuing", name)
                self._executor.submit(self._safe_invoke_handler, name, handler, pkt)
                results[name] = None
                continue

            # schedule and collect futures
            fut = self._executor.submit(self._safe_invoke_handler, name, handler, pkt)
            futures[name] = fut

        # Collect results
        if wait:
            for name, fut in futures.items():
                attempt = 0
                last_exc: Optional[Exception] = None
                while attempt <= retries:
                    try:
                        res_packet = fut.result(timeout=timeout)
                        self._metrics["responses"] += 1
                        # post-route hook
                        if self.post_route_hook:
                            try:
                                res_packet = self.post_route_hook(packet, res_packet)
                            except Exception as e:
                                logger.exception("post_route_hook failed for %s: %s", name, e)
                        results[name] = res_packet
                        break
                    except FuturesTimeout:
                        logger.warning("timeout waiting for module '%s' (t=%s)", name, timeout)
                        self._metrics["timeouts"] += 1
                        last_exc = RouterError(f"timeout after {timeout}s")
                        attempt += 1
                        if attempt <= retries:
                            sleep_time = backoff * (2 ** (attempt - 1))
                            time.sleep(sleep_time)
                            continue
                        results[name] = last_exc
                        break
                    except Exception as e:
                        logger.exception("handler exception for %s: %s", name, e)
                        self._metrics["errors"] += 1
                        last_exc = e
                        attempt += 1
                        if attempt <= retries:
                            sleep_time = backoff * (2 ** (attempt - 1))
                            time.sleep(sleep_time)
                            continue
                        results[name] = last_exc
                        break
        else:
            # non-wait: return None for those not immediate
            for name, fut in futures.items():
                if fut.done():
                    try:
                        res_packet = fut.result()
                        self._metrics["responses"] += 1
                        if self.post_route_hook:
                            try:
                                res_packet = self.post_route_hook(packet, res_packet)
                            except Exception as e:
                                logger.exception("post_route_hook failed for %s: %s", name, e)
                        results[name] = res_packet
                    except Exception as e:
                        results[name] = e
                        self._metrics["errors"] += 1
                else:
                    results[name] = None

        # metrics hook
        if self.metrics_hook:
            try:
                self.metrics_hook("send_complete", {"targets": list(results.keys()), "outcomes": {k: type(v).__name__ for k, v in results.items()}})
            except Exception:
                logger.exception("metrics_hook failed")

        return results

    # Convenience wrapper for single target request/response
    def request_response(
        self,
        packet: AICLPacket,
        target: str,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        backoff: Optional[float] = None,
    ) -> AICLPacket:
        """
        Send and wait for a single response from a target. Raises RouterError on failure.
        """
        res_map = self.send(packet, targets=[target], wait=True, timeout=timeout, retries=retries, backoff=backoff)
        res = res_map.get(target)
        if res is None:
            raise RouterError(f"No response from target '{target}' (async or dropped)")
        if isinstance(res, Exception):
            raise RouterError(f"error from target '{target}': {res}")
        if not isinstance(res, AICLPacket):
            raise RouterError(f"unexpected response type from '{target}': {type(res)}")
        return res

    # Broadcast helper
    def broadcast(self, packet: AICLPacket, exclude: Optional[Iterable[str]] = None, wait: bool = False, timeout: Optional[float] = None) -> Dict[str, Union[AICLPacket, Exception, None]]:
        exclude_set = set(exclude) if exclude else set()
        with self._lock:
            targets = [n for n in self.registry.list_all() if n not in exclude_set and self.registry.get(n).status == "healthy"]
        return self.send(packet, targets=targets, wait=wait, timeout=timeout)

    # Fetch inbox: delegate to registry ModuleRecord if supports inbox semantics
    def fetch_inbox(self, module_name: str) -> List[AICLPacket]:
        try:
            rec = self.registry.get(module_name)
        except Exception as e:
            raise RouterError(str(e))
        # ModuleRecord might not implement push_inbox/pop_all_inbox; be defensive
        inbox = []
        try:
            pop = getattr(rec, "pop_all_inbox", None)
            if callable(pop):
                inbox = pop()
            else:
                # no inbox feature
                inbox = []
        except Exception as e:
            logger.exception("failed to fetch inbox for %s: %s", module_name, e)
            raise RouterError(str(e))
        return inbox

    # Shutdown
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        try:
            self._executor.shutdown(wait=wait, timeout=timeout)
        except TypeError:
            # older Python versions
            self._executor.shutdown(wait=wait)
        logger.info("router shutdown complete")

    # expose metrics snapshot
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)


# =========================
# Standalone demo / tests
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("AICL.router")

    # Build a small registry and register demo handlers
    reg = Registry(start_health_monitor=False)  # disable health monitor for demo simplicity

    # example handlers
    def echo_handler(pkt: AICLPacket) -> AICLPacket:
        text = " ".join(pkt.symbols) if pkt.symbols else pkt.metadata.get("text", "")
        resp = AICLPacket(origin="echo", symbols=[f"ECHO:{text}"])
        resp.stamp("echo", "echoed")
        return resp

    def async_logger(pkt: AICLPacket) -> None:
        # simulate an async module that persists messages and doesn't return a packet
        logger.debug("[async_logger] stored: %s", pkt.pretty())
        return None

    # Register handlers via registry
    reg.register("echo", handler=echo_handler, capabilities=["utility.echo"], is_async=False, ttl=30.0)
    reg.register("logger", handler=async_logger, capabilities=["logging"], is_async=True, ttl=30.0)

    router = Router(registry=reg, max_workers=4, default_timeout=2.0, default_retries=1, default_backoff=0.2)

    pkt = AICLPacket(origin="user", symbols=["hello", "world"], intent="utility.echo", confidence=0.9)
    pkt.stamp("user", "created")

    print("\n-- single target request_response -->")
    res = router.request_response(pkt, target="echo")
    print("response.pretty():", res.pretty())

    print("\n-- broadcast (non-blocking) -->")
    br = router.broadcast(pkt, wait=False)
    print("broadcast results:", {k: type(v).__name__ for k, v in br.items()})

    print("\n-- send with explicit targets -->")
    s = router.send(pkt, targets=["echo", "logger"], wait=True)
    print("send results:", {k: (type(v).__name__ if v is not None else "None") for k, v in s.items()})

    router.shutdown()
    reg.shutdown()
