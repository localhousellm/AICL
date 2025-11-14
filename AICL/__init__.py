# AICL/__init__.py
#
# Adaptive Inter-Module Communication Language (AICL)
# Part of the LocalHouseLLM / AMA Modular AI Project
#
# SPDX-License-Identifier: MPL-2.0
#
# Copyright (c) 2025
# Vansh Bukkarwal — LocalHouseLLM Project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/

"""
AICL — Adaptive Inter-Module Communication Language
===================================================

AICL is a high-performance symbolic communication protocol designed for
modular AI systems such as ANVIRA / AMA-MRL.

It provides:
    • AICLPacket — The core message format used between modules
    • Router — A concurrent routing engine for modules
    • Registry — Persistent module metadata management
    • Safety — Validation and enforcement policies
    • Utils — Helper utilities for hashing, timestamps, formatting, etc.

AICL is designed to be:
    - Minimal yet expressive
    - Fast to encode/decode for AI modules
    - Extensible for research and production use
    - Fully open-source under MPL-2.0
"""

from AICL.packet import AICLPacket, AICLPacketError
from AICL.router import Router, RouterError
from AICL.registry import ModuleRegistry, RegistryError
from AICL.safety import SafetyRules, SafetyViolation
from AICL.utils.helpers import (
    generate_packet_id,
    now_ms,
    deep_freeze,
    deep_unfreeze,
)

__all__ = [
    # Packet
    "AICLPacket",
    "AICLPacketError",

    # Router
    "Router",
    "RouterError",

    # Registry
    "ModuleRegistry",
    "RegistryError",

    # Safety
    "SafetyRules",
    "SafetyViolation",

    # Utils
    "generate_packet_id",
    "now_ms",
    "deep_freeze",
    "deep_unfreeze",
]

# Package version
__version__ = "1.0.0"
