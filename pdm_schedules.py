"""Cosine ramp for the PDM directional KL loss."""

from __future__ import annotations

import math







def cosine_ramp_weight(
    iteration: int,
    maximum: float,
    warmup_end: int,
    ramp_end: int,
) -> float:
    """Zero warm-up followed by a half-cosine ramp to ``maximum``."""

    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if warmup_end < 0 or ramp_end < warmup_end:
        raise ValueError("require 0 <= warmup_end <= ramp_end")
    if iteration <= warmup_end:
        return 0.0
    if iteration >= ramp_end or ramp_end == warmup_end:
        return float(maximum)
    progress = (iteration - warmup_end) / float(ramp_end - warmup_end)
    return float(maximum) * 0.5 * (1.0 - math.cos(math.pi * progress))

