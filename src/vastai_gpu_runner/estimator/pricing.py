"""Live Vast.ai marketplace pricing queries.

Queries the Vast.ai CLI for current GPU availability and pricing,
with fallback to static prices when the CLI is unavailable.
"""

from __future__ import annotations

import logging
import statistics

from vastai_gpu_runner.estimator.core import (
    FALLBACK_PRICES,
    GPU_TYPES,
    PriceSummary,
)
from vastai_gpu_runner.providers.vastai import VastaiRunner
from vastai_gpu_runner.types import DeploymentConfig

logger = logging.getLogger(__name__)


def query_vastai_pricing(
    gpu_types: list[str] | None = None,
    *,
    max_cost_per_hour: float = 0.45,
    min_network_mbps: int = 800,
    min_reliability: float = 0.995,
) -> dict[str, PriceSummary]:
    """Query live Vast.ai marketplace pricing for GPU types.

    Args:
        gpu_types: GPU types to query. Defaults to all three.
        max_cost_per_hour: Maximum $/hr filter.
        min_network_mbps: Minimum download bandwidth.
        min_reliability: Minimum host reliability score.

    Returns:
        Dict mapping GPU type to PriceSummary.
        Falls back to hardcoded prices if Vast.ai CLI is unavailable.
    """
    if gpu_types is None:
        gpu_types = list(GPU_TYPES)

    results: dict[str, PriceSummary] = {}

    for gpu_type in gpu_types:
        config = DeploymentConfig(
            gpu_model=gpu_type,
            max_cost_per_hour=max_cost_per_hour,
            min_network_mbps=min_network_mbps,
            min_reliability=min_reliability,
        )
        runner = VastaiRunner(config)
        offers = runner.search_offers()

        if offers:
            prices = [float(str(o.get("dph_total", 0.0))) for o in offers if o.get("dph_total")]
            if prices:
                results[gpu_type] = PriceSummary(
                    gpu_type=gpu_type,
                    available_count=len(prices),
                    min_price_hr=min(prices),
                    max_price_hr=max(prices),
                    median_price_hr=statistics.median(prices),
                )
                continue

        # Fallback to static prices
        fb = FALLBACK_PRICES.get(gpu_type, 0.30)
        results[gpu_type] = PriceSummary(
            gpu_type=gpu_type,
            available_count=0,
            min_price_hr=fb,
            max_price_hr=fb,
            median_price_hr=fb,
        )

    return results
