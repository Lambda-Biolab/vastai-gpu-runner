"""Hybrid work splitting — divide items between local GPU and cloud shards.

Computes how many items the local GPU should process vs how many go to
cloud, accounting for GPU speed differences. The consuming project wires
up its own local worker using the split.

Usage::

    from vastai_gpu_runner.hybrid import HybridSplit, compute_hybrid_split

    split = compute_hybrid_split(
        total_items=200,
        cloud_gpus=8,
        cloud_gpu_type="RTX_3090",
        local_gpu_type="RTX_4090",
    )
    print(f"Local: {split.local_items}, Cloud: {split.cloud_items}")
    print(f"Items per shard: {split.items_per_shard}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vastai_gpu_runner.estimator.core import GPU_SPEED_FACTOR
from vastai_gpu_runner.types import ComputeMode


@dataclass
class HybridSplit:
    """Result of hybrid work splitting.

    Attributes:
        mode: Compute mode used.
        total_items: Total items to process.
        local_items: Items assigned to local GPU.
        cloud_items: Items assigned to cloud GPUs.
        cloud_gpus: Number of cloud GPU instances.
        items_per_shard: Items per cloud shard (list, one per shard).
        local_gpu_type: Local GPU model.
        cloud_gpu_type: Cloud GPU model.
    """

    mode: ComputeMode
    total_items: int
    local_items: int
    cloud_items: int
    cloud_gpus: int
    items_per_shard: list[int] = field(default_factory=list)
    local_gpu_type: str = "RTX_4090"
    cloud_gpu_type: str = "RTX_3090"


def compute_hybrid_split(
    total_items: int,
    cloud_gpus: int,
    *,
    mode: ComputeMode | None = None,
    cloud_gpu_type: str = "RTX_3090",
    local_gpu_type: str = "RTX_4090",
) -> HybridSplit:
    """Compute optimal work split between local GPU and cloud shards.

    Splits items proportionally by GPU speed. In hybrid mode, the local
    GPU gets ``ceil(total / effective_gpus)`` items; the rest go to cloud
    shards distributed evenly.

    Args:
        total_items: Total items to process.
        cloud_gpus: Number of cloud GPU instances.
        mode: Compute mode. If None, auto-selects based on cloud_gpus
            (0 = LOCAL, >0 = HYBRID).
        cloud_gpu_type: Cloud GPU model for speed factor lookup.
        local_gpu_type: Local GPU model for speed factor lookup.

    Returns:
        HybridSplit with item assignments.
    """
    if mode is None:
        mode = ComputeMode.LOCAL if cloud_gpus == 0 else ComputeMode.HYBRID

    local_speed = GPU_SPEED_FACTOR.get(local_gpu_type, 1.0)
    cloud_speed = GPU_SPEED_FACTOR.get(cloud_gpu_type, 1.0)

    if mode == ComputeMode.LOCAL or cloud_gpus == 0:
        return HybridSplit(
            mode=ComputeMode.LOCAL,
            total_items=total_items,
            local_items=total_items,
            cloud_items=0,
            cloud_gpus=0,
            items_per_shard=[],
            local_gpu_type=local_gpu_type,
            cloud_gpu_type=cloud_gpu_type,
        )

    if mode == ComputeMode.CLOUD:
        items_per_shard = _distribute_items(total_items, cloud_gpus)
        return HybridSplit(
            mode=ComputeMode.CLOUD,
            total_items=total_items,
            local_items=0,
            cloud_items=total_items,
            cloud_gpus=cloud_gpus,
            items_per_shard=items_per_shard,
            local_gpu_type=local_gpu_type,
            cloud_gpu_type=cloud_gpu_type,
        )

    # Hybrid: split proportionally by speed
    effective_total_speed = local_speed + cloud_gpus * cloud_speed
    local_share = local_speed / effective_total_speed
    local_items = math.ceil(total_items * local_share)
    cloud_items = total_items - local_items

    # Edge case: if local would get everything, give at least 1 to cloud
    if cloud_items == 0 and total_items > 1:
        local_items = total_items - 1
        cloud_items = 1

    items_per_shard = _distribute_items(cloud_items, cloud_gpus)

    return HybridSplit(
        mode=ComputeMode.HYBRID,
        total_items=total_items,
        local_items=local_items,
        cloud_items=cloud_items,
        cloud_gpus=cloud_gpus,
        items_per_shard=items_per_shard,
        local_gpu_type=local_gpu_type,
        cloud_gpu_type=cloud_gpu_type,
    )


def _distribute_items(total: int, n_shards: int) -> list[int]:
    """Distribute items evenly across shards (larger shards first)."""
    if n_shards <= 0:
        return []
    base = total // n_shards
    remainder = total % n_shards
    return [base + (1 if i < remainder else 0) for i in range(n_shards)]
