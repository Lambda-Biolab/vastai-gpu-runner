"""Tests for hybrid work splitting."""

from __future__ import annotations

from vastai_gpu_runner.hybrid import compute_hybrid_split
from vastai_gpu_runner.types import ComputeMode


class TestComputeHybridSplit:
    def test_local_only(self) -> None:
        split = compute_hybrid_split(100, cloud_gpus=0)
        assert split.mode == ComputeMode.LOCAL
        assert split.local_items == 100
        assert split.cloud_items == 0
        assert split.items_per_shard == []

    def test_cloud_only(self) -> None:
        split = compute_hybrid_split(100, cloud_gpus=4, mode=ComputeMode.CLOUD)
        assert split.mode == ComputeMode.CLOUD
        assert split.local_items == 0
        assert split.cloud_items == 100
        assert len(split.items_per_shard) == 4
        assert sum(split.items_per_shard) == 100

    def test_hybrid_splits_proportionally(self) -> None:
        # 1 local RTX 4090 (speed=1.0) + 4 cloud RTX 3090 (speed=0.77 each)
        # Effective: 1.0 + 4*0.77 = 4.08
        # Local share: 1.0/4.08 = ~24.5% -> ceil(100*0.245) = 25
        split = compute_hybrid_split(
            100, cloud_gpus=4, cloud_gpu_type="RTX_3090", local_gpu_type="RTX_4090"
        )
        assert split.mode == ComputeMode.HYBRID
        assert split.local_items > 0
        assert split.cloud_items > 0
        assert split.local_items + split.cloud_items == 100
        assert len(split.items_per_shard) == 4
        assert sum(split.items_per_shard) == split.cloud_items

    def test_hybrid_same_gpu_equal_split(self) -> None:
        # Same GPU type: 1 local + 4 cloud = 5 total, each gets 20%
        split = compute_hybrid_split(
            100, cloud_gpus=4, cloud_gpu_type="RTX_4090", local_gpu_type="RTX_4090"
        )
        assert split.local_items == 20  # ceil(100/5)
        assert split.cloud_items == 80

    def test_auto_mode_selects_hybrid(self) -> None:
        split = compute_hybrid_split(50, cloud_gpus=2)
        assert split.mode == ComputeMode.HYBRID

    def test_auto_mode_selects_local(self) -> None:
        split = compute_hybrid_split(50, cloud_gpus=0)
        assert split.mode == ComputeMode.LOCAL

    def test_single_item_hybrid(self) -> None:
        split = compute_hybrid_split(1, cloud_gpus=4)
        assert split.total_items == 1
        assert split.local_items + split.cloud_items == 1

    def test_shard_distribution_even(self) -> None:
        split = compute_hybrid_split(100, cloud_gpus=4, mode=ComputeMode.CLOUD)
        assert split.items_per_shard == [25, 25, 25, 25]

    def test_shard_distribution_uneven(self) -> None:
        split = compute_hybrid_split(10, cloud_gpus=3, mode=ComputeMode.CLOUD)
        assert sum(split.items_per_shard) == 10
        assert split.items_per_shard == [4, 3, 3]
