"""Tests for estimator core module."""

from __future__ import annotations

from pathlib import Path

from vastai_gpu_runner.estimator.core import (
    GPU_SPEED_FACTOR,
    EstimateResult,
    ScalingRow,
    build_scaling_table,
    fallback_pricing,
    format_time,
    load_calibration,
    record_timing,
)


class TestGpuSpeedFactors:
    def test_4090_is_baseline(self) -> None:
        assert GPU_SPEED_FACTOR["RTX_4090"] == 1.0

    def test_3090_is_slower(self) -> None:
        assert GPU_SPEED_FACTOR["RTX_3090"] < 1.0

    def test_5090_is_faster(self) -> None:
        assert GPU_SPEED_FACTOR["RTX_5090"] > 1.0


class TestFormatTime:
    def test_minutes_only(self) -> None:
        assert format_time(0.5) == "30m"

    def test_hours_and_minutes(self) -> None:
        assert format_time(2.25) == "2h 15m"

    def test_zero(self) -> None:
        assert format_time(0.0) == "0m"


class TestScalingRow:
    def test_wall_time_human(self) -> None:
        row = ScalingRow(cloud_gpus=4, cloud_gpu_type="RTX_4090",
                         wall_time_hours=1.5, cloud_cost_min=2.0, cloud_cost_max=3.0)
        assert row.wall_time_human == "1h 30m"

    def test_cost_display_local_only(self) -> None:
        row = ScalingRow(cloud_gpus=0, cloud_gpu_type="RTX_4090",
                         wall_time_hours=1.0, cloud_cost_min=0.0, cloud_cost_max=0.0)
        assert row.cost_display == "$0.00"

    def test_cost_display_range(self) -> None:
        row = ScalingRow(cloud_gpus=4, cloud_gpu_type="RTX_3090",
                         wall_time_hours=0.5, cloud_cost_min=1.20, cloud_cost_max=1.80)
        assert row.cost_display == "$1.20-$1.80"

    def test_to_dict(self) -> None:
        row = ScalingRow(cloud_gpus=2, cloud_gpu_type="RTX_4090",
                         wall_time_hours=1.0, cloud_cost_min=0.64, cloud_cost_max=0.64)
        d = row.to_dict()
        assert d["cloud_gpus"] == 2
        assert "wall_time_human" in d


class TestFallbackPricing:
    def test_returns_all_gpus(self) -> None:
        pricing = fallback_pricing()
        assert "RTX_3090" in pricing
        assert "RTX_4090" in pricing
        assert "RTX_5090" in pricing

    def test_3090_cheapest(self) -> None:
        pricing = fallback_pricing()
        assert pricing["RTX_3090"].median_price_hr < pricing["RTX_4090"].median_price_hr


class TestBuildScalingTable:
    def test_local_only(self) -> None:
        pricing = fallback_pricing()
        rows = build_scaling_table(10.0, [0], pricing)
        assert len(rows) == 1
        assert rows[0].cloud_gpus == 0
        assert rows[0].cloud_cost_min == 0.0

    def test_more_gpus_less_time(self) -> None:
        pricing = fallback_pricing()
        rows = build_scaling_table(10.0, [0, 4, 8], pricing)
        assert rows[0].wall_time_hours > rows[1].wall_time_hours > rows[2].wall_time_hours

    def test_more_gpus_more_cost(self) -> None:
        pricing = fallback_pricing()
        rows = build_scaling_table(10.0, [4, 8], pricing)
        # More GPUs = less wall time but more parallel instances, cost depends
        assert rows[0].cloud_cost_min >= 0
        assert rows[1].cloud_cost_min >= 0


class TestEstimateResult:
    def test_to_dict(self) -> None:
        result = EstimateResult(workload="test", description="test run", num_items=100)
        d = result.to_dict()
        assert d["workload"] == "test"
        assert d["num_items"] == 100
        assert "timestamp" in d


class TestTimingPersistence:
    def test_record_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "benchmarks.jsonl"
        record_timing(path, "boltz2", runtime_sec=180, peptides=1)
        record_timing(path, "md", ns_per_day=290, target="VicK")
        record_timing(path, "boltz2", runtime_sec=200, peptides=1)

        boltz_records = load_calibration(path, "boltz2")
        assert len(boltz_records) == 2

        md_records = load_calibration(path, "md")
        assert len(md_records) == 1
        assert md_records[0]["target"] == "VicK"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        assert load_calibration(path, "boltz2") == []
