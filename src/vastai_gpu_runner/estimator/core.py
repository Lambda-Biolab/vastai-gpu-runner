"""Generic GPU cost/time estimation framework.

Provides GPU speed factors, pricing snapshots, scaling table math,
and result formatting. Domain-specific estimation logic (workload
baselines, atom-count models) lives in the consuming project.

Usage::

    from vastai_gpu_runner.estimator.core import (
        GPU_SPEED_FACTOR,
        build_scaling_table,
        EstimateResult,
    )

    rows = build_scaling_table(
        total_work_hours_base=10.0,
        cloud_gpu_counts=[0, 4, 8],
        pricing=pricing,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU speed factors — dimensionless multipliers relative to RTX 4090 = 1.0
# ---------------------------------------------------------------------------

GPU_SPEED_FACTOR: dict[str, float] = {
    "RTX_3090": 0.77,
    "RTX_4090": 1.0,
    "RTX_5090": 1.43,
}

GPU_TYPES: list[str] = ["RTX_3090", "RTX_4090", "RTX_5090"]
LOCAL_GPU_TYPE = "RTX_4090"

# Fallback static cost rates (when Vast.ai CLI is unavailable)
FALLBACK_PRICES: dict[str, float] = {
    "RTX_3090": 0.15,
    "RTX_4090": 0.32,
    "RTX_5090": 0.60,
}


# ---------------------------------------------------------------------------
# Timing persistence
# ---------------------------------------------------------------------------


def record_timing(
    benchmarks_path: Path,
    workload: str,
    **metrics: float | int | str,
) -> None:
    """Append one timing record to a benchmarks JSONL file.

    Args:
        benchmarks_path: Path to the JSONL file.
        workload: Workload identifier (e.g. "boltz2", "md").
        **metrics: Workload-specific metrics.
    """
    benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "workload": workload,
        "ts": datetime.now(tz=UTC).isoformat(),
        **metrics,
    }
    with benchmarks_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_calibration(
    benchmarks_path: Path,
    workload: str,
) -> list[dict[str, object]]:
    """Load timing records for a workload from a benchmarks JSONL file.

    Args:
        benchmarks_path: Path to the JSONL file.
        workload: Workload identifier.

    Returns:
        List of record dicts matching the workload.
    """
    if not benchmarks_path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in benchmarks_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("workload") == workload:
                records.append(rec)
        except json.JSONDecodeError:
            continue
    return records


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass
class PriceSummary:
    """Live pricing snapshot for one GPU type."""

    gpu_type: str
    available_count: int
    min_price_hr: float
    max_price_hr: float
    median_price_hr: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dict."""
        return {
            "gpu_type": self.gpu_type,
            "available_count": self.available_count,
            "min_price_hr": round(self.min_price_hr, 3),
            "max_price_hr": round(self.max_price_hr, 3),
            "median_price_hr": round(self.median_price_hr, 3),
        }


def fallback_pricing(gpu_types: list[str] | None = None) -> dict[str, PriceSummary]:
    """Return hardcoded pricing without querying Vast.ai.

    Args:
        gpu_types: GPU types to include. Defaults to all.

    Returns:
        Dict mapping GPU type to PriceSummary.
    """
    if gpu_types is None:
        gpu_types = list(GPU_TYPES)
    results: dict[str, PriceSummary] = {}
    for gpu_type in gpu_types:
        price = FALLBACK_PRICES.get(gpu_type, 0.30)
        results[gpu_type] = PriceSummary(
            gpu_type=gpu_type,
            available_count=0,
            min_price_hr=price,
            max_price_hr=price,
            median_price_hr=price,
        )
    return results


def cheapest_gpu_type(pricing: dict[str, PriceSummary]) -> str:
    """Return the GPU type with the lowest median price.

    Args:
        pricing: Pricing snapshots.

    Returns:
        GPU type string (e.g. "RTX_3090").
    """
    if not pricing:
        return LOCAL_GPU_TYPE
    return min(pricing, key=lambda k: pricing[k].median_price_hr)


# ---------------------------------------------------------------------------
# Scaling table
# ---------------------------------------------------------------------------


def format_time(hours: float) -> str:
    """Format hours as '2h 15m' or '45m'."""
    h = int(hours)
    m = int((hours - h) * 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


@dataclass
class ScalingRow:
    """One row in the scaling table."""

    cloud_gpus: int
    cloud_gpu_type: str
    wall_time_hours: float
    cloud_cost_min: float
    cloud_cost_max: float
    notes: str = ""
    feasible: bool = True

    @property
    def wall_time_human(self) -> str:
        """Format wall time as '2h 15m' or '45m'."""
        return format_time(self.wall_time_hours)

    @property
    def cost_display(self) -> str:
        """Format cost range as '$1.34-$1.72' or '$0.00'."""
        if self.cloud_gpus == 0:
            return "$0.00"
        if abs(self.cloud_cost_min - self.cloud_cost_max) < 0.01:
            return f"${self.cloud_cost_min:.2f}"
        return f"${self.cloud_cost_min:.2f}-${self.cloud_cost_max:.2f}"

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dict."""
        return {
            "cloud_gpus": self.cloud_gpus,
            "cloud_gpu_type": self.cloud_gpu_type,
            "wall_time_hours": round(self.wall_time_hours, 3),
            "wall_time_human": self.wall_time_human,
            "cloud_cost_min": round(self.cloud_cost_min, 2),
            "cloud_cost_max": round(self.cloud_cost_max, 2),
            "cost_display": self.cost_display,
            "notes": self.notes,
            "feasible": self.feasible,
        }


@dataclass
class EstimateResult:
    """Complete estimation result for a workload."""

    workload: str
    description: str
    num_items: int
    target: str = ""
    local_gpu: str = LOCAL_GPU_TYPE
    local_time_hours: float = 0.0
    pricing: dict[str, PriceSummary] = field(default_factory=dict)
    scaling_table: list[ScalingRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: str = ""
    confidence: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dict for --json output."""
        return {
            "workload": self.workload,
            "description": self.description,
            "num_items": self.num_items,
            "target": self.target,
            "local_gpu": self.local_gpu,
            "local_time_hours": round(self.local_time_hours, 3),
            "pricing": {k: v.to_dict() for k, v in self.pricing.items()},
            "scaling_table": [r.to_dict() for r in self.scaling_table],
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    def to_rich_table(self) -> Table:
        """Render as a Rich Table for terminal display."""
        from rich.table import Table  # pyright: ignore[reportMissingImports]

        table = Table(
            title=f"Scaling table (1 local {self.local_gpu} + N cloud GPUs)",
            show_lines=False,
        )
        table.add_column("Cloud GPUs", style="cyan", justify="right")
        table.add_column("GPU type")
        table.add_column("Wall time", justify="right")
        table.add_column("Est. cost", justify="right", style="green")
        table.add_column("Notes", style="dim")

        for row in self.scaling_table:
            gpu_label = "0 (local)" if row.cloud_gpus == 0 else str(row.cloud_gpus)
            style = "" if row.feasible else "dim strike"
            table.add_row(
                gpu_label,
                row.cloud_gpu_type,
                row.wall_time_human,
                row.cost_display,
                row.notes,
                style=style,
            )

        return table


def build_scaling_table(
    total_work_hours_base: float,
    cloud_gpu_counts: list[int],
    pricing: dict[str, PriceSummary],
    *,
    local_gpu_type: str = LOCAL_GPU_TYPE,
    cloud_gpu_type: str | None = None,
    cost_band: tuple[float, float] = (1.0, 1.0),
) -> list[ScalingRow]:
    """Build a scaling table for a GPU workload.

    Computes wall time and cost for each cloud GPU count, assuming
    1 local GPU + N cloud GPUs working in parallel.

    Args:
        total_work_hours_base: Total work in RTX 4090-equivalent hours.
        cloud_gpu_counts: List of cloud GPU counts to estimate (e.g. [0, 4, 8]).
        pricing: Live or fallback pricing snapshots.
        local_gpu_type: Local GPU type (default RTX 4090).
        cloud_gpu_type: Preferred cloud GPU type. If None, uses cheapest available.
        cost_band: (low_multiplier, high_multiplier) for cost range. Use
            (1.0, 1.0) for point estimates, (0.83, 1.26) for confidence bands.

    Returns:
        List of ScalingRow objects.
    """
    if cloud_gpu_type is None:
        cloud_gpu_type = cheapest_gpu_type(pricing)

    local_speed = GPU_SPEED_FACTOR.get(local_gpu_type, 1.0)
    cloud_speed = GPU_SPEED_FACTOR.get(cloud_gpu_type, 1.0)

    price = pricing.get(cloud_gpu_type)
    price_min = price.min_price_hr if price else FALLBACK_PRICES.get(cloud_gpu_type, 0.30)
    price_max = price.max_price_hr if price else price_min

    rows: list[ScalingRow] = []
    for n_cloud in cloud_gpu_counts:
        # Total effective GPU-hours: 1 local + N cloud
        effective_gpus = local_speed + n_cloud * cloud_speed
        wall_hours = total_work_hours_base / effective_gpus

        # Cost range
        cloud_hours = wall_hours * n_cloud
        cost_lo = cloud_hours * price_min * cost_band[0]
        cost_hi = cloud_hours * price_max * cost_band[1]

        notes = ""
        if n_cloud == 0:
            notes = "local only"

        rows.append(
            ScalingRow(
                cloud_gpus=n_cloud,
                cloud_gpu_type=cloud_gpu_type if n_cloud > 0 else local_gpu_type,
                wall_time_hours=wall_hours,
                cloud_cost_min=cost_lo,
                cloud_cost_max=cost_hi,
                notes=notes,
            )
        )

    return rows
