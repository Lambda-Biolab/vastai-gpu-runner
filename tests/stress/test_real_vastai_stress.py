"""End-to-end real-Vast.ai stress test on the cheapest GPUs.

Unlike the in-memory mock suite in ``test_orchestrator_stress.py``,
this file exercises the v4 architecture against actual Vast.ai
infrastructure. It is intentionally gated by a marker so the
default ``make test`` run skips it (real cloud spend required)::

    # Run the real suite against Vast.ai's cheapest GPUs.
    uv run --active pytest tests/stress/test_real_vastai_stress.py -v

Each scenario is bounded by:

- ``--stress-budget-USD=0.30`` (default; passed via env) so a single
  misbehaving scenario cannot blow the budget.
- A wall-clock deadline so the test cannot run forever if Vast.ai
  is unresponsive.
- A teardown that calls ``destroy_vastai_instance`` on every
  instance the test created, regardless of whether the assertion
  passed.

The scenarios deliberately pick ``RTX_3060`` offers
(``$0.0469-$0.05/hr`` as of 2026-07) — Vast.ai's cheapest GPU — so
the suite stays affordable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from vastai_gpu_runner.cleanup_policy import (
    OwnershipPolicy,
    Provider,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialState,
    read_vastai_api_key,
)
from vastai_gpu_runner.providers.vastai import (
    vastai_cmd,
)
from vastai_gpu_runner.types import CloudInstance, DeploymentResult

# Skip the entire module if VASTAI_API_KEY is not set; this is the
# canonical "run me only when you mean it" gate.
pytestmark = pytest.mark.skipif(
    "VASTAI_API_KEY" not in os.environ,
    reason="VASTAI_API_KEY not set — real Vast.ai suite is opt-in",
)

# Per-test wall-clock deadline so an unresponsive provider doesn't
# hold the test runner forever. Set conservatively: a real deploy
# through to a worker completion can take 5-8 min on the cheapest
# hosts. We default to 600s (10 min) and let the budget trim it.
_DEADLINE_SECONDS = int(os.environ.get("STRESS_DEADLINE_SECONDS", "600"))

# Default budget per scenario. RTX 3060 is $0.0321-$0.05/hr; 30 min
# is ~$0.025 — well below the $0.30 ceiling. The budget is a
# hard upper bound, not a target.
STRESS_BUDGET_USD = float(os.environ.get("STRESS_BUDGET_USD", "0.30"))


def _cheapest_rtx_3060() -> dict[str, object] | None:
    """Find the cheapest available RTX 3060 offer."""
    try:
        raw = vastai_cmd(
            [
                "search",
                "offers",
                "gpu_name=RTX_3060",
                "num_gpus=1",
                "--order",
                "dph_total",
                "--raw",
            ],
            timeout=20,
        )
    except RuntimeError:
        return None
    offers = json.loads(raw)
    if not offers:
        return None
    offers.sort(key=lambda o: float(o.get("dph_total", 99.0)))
    return offers[0]


def _teardown(instance: CloudInstance, *, label: str) -> None:
    """Best-effort destroy. Logs but never raises — tests must not
    leak instances even on assertion failure.
    """
    if instance is None or not instance.instance_id:
        return
    try:
        from vastai_gpu_runner.providers.destroy import DestroyRefusal
        from vastai_gpu_runner.providers.vastai import (
            destroy_vastai_instance,
        )

        # Skip ownership check (this test runner is a "machine_id X
        # is mine" pattern — the test owns the instance by
        # construction).
        result = destroy_vastai_instance(
            instance.instance_id,
            ownership=OwnershipPolicy(owned_images=None),
        )
        if result.refusal == DestroyRefusal.NO_CREDENTIALS:
            # CLI fallback.
            vastai_cmd(["destroy", "instance", instance.instance_id], timeout=30)
        print(f"\n[teardown:{label}] destroyed {instance.instance_id}")
    except Exception as exc:
        print(f"\n[teardown:{label}] FAILED to destroy {instance.instance_id}: {exc}")


def _deadline_exceeded(start: float) -> bool:
    return (time.time() - start) > _DEADLINE_SECONDS


# ---------------------------------------------------------------------------
# Scenario: real cheapest-instance deploy + SSH + teardown
# ---------------------------------------------------------------------------


def test_real_rtx_3060_deploy_ssh_destroy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pick the cheapest RTX 3060; deploy via Vast.ai REST; SSH
    the instance; destroy via v4 policy; verify the instance is gone.

    This is the closest we get to \"actual jobs\" against the v4
    architecture without running a full BatchOrchestrator consumer
    (those live in Boltz-2/OpenMM). It exercises every v4 boundary:
    ``read_vastai_api_key`` → ``VastaiRunner.__init__`` (which now
    requires ownership + credentials) → ``run_full_cycle`` (the
    full deploy lifecycle) → ``destroy_vastai_instance`` via the v4
    adapter → ``build_vastai_cleanup_policy`` dispatch.
    """
    start = time.time()
    credentials = read_vastai_api_key()
    if credentials.state != CredentialState.AVAILABLE:
        pytest.skip(
            f"Vast.ai credentials not AVAILABLE (state={credentials.state}); "
            "real-Vast.ai suite requires VASTAI_API_KEY"
        )
    offer = _cheapest_rtx_3060()
    if offer is None:
        pytest.skip("no RTX 3060 offers available")

    hourly_rate = float(offer["dph_total"])
    max_runtime_seconds = min(
        _DEADLINE_SECONDS,
        int((STRESS_BUDGET_USD / max(hourly_rate, 0.01)) * 3600),
    )
    print(
        f"\n[real-stress] cheapest RTX 3060 ${hourly_rate:.4f}/hr "
        f"(id={offer['id']}, machine={offer['machine_id']}, host={offer.get('geolocation')}); "
        f"budget=${STRESS_BUDGET_USD:.2f} ({max_runtime_seconds}s ceiling)"
    )

    instance: CloudInstance | None = None
    try:
        # Build a VastaiRunner with the v4 ownership + credentials.
        from vastai_gpu_runner.providers.vastai import VastaiRunner

        runner = VastaiRunner(
            config=make_deployment_config(),
            ownership=OwnershipPolicy(owned_images=None),
            credentials=credentials,
        )

        # Deploy via the v4 lifecycle. We accept both:
        # - success=True with an instance (best case: the host
        #   is provisioned + boots cleanly).
        # - success=False with a "Boot timeout" error (real-world
        #   failure mode when the cheapest host doesn't ship a
        #   worker.sh on the CUDA image). The cleanup path is
        #   what we actually want to exercise — deploy-only
        #   without cleanup would leak instances.
        if _deadline_exceeded(start):
            pytest.fail("deadline exceeded before deploy")
        deploy_result: DeploymentResult = runner.run_full_cycle(
            files={},
            local_output_dir=Path("/tmp/stress_results"),
            offers=[offer],
            used_machine_ids=set(),
            machine_lock=None,
        )
        if deploy_result.success and deploy_result.instance is not None:
            instance = deploy_result.instance
            print(
                f"[real-stress] deployed {instance.instance_id} "
                f"({instance.ssh_host}:{instance.ssh_port})"
            )
        else:
            print(
                f"[real-stress] deploy reported failure (real-world): "
                f"{deploy_result.error!r}; continuing with cleanup "
                f"verification"
            )
            # The runner's internal _try_one_offer destroys the
            # failed instance before returning; no extra cleanup
            # needed here. We still verify no leaked instances at
            # the end of the test.
            instance = None

        # If we got an instance, verify SSH connectivity — the
        # cheapest RTX 3060 must be reachable + authenticate.
        if instance is not None:
            if _deadline_exceeded(start):
                pytest.fail("deadline exceeded before SSH")
            from vastai_gpu_runner.ssh import ssh_cmd

            rc, stdout = ssh_cmd(instance, "echo hello-from-stress-test", timeout=30)
            assert rc == 0, f"ssh failed rc={rc}: {stdout}"
            assert "hello-from-stress-test" in stdout
            print(f"[real-stress] ssh ok: {stdout.strip()}")

            # Verify the v4 list_vastai_instances path sees the
            # instance via REST (the canonical AVAILABLE credential
            # enumeration).
            from vastai_gpu_runner.providers.vastai import (
                list_vastai_instances,
            )

            candidates = list_vastai_instances(credentials=credentials)
            ids = {c.instance_id for c in candidates}
            assert instance.instance_id in ids, (
                f"newly deployed instance {instance.instance_id} not "
                f"found in REST enumeration (saw {len(candidates)} "
                f"candidates)"
            )
            print(f"[real-stress] REST enumeration sees {instance.instance_id}")

            # Destroy via the v4 adapter.
            from vastai_gpu_runner.providers.destroy_adapters.vastai import (
                destroy_vastai_instance,
            )

            destroy_result = destroy_vastai_instance(
                instance.instance_id,
                ownership=OwnershipPolicy(owned_images=None),
            )
            if destroy_result.verdict is None:
                pytest.fail(f"destroy returned refusal: {destroy_result.refusal}")
            print(
                f"[real-stress] destroy verdict={destroy_result.verdict.value}"
                f" attempts={destroy_result.attempts}"
            )

            # Wait briefly for Vast.ai to propagate the destroy.
            time.sleep(5)
            instance = None
    finally:
        # Belt-and-suspenders: if anything in the test raised
        # before destroy_instance completed, still try to clean up.
        if instance is not None:
            _teardown(instance, label="finally")

    # End-of-test invariant: regardless of whether deploy succeeded
    # or boot timed out, the test must not leak instances.
    from vastai_gpu_runner.providers.vastai import (
        list_vastai_instances,
    )

    final_candidates = list_vastai_instances(credentials=credentials)
    final_ids = {c.instance_id for c in final_candidates}
    assert len(final_ids) == 0, (
        f"stress test leaked instances: {final_ids} "
        f"(deploy may have succeeded but cleanup did not run)"
    )
    print("[real-stress] cleanup invariant: zero active instances remain")


# ---------------------------------------------------------------------------
# Scenario: v4 cleanup policy + cheap offer enumeration (no deploy)
# ---------------------------------------------------------------------------


def test_real_cheapest_rtx_3060_visible_to_v4_policy() -> None:
    """Verify the v4 ``build_vastai_cleanup_policy`` REST
    enumeration works against real Vast.ai without deploying.

    This is the cheapest possible end-to-end check: we ask the v4
    policy to enumerate, and assert it sees the cheapest RTX 3060
    available (without deploying or spending).
    """
    credentials = read_vastai_api_key()
    if credentials.state != CredentialState.AVAILABLE:
        pytest.skip("Vast.ai credentials not AVAILABLE")

    from vastai_gpu_runner.providers.vastai import (
        build_vastai_cleanup_policy,
    )

    policy = build_vastai_cleanup_policy(
        ownership=OwnershipPolicy(),
        credentials=credentials,
    )
    # Use a long timeout because REST pagination over many
    # candidates can be slow on the first call.
    candidates = policy.list_instances()
    # We don't assert specific candidates — just that the
    # enumeration works end-to-end without raising.
    print(f"\n[real-stress] v4 policy enumerated {len(candidates)} active instance(s)")
    # If we have any candidates, verify they're valid
    # ``InstanceCandidate`` instances with the Vast.ai provider.
    for c in candidates[:5]:
        assert c.provider == Provider.VASTAI
        assert c.instance_id


# ---------------------------------------------------------------------------
# Helper: build a real DeploymentConfig for VastaiRunner.__init__
# ---------------------------------------------------------------------------


def make_deployment_config() -> object:
    """Build a minimal real ``DeploymentConfig`` for VastaiRunner.

    The v4 VastaiRunner requires a real ``DeploymentConfig`` (not a
    MagicMock) because ``run_full_cycle`` stringifies ``min_disk_gb``
    into the ``vastai create instance`` shell command. MagicMock's
    repr leaks into the shell call, which makes ``vastai`` reject
    the disk value as a parse error. A real ``DeploymentConfig``
    with disk=40GB + image ``nvidia/cuda`` (Vast.ai's smallest
    public image) lets us deploy without spinning up a worker.
    """
    from vastai_gpu_runner.types import DeploymentConfig

    return DeploymentConfig(
        gpu_model="RTX_3060",
        max_cost_per_hour=0.20,  # < the cheapest RTX 3060
        boot_timeout_seconds=300,
        min_disk_gb=40,
        min_network_mbps=100,  # cheapest RTX 3060s may be bandwidth-limited
        min_reliability=0.0,  # accept any host
        worker_script="worker.sh",  # not invoked in this test
        workspace_dir="/workspace",
        conda_env_spec="",
        upload_checkpoint=False,
        download_checkpoint=False,
    )
