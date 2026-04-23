# Architecture v2 (target)

This doc describes the **target architecture** after roadmap items 1–3 land. For the current-state architecture (today's code) see `architecture.md`. For scope and sequencing see `roadmap.md`.

## What changes vs v1

In one paragraph: `CloudRunner` stays as the single ABC for SSH-lifecycle providers and gains two siblings — `LocalRunner` (subprocess) and `RunPodRunner`. A new `vastai_gpu_runner.inference` module appears alongside `storage/` as a worker-side capability, not a runner. Serverless-GPU providers (Modal, Beam, Replicate) are explicitly out of scope for `CloudRunner` and flagged as needing a second ABC — deferred.

Diff vs v1:

- **+** `providers/local.py` (`LocalRunner`)
- **+** `providers/runpod.py` (`RunPodRunner`)
- **+** `inference/` package (`client.py`, `providers.py`)
- **+** `Provider.LOCAL`, `Provider.RUNPOD` enum members
- **~** `cli.py check` iterates configured providers instead of assuming Vast.ai
- **—** No changes to `runner.py` ABC signature
- **—** No changes to `state.py`, `batch.py`, `storage/r2.py`, `worker/base.py`

## Provider taxonomy

Three lanes. Only Lane A is served by `CloudRunner`.

### Lane A — SSH-lifecycle providers

Full `CloudRunner` ABC implementation. Lifecycle: search → create → wait → deploy → launch → poll → download → destroy.

| Provider | Module | Status |
|---|---|---|
| Vast.ai | `providers/vastai.py` | Shipped |
| Local (subprocess) | `providers/local.py` | Roadmap item 1 |
| RunPod | `providers/runpod.py` | Roadmap item 2 |
| TensorDock / Lambda Labs / CoreWeave / Crusoe / Paperspace | — | Deferred |

### Lane B — Serverless-GPU providers (deferred)

Providers whose primitive is "invoke a function on a GPU", not "provision a VM". Examples: Modal, Beam.cloud, Replicate, Baseten, Fal.ai.

The `CloudRunner` lifecycle does not map: there is no SSH, no `create_instance`, no `destroy_instance` — instead there is `deploy function → call function → receive result`. Shoehorning them into `CloudRunner` would either degrade the abstraction or introduce no-op methods that hide real lifecycle differences.

**Decision**: a second ABC (provisionally `ServerlessRunner`) is the right answer. **Deferred** until a concrete user asks for it.

### Lane C — Inference APIs

Not runners. HTTP endpoints that serve models: Groq, Cerebras, OpenRouter (committed); HuggingFace, NVIDIA NIM, Fireworks, Together, DeepInfra, Novita, Lepton, GitHub Models (deferred).

Lives in `vastai_gpu_runner.inference` and is **imported by workers**, not orchestrated by the framework. The framework never calls inference on the user's behalf.

## Layered design (v2)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Deploy/poll/collect many units
├─────────────────────────────────────────────────┤
│  Orchestrator utils (orchestrator.py)           │  Zombie sweep, detach, budget
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
├──────────────┴──────────────┴───────────────────┤
│  SSH (ssh.py)      — used by Vast.ai, RunPod    │
│  subprocess        — used by Local              │
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
│    └── imports inference/ (Lane C, optional)    │
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py)                        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘

Lane B (ServerlessRunner ABC) — deferred, not drawn.
```

## `LocalRunner` shape

Each ABC method's local-subprocess override. All run on the host; no SSH, no network.

| ABC method | `LocalRunner` behavior |
|---|---|
| `search_offers` | Return `[{"machine_id": "local", "dph_total": 0.0}]` (single synthetic offer, price 0) |
| `create_instance` | Return `CloudInstance(provider=Provider.LOCAL, instance_id="local", ssh_host="localhost", ...)`; allocate a tempdir workspace |
| `wait_for_boot` | Return `True` immediately |
| `verify_gpu` | `subprocess.run(["nvidia-smi"])`; log and return `True` on failure (CI-friendly) |
| `deploy_files` | `shutil.copy` each source into the tempdir workspace |
| `setup_environment` | Return `True` (no-op) |
| `launch_worker` | `subprocess.Popen(["bash", "worker.sh"], cwd=workspace)`; persist PID |
| `check_progress` | Poll PID liveness + presence of a `DONE` marker in the workspace |
| `list_remote_files` | `os.listdir(workspace)` |
| `download_file` | `shutil.copy(workspace/name, local_path)` |
| `destroy_instance` | Kill process if alive; remove workspace; return `True` |
| `capture_deploy_failure_diagnostics` | Inherited no-op (nothing to capture) |

The `CloudInstance.ssh_host`/`ssh_port`/`ssh_user` fields remain at defaults; nothing in `LocalRunner` invokes `ssh.py`. No ABC change.

## `RunPodRunner` shape

Each ABC method's RunPod backing call.

| ABC method | `RunPodRunner` behavior |
|---|---|
| `search_offers` | Query RunPod GPU-types endpoint; filter by `gpu_model` and `max_cost_per_hour` |
| `create_instance` | POST to pods endpoint with Docker image, env, GPU type; return `CloudInstance` with returned SSH host/port |
| `wait_for_boot` | Poll pod status until `RUNNING`, then retry SSH until responsive (reuse existing `ssh.py` pattern) |
| `verify_gpu` | `ssh_cmd(nvidia-smi)` via `ssh.py` |
| `deploy_files` | `scp_upload` from `ssh.py` |
| `setup_environment` | Optional image-specific setup (same hook Vast.ai uses) |
| `launch_worker` | `ssh_cmd("bash worker.sh &")` |
| `check_progress` | R2-first poll, SSH fallback (mirror `VastaiRunner`) |
| `list_remote_files` | `ssh_cmd("ls /workspace")` |
| `download_file` | `scp_download` |
| `destroy_instance` | Belt-and-suspenders: stop endpoint → terminate endpoint → verify → re-terminate if resurrected. Apply `allowed_images` ownership guard. |
| `capture_deploy_failure_diagnostics` | Fetch pod logs via API; attach to failure record |

Credentials: `RUNPOD_API_KEY` env var or `~/.cloud-credentials`. Image whitelist (`allowed_images`) is ported verbatim from `VastaiRunner`.

## Inference module shape

```
inference/
    __init__.py          # Re-exports InferenceClient, InferenceProvider
    providers.py         # Enum + base URL + auth env var per provider
    client.py            # Thin OpenAI SDK wrapper
```

Minimal public surface:

```python
from vastai_gpu_runner.inference import InferenceClient, InferenceProvider

client = InferenceClient(provider=InferenceProvider.GROQ)
reply = client.chat(
    messages=[{"role": "user", "content": "hello"}],
    model="llama-3.3-70b-versatile",
)
```

Under the hood, each provider entry is `(base_url, auth_env_var)`. Adding a new OpenAI-compatible provider is one line. Non-OpenAI-compatible providers (Replicate, Fal.ai) are out of scope.

Credentials: env var first, `~/.cloud-credentials` second. Same precedence as Vast.ai and R2.

OpenAI SDK is an optional extra: `uv add "vastai-gpu-runner[inference]"`.

## ABC changes required

**None for Lanes A and C.** The research pass confirmed the existing `CloudRunner` ABC hosts `LocalRunner` and `RunPodRunner` without modification. `capture_deploy_failure_diagnostics` is already an optional override. `search_offers` already defaults to `[]`. `destroy_instance` can return `True` for trivially destroyable backends.

Lane B (serverless) is the only case requiring a second ABC. That ABC is **not designed here** — this doc deliberately leaves it unspecified until a concrete need appears.

## Open questions

These are flagged here for resolution as roadmap items land. They are **not** blocking for items 1–3.

### Credentials unification

Vast.ai uses its own CLI config; R2 uses `~/.cloud-credentials`; RunPod will want `RUNPOD_API_KEY`; inference providers want their own keys. A single loader (env-var-first, file-second, per-provider namespace) would be cleaner than each subclass re-inventing it. Design deferred until item 2 lands and we have a concrete second case.

### Cost reporting across providers

The estimator today queries Vast.ai's live marketplace. A multi-provider estimator needs a pluggable pricing interface. Not blocking — the estimator can remain Vast.ai-specific until a user asks to compare.

### Test matrix

How many providers must pass the ABC contract test? Starting answer: all Lane A providers run an identical happy-path test against a recorded fixture, plus each has its own provider-specific integration test gated on credentials.

### Package rename

With `LocalRunner` and `RunPodRunner` in-tree, the package name `vastai-gpu-runner` becomes slightly misleading. Not renaming now — rename churn is expensive and the name is load-bearing for existing users. Revisit after item 2 ships.
