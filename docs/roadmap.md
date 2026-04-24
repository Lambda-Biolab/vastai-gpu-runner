# Roadmap

**Scope**: committed near-term work only. Deferred items are listed once at the bottom with a single-line reason each. See `vision.md` for horizon framing and `architecture-v2.md` for the target architecture.

## Committed items

| # | Item | Rationale |
|---|---|---|
| 1 | `LocalRunner` (subprocess MVP) | Zero-cost CI + dev loop; proves `CloudRunner` ABC is genuinely provider-agnostic |
| 2 | `RunPodRunner` | Second cloud backend; closest Vast.ai analogue (SSH + Docker + spot + per-second) |
| 3 | `vastai_gpu_runner.inference` helper | OpenAI-compatible client (Groq / Cerebras / OpenRouter) callable from inside workers |

Order is prescriptive: item 1 first (no external dependency, validates ABC), then item 2 (second provider, proves portability), then item 3 (orthogonal, worker-side capability).

---

## Item 1 — `LocalRunner`

### Scope

- New module `vastai_gpu_runner/providers/local.py`
- Subclass of `CloudRunner` that executes the full lifecycle on the host via `subprocess`
- `search_offers` returns a synthetic single offer `{"machine_id": "local"}`
- `create_instance` returns a `CloudInstance(provider=Provider.LOCAL, ssh_host="localhost", ...)` (fields kept as defaults; no SSH is actually used)
- `deploy_files` uses `shutil.copy` into a tempdir workspace
- `launch_worker` uses `subprocess.Popen(["bash", "worker.sh"], cwd=workspace)`
- `check_progress` polls PID liveness and a local `DONE` marker file
- `destroy_instance` is a no-op returning `True`
- New `Provider.LOCAL` enum member in `types.py`

### Exit criteria

- End-to-end run of an existing worker against `LocalRunner` with no cloud credentials present
- Pytest suite covering each overridden method
- One worked example in `docs/extending.md` or `docs/guide.md` showing a local dry run
- No changes required to the `CloudRunner` ABC

### Out of scope

- GPU passthrough (works if host has NVIDIA + CUDA, otherwise silently CPU)
- Docker-based variant (deferred — see below)
- Concurrency / multi-worker-per-host (single job at a time)

### Dependencies

None. Does not block or depend on items 2 or 3.

---

## Item 2 — `RunPodRunner`

### Scope

- New module `vastai_gpu_runner/providers/runpod.py`
- Subclass of `CloudRunner` backed by the RunPod REST/SDK API
- Offer search via RunPod's GPU type/price query
- Instance creation via their pods endpoint; SSH via returned host+port
- File deployment reuses existing `ssh.py` (`scp_upload` / `scp_download`)
- Destroy uses RunPod's stop + terminate endpoints, mirroring Vast.ai's belt-and-suspenders pattern
- New `Provider.RUNPOD` enum member
- Credentials: `RUNPOD_API_KEY` env var + `~/.cloud-credentials` entry

### Exit criteria

- Feature parity with `VastaiRunner` for the core lifecycle
- Ownership guard ported (refuse to destroy pods running images outside `allowed_images`)
- Pytest suite with REST responses mocked
- One smoke-test run against a real RunPod account
- CLI command `vastai-gpu-runner check` reports RunPod credentials alongside Vast.ai + R2

### Out of scope

- RunPod serverless endpoints (they require a different lifecycle — deferred)
- Network volumes / persistent storage (R2 remains the result sink)
- CLI rename (the CLI remains `vastai-gpu-runner` for now; see open questions in `architecture-v2.md`)

### Dependencies

Should land **after** item 1. `LocalRunner` is the cheapest way to verify new ABC edge cases surface by adding a second backend.

---

## Item 3 — `vastai_gpu_runner.inference` helper

### Scope

- New package `vastai_gpu_runner/inference/`
- `client.py` — thin wrapper around the OpenAI SDK pointed at any OpenAI-compatible endpoint
- `providers.py` — enum + base URL + auth-env-var map for Groq, Cerebras, OpenRouter
- Public surface: `InferenceClient(provider=InferenceProvider.GROQ).chat(messages, model=...)`
- Importable from *inside* a `BaseWorker` — not orchestrated by `CloudRunner`
- Credentials read from env vars (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENROUTER_API_KEY`) or `~/.cloud-credentials`
- Optional: embeddings method for providers that support it (OpenRouter, DeepInfra)

### Exit criteria

- Worked example in `docs/extending.md`: a worker that calls Groq instead of running a local model
- Pytest suite with mocked HTTP responses
- Providers list documented in `api.md`
- No new non-optional dependency (OpenAI SDK added as `[inference]` extra)

### Out of scope

- Streaming responses (add when a consumer needs it)
- Model catalog / routing logic (OpenRouter already does this)
- Non-OpenAI-compatible providers (Replicate, Fal.ai) — deferred
- Orchestration side (the framework does not call inference on the user's behalf)

### Dependencies

None. Orthogonal to items 1 and 2. Can land in parallel.

---

## Deferred (documented, not committed)

Items explored during provider research, not scheduled. Revisit when a user need surfaces.

| Item | Reason for deferral |
|---|---|
| `TensorDockRunner` | Viable (marketplace + SSH), but RunPod covers the need until demand surfaces for a second spot-priced backend |
| `LambdaLabsRunner` | Premium/reliable but no spot instances; revisit when an H100/B200 use case lands |
| `CoreWeaveRunner` | Enterprise-gated onboarding; not self-serve |
| `CrusoeRunner` | Smaller ecosystem; revisit after RunPod validates the ABC |
| `PaperspaceRunner` | Limited regions; acceptable but lower priority than RunPod |
| Docker-based `LocalRunner` | Subprocess variant covers 90% of CI/dev need; Docker adds container-contract parity but also daemon + toolkit friction |
| e2b / Daytona sandbox backend | CPU-only preflight use case is largely absorbed by `LocalRunner` |
| Serverless-GPU providers (Modal, Beam, Replicate, Baseten, Fal.ai) | Lifecycle does not map onto `CloudRunner`; requires a second ABC (see `architecture-v2.md` Lane B) |
| HuggingFace Inference Endpoints | Covered transitively via HF routing; dedicated wrapper deferred |
| NVIDIA NIM (`build.nvidia.com`) | Enterprise pricing model unclear for self-serve users; deferred until a user need is concrete |
| GitHub Models | Thin-wrapper value low; OpenRouter covers the same breadth |
