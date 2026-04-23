# Vision

## Purpose

`vastai-gpu-runner` is a **GPU batch orchestration framework**. It provisions short-lived GPU instances, deploys a worker, collects results into object storage, and destroys the instance — with crash recovery and cost transparency along the way.

It is **not**:

- a long-running service scheduler
- a training framework (PyTorch Lightning, Ray, Accelerate)
- a Kubernetes-native orchestrator
- an inference gateway or model router

The framework owns the **lifecycle of ephemeral GPUs**. Everything the worker does once it has a GPU is the consumer's concern.

## Principles

| Principle | What it means |
|---|---|
| Provider-agnostic | `CloudRunner` ABC is the contract. Vast.ai is one implementation among several. |
| Crash-recoverable | Atomic JSON state. Any process can die at any point and resume. |
| Cost-transparent | Every run has a known price ceiling before it starts. Live pricing, scaling tables. |
| Local-first dev loop | Full lifecycle runnable without a cloud account or a GPU. CI and debug cost $0. |
| No lock-in | Consumer owns state, owns workers, owns storage format. Framework exposes primitives. |

## Non-goals

- **Multi-tenant SaaS.** Single-operator tool; auth is the operator's SSH/API keys.
- **Long-running services.** Workers self-destruct after one job. No reconnect.
- **Kubernetes.** Pods, operators, CRDs are out of scope; `CloudInstance` is deliberately flat.
- **Replacing hosted inference.** When a job is "call a model", the framework exposes an inference helper — it does not provision a GPU for it.

## Three-horizon picture

### Today

- One provider: Vast.ai via `VastaiRunner`.
- One storage backend: R2 via `R2Sink`.
- Workers run SSH + Docker.
- Local dev requires a Vast.ai account.

### Near-term (committed — see `roadmap.md`)

1. **`LocalRunner`** — subprocess backend for CI and offline dev. Zero cost.
2. **`RunPodRunner`** — second cloud backend. Validates the ABC is actually provider-agnostic.
3. **`vastai_gpu_runner.inference` helper** — OpenAI-compatible client (Groq / Cerebras / OpenRouter) callable *from inside* workers. Not a runner.

### Exploratory (deferred)

- Additional SSH-lifecycle providers: TensorDock, Lambda Labs, CoreWeave.
- Sandbox backends (e2b, Daytona) for CPU-only preflight.
- Docker-based `LocalRunner` variant for container-contract parity.
- A **second ABC** for serverless-GPU providers (Modal, Beam, Replicate) whose lifecycle does not map onto `CloudRunner`.
- Hosted endpoints: HuggingFace Inference Endpoints, NVIDIA NIM.

Deferred items are documented, not scheduled. They ship when a concrete user need surfaces.

## Success criteria

The framework is succeeding when:

- A new contributor can add a provider by subclassing `CloudRunner` and writing <500 lines.
- A user can run a worker locally with zero cloud credentials.
- A user can switch from Vast.ai to RunPod by changing one import.
- Cost estimates before a run match actual billing within 10%.
- A crashed orchestrator resumes without losing work or double-billing.
