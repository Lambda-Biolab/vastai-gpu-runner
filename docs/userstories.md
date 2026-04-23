# User stories

Three personas drive the near-term roadmap. Each story uses `As a <persona>, I want <capability> so that <outcome>` and is tagged with the roadmap item it maps to — `[R1]` `LocalRunner`, `[R2]` `RunPodRunner`, `[R3]` inference helper, `[future]` deferred.

See `roadmap.md` for item scope and `vision.md` for non-goals.

## Persona 1 — Batch-ML researcher (end user)

Runs workloads on rented GPUs. Cares about cost, crash recovery, and getting results into R2 without babysitting. Does not want to know which cloud is cheapest this week — wants the framework to choose.

- As a researcher, I want to switch my batch from Vast.ai to RunPod by changing one import so that I can follow availability and price without rewriting workers. `[R2]`
- As a researcher, I want `vastai-gpu-runner check` to verify *all* configured providers' credentials so that I know before launch which backends are usable. `[R2]`
- As a researcher, I want cost estimates to work across providers so that I can compare before committing. `[future]`
- As a researcher, I want my worker to call a hosted LLM (Groq, Cerebras) via a single import so that I don't provision a GPU for an LLM call. `[R3]`
- As a researcher, I want to keep my existing `BatchState` and R2 results format when I change provider so that historical runs remain comparable. `[R2]`
- As a researcher, I want crash recovery to behave identically regardless of provider so that orchestrator restarts don't double-bill. `[R2]`
- As a researcher, I want the `allowed_images` ownership guard on every provider so that cleanup can't destroy an unrelated run. `[R2]`
- As a researcher, I want to dry-run a batch locally before launching 100 cloud instances so that I catch worker bugs for free. `[R1]`

## Persona 2 — Framework extender (contributor)

Wants to add a new provider, swap the storage backend, or integrate a new inference API. Cares about ABC clarity, test ergonomics, and being able to iterate without cloud credentials.

- As a contributor, I want a reference non-trivial `CloudRunner` besides Vast.ai so that I have a pattern to copy when adding a third provider. `[R2]`
- As a contributor, I want to develop my new provider against `LocalRunner`'s test harness so that I can assert lifecycle invariants without signing up for an account. `[R1]`
- As a contributor, I want a documented mapping of each ABC method to its `LocalRunner` override so that I know which methods matter and which can no-op. `[R1]`
- As a contributor, I want to add a new inference provider by adding one enum member and a base URL so that parity with Groq/Cerebras is trivial. `[R3]`
- As a contributor, I want ABC-level tests that all providers must pass so that I catch contract drift when I subclass. `[R2]`
- As a contributor, I want the serverless-GPU lane (Modal, Beam, Replicate) to be documented as a *separate* ABC so that I don't try to force-fit it into `CloudRunner`. `[future]`
- As a contributor, I want clear credentials-loading conventions so that my new provider fits alongside Vast.ai, RunPod, and R2 without bespoke config. `[R2]`

## Persona 3 — CI/local developer

Iterates on worker logic, fixes bugs, writes tests. Does not want GPU cost, does not want network calls, does not want to wait for an instance to boot.

- As a CI user, I want to run the full `run_full_cycle` path against `LocalRunner` so that PRs can validate workers without cloud access. `[R1]`
- As a developer, I want `LocalRunner` to skip the GPU health check gracefully on machines without CUDA so that CI runners pass. `[R1]`
- As a developer, I want `destroy_instance` on `LocalRunner` to be a no-op so that I don't leak state when a test fails. `[R1]`
- As a developer, I want `deploy_files` on `LocalRunner` to copy into a tempdir so that parallel test runs don't collide. `[R1]`
- As a developer, I want `launch_worker` to return immediately with a PID I can poll so that my test loop mirrors the cloud path exactly. `[R1]`
- As a developer, I want inference helper tests to run fully offline against mocked HTTP so that CI works without a Groq key. `[R3]`
- As a developer, I want to inspect the workspace directory after a local run so that I can debug worker output without SSH. `[R1]`
- As a developer, I want a Docker-based `LocalRunner` variant so that I can validate the container contract end-to-end. `[future]`

## Future (unmapped)

Stories that don't map to a committed roadmap item. Listed for visibility; not planned.

- As a cost-conscious user, I want cross-provider cost estimation so that I can bin-pack workloads by current spot price. `[future]`
- As an operator, I want a TUI showing active instances across all providers so that I can triage zombies centrally. `[future]`
- As a user with sensitive data, I want an e2b or Daytona sandbox provider so that I can run untrusted worker scripts with stronger isolation than a local subprocess. `[future]`
- As a model-ops user, I want a `ServerlessRunner` ABC so that Modal / Beam / Replicate can share an orchestrator path distinct from `CloudRunner`. `[future]`
