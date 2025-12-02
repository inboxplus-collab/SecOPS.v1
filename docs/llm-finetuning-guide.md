# LLM Training & Fine-Tuning Integration Guide

This guide summarizes how to integrate popular open-source LLM training stacks into this repository.
Each subsection highlights the upstream project, when to use it, and the minimal steps to align it with our codebase and infrastructure.

## Unsloth (https://github.com/unslothai/unsloth)
- **Best for:** Rapid LoRA fine-tuning on consumer GPUs with aggressive memory optimizations.
- **Setup:**
  1. Create a new Python virtual environment inside `backend` and install `unsloth` plus `transformers` extras: `pip install unsloth[full] accelerate bitsandbytes`.
  2. Add a training entrypoint under `backend/src/training/unsloth_runner.py` (package scaffolded) that wraps `unsloth.train()` with our dataset loader.
     Reuse the shared config pattern from `backend/src/config.py` for paths and hyperparameters.
  3. Log metrics to our existing observability stack (Prometheus/Grafana) via `backend/src/telemetry` to keep dashboards consistent.
- **Data format:** Use Hugging Face `datasets`; for chat-style data, convert to the `messages` format Unsloth expects.

## nanoGPT (https://github.com/karpathy/nanoGPT)
- **Best for:** Educational or small-character-level GPTs where we control the entire training loop.
- **Setup:**
  1. Vendor the `data/` preprocessing scripts into `backend/scripts/nanogpt/` to keep them isolated from production code.
  2. Add a thin wrapper `backend/src/training/nanogpt_runner.py` that parses our YAML config and invokes nanoGPT's `train.py` with the generated config file.
  3. Mount our object storage bucket via fstab or s3fs in training nodes so checkpoints land in the `/data/checkpoints/nanogpt/` prefix used by the rest of the platform.
- **Data format:** Plain text files. Add a `make_corpus.py` helper to assemble corpora from our data lake exports.

## LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory)
- **Best for:** Instruction tuning and RLHF-style pipelines with GUI support.
- **Setup:**
  1. Run the FastAPI server in a sidecar container and expose it behind the internal ingress so the frontend can trigger jobs.
  2. Map the factory's `data` and `outputs` directories to our shared NFS path (e.g., `/mnt/training/llama_factory/`).
  3. Implement a controller in `backend/src/training/llama_factory_client.py` that posts job specs to the server and streams logs back to our log collector.
- **Data format:** JSON/JSONL with `instruction`, `input`, `output` fields. Align with our existing labeling schema in `backend/src/datasets/schemas.py`.

## LLMs From Scratch (https://github.com/rasbt/LLMs-from-scratch)
- **Best for:** Research prototypes and educational deep dives where we need maximal transparency.
- **Setup:**
  1. Keep experiments under `research/notebooks/` and check large artifacts into object storage instead of Git.
  2. Reuse the math/attention modules as reference implementations for unit tests in `backend/tests/` (e.g., gradient checks).
  3. Add a `Makefile` target `make scratch-run` that executes the main training script with smaller configs for CI smoke tests.
- **Data format:** Flexible; prefer the same tokenization pipeline used in production to keep comparability.

## Cross-cutting integration notes
- Use environment variables for cloud credentials and data paths; avoid hardcoding secrets.
- Standardize config via YAML files stored in `backend/config/training/`.
  Each runner should accept a `--config` path and produce metrics to a common JSON log schema.
- Keep Python dependencies optional by grouping extras in `backend/pyproject.toml` (e.g., `[project.optional-dependencies]` sections for `unsloth`, `nanogpt`, `llama-factory`).
- Document hardware requirements per stack in `backend/docs/hardware.md` (to be created) to help platform ops schedule jobs.

## Quick-start matrix
| Stack | Optimal Use Case | GPU Profile | Time to First Run |
| --- | --- | --- | --- |
| Unsloth | Fast LoRA fine-tunes | 1× consumer (>=12GB) | ~10 min |
| nanoGPT | Small language models | 1× midrange | ~15 min |
| LLaMA-Factory | Instruction/RLHF | 2× A100/consumer with offloading | ~20 min |
| LLMs from Scratch | Research/education | CPU/GPU | ~20 min |

## Next steps
1. Select the runner based on model size and latency requirements.
2. Add dataset loaders under `backend/src/datasets/` that emit the formats outlined above.
3. Implement CI smoke tests for each runner (config validation + 1 batch dry run).
4. Update platform documentation once the runners are wired into orchestration.
