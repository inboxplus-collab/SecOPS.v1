# LLM Fine-Tuning and Agent Playbook

This playbook links SecOps AI to leading open-source LLM training and agent frameworks. It focuses on reproducible fine-tuning, autonomous evolution loops, and self-improving alignment that avoid conflicts with existing data and configuration.

## Goals for SecOps AI
- **Tight backend integration:** Expose tuned models behind FastAPI without breaking existing routes or embeddings.
- **Reproducibility:** Use versioned configs, seeds, and dataset manifests; never mutate the raw security datasets in place.
- **Safety and alignment:** Apply preference data (self-generated or curated) plus guardrails before exposing new models to the frontend.
- **Observability:** Capture training/eval metrics and agent traces; ship logs to the existing observability stack.

## Recommended frameworks
| Capability | Framework | How it fits | Conflict-avoidance tips |
| --- | --- | --- | --- |
| Fast instruction/adapter fine-tuning with low VRAM | [Unsloth](https://github.com/unslothai/unsloth) | One-click LoRA/QLoRA runs; great for quick SecOps remediation experts. | Pin model + tokenizer versions; store LoRA deltas separately and register them in a new model card instead of overwriting base weights. |
| Lightweight GPT-like pretraining | [nanoGPT](https://github.com/karpathy/nanoGPT) | Small-scale pretraining on synthetic/obfuscated security logs. | Keep a dedicated `data/nanogpt/` split; never mix with production embeddings. |
| Instruction tuning + RLHF/ORPO | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Unified pipelines for SFT, DPO, ORPO, PPO; works with Hugging Face. | Track dataset versions via JSON manifest; export trained checkpoints to `models/llama-factory/<run-id>/` and load via env vars. |
| Educational/custom components | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Reference implementations for bespoke blocks (e.g., custom attention or gating). | Use as a sandbox; keep experimental modules in `experiments/` and behind feature flags before merging into backend serving. |

## Autonomous evolution engine (LLM agents)
An agent = perception + reasoning (LLM) + memory + action. Recommended blueprint:
- **Perception:** Tool outputs (scanner reports, CVE lookups), user prompts, and execution logs are normalized into structured observations.
- **Memory:**
  - Short-term: scratchpad in the conversation state; truncate deterministically.
  - Long-term: vector store of remediation snippets and past actions.
- **Reasoning:** Chain-of-thought and reflection prompts drive planning; critique steps are cached to improve determinism.
- **Action:** Tools for code generation, command execution in sandboxes, and search. Actions are idempotent where possible to avoid state conflicts.
- **Evolution loop:**
  1. Plan → act (tool call or code patch).
  2. Observe runtime/output.
  3. Reflect (self-critique) and score.
  4. Rewrite/continue until success or budget is hit.

### Integration hooks
- **Backend:** Add an `agents` router with tool registries and a vector-backed memory provider. Use pydantic models to validate observations/actions.
- **Frontend:** Stream agent steps over SSE/WebSocket; render scratchpad, plan, and tool outputs without altering existing views.
- **Storage:** Save trajectories to `storage/agent_runs/<run-id>.jsonl` for replay and preference data generation.

## Self-growth mechanisms
### Meta-rewarding (self-improving alignment)
1. **Actor:** Generate N responses per prompt.
2. **Judge:** Score for helpfulness, safety, and coherence.
3. **Meta-judge:** Critique the judge’s scoring and adjust rewards.
4. **Fine-tune:** Convert comparisons into preference datasets (e.g., for DPO/ORPO) and run via LLaMA-Factory or Unsloth. Keep preference data versioned and immutable.

### Self-correction and reflection
- Intrinsic loop: draft → critique → revise in a single call; expose both critique and final answer to observability.
- External loop: run tools (scanner, search), verify claims, then regenerate with evidence citations.

### Continual learning without conflicts
- Use **frozen base models** plus **adapters** for incremental tasks; merge only after eval passes.
- Maintain **evaluation gates** (toxicity, hallucination, policy alignment) before promotion.
- Archive older adapters instead of deleting to prevent regression and enable rollbacks.

## Delivery checklist (zero-conflict posture)
- [ ] New checkpoints live under `models/<provider>/<run-id>/` with a model card (YAML/Markdown) describing data, hyperparams, and evals.
- [ ] No raw dataset overwrites; use manifests and `data/<provider>/<dataset>-vX` folders.
- [ ] Backend model loading toggled via environment variables; defaults remain unchanged.
- [ ] Add unit/integration tests for agent routes and tool safety rails before enabling by default.
- [ ] Document frontend changes and avoid breaking existing API contracts when streaming agent events.

## Example repo mapping
- **Training jobs:** `infra/` can host Dockerfiles for Unsloth or LLaMA-Factory; mount data read-only and output to `models/`.
- **Serving:** Backend loads adapters via Hugging Face `peft` when env vars point to a run-id; fallback to the current base model otherwise.
- **Agent runs:** Persist traces and use them to bootstrap preference data for future alignment cycles.

Use this playbook as the canonical integration guide so that new fine-tuning or agent work complements existing SecOps capabilities without configuration or data conflicts.
