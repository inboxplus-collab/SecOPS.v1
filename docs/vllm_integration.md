# vLLM Integration Guide

This repository can pair the SecOPS RAG/agent stack with [vLLM](https://github.com/vllm-project/vllm) to accelerate inference and reduce GPU memory pressure. vLLM's PagedAttention kernel and continuous batching make it well suited for multi-tenant workloads where the backend receives many concurrent chat or scoring requests.

## Why vLLM for SecOPS
- **High throughput:** PagedAttention and continuous batching allow the engine to reuse KV cache blocks efficiently, delivering strong tokens-per-second rates even as the queue grows.
- **Memory efficiency:** The paged KV cache minimizes fragmentation and helps large-context prompts fit on commodity GPUs.
- **OpenAI-compatible serving:** The `vllm serve` / `vllm openai` entrypoints expose OpenAI-style chat/completions endpoints, so the backend can talk to vLLM with the existing OpenAI client wrapper.
- **Plugin support:** vLLM supports tensor parallelism, quantized weights, and LoRA adapters, letting SecOPS serve fine-tuned artifacts without changing application code.

## Running a vLLM endpoint
The upstream Docker image ships with an OpenAI-compatible server. A minimal single-GPU launch looks like:

```bash
# Example: host a quantized Llama 3 model with OpenAI-compatible routes
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0 \
  docker run --gpus all --rm -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --enforce-eager
```

Key flags:
- `--model`: HF Hub path or local checkpoint directory.
- `--tensor-parallel-size`: number of GPUs used for tensor parallelism.
- `--max-model-len`: context length; keep aligned with the backend prompt windows.
- `--enforce-eager`: keeps kernels in eager mode for easier debugging (remove for peak throughput).

## Wiring the SecOPS backend to vLLM
1. Set the backend to use the OpenAI transport that points at the vLLM server:
   - `OPENAI_API_BASE=http://localhost:8001/v1`
   - `OPENAI_API_KEY=dummy` (vLLM does not enforce keys by default; keep consistent with backend expectations).
2. Confirm the backend RAG components rely on the OpenAI-compatible client (see `backend/src/rag` and `backend/src/integrations`). No code changes are required as long as the base URL is set.
3. For production, place the vLLM service behind an internal load balancer and add authentication middleware (e.g., API gateway key check or mTLS).

## Deployment patterns
- **Docker Compose extension:** Add a `vllm` service alongside `backend` and `frontend`, exposing port `8001` and mounting a local model cache directory for faster cold starts.
- **Kubernetes:** Run the upstream `vllm/vllm-openai` image as a GPU-enabled Deployment. Set resource requests/limits aligned with your GPU type (e.g., `nvidia.com/gpu: 1`) and surface it as a ClusterIP service. Point `OPENAI_API_BASE` in the backend deployment to the in-cluster service DNS.
- **Autoscaling:** Pair the Deployment with an HPA that scales on GPU or request metrics; continuous batching keeps latency stable as replicas increase.

## Fine-tuning and adapters
- Train LoRA adapters using your preferred framework (e.g., PEFT) and mount them into the vLLM container with `--lora-path /models/lora-secops`. The serving layer can then hot-swap adapters without redeploying the backend.
- For full-model fine-tunes, export the checkpoint to a Hugging Face format that vLLM understands and update the `--model` path.
- Keep prompt templates and guardrails in `docs/llm-finetuning-and-agents.md` synchronized with any adapter-specific behaviors.

## Observability and safety
- Enable request/response logging at the vLLM layer to feed SecOPS audit trails.
- Use low-level metrics (KV cache usage, batch queue depth, tokens/sec) to size GPUs and trigger autoscaling.
- Combine vLLM's streaming responses with the frontend's event-driven UI to give analysts immediate feedback during long runs.
