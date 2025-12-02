# SecOPS.v1

This repository hosts SecOPS.v1. For background on the autonomous evolution engine and LLM agent concepts guiding this work, see the [autonomous evolution engine documentation](docs/autonomous_evolution_engine.md).
SecOPS.v1 is an experimental security automation platform that layers a multi-LLM Retrieval-Augmented Generation (RAG) engine on top of infrastructure, code, and operational data. The goal is to give SecOps teams a single interface for questions such as vulnerability triage, playbook generation, and environment introspection.

## Multi-LLM RAG inspirations

The RAG stack in this repository is designed to stay provider-agnostic and can borrow ideas from several leading open-source projects:

- **LangChain** ([langchain-ai/langchain](https://github.com/langchain-ai/langchain)) for modular chains, tool invocation, and prompt/LLM abstractions.
- **LlamaIndex** ([run-llama/llama_index](https://github.com/run-llama/llama_index)) for ingestion pipelines, retriever composition, and response synthesizers.
- **GraphRAG** ([microsoft/graphrag](https://github.com/microsoft/graphrag)) for graph-aware retrieval, semantic neighborhoods, and multi-hop reasoning.

The intent is not to fork these projects, but to use their patterns to keep the `backend/src/rag` components composable and easy to swap between providers. See `docs/multi_llm_rag.md` for a deeper mapping between upstream ideas and the current engine.

## How to plug frameworks into the backend RAG engine

1. **Locate the orchestration layer.** The public API for the engine lives in `backend/src/rag/AdvancedRAGSystem.py`, which wires together retrieval (`SearchOrchestrator`), synthesis (`RAGSynthesizer`), and citation processing.
2. **Add retrievers or vector stores.** Implement new retriever classes under `backend/src/integrations` and expose them through `SearchOrchestrator` to mimic LangChain or LlamaIndex retriever patterns.
3. **Customize synthesis prompts.** `backend/src/rag/RAGSynthesizer.py` is where LangChain-style chains or LlamaIndex response synthesizers can be slotted in.
4. **Experiment with graph retrieval.** If adding GraphRAG-like context stitching, create graph-aware retrieval helpers in `backend/src/rag` and pass them into the orchestrator.

This setup keeps the project ready to incorporate best practices from multiple frameworks without tightly coupling to any single library.
SecOps AI is an end-to-end DevSecOps co-pilot that pairs a FastAPI backend with a Next.js 14 frontend. It automates security scanning, dependency auditing, CI/CD hardening, and delivers AI-powered remediation guidance.

## Repository layout
- **backend/** – FastAPI service with RAG integrations, platform routing, and observability hooks.
- **frontend/** – Next.js 14 app with TailwindCSS-driven UI for the SecOps console.
- **infra/** – Docker, Kubernetes, and CI/CD assets (including docker-compose for local orchestration).
- **docs/** – Architectural notes such as the transformer/MoE overview.

## Quickstart
### Prerequisites
- Python 3.10+
- Node.js 18+ and npm (or a compatible package manager)
- Docker (optional, for containerized runs)

### Backend (FastAPI)
1. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the API locally:
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend (Next.js)
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the dev server:
   ```bash
   npm run dev
   ```
   The app defaults to http://localhost:3000 and expects the backend at http://localhost:8000.

### Docker Compose
Bring up both services with the provided compose file:
```bash
docker compose -f infra/docker-compose.yaml up --build
```

## Additional resources
- Frontend integration guidance: `frontend/lib/appsSdkUiIntegration.ts`
- Transformer and MoE background: `docs/gpt-architecture.md`
- RAG and agentic integration blueprint: `docs/rag-agent-integration.md`
- Fine-tuning, agent blueprint, and alignment playbook: `docs/llm-finetuning-and-agents.md`
