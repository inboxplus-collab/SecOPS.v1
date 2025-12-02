# RAG and Agentic Integration Blueprint

This document outlines how to extend SecOps AI with multi-LLM retrieval-augmented generation (RAG) and agentic workflow frameworks while keeping the existing data contracts intact. The goal is to reuse current embeddings, metadata, and orchestration code without introducing breaking changes.

## Principles for compatibility
- **Preserve data schemas:** Keep existing vector store schemas, metadata keys, and retrieval interfaces stable. Add adapters instead of changing schemas when integrating new frameworks.
- **Layered abstraction:** Route all new integrations through the existing backend service layer to avoid frontend or infra churn.
- **Isolated configuration:** Use feature flags and environment variables to toggle frameworks (e.g., `RAG_PROVIDER=langchain|llama_index|graphrag`). Default to the current provider to avoid surprises.
- **Model selection parity:** Map existing model aliases to new framework model identifiers so prompts and tools continue to work.

## Multi-LLM RAG systems
- **LangChain (`langchain-ai/langchain`):**
  - Reuse current document loaders and embeddings by wrapping them with `BaseRetriever` implementations.
  - Connect to the existing vector store via the standard `VectorStore` interface and expose the chain through a FastAPI route (e.g., `/rag/langchain/query`).
  - Keep prompt templates in the backend so the frontend payloads remain unchanged.
- **LlamaIndex (`run-llama/llama_index`):**
  - Build an `IndexStruct` from the current corpus using the same embedding function; persist to the shared storage location already used by the RAG cache.
  - Provide a FastAPI endpoint (e.g., `/rag/llama-index/query`) that maps the request DTO to an `index.as_query_engine()` call.
  - Align response schema with the existing RAG response shape to avoid breaking consumers.
- **GraphRAG (`microsoft/graphrag`):**
  - Generate graphs from the current knowledge base using the existing document metadata for nodes and edges to maintain referential integrity.
  - Serve queries via a backend handler that flattens graph responses into the existing answer + citations format.
  - Gate graph building with an environment flag to avoid long startup times in environments that do not need graph mode.

## AI agentic workflows
- **AutoGen (`microsoft/autogen`):**
  - Implement agents as pluggable tools within the backend, sharing the same logging and tracing utilities used by current pipelines.
  - Enforce existing permission and audit hooks when agents execute external actions.
- **LangGraph (`langchain-ai/langgraph`):**
  - Model current multi-step workflows as graphs; wrap each node with existing service functions to preserve side-effect behavior.
  - Version graph definitions and register them via a dispatcher so older flows remain available.
- **MetaGPT (`geekan/metagpt`):**
  - Constrain role definitions to mirror current capabilities (e.g., scanning, dependency analysis) and reuse the established task queue for coordination.
  - Map MetaGPT outputs back into the existing task result schema before returning to clients.
- **CrewAI (`crewAIInc/crewAI`):**
  - Use Crews to orchestrate cooperative tasks while preserving the existing tool catalog; expose them behind new API routes such as `/agents/crew/{crew_name}`.
  - Keep message histories in the current persistence layer to avoid format drift.

## Rollout checklist
1. Add provider selection flags and default values in the backend configuration module.
2. Implement adapter classes for each framework that conform to the current RAG and agent interfaces.
3. Add FastAPI routers that expose the new providers without changing request/response DTOs.
4. Extend observability (metrics, traces, logs) using the existing instrumentation decorators.
5. Write smoke tests that call each provider behind a feature flag to verify schema compatibility.
6. Document operational runbooks for each provider and keep them in sync with the `.env.example` defaults.
