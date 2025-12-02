# SecOPS.v1

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
