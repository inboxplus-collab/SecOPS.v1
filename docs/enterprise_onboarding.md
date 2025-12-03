# SecOpsAI Enterprise Onboarding Guide

## 1️⃣ Overview

SecOpsAI installs into customer environments to provide:
- Continuous vulnerability detection
- AI-guided remediation
- Autonomous infrastructure healing
- CI/CD failure diagnosis
- RAG-based explanations for issues
- Secure multi-tenant access

## 2️⃣ Installation Requirements

**Server Requirements**
- 2 vCPU, 4GB RAM (minimum)
- Docker or Kubernetes
- Outbound HTTPS allowed

**Optional Integrations**
- GitHub / GitLab
- Kubernetes Clusters
- Jenkins / GitHub Actions
- Supabase / Postgres
- AWS / GCP / Azure

## 3️⃣ Deploy Options

### A. Docker Compose (simple)

Run:

```bash
docker compose -f docker-compose.prod.yml up -d
```

Visit:

```
https://yourdomain.com
```

### B. Kubernetes (Helm Chart)

```bash
helm install secopsai ./helm/secopsai
```

## 4️⃣ Registering Servers / Agents

One-line installer:

```bash
curl -sL https://api.secops.ai/install | bash
```

Agents will appear under:

```
Console → Settings → Agents
```

## 5️⃣ Connecting Systems

**Connect GitHub**

Console → Integrations → GitHub → “Connect”

**Connect Kubernetes**

Provide kubeconfig file

**Connect Databases**

Paste database credentials

SecOpsAI does:
- schema inspection
- data flow risk analysis
- automatic repair suggestions
