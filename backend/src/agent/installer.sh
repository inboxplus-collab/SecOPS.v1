#!/bin/bash
set -e

echo "[SecOpsAI Installer] Starting installation..."

SERVER_URL="https://api.secops.ai"
INSTALL_DIR="/opt/secopsai"
PYTHON_BIN="$(command -v python3 || command -v python)"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

curl -fsSL "$SERVER_URL/agent/latest" > agent.py
curl -fsSL "$SERVER_URL/agent/config" > agent.conf

"$PYTHON_BIN" -m pip install --upgrade secopsagent 2>/dev/null || \
  pip install secopsagent

echo "[SecOpsAI Installer] Registration…"

NODE_ID=$("$PYTHON_BIN" agent.py register)

echo "NODE_ID=$NODE_ID" >> agent.conf

echo "[SecOpsAI Installer] Creating systemd service..."
cat <<SERVICE >/etc/systemd/system/secopsai.service
[Unit]
Description=SecOpsAI Agent
After=network.target

[Service]
ExecStart=$PYTHON_BIN $INSTALL_DIR/agent.py run
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable secopsai
systemctl start secopsai

echo "Installation complete. This server is now protected by SecOpsAI."
