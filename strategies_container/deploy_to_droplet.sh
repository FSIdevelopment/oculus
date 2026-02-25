#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Deploy the Strategies Container Build Agent to the
# signalSynk-Strategies Droplet.
#
# Run from the project root:
#   bash strategies_container/deploy_to_droplet.sh
#
# Required env vars (set in .env or export before running):
#   DROPLET_IP          - Droplet public IP address
#
# Optional env vars:
#   SSH_KEY_PATH        - Path to SSH private key (default: ~/.ssh/id_rsa)
#   SSH_USER            - SSH user on Droplet (default: root)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load .env from project root ───────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ── Config ────────────────────────────────────────────────────
DROPLET_IP="${DROPLET_IP:-}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
SSH_USER="${SSH_USER:-root}"
REMOTE_TMP="/tmp/build-agent-deploy"
AGENT_DIR="/opt/strategies-build-agent"

# ── Validation ────────────────────────────────────────────────
if [ -z "$DROPLET_IP" ]; then
    echo "ERROR: DROPLET_IP is not set."
    echo "  Add DROPLET_IP=<your-droplet-ip> to your .env file, or:"
    echo "  DROPLET_IP=x.x.x.x bash strategies_container/deploy_to_droplet.sh"
    exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
if [ -f "$SSH_KEY_PATH" ]; then
    SSH_OPTS+=(-i "$SSH_KEY_PATH")
fi

SSH_TARGET="$SSH_USER@$DROPLET_IP"

echo "═══════════════════════════════════════════════════"
echo "  Deploying Strategies Container Build Agent"
echo "  Target: $SSH_TARGET"
echo "  Agent will run at: http://$DROPLET_IP:8088"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Verify SSH connectivity ──────────────────────────
echo "[1/4] Verifying SSH connection..."
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "echo 'SSH OK'" || {
    echo "ERROR: Cannot SSH to $SSH_TARGET"
    echo "  Check DROPLET_IP and that your SSH key is authorised on the Droplet."
    exit 1
}

# ── Step 2: Copy agent files to Droplet ──────────────────────
echo "[2/4] Copying agent files to Droplet..."
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p $REMOTE_TMP"

scp "${SSH_OPTS[@]}" \
    "$SCRIPT_DIR/main.py" \
    "$SCRIPT_DIR/builder.py" \
    "$SCRIPT_DIR/requirements.txt" \
    "$SCRIPT_DIR/deploy.sh" \
    "$SSH_TARGET:$REMOTE_TMP/"

echo "      Files copied: main.py  builder.py  requirements.txt  deploy.sh"

# ── Step 3: Run setup script on Droplet ──────────────────────
echo "[3/4] Running deploy.sh on Droplet..."
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd $REMOTE_TMP && bash deploy.sh"

# ── Step 4: Verify health endpoint ───────────────────────────
echo ""
echo "[4/4] Verifying build agent health..."
sleep 2  # give uvicorn a moment to start

HEALTH=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "curl -sf http://localhost:8088/health || echo 'UNREACHABLE'")

if echo "$HEALTH" | grep -q '"ok"'; then
    echo "      Health check: OK ✓"
else
    echo "      WARNING: Health check did not return ok — service may still be starting."
    echo "      Response: $HEALTH"
    echo "      Check with: ssh $SSH_TARGET 'systemctl status strategies-build-agent'"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  IMPORTANT: If this is your first deploy, set real"
echo "  credentials on the Droplet and restart the service:"
echo ""
echo "    ssh $SSH_TARGET"
echo "    nano $AGENT_DIR/.env"
echo "    systemctl restart strategies-build-agent"
echo ""
echo "  Then add these to your App Platform env vars:"
echo "    BUILD_AGENT_URL=http://$DROPLET_IP:8088"
echo "    BUILD_AGENT_API_KEY=<the key in $AGENT_DIR/.env>"
echo "═══════════════════════════════════════════════════"
